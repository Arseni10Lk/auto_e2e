"""Differentiable camera projection operators for BEV view fusion.

``BEVViewFusion`` consumes geometry as a *projection operator*
(:class:`CameraProjectionModel`) rather than a fixed ``[B, V, 3, 4]`` matrix:
it maps ego-frame BEV reference points to sampling coordinates on each camera's
model-input image plane, plus a visibility mask. A pinhole ``K @ T`` matrix is
the linear fast path (:class:`PinholeProjection`); native fisheye uses the
non-linear :class:`FThetaProjection` so it keeps its full FOV; a
calibration-free run uses the learnable :class:`PseudoProjection`. All expose
the same :meth:`project_ego_to_image`, so the fusion module never branches on
camera model.

Projection targets the *model-input* image frame, so resize/crop/pad and
rectification are part of the contract via :class:`ImageTransform`; a
projection that is correct on the raw image but not on the resized tensor
would sample the wrong pixels.

Coordinate convention: reference points are in the ego/vehicle frame,
X=forward, Y=left, Z=up, in metres. :meth:`project_ego_to_image` returns pixel
coordinates normalized to ``[0, 1]`` by the model-input width/height, a
visibility mask, the per-point depth/range, and a metadata dict recording the
geometry that produced the result.

Reference:
    - BEVFormer (Li et al., ECCV 2022): spatial cross-attention.
    - NVIDIA PhysicalAI-AV sensor model: native f-theta (fisheye).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import torch

# Geometry labels. Each projection operator carries exactly one so experiment
# metadata records which geometry produced a run.
GEOMETRY_PINHOLE = "pinhole"
GEOMETRY_RECTIFIED_PINHOLE = "rectified_pinhole"
GEOMETRY_FTHETA = "ftheta"
GEOMETRY_PSEUDO = "pseudo"

VALID_GEOMETRY_TYPES = (
    GEOMETRY_PINHOLE,
    GEOMETRY_RECTIFIED_PINHOLE,
    GEOMETRY_FTHETA,
    GEOMETRY_PSEUDO,
)

# Points with camera-frame depth below this are treated as behind the camera.
_DEPTH_EPS = 1e-5


@dataclass(frozen=True)
class ImageTransform:
    """Describes the model-input image frame a projection must target.

    Projected ``uv`` must be expressed in the frame of the tensor that enters
    the backbone, after any resize/crop/pad or fisheye rectification.

    Attributes:
        model_input_size: ``(width, height)`` of the tensor fed to the backbone.
            uv is normalized by this (per-axis), so non-square inputs are handled.
        original_size: ``(width, height)`` of the raw image, if known (metadata).
        rectification: label of any rectification applied to the raw image before
            the backbone (e.g. "ftheta_to_pinhole"), or None. Metadata only;
            the geometry itself lives in the projection operator.
    """

    model_input_size: tuple[int, int]
    original_size: Optional[tuple[int, int]] = None
    rectification: Optional[str] = None

    @classmethod
    def square(cls, size: int) -> "ImageTransform":
        """Convenience for the common square model input (e.g. 256x256 shards)."""
        return cls(model_input_size=(int(size), int(size)))

    @property
    def wh(self) -> tuple[float, float]:
        return float(self.model_input_size[0]), float(self.model_input_size[1])


def _as_image_transform(image_transform) -> ImageTransform:
    """Coerce a bare int/float (square size) into an :class:`ImageTransform`."""
    if isinstance(image_transform, ImageTransform):
        return image_transform
    return ImageTransform.square(int(image_transform))


@dataclass
class ProjectionResult:
    """Output of projecting ego-frame reference points onto camera images.

    Shapes use ``M`` for the flattened reference-point count (``N * num_z`` in
    BEVViewFusion terms) so operators stay agnostic to the BEV grid layout; the
    fusion module reshapes ``M -> (N, num_z)`` afterwards.

    Attributes:
        uv_norm: ``[Bp, V, M, 2]`` pixel coordinates normalized to ``[0, 1]``.
            ``Bp`` is the operator's batch dim (real ``B`` for calibrated
            cameras, ``1`` for the batch-independent pseudo operator, which then
            broadcasts across the batch in the sampling loop).
        valid_mask: ``[Bp, V, M]`` bool — True where the point is in front of the
            camera AND lands within the image bounds.
        depth: ``[Bp, V, M]`` per-point depth/range along the optical axis
            (metres for calibrated cameras). Diagnostics only; not required by
            the sampler.
        metadata: geometry that produced this result (geometry_type,
            model_input_size, rectification).
    """

    uv_norm: torch.Tensor
    valid_mask: torch.Tensor
    depth: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class CameraProjectionModel(Protocol):
    """The BEV-fusion geometry contract: ego points -> image sampling coords.

    Implementations (:class:`PinholeProjection`, :class:`FThetaProjection`,
    :class:`PseudoProjection`) differ only in the camera model; the fusion module
    treats them uniformly.
    """

    geometry_type: str

    @property
    def num_views(self) -> int:
        ...

    def project_ego_to_image(
        self, points_ego: torch.Tensor, image_transform
    ) -> ProjectionResult:
        ...


def _homogenize(points_ego: torch.Tensor) -> torch.Tensor:
    """``[M, 3]`` ego points -> ``[M, 4]`` homogeneous (append a ones column)."""
    if points_ego.shape[-1] == 4:
        return points_ego
    ones = torch.ones(
        *points_ego.shape[:-1], 1, device=points_ego.device, dtype=points_ego.dtype
    )
    return torch.cat([points_ego, ones], dim=-1)


def _finalize_linear(projected, image_transform, geometry_type) -> ProjectionResult:
    """Perspective-divide, per-axis normalize by model-input size, build mask.

    Shared by the pinhole fast path. ``projected`` is ``[Bp, V, M, 3]`` =
    ``[u*d, v*d, d]`` in the model-input pixel frame.
    """
    it = _as_image_transform(image_transform)
    w, h = it.wh
    depth = projected[..., 2]                         # [Bp, V, M]
    valid_depth = depth > _DEPTH_EPS                  # in front of the camera
    depth_safe = depth.clamp(min=_DEPTH_EPS).unsqueeze(-1)  # avoid div-by-zero
    # Divide and normalize in fp32: under fp16 autocast a near-_DEPTH_EPS depth
    # overflows the divide (and its 1/d^2 backward) to inf, and the unnormalized
    # pixel coordinate can exceed fp16 max. The result also stays fp32 because
    # bf16 quantization of [0, 1] is several pixels at 512/1024 resolution.
    uv_f = projected[..., :2].float() / depth_safe.float()
    wh = torch.tensor([w, h], device=uv_f.device, dtype=torch.float32)
    uv_norm = uv_f / wh
    in_bounds = (
        (uv_norm[..., 0] >= 0) & (uv_norm[..., 0] <= 1)
        & (uv_norm[..., 1] >= 0) & (uv_norm[..., 1] <= 1)
    )
    mask = valid_depth & in_bounds
    meta = {"geometry_type": geometry_type, "model_input_size": it.model_input_size,
            "rectification": it.rectification}
    return ProjectionResult(uv_norm=uv_norm, valid_mask=mask, depth=depth, metadata=meta)


class PinholeProjection:
    """Linear ego-to-pixel projection: the ``K @ T`` fast path.

    Construct either from a combined ego-to-pixel matrix (``matrix`` =
    ``intrinsic @ extrinsic``, ``[B, V, 3, 4]``, already scaled to the
    model-input image) or from separate intrinsics ``K`` ``[B, V, 3, 3]`` and an
    ego->camera transform ``T_camera_ego`` ``[B, V, 4, 4]`` via :meth:`from_KT`.
    Keeping K and T separable lets a rectified fisheye be expressed as
    ``PinholeProjection.from_KT(K_rectified, T, geometry_type="rectified_pinhole")``
    instead of pretending a fisheye is a pinhole.
    """

    def __init__(self, matrix, geometry_type: str = GEOMETRY_PINHOLE,
                 K=None, T_camera_ego=None):
        if matrix.dim() != 4 or matrix.shape[-2:] != (3, 4):
            raise ValueError(
                f"PinholeProjection matrix must be [B, V, 3, 4], got {tuple(matrix.shape)}"
            )
        if geometry_type not in (GEOMETRY_PINHOLE, GEOMETRY_RECTIFIED_PINHOLE):
            raise ValueError(
                f"PinholeProjection geometry_type must be 'pinhole' or "
                f"'rectified_pinhole', got {geometry_type!r}"
            )
        self.matrix = matrix              # [B, V, 3, 4] combined (used by project)
        self.K = K                        # [B, V, 3, 3] or None (introspection)
        self.T_camera_ego = T_camera_ego  # [B, V, 4, 4] or None
        self.geometry_type = geometry_type

    @classmethod
    def from_KT(cls, K, T_camera_ego, geometry_type: str = GEOMETRY_PINHOLE):
        """Build from separate intrinsics ``K`` [B,V,3,3] and ego->camera
        transform ``T_camera_ego`` [B,V,4,4]; keeps both for introspection."""
        if K.dim() != 4 or K.shape[-2:] != (3, 3):
            raise ValueError(f"K must be [B, V, 3, 3], got {tuple(K.shape)}")
        if T_camera_ego.dim() != 4 or T_camera_ego.shape[-2:] != (4, 4):
            raise ValueError(
                f"T_camera_ego must be [B, V, 4, 4], got {tuple(T_camera_ego.shape)}"
            )
        # matrix = K @ T[:, :, :3, :]  -> [B, V, 3, 4]. Coerce T to K's dtype/device
        # first so mismatched inputs don't crash the einsum.
        T3 = T_camera_ego[:, :, :3, :].to(dtype=K.dtype, device=K.device)
        matrix = torch.einsum("bvij,bvjk->bvik", K, T3)
        return cls(matrix, geometry_type=geometry_type, K=K, T_camera_ego=T_camera_ego)

    @property
    def num_views(self) -> int:
        return self.matrix.shape[1]

    def to(self, device) -> "PinholeProjection":
        def _mv(x):
            return x.to(device) if torch.is_tensor(x) else x
        return PinholeProjection(
            self.matrix.to(device), geometry_type=self.geometry_type,
            K=_mv(self.K), T_camera_ego=_mv(self.T_camera_ego),
        )

    def to_spec(self) -> dict:
        """Serialize to a JSON-able manifest spec (batch dim dropped)."""
        return {
            "type": self.geometry_type,
            "matrix": self.matrix[0].detach().cpu().tolist(),  # [V, 3, 4]
        }

    def project_ego_to_image(self, points_ego, image_transform) -> ProjectionResult:
        """Project ego points ``[M, 3]`` (or homogeneous ``[M, 4]``) onto each
        camera. ``B, V`` come from the matrix — runtime ``V`` is derived here."""
        pts = _homogenize(points_ego)
        with torch.autocast(device_type=pts.device.type, enabled=False):
            proj = self.matrix.to(device=pts.device, dtype=torch.float32)
            # out[b, v, m, i] = sum_j proj[b, v, i, j] * points[m, j]
            projected = torch.einsum(
                "bvij,mj->bvmi",
                proj,
                pts.float(),
            )
        return _finalize_linear(projected, image_transform, self.geometry_type)


class PseudoProjection:
    """Learnable calibration-free fallback (shape-testing / ablation only).

    This is not real geometry but a learnable spatial prior that lets the
    module run without calibration. A single shared ``[3, 4]`` matrix is expanded
    to ``V`` views at projection time, so one instance serves any view count.
    Pixel coordinates are squashed with ``sigmoid`` (the raw matrix is unbounded)
    rather than normalized by the model-input size.

    The learnable tensor is owned by :class:`BEVViewFusion` (a leaf Parameter, so
    the optimizer sees it); this operator wraps it per forward. Callers must
    request the pseudo path explicitly.
    """

    geometry_type = GEOMETRY_PSEUDO

    def __init__(self, matrix: torch.Tensor, num_views: int):
        # The prior is view-independent; a per-view [V, 3, 4] would silently
        # mis-reshape, so only [3, 4] or [1, 3, 4] is accepted.
        if tuple(matrix.shape) not in ((3, 4), (1, 3, 4)):
            raise ValueError(
                f"PseudoProjection matrix must be [3, 4] or [1, 3, 4], "
                f"got {tuple(matrix.shape)}"
            )
        self.matrix = matrix
        self.num_views = num_views

    def project_ego_to_image(self, points_ego, image_transform) -> ProjectionResult:
        pts = _homogenize(points_ego)
        # Expand the shared [3, 4] prior to [1, V, 3, 4]: batch dim 1 broadcasts
        # across the real batch in the sampling loop (prior is batch- and
        # view-independent by construction).
        with torch.autocast(device_type=pts.device.type, enabled=False):
            proj = self.matrix.reshape(3, 4).unsqueeze(0).unsqueeze(0)
            proj = proj.expand(1, self.num_views, 3, 4).to(
                device=pts.device,
                dtype=torch.float32,
            )
            projected = torch.einsum(
                "bvij,mj->bvmi",
                proj,
                pts.float(),
            )

        depth = projected[..., 2]
        valid_depth = depth > _DEPTH_EPS
        depth_safe = depth.clamp(min=_DEPTH_EPS).unsqueeze(-1)
        # Divide and sigmoid in fp32: under fp16 the near-_DEPTH_EPS divide
        # overflows to inf while sigmoid saturates, giving 0*inf = nan grads that
        # make the AMP GradScaler skip every optimizer step.
        uv = projected[..., :2].float() / depth_safe.float()
        # Unbounded pseudo outputs → sigmoid to keep coords in (0, 1). in-bounds
        # is then trivially satisfied, so the mask reduces to the depth check.
        uv_norm = uv.sigmoid()
        it = _as_image_transform(image_transform)
        meta = {"geometry_type": self.geometry_type,
                "model_input_size": it.model_input_size, "rectification": None}
        return ProjectionResult(uv_norm=uv_norm, valid_mask=valid_depth,
                                depth=depth, metadata=meta)


class FThetaProjection:
    """Non-linear f-theta (fisheye) projection, native to NVIDIA PhysicalAI-AV.

    Maps ego points to pixels without flattening the fisheye to a pinhole, so a
    wide-FOV camera keeps its full field of view. The
    forward polynomial maps the incidence angle ``theta`` (angle between the
    camera-frame ray and the optical +Z axis) to a pixel radius::

        r(theta) = c0 + c1*theta + c2*theta^2 + ...
        u = cx + r * (x_cam / rho),  v = cy + r * (y_cam / rho)

    where ``rho = sqrt(x_cam^2 + y_cam^2)``. Matches the SDK's
    ``FThetaCameraModel.ray2pixel``.

    ``cx/cy`` and ``fw_poly`` are kept in the camera's native pixel resolution
    ``image_wh = (W, H)``, not pre-scaled to the model input. The projected
    ``(u, v)`` is normalized per-axis by the native ``(W, H)``. Under a plain
    aspect-changing resize this is exact (``u_native / W == u_model / W_model``),
    so the isotropic radial polynomial never needs an anisotropic approximation.

    Parameters:
        t_camera_ego: ``[B, V, 4, 4]`` ego->camera rigid transform.
        fw_poly: ``[K]`` or ``[B, V, K]`` forward polynomial coefficients
            (ascending powers of theta), radius in native pixels.
        cx, cy: principal point in native pixels, scalar or ``[B, V]``.
        image_wh: native ``(W, H)`` the intrinsics are expressed in; used to
            normalize ``(u, v)`` to ``[0, 1]``. Scalar ``(W, H)`` or per-view
            ``([B,]V, 2)``.
        max_theta: incidence-angle FOV cutoff (radians); the upper bound of the
            valid theta range. Points beyond it are masked (outside the lens).
            When None, the operator falls back to the +Z hemisphere (z>0).
        bw_poly: optional backward polynomial (radius -> theta) for inverse/remap
            (e.g. rectification); not used by the forward projection.
    """

    geometry_type = GEOMETRY_FTHETA

    def __init__(self, t_camera_ego, fw_poly, cx, cy, image_wh=(256.0, 256.0),
                 max_theta=None, bw_poly=None):
        if t_camera_ego.dim() != 4 or t_camera_ego.shape[-2:] != (4, 4):
            raise ValueError(
                f"FThetaProjection t_camera_ego must be [B, V, 4, 4], "
                f"got {tuple(t_camera_ego.shape)}"
            )
        self.t_camera_ego = t_camera_ego
        self.fw_poly = fw_poly
        self.cx = cx
        self.cy = cy
        self.image_wh = image_wh     # native (W, H) the intrinsics live in
        self.max_theta = max_theta   # upper bound of the valid theta range (FOV)
        self.bw_poly = bw_poly       # radius -> theta, for inverse/remap only

    @property
    def num_views(self) -> int:
        return self.t_camera_ego.shape[1]

    def to(self, device) -> "FThetaProjection":
        def _mv(x):
            return x.to(device) if torch.is_tensor(x) else x
        return FThetaProjection(
            self.t_camera_ego.to(device), _mv(self.fw_poly),
            _mv(self.cx), _mv(self.cy), _mv(self.image_wh),
            max_theta=_mv(self.max_theta), bw_poly=_mv(self.bw_poly),
        )

    def to_spec(self) -> dict:
        """Serialize to a JSON-able manifest spec (batch dim dropped).

        Fields differ in their canonical *batched* rank, so a rank alone cannot
        tell apart e.g. a per-view ``fw_poly [V, K]`` (keep whole) from a batched
        ``cx [B, V]`` (drop batch). Drop the leading dim only when the field is
        at its batched rank; otherwise keep it whole (a scalar becomes a float).
        """
        def _ser(x, batched_rank):
            if not torch.is_tensor(x):
                return x
            if x.dim() == 0:
                return x.item()
            if x.dim() == batched_rank:
                return x[0].detach().cpu().tolist()   # drop leading batch dim
            return x.detach().cpu().tolist()          # unbatched — keep whole
        return {
            "type": self.geometry_type,
            "t_camera_ego": _ser(self.t_camera_ego, 4),  # [B,V,4,4] -> [V,4,4]
            "fw_poly": _ser(self.fw_poly, 3),            # [B,V,K] -> [V,K]; [V,K]/[K] kept
            "cx": _ser(self.cx, 2),                      # [B,V] -> [V]; [V] kept
            "cy": _ser(self.cy, 2),
            "image_wh": _ser(self.image_wh, 3),          # native (W,H): [B,V,2]->[V,2]; kept otherwise
            "max_theta": _ser(self.max_theta, 2),        # scalar/list, never a raw tensor
        }

    def _radius(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the forward polynomial r(theta) via Horner's method.

        ``fw_poly`` is ascending-power coefficients. Accepts a shared ``[K]``
        vector, a per-view ``[V, K]``, or a batched ``[B, V, K]`` — all are
        normalized to broadcast against ``theta`` ``[B, V, M]`` (e.g. when
        reloaded from a manifest).
        """
        coeffs = torch.as_tensor(self.fw_poly, device=theta.device, dtype=theta.dtype)
        if coeffs.dim() == 1:
            r = torch.zeros_like(theta)
            for c in reversed(coeffs.unbind(0)):
                r = r * theta + c
            return r
        if coeffs.dim() == 2:
            coeffs = coeffs.unsqueeze(0)   # [V, K] -> [1, V, K]
        elif coeffs.dim() != 3:
            raise ValueError(
                f"fw_poly must be [K], [V, K] or [B, V, K], got {tuple(coeffs.shape)}"
            )
        # [B, V, K] -> Horner over the last dim, broadcasting on M via [B, V, 1, K].
        coeffs = coeffs.unsqueeze(2)  # [B, V, 1, K]
        r = torch.zeros_like(theta)
        for k in reversed(range(coeffs.shape[-1])):
            r = r * theta + coeffs[..., k]
        return r

    def project_ego_to_image(self, points_ego, image_transform) -> ProjectionResult:
        pts = _homogenize(points_ego)
        it = _as_image_transform(image_transform)
        with torch.autocast(device_type=pts.device.type, enabled=False):
            T = self.t_camera_ego.to(
                device=pts.device,
                dtype=torch.float32,
            )
            # camera-frame points: [B, V, M, 4] then drop homogeneous w.
            cam = torch.einsum(
                "bvij,mj->bvmi",
                T,
                pts.float(),
            )[..., :3]
        x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
        rho = torch.sqrt(x * x + y * y).clamp(min=_DEPTH_EPS)
        theta = torch.atan2(rho, z)                     # incidence angle from +Z
        r = self._radius(theta)                         # pixel radius (native px)
        cx = torch.as_tensor(self.cx, device=cam.device, dtype=cam.dtype)
        cy = torch.as_tensor(self.cy, device=cam.device, dtype=cam.dtype)
        if cx.dim() > 0:
            cx = cx.unsqueeze(-1)  # [B, V] -> [B, V, 1] to broadcast on M
            cy = cy.unsqueeze(-1)
        u = cx + r * (x / rho)     # native pixel u
        v = cy + r * (y / rho)     # native pixel v
        # Normalize per-axis by the native (W, H) the intrinsics live in.
        wh = torch.as_tensor(self.image_wh, device=cam.device, dtype=cam.dtype)
        if wh.dim() >= 1 and wh.shape[-1] == 2 and wh.dim() > 0:
            # scalar (2,) -> broadcast; per-view ([B,]V,2) -> add M axis
            if wh.dim() == 1:
                w_n, h_n = wh[0], wh[1]
            else:
                w_n = wh[..., 0].unsqueeze(-1)  # [.,V] -> [.,V,1]
                h_n = wh[..., 1].unsqueeze(-1)
        else:
            raise ValueError(f"image_wh must end in (W, H) size-2, got {tuple(wh.shape)}")
        uv_norm = torch.stack([u / w_n, v / h_n], dim=-1)

        depth = z                                       # optical-axis depth
        in_bounds = (
            (uv_norm[..., 0] >= 0) & (uv_norm[..., 0] <= 1)
            & (uv_norm[..., 1] >= 0) & (uv_norm[..., 1] <= 1)
        )
        # A fisheye sees rays beyond the +Z hemisphere (theta up to its real FOV,
        # which can exceed 90°), so gating on z > 0 would impose a 180° ceiling.
        # Gate on the lens FOV (max_theta) when known; otherwise fall back to the
        # +Z hemisphere, since wide rays cannot be validated without a bound.
        if self.max_theta is not None:
            # Accept a Python scalar, a list (e.g. reloaded from a manifest), or a
            # tensor. A per-view bound must broadcast against theta [B, V, M], so
            # add the M axis (mirrors cx/cy); a scalar broadcasts as-is.
            max_theta = torch.as_tensor(self.max_theta, device=theta.device, dtype=theta.dtype)
            if max_theta.dim() == 1:
                max_theta = max_theta.reshape(1, -1)   # [V] -> [1, V]
            if max_theta.dim() > 0:
                max_theta = max_theta.unsqueeze(-1)    # [B, V] -> [B, V, 1]
            # A non-finite per-view bound means that lens has no FOV bound and
            # falls back to the +Z hemisphere gate; theta <= inf would otherwise
            # admit rays from behind that lens in a mixed rig.
            angle_ok = torch.where(torch.isfinite(max_theta),
                                   theta <= max_theta, z > _DEPTH_EPS)
            mask = in_bounds & angle_ok
        else:
            mask = in_bounds & (z > _DEPTH_EPS)
        meta = {"geometry_type": self.geometry_type,
                "model_input_size": it.model_input_size, "rectification": it.rectification}
        return ProjectionResult(uv_norm=uv_norm, valid_mask=mask, depth=depth, metadata=meta)
