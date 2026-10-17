import timm

BACKBONE_REGISTRY = {
    "swin_v2_tiny": lambda pretrained=True, **kwargs: timm.create_model(
        "swinv2_tiny_window8_256", pretrained=pretrained, features_only=True, **kwargs
    ),
    "conv_next_v2_tiny": lambda pretrained=True, **kwargs: timm.create_model(
        "convnextv2_tiny", pretrained=pretrained, features_only=True, **kwargs
    ),
    "res_net_50": lambda pretrained=True, **kwargs: timm.create_model(
        "resnet50", pretrained=pretrained, features_only=True, **kwargs
    ),
}


def build_backbone(backbone, pretrained=True, **kwargs):
    """Construct a multi-scale feature backbone by registry name.

    Args:
        backbone: Registry key (see ``BACKBONE_REGISTRY`` for available names).
        pretrained: Whether to load pretrained weights.
        **kwargs: Forwarded to the underlying constructor.

    Channel discovery convention:
        :class:`Backbone` reads per-stage channel counts from a timm-style
        ``feature_info`` (entries with a ``num_chs`` key). Without it,
        ``Backbone`` runs one dummy forward pass and reads the channel dimension
        of every returned (channels-first) feature map.
    """
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[backbone](pretrained=pretrained, **kwargs)
