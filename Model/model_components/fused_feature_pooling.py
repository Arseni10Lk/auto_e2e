import torch
import torch.nn as nn


class FusedFeaturePooling(nn.Module):
    """Pool fused image/map BEV features into a planner feature vector."""

    def __init__(self, embed_dim = 256):
        super(FusedFeaturePooling, self).__init__()

        # [B, C, H, W] -> [B, 1, H, W] -> [B, 1, 75, 50]
        self.reduce_channels = nn.Conv2d(embed_dim, 1, 3, 1, 1)
        self.reduce_bev = nn.AdaptiveAvgPool2d((75, 50))

    def forward(self, fused_features):
        features_channel_compressed = self.reduce_channels(fused_features)
        features_bev_compressed = self.reduce_bev(features_channel_compressed)
        # [B, 3750]
        reduced_features = torch.flatten(features_bev_compressed, start_dim=1)

        return reduced_features
