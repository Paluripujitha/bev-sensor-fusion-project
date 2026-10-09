"""
Feature Extractors for BEV Sensor Fusion
==========================================
CameraFeatureExtractor  —  RGB image → camera BEV feature map
LidarFeatureExtractor   —  LiDAR pseudo-image → lidar BEV feature map

Both produce output of shape [B, out_channels, BEV_H, BEV_W].
FullBEVModel in fusion.py instantiates these with out_channels=32, bev_size=(100,100).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraFeatureExtractor(nn.Module):
    """
    Lightweight image-to-BEV feature extractor.
    Inspired by Lift-Splat-Shoot (LSS) view transformer concept.
    
    For a student project without calibration-based voxel lifting, we use:
      1. A CNN backbone to extract image-space features.
      2. A learned linear projection to BEV grid size.
    
    This is honestly described as a 'simplified LSS-inspired view transformer'.
    """
    def __init__(self, out_channels=32, bev_size=(100, 100)):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.out_channels = out_channels

        # Backbone: 3-stage feature extraction from RGB image
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # /2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # /8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # View transformer: project feature channels to BEV channel count
        self.view_transformer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, images):
        """images: [B, 3, H, W] → [B, out_channels, BEV_H, BEV_W]"""
        feat = self.backbone(images)
        # Bilinear interpolation to target BEV grid resolution
        feat = F.interpolate(feat, size=(self.bev_h, self.bev_w),
                             mode='bilinear', align_corners=False)
        return self.view_transformer(feat)


class LidarFeatureExtractor(nn.Module):
    """
    Lightweight LiDAR BEV feature extractor.
    Inspired by PointPillars pseudo-image processing.
    
    Input: a 2D BEV density/height map derived from the raw point cloud.
    Output: learned feature map in the same BEV grid.
    """
    def __init__(self, out_channels=32, bev_size=(100, 100)):
        super().__init__()
        self.bev_h, self.bev_w = bev_size

        # PointPillars-inspired 2D backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar_pseudo_image):
        """
        lidar_pseudo_image: [B, 1, H, W] BEV density map
        Returns: [B, out_channels, BEV_H, BEV_W]
        """
        feat = self.backbone(lidar_pseudo_image)
        # Resize to target BEV resolution if input differs
        if feat.shape[-2:] != (self.bev_h, self.bev_w):
            feat = F.interpolate(feat, size=(self.bev_h, self.bev_w),
                                 mode='bilinear', align_corners=False)
        return feat
