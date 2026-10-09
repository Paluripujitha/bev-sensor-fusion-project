import torch
import torch.nn as nn
import torch.nn.functional as F

class DHIP(nn.Module):
    """
    Dual Hard Instance Probing (DHIP)
    Identifies 'hard' regions based on feature entropy or spatial gradient variance.
    """
    def __init__(self, channels):
        super(DHIP, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is a feature map [B, C, H, W]
        # We calculate an "instance difficulty" map
        difficulty = self.sigmoid(self.conv(x))
        return difficulty

class DAFusionNet(nn.Module):
    """
    Deformable Attention Fusion Network (Simplified for CPU/Software-only constraint)
    Fuses Camera BEV and LiDAR BEV using multi-head attention over the grid.
    """
    def __init__(self, in_channels, out_channels):
        super(DAFusionNet, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.val_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, cam_feat, lidar_feat):
        # Flatten spatial dimensions for attention
        B, C, H, W = cam_feat.shape
        
        # In a true deformable attention, this acts locally. 
        # For our lightweight proxy, we simulate this with standard learned cross-attention.
        query = self.query_conv(cam_feat).view(B, C, -1).permute(0, 2, 1) # [B, H*W, C]
        key = self.key_conv(lidar_feat).view(B, C, -1) # [B, C, H*W]
        value = self.val_conv(lidar_feat).view(B, C, -1).permute(0, 2, 1) # [B, H*W, C]
        
        attention = F.softmax(torch.bmm(query, key), dim=-1) # [B, H*W, H*W]
        out = torch.bmm(attention, value) # [B, H*W, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        
        return self.out_conv(out + cam_feat) # Residual connection

class AdaptiveWeatherFusion(nn.Module):
    """
    Combines the baseline BEVFusion with Extension 1 (Adaptive) and Extension 2 (Weather-Robust).
    """
    def __init__(self, channels=64, hard_threshold=0.5):
        super(AdaptiveWeatherFusion, self).__init__()
        self.dhip = DHIP(channels * 2) 
        self.da_fusion = DAFusionNet(channels, channels)
        
        # Lightweight fusion for easy instances (Extension 1)
        self.lightweight_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        
        self.hard_threshold = hard_threshold

    def forward(self, cam_bev, lidar_bev, weather_severity=0.0):
        """
        weather_severity: 0.0 (Clear) to 1.0 (Severe weather where camera is useless)
        """
        # Extension 2: Weather-Robust Feature Re-weighting
        # When weather is severe, we suppress camera features.
        cam_weight = max(0.0, 1.0 - weather_severity)
        weighted_cam_bev = cam_bev * cam_weight
        
        # Concatenate features
        concat_feat = torch.cat([weighted_cam_bev, lidar_bev], dim=1)
        
        # Extension 1: Adaptive Lightweight Fusion using DHIP
        difficulty_map = self.dhip(concat_feat) # [B, 1, H, W]
        
        # Hard regions go through DAFusionNet
        hard_mask = (difficulty_map >= self.hard_threshold).float()
        easy_mask = 1.0 - hard_mask
        
        # Calculate both paths (In real deployment, we'd use sparse convolutions to save time)
        hard_fused = self.da_fusion(weighted_cam_bev, lidar_bev)
        easy_fused = self.lightweight_fusion(concat_feat)
        
        # Combine dynamically
        final_fusion = (hard_mask * hard_fused) + (easy_mask * easy_fused)
        
        # Return fused features and the difficulty map for visualization
        return final_fusion, difficulty_map

class SimpleDetectionHead(nn.Module):
    """
    Decodes the fused BEV features into 3D Detections (Class, Position, Size, Yaw).
    """
    def __init__(self, in_channels, num_classes=10):
        super(SimpleDetectionHead, self).__init__()
        # Heatmap for class prediction and confidence
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        
        # Regression head for (x, y, z, w, l, h, sin(yaw), cos(yaw))
        self.regression_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 8, kernel_size=1)
        )

    def forward(self, x):
        heatmap = torch.sigmoid(self.heatmap_head(x))
        regression = self.regression_head(x)
        return heatmap, regression

class FullBEVModel(nn.Module):
    """
    Genuine End-to-End BEV Sensor Fusion Model.
    Camera Image + LiDAR Point Cloud Pseudo-Image -> 3D Detections.
    """
    def __init__(self, num_classes=10):
        super(FullBEVModel, self).__init__()
        import backend.model.feature_extractor as fe
        self.cam_extractor = fe.CameraFeatureExtractor(out_channels=32, bev_size=(100, 100))
        self.lidar_extractor = fe.LidarFeatureExtractor(out_channels=32, bev_size=(100, 100))
        self.fusion = AdaptiveWeatherFusion(channels=32, hard_threshold=0.5)
        self.head = SimpleDetectionHead(in_channels=32, num_classes=num_classes)
        
    def forward(self, img_tensor, lidar_pseudo, weather_severity=0.0):
        # 1. Extract Features
        cam_feat = self.cam_extractor(img_tensor)
        lidar_feat = self.lidar_extractor(lidar_pseudo)
        
        # 2. BEV Fusion (Adaptive + Weather-Robust)
        fused_bev, difficulty_map = self.fusion(cam_feat, lidar_feat, weather_severity)
        
        # 3. Detection Head
        heatmap, regression = self.head(fused_bev)
        
        return heatmap, regression, fused_bev, cam_feat, lidar_feat, difficulty_map
