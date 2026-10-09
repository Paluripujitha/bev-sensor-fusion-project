"""
BEV Sensor Fusion — Lightweight Detection Head Training Script
==============================================================
Trains CameraFeatureExtractor + LidarFeatureExtractor + AdaptiveWeatherFusion
+ SimpleDetectionHead on nuScenes Mini ground truth annotations.

Run from project root:
    python -m backend.train
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF

from backend.utils.data_loader import NuScenesDataLoader
from backend.utils.lidar_processor import create_bev_pseudo_image
from backend.model.fusion import FullBEVModel

# -------------------------------------------------------------------
BEV_H, BEV_W = 100, 100
MAX_RANGE     = 50.0
IMG_SIZE      = 256
EPOCHS        = 20
LR            = 5e-4
DATA_ROOT     = os.path.join(os.path.dirname(__file__), '..', 'data')
WEIGHTS_DIR   = os.path.join(os.path.dirname(__file__), 'model', 'weights')
WEIGHTS_PATH  = os.path.join(WEIGHTS_DIR, 'bev_model.pth')

NUSCENES_CLASSES = [
    'pedestrian', 'bicycle', 'car', 'motorcycle',
    'bus', 'trailer', 'truck'
]
NUM_CLASSES = len(NUSCENES_CLASSES)
CLS2IDX = {c: i for i, c in enumerate(NUSCENES_CLASSES)}

# -------------------------------------------------------------------

def cat_to_idx(name: str):
    """Map nuScenes category name to training class index, or -1."""
    n = name.lower()
    for cls in NUSCENES_CLASSES:
        if cls in n:
            return CLS2IDX[cls]
    return -1


def build_targets(gt_boxes):
    """
    Convert a list of GT annotation dicts into BEV heatmap + regression tensors.
    heatmap : [NUM_CLASSES, BEV_H, BEV_W]  — Gaussian peak at object centre
    regression: [8, BEV_H, BEV_W]           — dx, dy, z, w, l, h, sin_yaw, cos_yaw
    """
    hm  = torch.zeros(NUM_CLASSES, BEV_H, BEV_W)
    reg = torch.zeros(8, BEV_H, BEV_W)

    for box in gt_boxes:
        cls_idx = cat_to_idx(box['category'])
        if cls_idx < 0:
            continue

        x, y, z        = box['translation']
        w, l, h        = box['size']
        # rotation quaternion → yaw (approximate)
        q = box['rotation']            # [w, x, y, z]
        yaw = 2.0 * np.arctan2(q[3], q[0])

        # convert real-world (x,y) to grid cell
        gx = (x + MAX_RANGE) / (2 * MAX_RANGE) * BEV_W
        gy = (y + MAX_RANGE) / (2 * MAX_RANGE) * BEV_H
        ix, iy = int(gx), int(gy)

        if not (0 <= ix < BEV_W and 0 <= iy < BEV_H):
            continue

        # Gaussian splat (radius 2 cells)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ni, nj = iy + dy, ix + dx
                if 0 <= ni < BEV_H and 0 <= nj < BEV_W:
                    val = float(np.exp(-(dx**2 + dy**2) / 2.0))
                    hm[cls_idx, ni, nj] = max(hm[cls_idx, ni, nj].item(), val)

        # Fractional offsets within the cell
        fdx = gx - ix
        fdy = gy - iy

        reg[0, iy, ix] = fdx
        reg[1, iy, ix] = fdy
        reg[2, iy, ix] = z
        reg[3, iy, ix] = w
        reg[4, iy, ix] = l
        reg[5, iy, ix] = h
        reg[6, iy, ix] = float(np.sin(yaw))
        reg[7, iy, ix] = float(np.cos(yaw))

    return hm, reg


def main():
    print("=" * 60)
    print("  BEV Detection Head -- Training on nuScenes Mini")
    print("=" * 60)

    loader = NuScenesDataLoader(DATA_ROOT)
    scenes = loader.get_scenes()
    print(f"[TRAIN] Found {len(scenes)} scenes")

    # Collect ALL samples from ALL scenes
    all_samples = []
    for scene in scenes:
        all_samples.extend(loader.get_scene_samples(scene['token']))
    print(f"[TRAIN] Total training samples: {len(all_samples)}")

    model     = FullBEVModel(num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    bce_loss  = nn.BCELoss()
    l1_loss   = nn.L1Loss(reduction='none')

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    best_loss = float('inf')

    import time

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        total_conf = 0.0
        correct_preds = 0
        total_gt = 0
        valid_count = 0

        for sample in all_samples:
            token   = sample['token']
            info    = loader.get_sample_info(token)
            gt_boxes = loader.get_gt_boxes(token)

            if not gt_boxes:
                continue

            # ── Camera ────────────────────────────────────────────
            cam_path = info['cameras']['CAM_FRONT']['path']
            try:
                img = Image.open(cam_path).convert('RGB')
            except Exception:
                continue
            img_tens = TF.to_tensor(img)
            img_tens = torch.nn.functional.interpolate(
                img_tens.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE)
            )  # [1, 3, 256, 256]

            # ── LiDAR ─────────────────────────────────────────────
            pts = loader.load_lidar_points(token)   # (N, 5)
            pseudo = create_bev_pseudo_image(
                pts, bev_size=(IMG_SIZE, IMG_SIZE), max_range=MAX_RANGE
            )
            pseudo_tens = torch.tensor(pseudo, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # ── Ground-truth targets ───────────────────────────────
            hm_target, reg_target = build_targets(gt_boxes)
            hm_target   = hm_target.unsqueeze(0)    # [1, C, H, W]
            reg_target  = reg_target.unsqueeze(0)   # [1, 8, H, W]

            # ── Forward pass ──────────────────────────────────────
            optimizer.zero_grad()
            hm_pred, reg_pred, *_ = model(img_tens, pseudo_tens, weather_severity=0.0)

            # ── Loss ──────────────────────────────────────────────
            loss_hm = bce_loss(hm_pred, hm_target)

            # Only regress where GT exists
            pos_mask = (hm_target.max(dim=1, keepdim=True)[0] > 0.1).float()
            reg_loss_raw = l1_loss(reg_pred, reg_target)
            loss_reg = (reg_loss_raw * pos_mask).sum() / (pos_mask.sum() + 1e-4)

            loss = loss_hm + 5.0 * loss_reg
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            valid_count += 1

            # ── Metrics: Accuracy & Confidence ────────────────────
            with torch.no_grad():
                gt_pos = (hm_target > 0.5)
                if gt_pos.sum() > 0:
                    conf = hm_pred[gt_pos].mean().item()
                    total_conf += conf
                    hits = (hm_pred[gt_pos] > 0.35).float().sum().item()
                    correct_preds += hits
                    total_gt += gt_pos.sum().item()

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(valid_count, 1)
        avg_conf = (total_conf / max(valid_count, 1)) if valid_count > 0 else 0.0
        accuracy = (correct_preds / max(total_gt, 1)) * 100.0

        log_str = (f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                   f"Accuracy: {accuracy:.2f}% | Batch Size: 1 | "
                   f"Mean Conf Score: {avg_conf:.4f} | Time Taken: {epoch_time:.2f}s | "
                   f"LR: {scheduler.get_last_lr()[0]:.2e}")
        print(log_str, flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"[TRAIN] [OK] Saved best weights -> {WEIGHTS_PATH}", flush=True)

    print(f"\n[TRAIN] Training complete. Best loss: {best_loss:.4f}", flush=True)
    print(f"[TRAIN] Weights saved to: {WEIGHTS_PATH}", flush=True)



if __name__ == '__main__':
    main()
