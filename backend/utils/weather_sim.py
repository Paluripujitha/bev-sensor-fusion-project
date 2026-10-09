import cv2
import numpy as np
from PIL import Image

def add_fog(image_bgr, severity):
    fog = np.ones_like(image_bgr, dtype=np.uint8) * 200
    out = cv2.addWeighted(image_bgr, 1.0 - severity * 0.85, fog, severity * 0.85, 0)
    noise = (np.random.randn(*image_bgr.shape) * 12 * severity).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # gaussian blur simulates scattering
    ksize = max(1, int(severity * 20) | 1)
    out = cv2.GaussianBlur(out, (ksize, ksize), 0)
    return out

def add_rain(image_bgr, severity):
    out = image_bgr.copy()
    h, w = out.shape[:2]
    n_streaks = int(severity * 800)
    for _ in range(n_streaks):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        length = np.random.randint(10, 30)
        x2 = min(w-1, x1 + np.random.randint(-3, 3))
        y2 = min(h-1, y1 + length)
        alpha = np.random.uniform(0.4, 0.9)
        cv2.line(out, (x1, y1), (x2, y2), (200, 200, 220), 1)
    # darken slightly
    out = cv2.addWeighted(out, 1.0 - severity * 0.3, np.zeros_like(out), severity * 0.3, 0)
    return out

def compute_image_quality(image_bgr):
    """Return 0-1 quality estimate (1=pristine). Based on local variance / Laplacian."""
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(grey, cv2.CV_64F).var()
    # scale: 25+ = sharp (typical for nuScenes); 0 = blurred/foggy
    quality = float(np.clip(lap / 25.0, 0, 1))
    return quality

def apply_weather(image_path_or_array, weather_type='clear', severity=0.0):
    """
    Apply weather degradation.
    Returns (degraded_rgb_np, camera_quality 0-1, cam_weight, lidar_weight)
    """
    if isinstance(image_path_or_array, str):
        bgr = cv2.imread(image_path_or_array)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read {image_path_or_array}")
    else:
        bgr = cv2.cvtColor(np.array(image_path_or_array), cv2.COLOR_RGB2BGR)

    if weather_type == 'fog' and severity > 0:
        bgr = add_fog(bgr, severity)
    elif weather_type == 'rain' and severity > 0:
        bgr = add_rain(bgr, severity)
    elif weather_type in ('poor_visibility', 'night') and severity > 0:
        dark = (1.0 - severity * 0.8)
        bgr = np.clip(bgr * dark, 0, 255).astype(np.uint8)

    cam_quality = compute_image_quality(bgr)
    cam_weight   = float(np.clip(cam_quality, 0.05, 1.0))
    lidar_weight  = float(np.clip(1.0 - cam_quality * 0.5, 0.5, 1.0))  # LiDAR stays reliable

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, cam_quality, cam_weight, lidar_weight
