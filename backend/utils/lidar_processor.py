import numpy as np
from PIL import Image
import io, base64

def load_point_cloud(lidar_path):
    """Load nuScenes .pcd.bin → (N,5) float32: x y z intensity ring."""
    scan = np.fromfile(lidar_path, dtype=np.float32)
    return scan.reshape(-1, 5)

def create_bev_image(points, bev_size=(256, 256), max_range=50.0,
                     color_by='height'):
    """
    Convert (N,5) point cloud to a colour BEV PNG (RGB numpy array).
    - color_by='height'     → height-coloured heatmap  (blue→green→red)
    - color_by='density'    → greyscale density map
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (np.abs(x) < max_range) & (np.abs(y) < max_range)
    x, y, z = x[mask], y[mask], z[mask]

    # Grid indices
    xi = np.clip(((x + max_range) / (2 * max_range) * bev_size[1]).astype(np.int32),
                 0, bev_size[1] - 1)
    yi = np.clip(((y + max_range) / (2 * max_range) * bev_size[0]).astype(np.int32),
                 0, bev_size[0] - 1)

    img = np.zeros((bev_size[0], bev_size[1], 3), dtype=np.uint8)

    if color_by == 'density':
        density = np.zeros(bev_size, dtype=np.float32)
        np.add.at(density, (yi, xi), 1.0)
        m = density.max()
        if m > 0:
            density /= m
        grey = (density * 255).astype(np.uint8)
        img[:, :, 0] = grey
        img[:, :, 1] = grey
        img[:, :, 2] = grey
    else:
        # Height-coloured: accumulate max z per cell then colour
        z_grid = np.full(bev_size, -np.inf, dtype=np.float32)
        np.maximum.at(z_grid, (yi, xi), z)
        valid = z_grid > -np.inf

        z_min, z_max = -2.0, 5.0
        norm = np.clip((z_grid - z_min) / (z_max - z_min), 0, 1)  # 0→1

        # Hue ramp: blue(0) → cyan(0.25) → green(0.5) → yellow(0.75) → red(1)
        def _plasma(t):
            # simple blue→green→red colormap
            r = np.clip(2.0 * t - 0.5, 0, 1)
            g = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0, 1)
            b = np.clip(1.0 - 2.0 * t, 0, 1)
            return (r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)

        r, g, b = _plasma(norm)
        img[valid, 0] = r[valid]
        img[valid, 1] = g[valid]
        img[valid, 2] = b[valid]

    # ego vehicle dot at centre
    cx, cy = bev_size[1]//2, bev_size[0]//2
    img[cy-3:cy+3, cx-3:cx+3] = [0, 255, 255]   # cyan square

    return img   # (H,W,3) uint8 RGB


def create_bev_pseudo_image(points, bev_size=(200, 200), max_range=50.0):
    """Greyscale normalised BEV array (float32 0-1) kept for model compatibility."""
    x = points[:, 0]; y = points[:, 1]
    mask = (np.abs(x) < max_range) & (np.abs(y) < max_range)
    x, y = x[mask], y[mask]
    xi = np.clip(((x + max_range) / (2 * max_range) * bev_size[1]).astype(np.int32), 0, bev_size[1]-1)
    yi = np.clip(((y + max_range) / (2 * max_range) * bev_size[0]).astype(np.int32), 0, bev_size[0]-1)
    grid = np.zeros(bev_size, dtype=np.float32)
    np.add.at(grid, (yi, xi), 1.0)
    m = grid.max()
    if m > 0: grid /= m
    return grid


def numpy_to_base64(rgb_array):
    """(H,W,3) uint8 → 'data:image/png;base64,…'"""
    pil = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{b64}'


def grey_to_base64(mono_array):
    """(H,W) float32 0-1 → 'data:image/png;base64,…'"""
    arr = (np.clip(mono_array, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{b64}'
