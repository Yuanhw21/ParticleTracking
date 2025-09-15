from typing import Tuple

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def gaussian_heatmap(points_xy: np.ndarray, shape: Tuple[int,int], sigma: float = 2.0) -> np.ndarray:
    """
    Render a heatmap with Gaussian peaks at (x, y) particle centers.
    points_xy: array of shape (N, 2) with (x, y) in pixel coords (float)
    shape: (H, W)
    Return: (H, W) float32 in [0,1]
    """
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    heat = np.zeros((H, W), dtype=np.float32)
    two_sigma2 = 2.0 * sigma * sigma
    for p in points_xy:
        x, y = float(p[0]), float(p[1])
        # accumulate (use max to get distinct peaks)
        g = np.exp(-((xx - x)**2 + (yy - y)**2)/two_sigma2)
        heat = np.maximum(heat, g)
    heat = np.clip(heat, 0.0, 1.0).astype(np.float32)
    return heat

def extract_patches(img: np.ndarray, centers_xy: np.ndarray, patch: int = 21) -> np.ndarray:
    """
    Extract square patches centered at subpixel (x,y). We round to nearest pixel.
    img: (H,W) uint or float
    centers_xy: (N,2) where columns are (x,y)
    Return: (N, patch, patch, 1)
    """
    H, W = img.shape
    r = patch // 2
    pads = ((r, r), (r, r))
    pad_img = np.pad(img, pads, mode='reflect')
    patches = []
    for (x, y) in centers_xy:
        cx, cy = int(round(x)), int(round(y))
        # shift due to padding
        cx += r
        cy += r
        cut = pad_img[cy - r: cy + r + 1, cx - r: cx + r + 1]
        if cut.shape != (patch, patch):
            # If at boundaries and something odd happened, skip
            continue
        patches.append(cut[..., None])
    if len(patches) == 0:
        return np.zeros((0, patch, patch, 1), dtype=np.float32)
    patches = np.stack(patches, axis=0).astype(np.float32)
    # normalize per-patch to [0,1]
    pmin = patches.min(axis=(1,2,3), keepdims=True)
    pmax = patches.max(axis=(1,2,3), keepdims=True)
    patches = (patches - pmin) / (pmax - pmin + 1e-6)
    return patches

def non_maximum_suppression(heat: np.ndarray, thr: float = 0.3, radius: int = 3, topk: int = 1024) -> np.ndarray:
    """
    Simple NMS on heatmap.
    Return detected centers as (N,2) array of (x,y) in float (pixel coords).
    """
    from scipy.ndimage import maximum_filter

    H, W = heat.shape
    # suppress small values
    mask = heat >= thr
    if not mask.any():
        return np.zeros((0,2), dtype=np.float32)
    # local maxima
    pooled = maximum_filter(heat, size=(2*radius+1))
    peaks = (heat == pooled) & mask
    ys, xs = np.where(peaks)
    scores = heat[ys, xs]
    order = np.argsort(-scores)[:topk]
    xs = xs[order].astype(np.float32)
    ys = ys[order].astype(np.float32)
    return np.stack([xs, ys], axis=1)

def overlay_detections(img: np.ndarray, centers_xy: np.ndarray, zs: np.ndarray = None, save_path: str = None):
    """
    Plot detections on the image, optionally annotate z.
    """
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray', vmin=img.min(), vmax=img.max())
    if centers_xy is not None and len(centers_xy) > 0:
        xs, ys = centers_xy[:,0], centers_xy[:,1]
        plt.scatter(xs, ys, s=20, facecolors='none', edgecolors='lime', linewidths=1.0)
        if zs is not None and len(zs) == len(xs):
            for x,y,z in zip(xs,ys,zs):
                plt.text(x+2, y-2, f"{float(z):.1f}", color='yellow', fontsize=6)
    plt.title("Detections (x,y) with optional z")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()