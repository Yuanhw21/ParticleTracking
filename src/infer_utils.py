"""Utilities for particle detection/regression inference and visualization.

Functions included:
- denorm_z
- non_maximum_suppression
- extract_patches
- overlay
- match_and_metrics
- build_infer_functions (returns compiled det_step & z_step and Z_PATCH)
"""
from __future__ import annotations

from typing import Callable, Tuple, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree


def denorm_z(z_norm: np.ndarray | tf.Tensor, z_min: float = -30.0, z_max: float = 30.0) -> np.ndarray:
    """Denormalize z from [-1, 1] back to [z_min, z_max]."""
    return (np.asarray(z_norm) + 1.0) * (z_max - z_min) / 2.0 + z_min


def non_maximum_suppression(
    heat: np.ndarray,
    thr: float = 0.3,
    radius: int = 3,
    topk: int = 1024,
) -> np.ndarray:
    """Simple NMS on a 2D heatmap.

    Returns array of shape (M, 2) with [x, y] of peaks.
    """
    mask = heat >= thr
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float32)
    pooled = maximum_filter(heat, size=(2 * radius + 1))
    peaks = (heat == pooled) & mask
    ys, xs = np.where(peaks)
    scores = heat[ys, xs]
    order = np.argsort(-scores)[:topk]
    return np.stack(
        [xs[order].astype(np.float32), ys[order].astype(np.float32)], axis=1
    )


def extract_patches(img: np.ndarray, centers_xy: np.ndarray, patch: int = 21) -> np.ndarray:
    """Extract square patches centered at centers_xy from a single-channel image.

    Returns float32 array (M, patch, patch, 1) in [0,1].
    """
    H, W = img.shape
    r = patch // 2
    pad_img = np.pad(img, ((r, r), (r, r)), mode="reflect")
    outs = []
    for x, y in centers_xy:
        cx, cy = int(round(x)) + r, int(round(y)) + r
        cut = pad_img[cy - r : cy + r + 1, cx - r : cx + r + 1]
        if cut.shape == (patch, patch):
            outs.append(cut[..., None])
    if not outs:
        return np.zeros((0, patch, patch, 1), np.float32)
    outs = np.stack(outs).astype(np.float32)
    pmin = outs.min(axis=(1, 2, 3), keepdims=True)
    pmax = outs.max(axis=(1, 2, 3), keepdims=True)
    return (outs - pmin) / (pmax - pmin + 1e-6)


def overlay(
    ax: plt.Axes,
    img: np.ndarray,
    pred_xy: Optional[np.ndarray] = None,
    pred_z: Optional[np.ndarray] = None,
    gt_xy: Optional[np.ndarray] = None,
    gt_z: Optional[np.ndarray] = None,
    show_pred_z: bool = True,
    show_gt_z: bool = True,
    title: Optional[str] = None,
) -> None:
    ax.imshow(img, cmap="gray", origin="lower")
    # GT: red plus
    if gt_xy is not None and len(gt_xy) > 0:
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "r+", markersize=6, label="GT")
        if show_gt_z and gt_z is not None and len(gt_z) == len(gt_xy):
            for (x, y), z in zip(gt_xy, gt_z):
                ax.text(x + 2, y + 2, f"{z:.1f}", color="red", fontsize=7)
    # Pred: green hollow circles
    if pred_xy is not None and len(pred_xy) > 0:
        ax.scatter(
            pred_xy[:, 0],
            pred_xy[:, 1],
            s=20,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.0,
            label="Pred",
        )
        if show_pred_z and pred_z is not None and len(pred_z) == len(pred_xy):
            for (x, y), z in zip(pred_xy, pred_z):
                ax.text(x + 2, y - 10, f"{z:.1f}", color="lime", fontsize=7)
    if title:
        ax.set_title(title)
    ax.set_xlim(-10, img.shape[1] + 9)
    ax.set_ylim(-10, img.shape[0] + 9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def match_and_metrics(
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    pred_z: Optional[np.ndarray],
    gt_z: Optional[np.ndarray],
    px_tol: float = 3.0,
) -> dict:
    """Nearest-neighbor matching (unique on GT).

    Returns precision/recall/F1 and z MAE/RMSE on matched pairs.
    """
    if len(gt_xy) == 0 and len(pred_xy) == 0:
        return dict(
            precision=1.0, recall=1.0, f1=1.0, mae=np.nan, rmse=np.nan, matches=0
        )
    if len(pred_xy) == 0:
        return dict(
            precision=0.0,
            recall=0.0 if len(gt_xy) > 0 else 1.0,
            f1=0.0,
            mae=np.nan,
            rmse=np.nan,
            matches=0,
        )
    if len(gt_xy) == 0:
        return dict(precision=0.0, recall=1.0, f1=0.0, mae=np.nan, rmse=np.nan, matches=0)

    tree = cKDTree(gt_xy)
    dist, idx = tree.query(pred_xy, k=1)
    pairs = [(p_i, g_i) for p_i, (g_i, d) in enumerate(zip(idx, dist)) if d <= px_tol]

    # Ensure unique GT matches (keep closest when multiple preds map to same GT)
    by_gt = {}
    for p_i, g_i in pairs:
        d = np.linalg.norm(pred_xy[p_i] - gt_xy[g_i])
        if g_i not in by_gt or d < by_gt[g_i][1]:
            by_gt[g_i] = (p_i, d)
    chosen = [(p_i, g_i) for g_i, (p_i, _) in by_gt.items()]

    TP = len(chosen)
    FP = len(pred_xy) - TP
    FN = len(gt_xy) - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mae = rmse = np.nan
    if TP > 0 and pred_z is not None and gt_z is not None:
        pz = np.array([pred_z[p_i] for p_i, _ in chosen])
        gz = np.array([gt_z[g_i] for _, g_i in chosen])
        diff = pz - gz
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
    return dict(precision=precision, recall=recall, f1=f1, mae=mae, rmse=rmse, matches=TP)


def build_infer_functions(
    det_model: tf.keras.Model, z_model: tf.keras.Model
) -> Tuple[Callable[[tf.Tensor], tf.Tensor], Callable[[tf.Tensor], tf.Tensor], int]:
    """Create compiled tf.functions for detector and z regressor.

    Returns: (det_step, z_step, Z_PATCH)
    - det_step: (B,H,W,1) float32 -> (B,H,W) float32
    - z_step: (B,P,P,1) float32 -> (B,) float32
    - Z_PATCH: the expected patch size for z_model
    """
    DET_H, DET_W = det_model.input_shape[1], det_model.input_shape[2]
    Z_PATCH = z_model.input_shape[1]

    det_spec = tf.TensorSpec(shape=[None, DET_H, DET_W, 1], dtype=tf.float32)

    @tf.function(input_signature=[det_spec])
    def det_step(x: tf.Tensor) -> tf.Tensor:
        y = det_model(x, training=False)
        return tf.squeeze(y, axis=-1)

    z_spec = tf.TensorSpec(shape=[None, Z_PATCH, Z_PATCH, 1], dtype=tf.float32)

    @tf.function(input_signature=[z_spec])
    def z_step(patches: tf.Tensor) -> tf.Tensor:
        y = z_model(patches, training=False)
        return tf.reshape(y, [-1])

    return det_step, z_step, Z_PATCH


def make_browse(
    imgs: np.ndarray,
    label_heatmaps: np.ndarray,
    gt_xy_list: np.ndarray,
    gt_z_list: np.ndarray,
    det_step: Callable[[tf.Tensor], tf.Tensor],
    z_step: Callable[[tf.Tensor], tf.Tensor],
    Z_PATCH: int,
):
    """Create a browse(i, ...) function for interactive visualization.

    The returned function signature matches the original notebook implementation.
    """

    import tensorflow as tf  # local import to avoid hard dependency when unused

    def _browse(
        i: int = 0,
        thr: float = 0.3,
        px_tol: float = 3.0,
        patch_size: int = 21,
        show_label_heat: bool = True,
        show_pred_z: bool = True,
        show_gt_z: bool = True,
    ):
        img = imgs[i, ..., 0].astype(np.float32)

        # Detector forward
        img_b1 = img[None, ..., None]
        pred_heat = det_step(tf.convert_to_tensor(img_b1)).numpy()[0, ...]

        # NMS to get candidate centers
        pred_xy = non_maximum_suppression(pred_heat, thr=thr, radius=3)

        # Z regressor
        pred_z = None
        if len(pred_xy) > 0:
            patches = extract_patches(img, pred_xy, patch=patch_size)
            if len(patches) > 0:
                patches_tf = tf.image.resize(
                    tf.convert_to_tensor(patches), [Z_PATCH, Z_PATCH], method="bilinear"
                )
                patches_tf = tf.clip_by_value(tf.cast(patches_tf, tf.float32), 0.0, 1.0)
                z_pred_norm = z_step(patches_tf).numpy()
                pred_z = denorm_z(z_pred_norm)

        # Visualization and metrics (mirrors original logic)
        m = match_and_metrics(gt_xy_list[i], pred_xy, pred_z, gt_z_list[i], px_tol=px_tol)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        overlay(
            axs[0],
            img,
            pred_xy,
            pred_z,
            gt_xy_list[i],
            gt_z_list[i],
            show_pred_z=show_pred_z,
            show_gt_z=show_gt_z,
            title=f"Pred vs GT | P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}  z-MAE={m['mae']:.2f} z-RMSE={m['rmse']:.2f}",
        )
        axs[1].imshow(img, cmap="gray", origin="lower")
        axs[1].imshow(
            pred_heat, cmap="magma", origin="lower", alpha=0.45, interpolation="nearest"
        )
        if show_label_heat:
            axs[1].imshow(
                label_heatmaps[i, ..., 0],
                cmap="spring",
                origin="lower",
                alpha=0.35,
                interpolation="nearest",
            )
        gt_xy, gt_z = gt_xy_list[i], gt_z_list[i]
        if len(gt_xy) > 0:
            axs[1].plot(gt_xy[:, 0], gt_xy[:, 1], "r+", markersize=6)
            if show_gt_z and gt_z is not None and len(gt_z) == len(gt_xy):
                for (x, y), z in zip(gt_xy, gt_z):
                    axs[1].text(x + 2, y + 2, f"{z:.1f}", color="orange", fontsize=7)
        axs[1].set_title("Pred heat (magma) + Label heat (spring) + GT points")
        for ax in axs:
            ax.set_xlim(-10, img.shape[1] + 9)
            ax.set_ylim(-10, img.shape[0] + 9)
        plt.tight_layout()
        plt.show()

    return _browse


def launch_interact(browse_fn, N: int):
    """Launch ipywidgets interact UI for the provided browse function."""
    import ipywidgets as widgets
    from ipywidgets import interact

    return interact(
        browse_fn,
        i=widgets.IntSlider(min=0, max=N - 1, step=1, value=0, description="index"),
        thr=widgets.FloatSlider(
            min=0.05,
            max=0.8,
            step=0.05,
            value=0.3,
            readout_format=".2f",
            description="pred NMS",
        ),
        px_tol=widgets.FloatSlider(
            min=1.0,
            max=8.0,
            step=0.5,
            value=3.0,
            readout_format=".1f",
            description="match tol px",
        ),
        patch_size=widgets.IntSlider(min=15, max=41, step=2, value=21, description="patch"),
        show_label_heat=widgets.Checkbox(value=True, description="show label heat"),
        show_pred_z=widgets.Checkbox(value=True, description="annotate pred z"),
        show_gt_z=widgets.Checkbox(value=True, description="annotate GT z"),
    )
