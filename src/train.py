# --------------------------
# Training loops
# --------------------------
import os
import numpy as np
from tensorflow import keras

from .model import build_unet, build_z_regressor, HeatmapLoss
from .utilities import extract_patches

def train_detector(train_npz, val_npz, epochs=20, batch_size=8, lr=1e-3):
    data_tr = np.load(train_npz)
    data_va = np.load(val_npz)
    x_tr, y_tr = data_tr["imgs"], data_tr["heatmaps"]
    x_va, y_va = data_va["imgs"], data_va["heatmaps"]

    model = build_unet(input_shape=x_tr.shape[1:])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=HeatmapLoss())
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.path.dirname(train_npz), "detector.keras"),
        monitor="val_loss", save_best_only=True, save_weights_only=False)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = model.fit(x_tr, y_tr,
                     validation_data=(x_va, y_va),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=[ckpt, es])
    return model, hist


def _prepare_patches_and_znorm_from_with_gt(npz_path: str, patch_size: int = 21,
                                             z_min: float = -30.0, z_max: float = 30.0):
    """Helper: from *_with_gt.npz produce (patches, z_norm) in-memory."""
    data = np.load(npz_path, allow_pickle=True)
    imgs = data["imgs"].astype(np.float32)           # (N,H,W,1)
    gt_xy_list = data["gt_xy_list"]                  # object array
    gt_z_list  = data["gt_z_list"]                   # object array

    all_patches, all_z_norm = [], []
    for i in range(len(imgs)):
        img = imgs[i, ..., 0]
        xy  = gt_xy_list[i]
        zz  = gt_z_list[i]
        if xy is None or len(xy) == 0:
            continue
        patches_i = extract_patches(img, xy, patch=patch_size)
        if patches_i.shape[0] == 0:
            continue
        keep = min(patches_i.shape[0], len(zz))
        patches_i = patches_i[:keep]
        z_i = np.asarray(zz, dtype=np.float32)[:keep]
        z_norm_i = 2.0 * (z_i - z_min) / (z_max - z_min) - 1.0
        all_patches.append(patches_i.astype(np.float32))
        all_z_norm.append(z_norm_i.astype(np.float32))

    if len(all_patches) == 0:
        return (np.zeros((0, patch_size, patch_size, 1), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    patches = np.concatenate(all_patches, axis=0)
    z_norm = np.concatenate(all_z_norm, axis=0)
    return patches, z_norm


def train_z_regressor(train_npz, val_npz, patch_size=21, epochs=20, batch_size=64, lr=1e-3):
    # Supports two dataset types:
    # 1) Standard: contains "patches" and "z_norm"
    # 2) with_gt: contains "gt_xy_list" and "gt_z_list" (build patches & z_norm in memory)
    tr = np.load(train_npz, allow_pickle=True)
    va = np.load(val_npz, allow_pickle=True)

    if "patches" in tr and "z_norm" in tr:
        x_tr, z_tr = tr["patches"], tr["z_norm"]
    else:
        x_tr, z_tr = _prepare_patches_and_znorm_from_with_gt(train_npz, patch_size=patch_size)

    if "patches" in va and "z_norm" in va:
        x_va, z_va = va["patches"], va["z_norm"]
    else:
        x_va, z_va = _prepare_patches_and_znorm_from_with_gt(val_npz, patch_size=patch_size)

    model = build_z_regressor(patch_size=patch_size)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="huber")
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.path.dirname(train_npz), "z_regressor.keras"),
        monitor="val_loss", save_best_only=True, save_weights_only=False)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = model.fit(x_tr, z_tr,
                     validation_data=(x_va, z_va),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=[ckpt, es])
    return model, hist