import os, importlib.util, numpy as np, tensorflow as tf
from typing import Tuple
from tqdm import trange

# ensure functions are available if others import this module directly
__all__ = [
    "load_particle_generator",
    "gaussian_heatmap",
    "make_one_sample",
    "build_datasets_with_gt",
]
from pathlib import Path

def load_particle_generator(module_path: str):
    spec = importlib.util.spec_from_file_location("Particle_Tracking_Training_Data", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Particle_Tracking_Training_Data

def gaussian_heatmap(points_xy: np.ndarray, shape: Tuple[int,int], sigma: float = 2.0) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    heat = np.zeros((H, W), dtype=np.float32)
    two_sigma2 = 2.0 * sigma * sigma
    for x,y in points_xy:
        g = np.exp(-((xx - x)**2 + (yy - y)**2)/two_sigma2)
        heat = np.maximum(heat, g)
    return np.clip(heat, 0.0, 1.0).astype(np.float32)

def make_one_sample(gen,
                    kappa_range=(0.05, 0.25), a_range=(1.5,4.0),
                    Iback_range=(0.0, 0.5), Np_range=(5, 40), sigma_motion=1.5):
    kappa = tf.random.uniform([], *kappa_range); a = tf.random.uniform([], *a_range)
    Iback = tf.random.uniform([], *Iback_range)
    Np = tf.cast(tf.round(tf.random.uniform([], *Np_range)), tf.int32)
    I, labels, xi = gen(kappa, a, Iback, Np, tf.constant(sigma_motion, tf.float32))
    frame = I.numpy()[0].astype(np.float32)
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
    coords = xi.numpy()[0]  # (Np,3): x,y,z
    H, W = frame.shape
    m = (coords[:,0] >= 0) & (coords[:,0] < W) & (coords[:,1] >= 0) & (coords[:,1] < H)
    coords = coords[m]
    xy = coords[:, :2].astype(np.float32)
    z  = coords[:, 2].astype(np.float32)
    return frame, xy, z

def build_datasets_with_gt(generator_path: str,
                           out_dir: str = "./particles_data",
                           n_train: int = 500,
                           n_val: int = 100,
                           n_test: int = 100,
                           Nt: int = 1,
                           heat_sigma: float = 2.0,
                           seed: int = 123):
    np.random.seed(seed); tf.random.set_seed(seed)
    ParticleGen = load_particle_generator(generator_path)
    gen = ParticleGen(Nt=Nt, rings=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def do_split(name, N):
        imgs, heats = [], []
        gt_xy_list, gt_z_list = [], []
        for _ in trange(N, desc=f"building {name}"):
            img, xy, z = make_one_sample(gen)
            heat = gaussian_heatmap(xy, img.shape, sigma=heat_sigma)
            imgs.append(img[..., None]); heats.append(heat[..., None])
            gt_xy_list.append(xy); gt_z_list.append(z)
        imgs = np.stack(imgs).astype(np.float32)        # (N,H,W,1)
        heats = np.stack(heats).astype(np.float32)      # (N,H,W,1)
        # Save per-image variable-length GT as an object array
        gt_xy_obj = np.empty((len(gt_xy_list),), dtype=object); gt_xy_obj[:] = gt_xy_list
        gt_z_obj  = np.empty((len(gt_z_list),),  dtype=object); gt_z_obj[:]  = gt_z_list
        out = os.path.join(out_dir, f"{name}_with_gt.npz")
        np.savez_compressed(out, imgs=imgs, heatmaps=heats,
                            gt_xy_list=gt_xy_obj, gt_z_list=gt_z_obj)
        print(f"saved {name} -> {out} | imgs={imgs.shape}, gt-lists={len(gt_xy_obj)}")
    do_split("train", n_train)
    do_split("val",   n_val)
    do_split("test",  n_test)