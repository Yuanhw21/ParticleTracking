
import numpy as np
import tensorflow as tf

def gaussian2d(h, w, y, x, sigma):
    """Return (H,W) gaussian centered at (y,x)."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    g = np.exp(-((yy - y)**2 + (xx - x)**2) / (2.0 * sigma**2))
    return g

def build_targets_for_frame(h, w, tracks_in_frame, sigma_px=2.0, z_clip=None, ring_tau=None):
    """
    tracks_in_frame: list of dicts like {'x': float, 'y': float, 'z': float}
    Returns:
      heatmap: (H,W,1) in [0,1]
      zmap:    (H,W,1) real-valued; equals weighted average z at each pixel
      ring:    (H,W,1) optional, 1 if |z|>ring_tau where heatmap>0, else 0
    """
    hm = np.zeros((h, w), dtype=np.float32)
    zsum = np.zeros((h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)

    for tr in tracks_in_frame:
        x = tr['x']; y = tr['y']; z = tr['z']
        if z_clip is not None:
            z = float(np.clip(z, -z_clip, z_clip))
        g = gaussian2d(h, w, y, x, sigma_px).astype(np.float32)
        hm = np.maximum(hm, g)  # for detection we want peaks
        zsum += g * z
        wsum += g

    eps = 1e-6
    zmap = np.where(wsum > eps, zsum / (wsum + eps), 0.0).astype(np.float32)
    heatmap = hm[..., None]

    if ring_tau is not None:
        ring = (np.abs(zmap) > ring_tau).astype(np.float32) * (heatmap > 0).astype(np.float32)
        ring = ring[..., None]
        return heatmap, zmap[..., None], ring
    else:
        return heatmap, zmap[..., None], None

def dataset_from_tracks(images, tracks, sigma_px=2.0, z_clip=None, ring_tau=None, batch_size=4, shuffle=True):
    """
    images: list/array of shape [T, H, W] in [0,1] or [0..255]
    tracks: list of length T; each item is a list of dicts {'x','y','z'} for that frame
    """
    images = np.asarray(images, dtype=np.float32)
    if images.max() > 1.5:
        images = images / 255.0
    T, H, W = images.shape
    assert len(tracks) == T, "tracks length must match images length"

    heatmaps = np.zeros((T, H, W, 1), dtype=np.float32)
    zmaps    = np.zeros((T, H, W, 1), dtype=np.float32)
    if ring_tau is not None:
        rings = np.zeros((T, H, W, 1), dtype=np.float32)
    else:
        rings = None

    for t in range(T):
        hm, zm, rg = build_targets_for_frame(H, W, tracks[t], sigma_px=sigma_px, z_clip=z_clip, ring_tau=ring_tau)
        heatmaps[t] = hm
        zmaps[t]    = zm
        if rings is not None:
            rings[t] = rg

    if rings is None:
        y = (heatmaps, zmaps)
        output_sig = (tf.TensorSpec((None, H, W, 1), tf.float32),
                      tf.TensorSpec((None, H, W, 1), tf.float32))
    else:
        y = (heatmaps, zmaps, rings)
        output_sig = (tf.TensorSpec((None, H, W, 1), tf.float32),
                      tf.TensorSpec((None, H, W, 1), tf.float32),
                      tf.TensorSpec((None, H, W, 1), tf.float32))

    def gen():
        for t in range(T):
            x = images[t, ..., None]
            if rings is None:
                yield x, (heatmaps[t], zmaps[t])
            else:
                yield x, (heatmaps[t], zmaps[t], rings[t])

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((None, H, W, 1), tf.float32),
            output_sig
        )
    )
    if shuffle:
        ds = ds.shuffle(min(8*T, 1024), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
