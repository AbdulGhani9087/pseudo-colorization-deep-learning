"""Baseline pseudo-color algorithms: only LUT and level-slice 

This module exposes two functions used by the GUI and CLI:
- `apply_lut(gray, cmap='viridis' and more many)` -> RGB uint8
- `level_slice(gray, thresholds, colors)` -> RGB uint8
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from matplotlib import cm
from skimage import img_as_ubyte


def apply_lut(
    gray: np.ndarray,
    cmap: str | None = "viridis",
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    rescale: bool = True,
    gamma: float = 1.0,
    reverse: bool = False,
    se: bool = False,
    se_strength: float = 1.0,
) -> np.ndarray:
    """Apply a matplotlib colormap to a grayscale image .

    Parameters
    - gray: 2D uint8 or float array (0-255 or 0-1)
    - cmap: matplotlib colormap name or None (None will just replicate grayscale)
    - vmin/vmax: optional intensity range to map to [0,1]. If None uses arr min/max.
    - rescale: if True, rescale intensities to [0,1] before applying colormap.
    - gamma: gamma correction applied after rescaling (value >0), default 1.0 (no change).
    - reverse: if True, use reversed colormap.

    Returns HxWx3 uint8 RGB
    """
    arr = gray.astype(np.float32)

    # Force 2D input
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        # convert RGB to luminance
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]

    # Determine vmin/vmax
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())

    if rescale:
        # avoid divide-by-zero
        span = vmax - vmin
        if span <= 0:
            norm = np.clip(arr - vmin, 0, 1)
        else:
            norm = (arr - vmin) / span
        norm = np.clip(norm, 0.0, 1.0)
        if gamma != 1.0 and gamma > 0:
            norm = np.power(norm, float(gamma))
    else:
        # assume arr already in 0..1 when rescale=False
        norm = arr

    if cmap is None:
        # replicate grayscale into RGB
        out = np.stack([norm, norm, norm], axis=-1)
        return img_as_ubyte(out)

    cmap_fn = cm.get_cmap(cmap)
    if reverse:
        try:
            cmap_fn = cmap_fn.reversed()
        except Exception:
            pass

    colored = cmap_fn(norm)[:, :, :3]  # drop alpha
    out = img_as_ubyte(colored)
    if se:
        try:
            out = se_enhance(out, strength=se_strength)
        except Exception:
            pass
    return out


def level_slice(gray: np.ndarray, thresholds: list[int], colors: list[tuple[int, int, int]]) -> np.ndarray:
    """Map intensity ranges to fixed RGB colors.

    thresholds: list of ascending thresholds (0-255). There should be len(colors) ranges.
    colors: list of RGB tuples (0-255)
    """
    h, w = gray.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    prev = 0
    for th, col in zip(thresholds, colors):
        mask = (gray >= prev) & (gray < th)
        out[mask] = col
        prev = th
    # last range
    if len(colors) > len(thresholds):
        out[gray >= prev] = colors[-1]
    return out


def se_enhance(rgb: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    """Simple color enhancement algorithm to boost colorfulness."""
    arr = rgb.astype(np.float32) / 255.0
    # Global average per channel
    ch_mean = arr.reshape(-1, 3).mean(axis=0)
    # Center and scale -> produce logits
    logits = (ch_mean - ch_mean.mean()) * float(strength)
    # softmax to get positive weights summing to 1
    exps = np.exp(logits - np.max(logits))
    weights = exps / exps.sum()
    # Normalize weights so mean weight is 1.0 to preserve overall brightness
    weights = weights / weights.mean()
    # Apply channel-wise scaling
    out = arr * weights[None, None, :]
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)

