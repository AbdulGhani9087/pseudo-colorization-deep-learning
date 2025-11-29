"""
Triang model definition + lightweight inference utilities.

This module keeps everything self-contained so other parts of the project can
simply do:

    from model import load_triang_model, triang_infer

and never worry about PyTorch boilerplate. The actual architecture is a tiny
encoder-decoder that maps 1-channel grayscale inputs into 3-channel RGB outputs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - torch may be optional
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _ensure_torch():
    """Raise a clear error if PyTorch is missing."""
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for the Triang model. "
            "Install it via `pip install torch` (pick the build matching your GPU/CPU)."
        ) from _TORCH_IMPORT_ERROR


class ConvBlock(nn.Module):
    """Basic Conv-BN-ReLU block used by TriangNet."""

    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, s: int = 1):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TriangNet(nn.Module):
    """
    Tiny encoder-decoder network that learns pseudo-color mappings.

    - Input:  (N, 1, H, W) grayscale
    - Output: (N, 3, H, W) RGB in [0, 1]
    """

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock(1, 16), ConvBlock(16, 32, s=2))
        self.enc2 = nn.Sequential(ConvBlock(32, 64), ConvBlock(64, 64, s=2))
        self.bottleneck = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            ConvBlock(64, 64),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            ConvBlock(32, 32),
        )
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid(),  # ensures outputs are within 0..1
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        bottleneck = self.bottleneck(x2)
        y = self.dec1(bottleneck)
        y = self.dec2(y)
        out = self.head(y)
        return out


class SimpleTriangNet(nn.Module):
    """
    Alternative simpler encoder-decoder architecture for compatibility with
    models trained with encoder/decoder naming convention.
    
    - Input:  (N, 1, H, W) grayscale
    - Output: (N, 3, H, W) RGB in [0, 1]
    """
    
    def __init__(self):
        super().__init__()
        # Encoder: grayscale -> features
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        # Decoder: features -> RGB
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # ensures outputs are within 0..1
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out


def default_device() -> str:
    """Return 'cuda' if available, otherwise 'cpu'."""
    _ensure_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_triang_model(
    weights_path: Optional[str | os.PathLike] = None,
    *,
    device: Optional[str] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Build the Triang network and optionally load weights.
    Auto-detects model architecture based on state_dict keys.

    Parameters
    ----------
    weights_path : str or Path, optional
        Path to a `.pt` or `.pth` file. If None, the model will keep random
        initialization (uses TriangNet by default).
    device : str, optional
        Torch device string. Defaults to 'cuda' when available, else 'cpu'.
    strict : bool
        Passed through to `load_state_dict`. If False, allows partial loading.
    """
    _ensure_torch()
    device = device or default_device()
    
    # If no weights provided, use default TriangNet
    if weights_path is None:
        model = TriangNet().to(device)
        model.eval()
        return model
    
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot load Triang weights: {path} not found")
    
    loaded = torch.load(path, map_location=device)
    
    # Handle different save formats: might be just state_dict or wrapped dict
    if isinstance(loaded, dict):
        # Check if it's a checkpoint dict with 'model_state_dict' or 'state_dict' key
        if 'model_state_dict' in loaded:
            state = loaded['model_state_dict']
        elif 'state_dict' in loaded:
            state = loaded['state_dict']
        else:
            state = loaded
    else:
        state = loaded
    
    # Auto-detect architecture based on state_dict keys
    state_keys = set(state.keys())
    
    # Check if it's the simple encoder/decoder architecture
    if any(k.startswith("encoder.") or k.startswith("decoder.") for k in state_keys):
        model = SimpleTriangNet().to(device)
        try:
            model.load_state_dict(state, strict=strict)
        except RuntimeError as e:
            # If strict loading fails, try non-strict
            if strict:
                print(f"Warning: Strict loading failed, trying non-strict: {e}")
                model.load_state_dict(state, strict=False)
            else:
                raise
    else:
        # Default to TriangNet architecture
        model = TriangNet().to(device)
        try:
            model.load_state_dict(state, strict=strict)
        except RuntimeError as e:
            # If strict loading fails, try non-strict
            if strict:
                print(f"Warning: Strict loading failed, trying non-strict: {e}")
                model.load_state_dict(state, strict=False)
            else:
                raise
    
    model.eval()
    return model


def enhance_colors(rgb: np.ndarray, saturation: float = 1.5, intensity: float = 1.2) -> np.ndarray:
    """
    Enhance color saturation and intensity of an RGB image.
    
    Parameters
    ----------
    rgb : ndarray (H, W, 3) uint8
        Input RGB image.
    saturation : float
        Saturation multiplier (>1.0 increases saturation, <1.0 decreases).
    intensity : float
        Color intensity multiplier (>1.0 makes colors brighter/more vibrant).
    
    Returns
    -------
    ndarray (H, W, 3) uint8
        Enhanced RGB image.
    """
    # Convert to float for processing
    img = rgb.astype(np.float32) / 255.0
    
    # Convert RGB to HSV for saturation control
    # Simple RGB to HSV conversion
    hsv = np.zeros_like(img)
    max_val = img.max(axis=2, keepdims=True)
    min_val = img.min(axis=2, keepdims=True)
    delta = max_val - min_val
    
    # Value (brightness)
    v = max_val
    
    # Saturation
    s = np.where(max_val > 0, delta / (max_val + 1e-8), 0)
    
    # Hue (simplified, we'll preserve it)
    h = np.zeros_like(img[..., 0:1])
    
    # Boost saturation
    s_enhanced = np.clip(s * saturation, 0, 1)
    
    # Convert back to RGB (simplified - preserve hue roughly)
    # For better results, we'll use a simpler approach: boost color channels
    # relative to grayscale
    
    # Calculate grayscale luminance
    gray = 0.299 * img[..., 0:1] + 0.587 * img[..., 1:2] + 0.114 * img[..., 2:3]
    
    # Enhance colors by amplifying the difference from grayscale
    enhanced = gray + (img - gray) * saturation * intensity
    
    # Apply intensity boost to all channels
    enhanced = enhanced * intensity
    
    # Ensure we don't clip too much - normalize if needed
    max_enhanced = enhanced.max()
    if max_enhanced > 1.0:
        enhanced = enhanced / max_enhanced
    
    # Convert back to uint8
    return np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)


def triang_infer(
    gray: np.ndarray,
    *,
    model: Optional[nn.Module] = None,
    weights_path: Optional[str | os.PathLike] = None,
    device: Optional[str] = None,
    normalize: bool = True,
    saturation: float = 2.5,
    intensity: float = 1.8,
    apply_lut: bool = False,
    lut_cmap: Optional[str] = "viridis",
    lut_gamma: float = 1.0,
    lut_blend: float = 0.5,
) -> np.ndarray:
    """
    Run inference with the Triang model on a grayscale numpy image.

    Parameters
    ----------
    gray : ndarray (H, W) or (H, W, 1)
        Image data in uint8 or float form.
    model : nn.Module, optional
        Pre-loaded model. If None, `weights_path` must be provided (or random
        weights will be used).
    weights_path : str, optional
        Path to weights to load when `model` is None.
    device : str, optional
        Torch device string. Defaults to auto-detection.
    normalize : bool
        If True (default) scale uint8 inputs to 0..1.
    saturation : float
        Color saturation multiplier (default 2.5 for more vibrant colors).
    intensity : float
        Color intensity multiplier (default 1.8 for brighter colors).
    apply_lut : bool
        If True, apply LUT colormap to enhance colors further.
    lut_cmap : str, optional
        Matplotlib colormap name (e.g., 'viridis', 'magma', 'plasma').
    lut_gamma : float
        Gamma correction for LUT application.
    lut_blend : float
        Blend factor between model output and LUT (0.0 = model only, 1.0 = LUT only, 0.5 = equal blend).
    """
    _ensure_torch()
    arr = gray.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if normalize:
        arr = arr / 255.0

    if model is None:
        model = load_triang_model(weights_path, device=device)
    device = device or next(model.parameters()).device

    with torch.inference_mode():
        tensor = torch.from_numpy(arr)[None, None, ...].to(device)
        out = model(tensor)
        rgb = out.clamp(0, 1).cpu().numpy()[0]
        rgb = np.transpose(rgb, (1, 2, 0))  # HWC
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        # Always apply color enhancement for more vibrant colors
        rgb_uint8 = enhance_colors(rgb_uint8, saturation=saturation, intensity=intensity)
        
        # Apply LUT colormap if requested
        if apply_lut and lut_cmap:
            try:
                # Import baselines here to avoid circular imports
                import baselines
                
                # Convert RGB output to grayscale (luminance) for LUT application
                gray_from_rgb = (
                    0.2989 * rgb_uint8[..., 0] 
                    + 0.5870 * rgb_uint8[..., 1] 
                    + 0.1140 * rgb_uint8[..., 2]
                ).astype(np.uint8)
                
                # Apply LUT colormap
                lut_output = baselines.apply_lut(
                    gray_from_rgb,
                    cmap=lut_cmap,
                    gamma=lut_gamma,
                    se=False,
                )
                
                # Blend model output with LUT output
                blend_factor = np.clip(lut_blend, 0.0, 1.0)
                rgb_uint8 = (
                    rgb_uint8.astype(np.float32) * (1.0 - blend_factor)
                    + lut_output.astype(np.float32) * blend_factor
                ).astype(np.uint8)
            except Exception as e:
                # If LUT application fails, just use the model output
                print(f"Warning: LUT application failed: {e}")
        
        return rgb_uint8


__all__ = [
    "TriangNet",
    "SimpleTriangNet",
    "load_triang_model",
    "triang_infer",
    "default_device",
]

