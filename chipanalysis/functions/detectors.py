"""
detectors.py
============
Pluggable detection function factories.

Every factory returns a callable with the uniform signature::

    detect_fn(image: np.ndarray, px_um: float) -> np.ndarray (bool)

so that any detector can be passed to ``build_cell_dataframe`` or used
interchangeably in notebooks and batch scripts.

Available factories
-------------------
make_fluo_detector      – threshold-based fluorescence (no ML)
make_unet_detector      – wraps an arbitrary U-Net checkpoint
make_unet_pillar_detector – UNet + pillar-specific mask cleaning
"""

from __future__ import annotations

import numpy as np
from skimage import exposure
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes


# ──────────────────────────────────────────────────────────────────────────────
# Fluorescence threshold detector
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_fluo_threshold(image, temporal_mean=None, bg_sigma_um=1000.0, px_um=None):
    """
    Run the same pre-processing as ``make_fluo_detector`` on a single image
    and return the raw Otsu threshold on the background-removed result.

    Use this to measure a fixed threshold from a representative frame and then
    pass it as ``fixed_threshold`` to ``make_fluo_detector``.

    Parameters
    ----------
    image         : np.ndarray  2-D, any dtype
    temporal_mean : np.ndarray or None  – same scale as *image*
    bg_sigma_um   : float  – Gaussian σ in µm (must match the detector)
    px_um         : float  – pixel size in µm  (required)

    Returns
    -------
    otsu_value : float  – Otsu threshold on the BG-removed, normalised image
    """
    if px_um is None:
        raise ValueError("px_um is required for calibrate_fluo_threshold")
    img = image - temporal_mean if temporal_mean is not None else image
    img = exposure.rescale_intensity(img.astype(np.float32), out_range=(0.0, 1.0))
    sigma_px   = bg_sigma_um / px_um
    background = gaussian(img, sigma=sigma_px, preserve_range=True)
    bg_rem     = np.clip(img - background, 0, None)
    bg_rem     = exposure.rescale_intensity(bg_rem, out_range=(0.0, 1.0))
    return float(threshold_otsu(bg_rem))


def make_fluo_detector(
    temporal_mean=None,
    bg_sigma_um=1000.0,
    min_obj_um2=25.0,
    max_hole_um2=5.0,
    thresh_factor=0.2,
    fixed_threshold=None,
):
    """
    Factory: fluorescence threshold detector with optional temporal-mean subtraction.

    Steps inside the returned function:
      1. Subtract *temporal_mean* (removes static background)
      2. Subtract Gaussian spatial background (removes residual gradients)
      3. Threshold the BG-removed image — two modes:
           • **Otsu per-frame** (default): threshold = Otsu(frame) × *thresh_factor*
           • **Fixed threshold**: threshold = *fixed_threshold* (ignores *thresh_factor*)
             Set *fixed_threshold* once via ``calibrate_fluo_threshold`` on a
             representative frame so every frame uses the same absolute cutoff.
      4. Morphological clean-up

    Parameters
    ----------
    temporal_mean   : np.ndarray or None
        Pre-computed pixel-wise mean (same raw scale as input images).
        If None, step 1 is skipped.
    bg_sigma_um     : float  – σ of the Gaussian background kernel in µm
    min_obj_um2     : float  – minimum object area in µm²
    max_hole_um2    : float  – maximum hole area in µm² to fill
    thresh_factor   : float  – multiplied by the per-frame Otsu threshold.
                               Ignored when *fixed_threshold* is set.
    fixed_threshold : float or None
        Absolute threshold applied to the normalised BG-removed image [0, 1].
        If None (default), Otsu × thresh_factor is used per frame.
        Obtain a good value with::

            t = calibrate_fluo_threshold(ref_image, temporal_mean, bg_sigma_um, px_um)
            detector = make_fluo_detector(..., fixed_threshold=t * thresh_factor)

    Returns
    -------
    detect_fn : callable  (image, px_um) -> bool mask
    """
    def _detect(image: np.ndarray, px_um: float) -> np.ndarray:
        img = image - temporal_mean if temporal_mean is not None else image
        img = exposure.rescale_intensity(img.astype(np.float32), out_range=(0.0, 1.0))

        sigma_px   = bg_sigma_um / px_um
        background = gaussian(img, sigma=sigma_px, preserve_range=True)
        bg_rem     = np.clip(img - background, 0, None)
        bg_rem     = exposure.rescale_intensity(bg_rem, out_range=(0.0, 1.0))

        if fixed_threshold is not None:
            thr = fixed_threshold
        else:
            thr = thresh_factor * threshold_otsu(bg_rem)

        binary = bg_rem > thr

        min_px = max(1, int(np.round(min_obj_um2  / px_um ** 2)))
        max_px = max(1, int(np.round(max_hole_um2 / px_um ** 2)))
        binary = remove_small_objects(binary, max_size=min_px)
        binary = remove_small_holes(binary, max_size=max_px)
        return binary

    _thr_desc = (f"fixed={fixed_threshold:.4f}" if fixed_threshold is not None
                 else f"Otsu×{thresh_factor}")
    _detect.__doc__ = (
        f"Fluo detector  bg_sigma={bg_sigma_um} µm  "
        f"threshold={_thr_desc}  "
        f"min_obj={min_obj_um2} µm²"
    )
    return _detect


# ──────────────────────────────────────────────────────────────────────────────
# Generic U-Net detector
# ──────────────────────────────────────────────────────────────────────────────

def _build_unet(features=(32, 64, 128, 256)):
    """Build and return a UNet instance (defined locally to keep detectors.py self-contained)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class _UNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1):
            super().__init__()
            self.pool      = nn.MaxPool2d(2, 2)
            self.downs     = nn.ModuleList()
            self.ups_trans = nn.ModuleList()
            self.ups_conv  = nn.ModuleList()
            ch = in_channels
            for feat in features:
                self.downs.append(_DoubleConv(ch, feat)); ch = feat
            self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)
            for feat in reversed(features):
                self.ups_trans.append(nn.ConvTranspose2d(feat * 2, feat, 2, stride=2))
                self.ups_conv.append(_DoubleConv(feat * 2, feat))
            self.final = nn.Conv2d(features[0], out_channels, 1)

        def forward(self, x):
            skips = []
            for down in self.downs:
                x = down(x); skips.append(x); x = self.pool(x)
            x = self.bottleneck(x)
            for i, (up_t, up_c) in enumerate(zip(self.ups_trans, self.ups_conv)):
                x = up_t(x); s = skips[-(i + 1)]
                if x.shape != s.shape:
                    import torch.nn.functional as F
                    x = F.interpolate(x, size=s.shape[2:])
                x = torch.cat([s, x], dim=1); x = up_c(x)
            return self.final(x)

    return _UNet()


def _unet_predict_raw(model, device, image, threshold,
                      patch_size=256, patch_stride=128):
    """Run tiled inference and return a probability map (float32) and binary mask."""
    import torch

    img = image.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo > 0:
        img = (img - lo) / (hi - lo)
    img = img.astype(np.float32)
    H, W = img.shape

    def _pad(n):
        if n <= patch_size:
            return patch_size - n
        r = (n - patch_size) % patch_stride
        return 0 if r == 0 else patch_stride - r

    img_pad = np.pad(img, ((0, _pad(H)), (0, _pad(W))), mode="reflect")
    H_pad, W_pad = img_pad.shape

    prob_sum = np.zeros((H_pad, W_pad), np.float32)
    count    = np.zeros((H_pad, W_pad), np.float32)

    model.eval()
    with torch.no_grad():
        for r in range(0, H_pad - patch_size + 1, patch_stride):
            for c in range(0, W_pad - patch_size + 1, patch_stride):
                patch = img_pad[r:r + patch_size, c:c + patch_size]
                t     = torch.from_numpy(patch[np.newaxis, np.newaxis]).to(device)
                prob  = torch.sigmoid(model(t)).squeeze().cpu().numpy()
                prob_sum[r:r + patch_size, c:c + patch_size] += prob
                count   [r:r + patch_size, c:c + patch_size] += 1.0

    prob_map = (prob_sum / np.maximum(count, 1))[:H, :W]
    return prob_map, prob_map >= threshold


def make_unet_detector(
    model_path,
    threshold=0.5,
    patch_size=256,
    patch_stride=128,
    features=(32, 64, 128, 256),
    min_obj_um2=0.0,
    post_process_fn=None,
):
    """
    Factory: generic U-Net detector.

    Loads the model once at factory time; the returned callable is lightweight.

    Parameters
    ----------
    model_path      : str or Path – path to the ``.pth`` checkpoint
    threshold       : float       – sigmoid probability threshold
    patch_size      : int         – inference patch size (pixels)
    patch_stride    : int         – stride between patches
    features        : tuple       – U-Net feature sizes (must match checkpoint)
    min_obj_um2     : float       – remove connected components whose area is
                                    smaller than this value (µm²).  0 = disabled.
    post_process_fn : callable or None
        Optional  (binary_mask, px_um) -> binary_mask  applied after thresholding
        and min-size filtering.

    Returns
    -------
    detect_fn : callable  (image, px_um) -> bool mask
    """
    import torch

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = _build_unet(features).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    def _detect(image: np.ndarray, px_um: float) -> np.ndarray:
        _, binary = _unet_predict_raw(
            model, device, image, threshold, patch_size, patch_stride
        )
        if min_obj_um2 > 0.0:
            min_px = max(1, int(np.round(min_obj_um2 / px_um ** 2)))
            binary = remove_small_objects(binary, max_size=min_px)
        if post_process_fn is not None:
            binary = post_process_fn(binary, px_um)
        return binary

    _detect.__doc__ = (
        f"UNet detector  model={model_path}  threshold={threshold}"
        f"  min_obj={min_obj_um2} µm²"
    )
    return _detect


def make_unet_pillar_detector(
    model_path,
    threshold=0.5,
    patch_size=256,
    patch_stride=128,
    min_pillar_um2=1000.0,
    features=(32, 64, 128, 256),
):
    """
    Factory: UNet pillar detector with built-in mask cleaning.

    Identical to ``make_unet_detector`` but adds the pillar-specific
    morphological clean-up from ``chipanalysis.functions.pillar_detection``.

    Returns
    -------
    detect_fn : callable  (image, px_um) -> bool mask
    """
    from chipanalysis.functions.pillar_detection import clean_pillar_mask

    def _post(binary, px_um):
        return clean_pillar_mask(binary, px_um, min_area_um2=min_pillar_um2)

    return make_unet_detector(
        model_path=model_path,
        threshold=threshold,
        patch_size=patch_size,
        patch_stride=patch_stride,
        features=features,
        post_process_fn=_post,
    )
