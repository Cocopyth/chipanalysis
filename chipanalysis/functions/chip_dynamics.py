"""
chip_dynamics.py
================
Core pipeline functions for chip dynamics analysis.

All functions are pure numpy / pandas — no matplotlib — so this module is safe
to import on a cluster or inside batch scripts.

Typical pipeline
----------------
1.  czi, px_um, dim_sizes = load_czi(path)
2.  rotate_fn, align_result = get_rotation_fn(czi, bf_channel=2, scene=3)
3.  temporal_mean = compute_temporal_mean(czi, channel=1, scene=3, rotate_fn=rotate_fn)
4.  band_info    = align_result["band_info"]  # Fourier/delay channel geometry
6.  df_cells     = build_cell_dataframe(
                       czi, scene=3, channel=1, timepoints=[0,10,20],
                       rotate_fn=rotate_fn,
                       detect_fn=my_detect_fn,
                       band_info=align_result["band_info"],
                       px_um=px_um,
                   )

Detection function contract
---------------------------
Any object ``detect_fn`` must have the signature::

    def detect_fn(image: np.ndarray, px_um: float) -> np.ndarray:
        '''image is a 2-D float array (already rotated).
        Returns a boolean binary mask of the same spatial shape.'''

Use the factories in ``detectors.py`` to build these callables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from skimage.measure import label as sk_label, regionprops as sk_regionprops
from skimage.morphology import remove_small_objects
from skimage.transform import resize as sk_resize


# ──────────────────────────────────────────────────────────────────────────────
# CZI loading
# ──────────────────────────────────────────────────────────────────────────────

def load_czi(czi_path):
    """
    Open a CZI file and return basic metadata.

    Parameters
    ----------
    czi_path : str or Path

    Returns
    -------
    czi       : CziFile
    px_um     : float   – pixel size in µm (X axis)
    dim_sizes : dict    – dimension letter → size  e.g. {"T": 120, "S": 4, …}
    pixel_size : dict   – full pixel-size dict from get_pixel_sizes_um
    """
    from aicspylibczi import CziFile
    from chipanalysis.utils.file_reader import get_pixel_sizes_um

    czi = CziFile(Path(czi_path))
    pixel_size = get_pixel_sizes_um(czi)
    dim_sizes = dict(zip(czi.dims, czi.size))
    return czi, pixel_size["X"], dim_sizes, pixel_size


# ──────────────────────────────────────────────────────────────────────────────
# Rotation / alignment
# ──────────────────────────────────────────────────────────────────────────────

def get_rotation_fn(
    czi,
    bf_channel,
    scene,
    timepoint=0,
    px_um=None,
    debug=False,
    alignment_method="fourier_channel",
    alignment_kwargs=None,
    chip_geometry=None,
):
    """
    Estimate chip rotation from a brightfield frame.

    Parameters
    ----------
    czi         : CziFile
    bf_channel  : int   – brightfield channel index
    scene       : int or None
    timepoint   : int   – frame used for alignment (default 0)
    px_um       : float or None – if None, re-reads from czi
    debug       : bool  – if True, alignment generates diagnostic figures stored
                          in align_result['figures']
    alignment_method : {"fourier_channel", "fft_only"}
        Preferred default is "fourier_channel", which detects the main channel
        from Fourier periodicity and delay refinement. Use "fft_only" for the
        older rotation-only behavior.
    alignment_kwargs : dict or None
        Extra keyword arguments forwarded to the selected alignment function.
    chip_geometry : ChipGeometry or None
        Geometry selected from the manifest chip column. If omitted, the
        generic geometry is used.

    Returns
    -------
    rotate_fn    : callable  img -> rotated_img   (works on any dtype / shape)
    align_result : dict      full output of align_chip_to_image
    """
    from chipanalysis.utils.file_reader import get_frame, get_pixel_sizes_um
    from chipanalysis.chip_alignment import (
        align_chip_to_image,
        align_chip_to_image_fourier_channel,
        ChipGeometry,
    )

    if px_um is None:
        px_um = get_pixel_sizes_um(czi)["X"]

    if alignment_kwargs is None:
        alignment_kwargs = {}
    if chip_geometry is None:
        chip_geometry = ChipGeometry()

    img_bf, _ = get_frame(czi, timepoint, bf_channel, scene=scene)
    if alignment_method in ("fourier_channel", "fourier", "fft_channel"):
        align_result = align_chip_to_image_fourier_channel(
            img_bf,
            pixel_size_um=px_um,
            debug=debug,
            geom=chip_geometry,
            **alignment_kwargs,
        )
    elif alignment_method in ("fft_only", "legacy_fft", "orientation_only"):
        align_result = align_chip_to_image(
            img_bf,
            pixel_size_um=px_um,
            debug=debug,
            geom=chip_geometry,
            **alignment_kwargs,
        )
    else:
        raise ValueError(
            "alignment_method must be 'fourier_channel' or 'fft_only'; "
            f"got {alignment_method!r}"
        )
    return align_result["rotate_fn"], align_result


def get_rotated_frame(czi, timepoint, channel, scene, rotate_fn,
                      gamma=1.0, stretch_min=1, stretch_max=99.5):
    """
    Load a single frame and return (raw_rotated, display_rotated).

    Convenience wrapper around get_frame + rotate_fn.
    """
    from chipanalysis.utils.file_reader import get_frame

    raw, disp = get_frame(czi, timepoint, channel, scene=scene,
                          gamma=gamma,
                          stretch_min=stretch_min,
                          stretch_max=stretch_max)
    return rotate_fn(raw), rotate_fn(disp)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal mean
# ──────────────────────────────────────────────────────────────────────────────

def compute_temporal_mean(czi, channel, scene, rotate_fn, n_frames=10):
    """
    Pixel-wise mean over the first *n_frames* timepoints.

    Useful for static-background subtraction in fluorescence channels.

    Returns
    -------
    temporal_mean : np.ndarray float32  (H, W)
    """
    from chipanalysis.utils.file_reader import get_frame

    accum = None
    for t in range(n_frames):
        raw, _ = get_frame(czi, t, channel, scene=scene)
        frame = rotate_fn(raw.astype(np.float64))
        accum = frame if accum is None else accum + frame
    return (accum / n_frames).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Channel / band detection from pillar mask
# ──────────────────────────────────────────────────────────────────────────────

def detect_channel_from_mask(pillar_mask, px_um, crop_cols=150):
    """
    Find the empty channel separating the top and bottom pillar clusters
    using a row-projection of the binary mask.

    The mask is assumed to have two clusters of foreground pixels – one in
    the top half and one in the bottom half of the image – with a clear gap
    (the microfluidic channel) in between.

    Parameters
    ----------
    pillar_mask : np.ndarray bool  (H, W)
    px_um       : float
    crop_cols   : int  – columns trimmed on each side before projecting
                         (removes edge artefacts from rotation padding)

    Returns
    -------
    band_info : dict with keys
        band_top           – row index of the lower edge of the top cluster
        band_bottom        – row index of the upper edge of the bottom cluster
        band_centre_row    – float, midpoint between band_top and band_bottom
        band_half_width_um – half-width of the channel in µm
        band_width_um      – full channel width in µm
        band_width_px      – full channel width in pixels
        row_projection     – 1-D float32 array (smoothed) for diagnostics
    """
    mask = pillar_mask[:, crop_cols:-crop_cols] if crop_cols > 0 else pillar_mask
    row_fill = mask.mean(axis=1).astype(np.float32)
    row_smooth = uniform_filter1d(row_fill, size=5)

    mid = row_smooth.size // 2
    top_rows    = row_smooth[:mid]
    bottom_rows = row_smooth[mid:]

    filled_top = np.where(top_rows > 0)[0]
    filled_bot = np.where(bottom_rows > 0)[0]

    if len(filled_top) == 0 or len(filled_bot) == 0:
        raise ValueError(
            "detect_channel_from_mask: could not find pillar clusters in both "
            "halves of the image. Check the pillar detection or increase crop_cols."
        )

    band_top    = int(filled_top.max())
    band_bottom = int(filled_bot.min()) + mid

    return {
        "band_top":           band_top,
        "band_bottom":        band_bottom,
        "band_centre_row":    (band_top + band_bottom) / 2.0,
        "band_half_width_um": (band_bottom - band_top) / 2.0 * px_um,
        "band_width_um":      (band_bottom - band_top) * px_um,
        "band_width_px":      band_bottom - band_top,
        "row_projection":     row_smooth,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ──────────────────────────────────────────────────────────────────────────────

def build_cell_dataframe(
    czi,
    scene,
    channel,
    timepoints,
    rotate_fn,
    detect_fn,
    band_info,
    px_um,
    min_obj_um2=25.0,
    reference_shape=None,
    verbose=True,
):
    """
    Run *detect_fn* over a list of timepoints and build a position dataframe.

    Parameters
    ----------
    czi            : CziFile
    scene          : int or None
    channel        : int   – channel index to load for detection
    timepoints     : list[int]
    rotate_fn      : callable img -> rotated_img
    detect_fn      : callable (image, px_um) -> bool mask
                     Use ``make_fluo_detector`` or ``make_unet_detector`` from
                     ``chipanalysis.functions.detectors``.
    band_info      : dict   – main-channel geometry. Usually
                     align_result["band_info"] from Fourier/delay alignment;
                     legacy mask-based output from detect_channel_from_mask is
                     still accepted.
    px_um          : float
    min_obj_um2    : float  – objects smaller than this are discarded
    reference_shape: (H, W) or None – resize masks to this shape if given
    verbose        : bool

    Returns
    -------
    df_cells : pd.DataFrame with columns:
        timepoint, id, centroid_row, centroid_col, centroid_row_um, centroid_col_um,
        dist_to_band_px, dist_to_band_um, area_px, area_um2, time_hours
    """
    from chipanalysis.utils.file_reader import get_frame, get_timestamps_by_T

    band_centre_row = band_info["band_centre_row"]
    min_px = max(1, int(min_obj_um2 / px_um ** 2))
    records = []
    
    # Get timestamps for all timepoints to compute time in hours
    timestamps = get_timestamps_by_T(czi, S=scene if scene is not None else 0, C=channel, Z=0)
    time_dict = {t: dt for t, dt in timestamps if dt is not None}
    
    # Determine reference time (first timepoint or first available timestamp)
    if timepoints and timepoints[0] in time_dict:
        ref_time = time_dict[timepoints[0]]
    elif time_dict:
        ref_time = time_dict[min(time_dict.keys())]
    else:
        ref_time = None

    for t in timepoints:
        raw, _ = get_frame(czi, t, channel, scene=scene,
                           gamma=1.0, stretch_min=1, stretch_max=99.5)
        img_t = rotate_fn(raw)

        binary = detect_fn(img_t, px_um)

        if reference_shape is not None and binary.shape != reference_shape:
            binary = sk_resize(
                binary.astype(np.float32), reference_shape,
                order=0, preserve_range=True, anti_aliasing=False,
            ).astype(bool)

        binary   = remove_small_objects(binary, min_size=min_px)
        labeled  = sk_label(binary)
        regions  = sk_regionprops(labeled)
        
        # Calculate time in hours
        time_hours = None
        if ref_time is not None and t in time_dict:
            from datetime import datetime
            current_time = time_dict[t]
            if isinstance(current_time, datetime) and isinstance(ref_time, datetime):
                time_diff = (current_time - ref_time).total_seconds()
                time_hours = time_diff / 3600.0

        if verbose:
            time_str = f", time={time_hours:.2f}h" if time_hours is not None else ""
            print(f"  t={t:4d}  objects detected: {len(regions)}{time_str}")

        for reg in regions:
            cy, cx  = reg.centroid
            dist_px = cy - band_centre_row
            perimeter_px = float(reg.perimeter)
            circularity = (
                4.0 * np.pi * reg.area / (perimeter_px ** 2)
                if perimeter_px > 0.0 else np.nan
            )
            records.append({
                "timepoint":       t,
                "id":              reg.label,
                "centroid_row":    cy,
                "centroid_col":    cx,
                "centroid_row_um": cy * px_um,
                "centroid_col_um": cx * px_um,
                "dist_to_band_px": dist_px,
                "dist_to_band_um": dist_px * px_um,
                "area_px":         reg.area,
                "area_um2":        reg.area * px_um ** 2,
                "perimeter_px":    perimeter_px,
                "perimeter_um":    perimeter_px * px_um,
                "circularity":     circularity,
                "time_hours":      time_hours,
            })

    return pd.DataFrame(records)
