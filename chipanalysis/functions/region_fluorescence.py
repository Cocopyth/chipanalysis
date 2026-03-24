"""Region fluorescence analysis functions."""

import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from skimage import exposure
from skimage.feature import blob_log


def profile_mean(img, channel):
    """Compute mean intensity of an image region."""
    return np.mean(img)


def count_cells(img, channel, px_um=1.0, p_low=None, p_high=None):
    """
    Count cells/blobs in an image region using Laplacian of Gaussian.

    Blob size parameters are derived from expected physical cell sizes:
        - channel 1: ~10 µm width (e.g. D. discoideum)
        - other:     ~1-2 µm width (e.g. B. subtilis)

    Parameters
    ----------
    img : ndarray
        Image region to analyze.
    channel : int
        Channel index (determines expected cell size).
    px_um : float, optional
        Pixel size in µm. Used to convert cell size → sigma in pixels.
        Default is 1.0.
    p_low : float, optional
        Lower percentile value used to anchor intensity normalisation.
        When provided together with p_high, the image is rescaled to [0, 1]
        using these global bounds rather than the per-crop min/max, making
        ``blob_threshold`` consistent across all crops and timepoints.
    p_high : float, optional
        Upper percentile value (see p_low).
    blob_threshold : float, optional
        Absolute LoG response threshold. Stable as long as normalisation
        is consistent (i.e. p_low/p_high are supplied). Default is 0.05.

    Returns
    -------
    int
        Number of detected blobs.
    """
    img = img.astype(float)

    # ── Intensity normalisation ──────────────────────────────────────────────
    # Goal: map pixel values to [0, 1] using a *fixed* reference scale so that
    # blob_threshold (an absolute LoG response value) means the same thing for
    # every crop, every ROI, and every timepoint.
    #
    # Strategy:
    #   • p_low / p_high are the 1st and 99.5th percentiles computed ONCE on
    #     the chip ROI (the region encompassing all three sub-regions) from the
    #     first frame of each channel (see analyze_region_fluorescence or the
    #     debug notebook).  Using the chip ROI — rather than the full frame —
    #     ensures the reference is representative of the actual measurement area
    #     and is not skewed by off-chip background pixels.
    #   • rescale_intensity maps [p_low, p_high] → [0, 1].  Any value outside
    #     that range is clipped, not extrapolated.
    #   • If p_low/p_high are not supplied (e.g. standalone use), we fall back
    #     to per-crop min/max, which is less stable but still functional.
    #
    # Why NOT z-score?  Z-scoring uses the crop's own mean/std, so it changes
    # with every sub-image and makes an absolute threshold unreliable.
    # ─────────────────────────────────────────────────────────────────────────
    in_range = (p_low, p_high) if (p_low is not None and p_high is not None) else 'image'
    img = np.clip(
        exposure.rescale_intensity(img, in_range=in_range, out_range=(0.0, 1.0)),
        0.0, 1.0,
    )

    if channel == 1:
        # ~10 µm wide cells → radius ~5 µm
        cell_radius_um = 5.0
        blob_threshold=0.05
    else:
        # ~1-2 µm wide cells → radius ~0.75 µm
        cell_radius_um = 1.5
        blob_threshold=0.02


    # Convert physical radius to pixel sigma for LoG  (sigma = radius / sqrt(2))
    sigma_px = (cell_radius_um / px_um) / (2 ** 0.5)
    min_sigma = max(1.0, sigma_px * 0.6)
    max_sigma = max(2.0, sigma_px * 1.4)

    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=10, threshold=blob_threshold)
    return len(blobs)


def compute_profiles_over_time_roi(
    czi,
    times,
    channels,
    roi_selectors,
    roi_names,
    metrics=None,
    get_frame_fn=None,
    px_um=1.0,
    n_workers=1,
    norm_percentiles=None,
):
    """
    Compute metrics over time for multiple ROIs and channels.

    Parameters
    ----------
    czi : CziFile
        CZI file object.
    times : list of int
        Timepoints to process.
    channels : list of int
        Channel indices to process.
    roi_selectors : list of callable
        Functions that extract ROI from image.
    roi_names : list of str
        Names corresponding to roi_selectors.
    metrics : dict, optional
        Mapping of metric_name → callable(img, channel) → scalar.
        Defaults to {"mean": profile_mean}.
    get_frame_fn : callable, optional
        Function to load frames. Defaults to a simple lambda expecting
        get_frame(czi, t, channel, gamma=1) signature.
    px_um : float, optional
        Pixel size in µm. Forwarded to metric functions that accept it
        (e.g. count_cells). Default is 1.0.
    n_workers : int, optional
        Number of parallel worker threads for timestep processing.
        Each timestep (all channels and ROIs within it) is independent
        and can run concurrently. Default is 1 (sequential).
    norm_percentiles : dict, optional
        Mapping of channel → (p_low, p_high). When provided, functions
        that accept ``p_low``/``p_high`` keyword arguments (e.g. count_cells)
        receive the values for the current channel, anchoring intensity
        normalisation so that absolute thresholds are consistent across
        all crops and timepoints.

    Returns
    -------
    pd.DataFrame
        Columns: t, channel, roi_name, <metric1>, <metric2>, ...
    """
    if metrics is None:
        metrics = {"mean": profile_mean}

    if get_frame_fn is None:
        # Import here to avoid circular dependency
        from chipanalysis.utils.file_reader import get_frame
        get_frame_fn = get_frame

    # Pre-build a lookup for roi ordering so sort is stable after parallel gather
    roi_order = {name: i for i, name in enumerate(roi_names)}

    def _process_timestep(t):
        """Process all channels × ROIs for a single timestep."""
        rows = []
        for channel in channels:
            for roi_selector, roi_name in zip(roi_selectors, roi_names):
                img, _ = get_frame_fn(czi, t, channel, gamma=1)
                roi_chosen = roi_selector(img)

                row = {"t": t, "channel": channel, "roi_name": roi_name}

                for name, fn in metrics.items():
                    sig = inspect.signature(fn).parameters
                    kwargs = {}
                    if "px_um" in sig:
                        kwargs["px_um"] = px_um
                    if "p_low" in sig and norm_percentiles and channel in norm_percentiles:
                        kwargs["p_low"], kwargs["p_high"] = norm_percentiles[channel]
                    y = fn(roi_chosen, channel, **kwargs)
                    row[name] = y

                rows.append(row)
        return rows

    all_rows = []
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_timestep, t): t for t in times}
            for future in as_completed(futures):
                all_rows.extend(future.result())
        # Restore deterministic order: (t, channel, roi_name)
        all_rows.sort(key=lambda r: (r["t"], r["channel"], roi_order[r["roi_name"]]))
    else:
        for t in times:
            all_rows.extend(_process_timestep(t))

    return pd.DataFrame(all_rows)
