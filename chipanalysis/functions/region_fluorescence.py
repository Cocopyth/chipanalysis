"""Region fluorescence analysis functions."""

import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from skimage import exposure
from skimage.feature import blob_doh


def profile_mean(img, channel):
    """Compute mean intensity of an image region."""
    return np.mean(img)


def count_cells(img, channel, px_um=1.0):
    """
    Count cells/blobs in an image region using Determinant of Hessian.

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

    Returns
    -------
    int
        Number of detected blobs.
    """
    img = img.astype(float)
    img = exposure.rescale_intensity(img, in_range='image', out_range=(0, 1))
    img = (img - img.mean()) / (img.std() + 1e-8)

    if channel == 1:
        # ~10 µm wide cells → radius ~5 µm → sigma = radius / sqrt(2)
        cell_radius_um = 5.0
        threshold = 0.2
    else:
        # ~1-2 µm wide cells → radius ~0.75 µm
        cell_radius_um = 1.5
        threshold = 0.2

    # Convert physical radius to pixel sigma
    # For DoH, sigma ≈ radius / sqrt(2)
    sigma_px = (cell_radius_um / px_um) / (2 ** 0.5)
    min_sigma = max(1.0, sigma_px * 0.6)
    max_sigma = max(2.0, sigma_px * 1.4)

    blobs = blob_doh(img, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=10, threshold=threshold)
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

    Returns
    -------
    pd.DataFrame
        Columns: t, channel, roi_name, <metric1>, <metric2>, ...
    """
    if metrics is None:
        metrics = {"mean": profile_mean}

    if get_frame_fn is None:
        # Import here to avoid circular dependency
        from chipanalysis.utils.maye_video_axio import get_frame
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
                    # Forward px_um to functions that support it (e.g. count_cells)
                    if "px_um" in inspect.signature(fn).parameters:
                        y = fn(roi_chosen, channel, px_um=px_um)
                    else:
                        y = fn(roi_chosen, channel)
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
