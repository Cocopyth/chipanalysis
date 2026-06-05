"""
Chip alignment and bounding box detection.

Pipeline:
1. Determine image orientation via FFT
2. Extract horizontal band containing chip channels
3. Build theoretical interface comb from design parameters
4. Correlate design comb with measured signal
5. Find chip edges and compute bounding box
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import correlate, savgol_filter
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChipGeometry:
    """
    Explicit chip geometry and channel parameters.
    
    All dimensions in µm unless otherwise noted.
    """
    # Main channel geometry
    main_channel_width_um: float = 250.0
    side_sub_channel_width_um: float = 355.0
    
    # PPA substrate thickness (defines chip height)
    ppa_thickness_um: float = 1400.0
    
    # Periodicity detection parameters
    min_period_um: float = 40.0
    max_period_um: float = 100.0
    band_height_px: int = 50  # Height of horizontal strips for scoring (pixels, not µm)
    blur_sigma_bg: Tuple[int, int] = (5, 25)  # (y, x) for background subtraction
    
    # Design parameters for interface comb
    min_width_um: float = 10.0
    max_width_um: float = 50.0
    gap_um: float = 65.0
    total_length_um: float = 6000.0


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: ORIENTATION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_image_orientation(img: np.ndarray, nbins: int = 180) -> Tuple[float, float]:
    """
    Find dominant stripe direction via angular FFT spectrum.
    
    Parameters
    ----------
    img : np.ndarray
        2D grayscale image
    nbins : int
        Number of angular bins for histogram
        
    Returns
    -------
    peak_theta_deg : float
        Peak angle in FFT domain (degrees)
    spatial_dir_deg : float
        Estimated striping direction in image (perpendicular to FFT peak)
    """
    img_float = img.astype(np.float64)
    
    # Window to reduce edge leakage
    wy = np.hanning(img_float.shape[0])
    wx = np.hanning(img_float.shape[1])
    img_windowed = img_float * np.outer(wy, wx)
    img_windowed = img_windowed - img_windowed.mean()
    
    # FFT and magnitude
    F = np.fft.fftshift(np.fft.fft2(img_windowed))
    mag = np.log1p(np.abs(F))
    
    H, W = img_windowed.shape
    ky = np.fft.fftshift(np.fft.fftfreq(H, d=1.0))
    kx = np.fft.fftshift(np.fft.fftfreq(W, d=1.0))
    KX, KY = np.meshgrid(kx, ky)
    
    r = np.hypot(KX, KY)
    theta = np.arctan2(KY, KX)
    theta = np.mod(theta, np.pi)
    
    # Angular histogram (exclude very low frequencies)
    mask = (r >= 0.02) & (r <= 0.5)
    edges = np.linspace(0.0, np.pi, nbins + 1)
    bin_idx = np.digitize(theta[mask], edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)
    
    vals = mag[mask]
    sums = np.bincount(bin_idx, weights=vals, minlength=nbins)
    counts = np.bincount(bin_idx, minlength=nbins)
    ang_mean = sums / np.maximum(counts, 1)
    
    theta_centers = 0.5 * (edges[:-1] + edges[1:])
    peak_idx = np.argmax(ang_mean)
    peak_theta_deg = np.degrees(theta_centers[peak_idx])
    
    # Image striping direction is perpendicular to FFT peak
    spatial_dir_deg = (peak_theta_deg + 90) % 180
    
    _fft_debug = {
        "mag":           mag,
        "ang_mean":      ang_mean,
        "theta_centers": theta_centers,
    }
    return peak_theta_deg, spatial_dir_deg, _fft_debug


def rotate_image_to_horizontal(img: np.ndarray, spatial_dir_deg: float) -> np.ndarray:
    """
    Rotate image so stripes are horizontal.
    
    Parameters
    ----------
    img : np.ndarray
        2D grayscale image
    spatial_dir_deg : float
        Stripe direction in image (degrees)
        
    Returns
    -------
    img_rotated : np.ndarray
        Rotated image
    """
    # Stripes should be horizontal (0°), so rotate by -(spatial_dir_deg - 90)
    rotate_angle = spatial_dir_deg - 90
    return ndimage.rotate(img, rotate_angle, reshape=False)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: BAND EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def find_middle_channel_position(
    img_rotated: np.ndarray,
    pixel_size_um: float,
    geom: ChipGeometry = None,
) -> Tuple[float, float]:
    """
    Locate the middle zone (main channel + 2×side sub-channels) via periodicity scoring.
    Uses autocorrelation-based periodicity detection to find repeating channel structure.
    
    Parameters
    ----------
    img_rotated : np.ndarray
        Horizontally-rotated image
    pixel_size_um : float
        Physical pixel size (µm)
    geom : ChipGeometry, optional
        Chip geometry parameters. Defaults to ChipGeometry().
        
    Returns
    -------
    x_middle : float
        Y-position of middle zone center (pixels)
    middle_px : float
        Height of middle zone (pixels)
    """
    from scipy.signal import find_peaks
    
    if geom is None:
        geom = ChipGeometry()
    
    # Geometry
    middle_um = geom.main_channel_width_um + 2 * geom.side_sub_channel_width_um
    middle_px = middle_um / pixel_size_um
    min_period_px = geom.min_period_um / pixel_size_um
    max_period_px = geom.max_period_um / pixel_size_um
    
    # Background subtraction to remove slow illumination gradient
    img_float = img_rotated.astype(np.float64)
    bg = ndimage.gaussian_filter(img_float, geom.blur_sigma_bg)
    hp = img_float - bg
    
    # Periodicity score: autocorrelation peak prominence in target lag range
    def periodicity_score_from_autocorr(profile, min_period_px, max_period_px):
        x = profile.astype(np.float64)
        x = x - np.mean(x)
        if np.std(x) < 1e-12:
            return 0.0
        x = x / (np.std(x) + 1e-12)
        ac = np.correlate(x, x, mode="full")
        ac = ac[len(ac) // 2:]   # non-negative lags
        ac = ac / (ac[0] + 1e-12)
        lo = max(1, int(min_period_px))
        hi = min(len(ac), int(max_period_px))
        if hi <= lo + 2:
            return 0.0
        region = ac[lo:hi]
        peaks, props = find_peaks(region, prominence=0.02)
        if len(peaks) == 0:
            return 0.0
        return np.max(props["prominences"])
    
    # Compute periodicity scores for each horizontal band (on background-subtracted image)
    half = geom.band_height_px // 2
    ys = []
    pscore_list = []
    
    for y in range(half, hp.shape[0] - half, 1):
        strip = hp[y - half:y + half + 1, :]
        profile = np.mean(strip, axis=0)
        pscore_val = periodicity_score_from_autocorr(profile, min_period_px, max_period_px)
        ys.append(y)
        pscore_list.append(pscore_val)
    
    ys = np.array(ys)
    pscore = np.array(pscore_list)
    
    # Sliding window minimum: find where periodicity is lowest (channels disrupt it)
    def fit_box(x0, middle_px, signal):
        x1 = x0 + int(middle_px)
        return np.mean(signal[x0:x1])
    
    begin = int(500 / pixel_size_um)
    fits = [fit_box(x0, middle_px, pscore) for x0 in range(len(pscore) - int(middle_px))]
    
    # Exact notebook formula: argmin in slice [begin:], then add offsets
    x_middle = np.argmin(fits[begin:]) + middle_px / 2 + begin + ys[0]
    
    return float(x_middle), float(middle_px)


def extract_band_region(
    img_rotated: np.ndarray,
    x_middle: float,
    middle_px: float,
    pixel_size_um: float,
    geom: ChipGeometry = None,
) -> np.ndarray:
    """
    Extract the horizontal band containing main channel + side sub-channels.
    
    Parameters
    ----------
    img_rotated : np.ndarray
        Rotated image
    x_middle : float
        Y-center of middle zone (pixels)
    middle_px : float
        Height of middle zone (pixels)
    pixel_size_um : float
        Pixel size (µm)
    geom : ChipGeometry, optional
        Chip geometry. Defaults to ChipGeometry().
        
    Returns
    -------
    band : np.ndarray
        Extracted band (concatenated top + bottom side channels)
    """
    if geom is None:
        geom = ChipGeometry()
    
    side_px = geom.side_sub_channel_width_um / pixel_size_um
    
    # Top side channel
    y1_start = int(x_middle - middle_px / 2 - side_px / 10)
    y1_end = int(x_middle - middle_px / 2 + side_px + side_px / 10)
    band_top = img_rotated[y1_start:y1_end, :]
    
    # Bottom side channel
    y2_start = int(x_middle + middle_px / 2 - side_px - side_px / 10)
    y2_end = int(x_middle + middle_px / 2 + side_px / 10)
    band_bottom = img_rotated[y2_start:y2_end, :]
    
    band = np.concatenate([band_top, band_bottom], axis=0)
    return band


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: DESIGN COMB
# ─────────────────────────────────────────────────────────────────────────────

def build_ppa_interface_comb(
    min_width_um: float = 10.0,
    max_width_um: float = 50.0,
    gap_um: float = 65.0,
    total_length_um: float = 6000.0,
    sample_dx_um: float = 0.1,
    smoothing_sigma_px: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Build 1D interface comb for variable-width channel design.
    
    Parameters
    ----------
    min_width_um : float
        Narrowest channel width (µm)
    max_width_um : float
        Widest channel width (µm)
    gap_um : float
        Gap (pillar) width between channels (µm)
    total_length_um : float
        Total available length (µm)
    sample_dx_um : float
        Sampling resolution (µm)
    smoothing_sigma_px : float
        Gaussian broadening sigma (pixels)
        
    Returns
    -------
    positions_um : np.ndarray
        Position axis (µm)
    comb : np.ndarray
        Normalized comb signal [0, 1]
    interfaces_um : list[float]
        Interface positions (µm)
    widths_um : list[float]
        Channel widths (µm)
    """
    if min_width_um > max_width_um:
        min_width_um, max_width_um = max_width_um, min_width_um
    
    a, b, g = float(min_width_um), float(max_width_um), float(gap_um)
    
    # Find N channels that fit in total_length_um
    def total_extent(N: int) -> float:
        if N <= 0:
            return 0.0
        if N == 1:
            return a
        return N * (a + b) / 2.0 + (N - 1) * g
    
    N = 1
    while total_extent(N + 1) <= total_length_um + 1e-9:
        N += 1
    
    # Linear width progression
    widths_um = [a + (float(i) / (N - 1)) * (b - a) for i in range(N)] if N > 1 else [a]
    
    # Center pattern
    extent = sum(widths_um) + (N - 1) * g
    start = 0.5 * (total_length_um - extent)
    
    # Collect interfaces
    interfaces_um = []
    pos = start
    for i, w in enumerate(widths_um):
        interfaces_um.append(pos)
        interfaces_um.append(pos + w)
        pos += w
        if i < N - 1:
            pos += g
    
    # Build comb
    positions_um = np.arange(0.0, total_length_um + sample_dx_um, sample_dx_um)
    comb = np.zeros_like(positions_um)
    for iface in interfaces_um:
        idx = int(round(iface / sample_dx_um))
        if 0 <= idx < len(comb):
            comb[idx] = 1.0
    
    # Broaden
    if smoothing_sigma_px > 0:
        comb = gaussian_filter1d(comb, sigma=smoothing_sigma_px)
        comb /= comb.max() + 1e-12
    
    return positions_um, comb, interfaces_um, widths_um


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def extract_1d_signal(band: np.ndarray) -> np.ndarray:
    """
    Average band across height to get 1D signal.
    
    Parameters
    ----------
    band : np.ndarray
        2D band image
        
    Returns
    -------
    signal_1d : np.ndarray
        1D averaged signal
    """
    return np.mean(band, axis=0)


def compute_signal_peaks(
    signal_1d: np.ndarray,
    pixel_size_um: float,
    sg_window_um: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute second derivative of normalized signal to detect dark minima.
    
    Dark interfaces are concave-up (d²I/dx² > 0) → one peak per dip.
    Flat regions have d² ≈ 0.
    
    Parameters
    ----------
    signal_1d : np.ndarray
        1D signal
    pixel_size_um : float
        Pixel size (µm)
    sg_window_um : float
        Savitzky-Golay window (µm)
        
    Returns
    -------
    signal_inv : np.ndarray
        Normalized peak signal [0, 1]
    d2 : np.ndarray
        Raw second derivative
    """
    signal = signal_1d.astype(float)
    signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-12)
    
    sg_window_px = int(sg_window_um / pixel_size_um)
    sg_window_px += 1 - sg_window_px % 2  # Must be odd
    
    d2 = savgol_filter(signal_norm, window_length=sg_window_px, polyorder=3, deriv=2)
    signal_inv = np.maximum(0, d2)  # Keep only concave-up
    signal_inv /= signal_inv.max() + 1e-12
    
    return signal_inv, d2


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: CROSS-CORRELATION & ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def correlate_comb_to_signal(
    signal_inv: np.ndarray,
    comb_design: np.ndarray,
    interfaces_design_px: np.ndarray,
    template_offset: int,
) -> Dict:
    """
    Cross-correlate design comb (normal & flipped) with signal.
    
    Parameters
    ----------
    signal_inv : np.ndarray
        Signal peaks
    comb_design : np.ndarray
        Design comb template (already cropped)
    interfaces_design_px : np.ndarray
        Interface positions in full design (pixels)
    template_offset : int
        Start of crop window in full design (pixels)
        
    Returns
    -------
    result : dict
        Keys: shift_normal, score_normal, shift_flipped, score_flipped,
              best_shift, best_interfaces_px, is_flipped, orientation,
              aligned_comb
    """
    n_signal = len(signal_inv)
    n_comb = len(comb_design)
    
    xcorr_normal = correlate(signal_inv, comb_design, mode='full')
    xcorr_flipped = correlate(signal_inv, comb_design[::-1], mode='full')
    
    lags = np.arange(-(n_comb - 1), n_signal)
    
    best_idx_n = np.argmax(xcorr_normal)
    best_idx_f = np.argmax(xcorr_flipped)
    shift_normal = lags[best_idx_n]
    shift_flipped = lags[best_idx_f]
    score_normal = xcorr_normal[best_idx_n]
    score_flipped = xcorr_flipped[best_idx_f]
    
    # Select best orientation
    if score_normal >= score_flipped:
        best_shift = shift_normal
        best_comb = comb_design.copy()
        is_flipped = False
        orientation = "normal (min→max width)"
    else:
        best_shift = shift_flipped
        best_comb = comb_design[::-1].copy()
        is_flipped = True
        orientation = "flipped (max→min width)"
    
    # Map design interfaces to signal space
    crop_mask = interfaces_design_px >= template_offset
    ifaces_in_crop = interfaces_design_px[crop_mask] - template_offset
    
    if is_flipped:
        best_interfaces_px = (n_comb - 1 - ifaces_in_crop) + best_shift
    else:
        best_interfaces_px = ifaces_in_crop + best_shift
    
    valid = (best_interfaces_px >= 0) & (best_interfaces_px < n_signal)
    aligned_px = np.sort(best_interfaces_px[valid])
    
    # Place comb in signal space
    aligned_comb = np.zeros(n_signal)
    s = int(best_shift)
    sig_start = max(0, s)
    comb_start = max(0, -s)
    sig_end = min(n_signal, s + n_comb)
    comb_end = comb_start + (sig_end - sig_start)
    if sig_end > sig_start:
        aligned_comb[sig_start:sig_end] = best_comb[comb_start:comb_end]
    
    return {
        'shift_normal': shift_normal,
        'score_normal': score_normal,
        'shift_flipped': shift_flipped,
        'score_flipped': score_flipped,
        'best_shift': best_shift,
        'best_interfaces_px': best_interfaces_px,
        'aligned_px': aligned_px,
        'is_flipped': is_flipped,
        'orientation': orientation,
        'aligned_comb': aligned_comb,
        'valid_count': valid.sum(),
        'total_count': crop_mask.sum(),
        'xcorr_normal': xcorr_normal,
        'xcorr_flipped': xcorr_flipped,
        'lags': lags,
    }


def find_first_match(
    aligned_px: np.ndarray,
    aligned_comb: np.ndarray,
    signal_inv: np.ndarray,
    is_flipped: bool,
    pixel_size_um: float,
    match_threshold: float = 0.05,
    peak_window_um: float = 5.0,
) -> Optional[float]:
    """
    Find rightmost (normal) or leftmost (flipped) confident match.
    
    Parameters
    ----------
    aligned_px : np.ndarray
        Aligned interface positions (pixels)
    aligned_comb : np.ndarray
        Aligned comb in signal space
    signal_inv : np.ndarray
        Signal peaks
    is_flipped : bool
        Whether orientation is flipped
    pixel_size_um : float
        Pixel size (µm)
    match_threshold : float
        Threshold for comb×signal product
    peak_window_um : float
        Search window half-width (µm)
        
    Returns
    -------
    first_match_um : float or None
        Position of first match (µm), or None if not found
    """
    peak_window_px = max(1, int(peak_window_um / pixel_size_um))
    product = aligned_comb * signal_inv
    n_signal = len(signal_inv)
    
    # Normal: walk right→left; Flipped: walk left→right
    scan_order = aligned_px if is_flipped else aligned_px[::-1]
    
    for px in scan_order:
        ipx = int(round(px))
        lo = max(0, ipx - peak_window_px)
        hi = min(n_signal, ipx + peak_window_px + 1)
        if hi > lo and np.max(product[lo:hi]) >= match_threshold:
            return ipx * pixel_size_um
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def align_chip_to_image(
    img: np.ndarray,
    pixel_size_um: float,
    geom: ChipGeometry = None,
    crop_um: float = 1800.0,   # kept for API compatibility, unused
    debug: bool = False,
) -> Dict:
    """
    Determine chip rotation via FFT angular spectrum and return a rotate_fn.

    Only the FFT-based orientation step is performed — no band finding,
    no comb correlation.  The returned result dict is compatible with
    get_roi_from_result (those keys are None when not computed).

    Parameters
    ----------
    img           : np.ndarray  2-D grayscale image
    pixel_size_um : float
    geom          : ChipGeometry (unused, kept for API compatibility)
    crop_um       : float        (unused, kept for API compatibility)
    debug         : bool         show FFT orientation + rotated image figures

    Returns
    -------
    result : dict with keys
        rotate_fn, rotate_angle_deg, success, messages, scores, pixel_size_um
        (and 'figures' when debug=True)
        bounding_box, x_middle_px, middle_px, main_px, is_flipped → None
    """
    if geom is None:
        geom = ChipGeometry()

    figures  = {}
    messages = []
    scores   = {}

    try:
        # ── FFT orientation ───────────────────────────────────────────────────
        messages.append("Finding image orientation via FFT...")
        peak_theta, spatial_dir, _fft_dbg = find_image_orientation(img)
        messages.append(
            f"  → FFT peak: {peak_theta:.1f}°  spatial direction: {spatial_dir:.1f}°"
        )
        scores['orientation_confidence'] = 0.9

        # Wrap raw rotation (peak_theta − 90) into [−45°, +45°].
        # The FFT spectrum has π-periodicity, so a peak at e.g. 170° means
        # the same stripe as a peak at 0° — the ±90° fold resolves that.
        rotate_angle = peak_theta - 90.0
        if rotate_angle > 45.0:
            rotate_angle -= 90.0
        elif rotate_angle < -45.0:
            rotate_angle += 90.0
        messages.append(
            f"  → Raw rotation: {peak_theta - 90.0:.1f}°  →  applied: {rotate_angle:.1f}°"
        )
        scores['rotation_success'] = 1.0
        success = True

        if debug:
            _theta_deg = np.degrees(_fft_dbg["theta_centers"])

            fig, axes = plt.subplots(1, 3, figsize=(20, 5))

            axes[0].imshow(img, cmap="gray")
            axes[0].set_title(
                f"Original image\n"
                f"stripe dir: {spatial_dir:.1f}°  →  rotation to apply: {rotate_angle:.1f}°"
            )
            axes[0].axis("off")

            axes[1].imshow(_fft_dbg["mag"], cmap="inferno", origin="upper")
            axes[1].set_title("FFT log-magnitude spectrum")
            axes[1].axis("off")

            axes[2].plot(_theta_deg, _fft_dbg["ang_mean"], color="steelblue", lw=1.5)
            axes[2].axvline(
                peak_theta, color="red", lw=1.5, ls="-",
                label=f"FFT peak  {peak_theta:.1f}°  (raw rot {peak_theta - 90:.1f}°)",
            )
            axes[2].axvline(
                rotate_angle + 90.0, color="orange", lw=1.5, ls="--",
                label=f"Effective peak  →  rotation {rotate_angle:.1f}°",
            )
            axes[2].set_xlabel("Angle (degrees)")
            axes[2].set_ylabel("Mean FFT magnitude")
            axes[2].set_title("Angular power spectrum")
            axes[2].legend(fontsize=9)
            axes[2].grid(True, alpha=0.3)
            plt.suptitle(
                f"FFT orientation  (raw {peak_theta - 90:.1f}° → wrapped {rotate_angle:.1f}°)",
                fontsize=10,
            )
            plt.tight_layout()
            figures['00_fft_orientation'] = fig

            img_rotated_dbg = ndimage.rotate(img, rotate_angle, reshape=False)
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            ax2.imshow(img_rotated_dbg, cmap='gray')
            ax2.set_title(f'Rotated image  (angle = {rotate_angle:.1f}°)')
            figures['02_rotated'] = fig2

    except Exception as e:
        messages.append(f"ERROR: {str(e)}")
        rotate_angle = 0.0
        success = False

    def rotate_fn(other_img: np.ndarray) -> np.ndarray:
        """Apply the rotation found during FFT alignment to another image."""
        return ndimage.rotate(other_img, rotate_angle, reshape=False)

    result = {
        'success':          success,
        'rotate_angle_deg': rotate_angle,
        'rotate_fn':        rotate_fn,
        'pixel_size_um':    pixel_size_um,
        'scores':           scores,
        'messages':         messages,
        # Fields expected by get_roi_from_result — not computed in FFT-only mode
        'bounding_box':     None,
        'x_middle_px':      None,
        'middle_px':        None,
        'main_px':          None,
        'is_flipped':       None,
    }

    if debug:
        result['figures'] = figures

    return result


def get_roi_from_result(
    result: Dict,
    img_rotated: np.ndarray,
    region: str = "full",
    pad_left_um: float = 0.0,
    pad_right_um: float = 0.0,
    pad_top_um: float = 0.0,
    pad_bottom_um: float = 0.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Crop a ROI from the rotated image using the alignment result.

    Can extract the full chip, main channel only, or one of the side
    sub-channels, all with configurable padding.

    Parameters
    ----------
    result : dict
        Output of ``align_chip_to_image``.
    img_rotated : np.ndarray
        Rotated image (apply ``result['rotate_fn']`` to the raw image first).
    region : str
        Which region to extract: 'full' (all channels), 'main' (main channel only),
        'top' (top side sub-channel), or 'bottom' (bottom side sub-channel).
        Default: 'full'.
    pad_left_um : float
        Extra padding on left edge of chip box (µm). Positive extends outward;
        negative shrinks.
    pad_right_um : float
        Extra padding on right edge of chip box (µm).
    pad_top_um : float
        Extra padding on the selected region's top (µm).
    pad_bottom_um : float
        Extra padding on the selected region's bottom (µm).
        Defaults to ``pad_top_um`` when 0 (symmetric).

    Returns
    -------
    roi : np.ndarray
        Cropped image patch.
    roi_coords : dict
        Pixel and physical coordinates of the crop:
        ``x0_px, x1_px, y0_px, y1_px`` (pixels in rotated image) and
        ``left_um, right_um, top_um, bottom_um`` (µm).

    Raises
    ------
    ValueError
        If ``region`` is not one of 'full', 'main', 'top', 'bottom'.
    """
    if region not in ('full', 'main', 'top', 'bottom'):
        raise ValueError(
            f"region must be one of 'full', 'main', 'top', 'bottom'; got {region!r}"
        )

    box = result['bounding_box']
    pixel_size_um = result['pixel_size_um']
    x_middle_px = result['x_middle_px']
    middle_px   = result['middle_px']
    is_flipped = result.get('is_flipped', True)
    if pad_bottom_um == 0.0:
        pad_bottom_um = pad_top_um
    # When flipped, top/bottom regions are swapped because the chip
    # orientation is reversed. Swap them now so geometry is consistent.
    if not is_flipped and region in ('top', 'bottom'):
        region = 'bottom' if region == 'top' else 'top'
    
    # # When not flipped, we apply 180° rotation at the end, so padding
    # # directions also rotate: top↔bottom, left↔right.
    if not is_flipped:
        pad_left_um, pad_right_um = pad_right_um, pad_left_um
        pad_top_um, pad_bottom_um = pad_bottom_um, pad_top_um

    # X extent: chip left/right + padding (always the same)
    left_um  = box['left_um']  - pad_left_um
    right_um = box['right_um'] + pad_right_um

    # Y extent: depends on region selection
    half_main_px = result['main_px'] / 2.0


    if region == 'full':
        # Entire middle zone (main + both side sub-channels)
        top_um    = x_middle_px * pixel_size_um - pad_top_um
        bottom_um = x_middle_px * pixel_size_um + pad_bottom_um

    elif region == 'main':
        # Main channel only (centre half_main_px on each side)
        top_um    = (x_middle_px - half_main_px) * pixel_size_um - pad_top_um
        bottom_um = (x_middle_px + half_main_px) * pixel_size_um + pad_bottom_um

    elif region == 'top':
        # Top side sub-channel (above main, height = half_main_px)
        top_um    = (x_middle_px - middle_px / 2.0) * pixel_size_um - pad_top_um
        bottom_um = (x_middle_px - half_main_px) * pixel_size_um + pad_bottom_um

    elif region == 'bottom':
        # Bottom side sub-channel (below main, height = half_main_px)
        top_um    = (x_middle_px + half_main_px) * pixel_size_um - pad_top_um
        bottom_um = (x_middle_px + middle_px / 2.0) * pixel_size_um + pad_bottom_um

    # Convert to pixel indices (clip to image bounds)
    H, W = img_rotated.shape[:2]
    x0_px = int(np.clip(round(left_um  / pixel_size_um), 0, W - 1))
    x1_px = int(np.clip(round(right_um / pixel_size_um), 0, W))
    y0_px = int(np.clip(round(top_um    / pixel_size_um), 0, H - 1))
    y1_px = int(np.clip(round(bottom_um / pixel_size_um), 0, H))

    roi = img_rotated[y0_px:y1_px, x0_px:x1_px]

    # Apply 180° rotation if needed (at the end, after region selection)
    if not is_flipped:
        roi = np.rot90(roi, 2)

    roi_coords = {
        'x0_px': x0_px, 'x1_px': x1_px,
        'y0_px': y0_px, 'y1_px': y1_px,
        'left_um':   x0_px * pixel_size_um,
        'right_um':  x1_px * pixel_size_um,
        'top_um':    y0_px * pixel_size_um,
        'bottom_um': y1_px * pixel_size_um,
    }
    return roi, roi_coords
