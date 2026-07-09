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


@dataclass(frozen=True)
class ChipGeometry:
    """
    Explicit chip geometry and channel parameters.
    
    All dimensions in µm unless otherwise noted.
    """
    name: str = "generic"

    # Main channel geometry
    main_channel_width_um: float = 250.0
    expected_main_width_um: float = 450.0 * 0.87
    side_sub_channel_width_um: float = 355.0
    
    # PPA substrate thickness (defines chip height)
    ppa_thickness_um: float = 1400.0

    # Fourier/delay alignment parameters for the visible pillar periodicity.
    target_period_um: float = 405.0
    
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

    def fourier_alignment_kwargs(self) -> Dict[str, float]:
        """Return geometry parameters consumed by Fourier channel alignment."""
        return {
            "target_period_um": self.target_period_um,
            "expected_main_width_um": self.expected_main_width_um,
        }


CHIP_GEOMETRIES: Dict[str, ChipGeometry] = {
    "ppa_chip_may25_onelayer": ChipGeometry(
        name="PPA_Chip_May25_onelayer",
        main_channel_width_um=450.0 * 0.90,
        expected_main_width_um=450.0 * 0.90,
        target_period_um=405.0,
    ),
}


def get_chip_geometry(chip_name: Optional[str]) -> Optional[ChipGeometry]:
    """
    Look up a named chip geometry from the registry.

    The manifest uses human-readable names (for example
    ``PPA_Chip_May25_onelayer``). Matching is case-insensitive.
    """
    if chip_name is None:
        return None
    key = str(chip_name).strip().lower()
    if not key or key == "nan":
        return None
    return CHIP_GEOMETRIES.get(key)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: ORIENTATION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_orientation_nbins(
    img_shape: Tuple[int, int],
    edge_drift_px: float = 20.0,
    min_bins: int = 360,
    max_bins: int = 2880,
    round_to: int = 180,
) -> int:
    """
    Choose an angular histogram size from the image size.

    One angular bin is chosen so that it corresponds to roughly `edge_drift_px`
    lateral displacement across the longest image axis. For a 4928 px image and
    the default 20 px drift, this gives 720 bins.
    """
    if len(img_shape) < 2:
        raise ValueError("img_shape must contain at least two dimensions")

    longest_axis_px = float(max(img_shape[:2]))
    if longest_axis_px <= 0:
        return int(min_bins)

    raw_bins = np.pi * longest_axis_px / max(float(edge_drift_px), 1.0)
    if round_to and round_to > 1:
        raw_bins = round_to * round(raw_bins / round_to)

    return int(np.clip(raw_bins, min_bins, max_bins))


def find_image_orientation(
    img: np.ndarray,
    nbins: Optional[int] = None,
    pixel_size_um: Optional[float] = None,
    expected_period_um: Optional[float] = None,
    period_tolerance_fraction: float = 0.25,
    frequency_range_cyc_per_um: Optional[Tuple[float, float]] = None,
    bin_edge_drift_px: float = 20.0,
    background_kernel_um: Optional[float] = None,
    angle_method: str = "histogram_peak",
    centroid_floor_percentile: float = 50.0,
) -> Tuple[float, float, Dict]:
    """
    Find dominant stripe direction via angular FFT spectrum.
    
    Parameters
    ----------
    img : np.ndarray
        2D grayscale image
    nbins : int, optional
        Number of angular bins for histogram. If None, it is estimated from the
        longest image axis with estimate_orientation_nbins().
    pixel_size_um : float, optional
        Pixel size, required when expected_period_um or
        frequency_range_cyc_per_um is used.
    expected_period_um : float, optional
        Expected spatial period to use for radial FFT masking.
    period_tolerance_fraction : float
        Half-width around expected_period_um expressed as a frequency fraction.
    frequency_range_cyc_per_um : tuple[float, float], optional
        Explicit radial frequency range in cycles/um. Takes precedence over
        expected_period_um when provided.
    bin_edge_drift_px : float
        Target lateral displacement across the longest image axis represented by
        one angular bin when nbins is estimated.
    background_kernel_um : float, optional
        If provided, subtract a fast uniform-filter background with this physical
        kernel size before computing the FFT orientation.
    angle_method : {"centroid", "histogram_peak"}
        How to estimate the FFT peak angle. "centroid" computes a weighted axial
        circular mean inside the frequency mask, giving sub-bin angular estimates.
        "histogram_peak" uses the older angular histogram maximum.
    centroid_floor_percentile : float
        Power percentile to subtract before centroiding. This reduces isotropic
        background influence while keeping broad peak leakage useful for sub-bin
        estimation.
        
    Returns
    -------
    peak_theta_deg : float
        Peak angle in FFT domain (degrees)
    spatial_dir_deg : float
        Estimated striping direction in image (perpendicular to FFT peak)
    """
    img_float = img.astype(np.float64)
    if nbins is None:
        nbins = estimate_orientation_nbins(
            img_float.shape,
            edge_drift_px=bin_edge_drift_px,
        )

    background_kernel_px = None
    if background_kernel_um is not None and background_kernel_um > 0:
        if pixel_size_um is None:
            raise ValueError("pixel_size_um is required with background_kernel_um")
        background_kernel_px = max(3, int(round(background_kernel_um / pixel_size_um)))
        if background_kernel_px % 2 == 0:
            background_kernel_px += 1
        background = ndimage.uniform_filter(
            img_float,
            size=background_kernel_px,
            mode="nearest",
        )
        img_float = img_float - background
    
    # Window to reduce edge leakage
    wy = np.hanning(img_float.shape[0])
    wx = np.hanning(img_float.shape[1])
    img_windowed = img_float * np.outer(wy, wx)
    img_windowed = img_windowed - img_windowed.mean()
    
    # FFT and magnitude
    F = np.fft.fftshift(np.fft.fft2(img_windowed))
    fft_abs = np.abs(F)
    mag = np.log1p(fft_abs)
    power = fft_abs ** 2
    
    H, W = img_windowed.shape
    ky = np.fft.fftshift(np.fft.fftfreq(H, d=1.0))
    kx = np.fft.fftshift(np.fft.fftfreq(W, d=1.0))
    KX, KY = np.meshgrid(kx, ky)
    
    r = np.hypot(KX, KY)
    theta = np.arctan2(KY, KX)
    theta = np.mod(theta, np.pi)
    
    # Angular histogram. By default, exclude very low frequencies. If a target
    # period/frequency is provided, restrict the radial FFT band around it.
    if frequency_range_cyc_per_um is not None:
        if pixel_size_um is None:
            raise ValueError("pixel_size_um is required with frequency_range_cyc_per_um")
        f_lo_um, f_hi_um = frequency_range_cyc_per_um
        f_lo_px = min(f_lo_um, f_hi_um) * pixel_size_um
        f_hi_px = max(f_lo_um, f_hi_um) * pixel_size_um
        mask = (r >= f_lo_px) & (r <= f_hi_px)
    elif expected_period_um is not None:
        if pixel_size_um is None:
            raise ValueError("pixel_size_um is required with expected_period_um")
        target_freq_px = pixel_size_um / expected_period_um
        bandwidth_px = max(
            abs(float(period_tolerance_fraction)) * target_freq_px,
            2.0 / max(img_float.shape[:2]),
        )
        mask = (r >= target_freq_px - bandwidth_px) & (r <= target_freq_px + bandwidth_px)
    else:
        mask = (r >= 0.02) & (r <= 0.5)

    if not np.any(mask):
        raise ValueError("No FFT samples found in the requested orientation frequency band")

    edges = np.linspace(0.0, np.pi, nbins + 1)
    bin_idx = np.digitize(theta[mask], edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)
    
    hist_vals = mag[mask]
    sums = np.bincount(bin_idx, weights=hist_vals, minlength=nbins)
    counts = np.bincount(bin_idx, minlength=nbins)
    ang_mean = sums / np.maximum(counts, 1)
    
    theta_centers = 0.5 * (edges[:-1] + edges[1:])
    histogram_peak_idx = int(np.argmax(ang_mean))
    histogram_peak_theta_deg = float(np.degrees(theta_centers[histogram_peak_idx]))

    angle_method = angle_method.lower()
    if angle_method == "histogram_peak":
        peak_theta_deg = histogram_peak_theta_deg
        centroid_strength = np.nan
    elif angle_method == "centroid":
        theta_masked = theta[mask]
        centroid_weights = power[mask].astype(float)
        if centroid_floor_percentile is not None:
            floor = np.percentile(centroid_weights, centroid_floor_percentile)
            centroid_weights = np.maximum(centroid_weights - floor, 0.0)
        if np.sum(centroid_weights) <= 0:
            centroid_weights = power[mask].astype(float)

        # Axial circular mean: doubling angles handles the 180° FFT symmetry.
        z = np.sum(centroid_weights * np.exp(2j * theta_masked))
        centroid_strength = float(np.abs(z) / (np.sum(centroid_weights) + 1e-12))
        if not np.isfinite(centroid_strength) or centroid_strength < 1e-12:
            peak_theta_deg = histogram_peak_theta_deg
        else:
            peak_theta_rad = 0.5 * np.angle(z)
            if peak_theta_rad < 0:
                peak_theta_rad += np.pi
            peak_theta_deg = float(np.degrees(peak_theta_rad))
    else:
        raise ValueError("angle_method must be 'centroid' or 'histogram_peak'")
    
    # Image striping direction is perpendicular to FFT peak
    spatial_dir_deg = (peak_theta_deg + 90) % 180
    
    _fft_debug = {
        "mag":           mag,
        "ang_mean":      ang_mean,
        "theta_centers": theta_centers,
        "nbins":         nbins,
        "frequency_mask": mask,
        "img_for_fft":    img_float,
        "background_kernel_um": background_kernel_um,
        "background_kernel_px": background_kernel_px,
        "angle_method": angle_method,
        "histogram_peak_theta_deg": histogram_peak_theta_deg,
        "centroid_strength": centroid_strength,
        "centroid_floor_percentile": centroid_floor_percentile,
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
    # ndimage.rotate's sign convention makes a small positive image-space stripe
    # angle horizontal after applying that same positive angle. Wrap the
    # 180-degree direction ambiguity to the smallest equivalent rotation.
    rotate_angle = spatial_dir_deg
    if rotate_angle > 90.0:
        rotate_angle -= 180.0
    elif rotate_angle < -90.0:
        rotate_angle += 180.0
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
    orientation_nbins: Optional[int] = None,
    orientation_expected_period_um: Optional[float] = None,
    orientation_period_tolerance_fraction: float = 0.25,
    orientation_frequency_range_cyc_per_um: Optional[Tuple[float, float]] = None,
    orientation_bin_edge_drift_px: float = 20.0,
    orientation_background_kernel_um: Optional[float] = None,
    orientation_angle_method: str = "histogram_peak",
    orientation_centroid_floor_percentile: float = 50.0,
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
    orientation_nbins : int, optional
        Angular histogram bins. If None, estimated from image size.
    orientation_expected_period_um : float, optional
        Restrict FFT orientation detection to frequencies near this period.
    orientation_period_tolerance_fraction : float
        Frequency half-width around orientation_expected_period_um.
    orientation_frequency_range_cyc_per_um : tuple, optional
        Explicit FFT radial frequency range in cycles/µm. Takes precedence over
        orientation_expected_period_um.
    orientation_bin_edge_drift_px : float
        Target lateral displacement across the longest image axis represented by
        one estimated angular bin.
    orientation_background_kernel_um : float, optional
        If provided, subtract a fast uniform-filter background with this kernel
        size before computing the FFT orientation.
    orientation_angle_method : {"centroid", "histogram_peak"}
        Angle estimator used inside the FFT frequency mask.
    orientation_centroid_floor_percentile : float
        Power percentile subtracted before centroiding in centroid mode.

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
        peak_theta, spatial_dir, _fft_dbg = find_image_orientation(
            img,
            nbins=orientation_nbins,
            pixel_size_um=pixel_size_um,
            expected_period_um=orientation_expected_period_um,
            period_tolerance_fraction=orientation_period_tolerance_fraction,
            frequency_range_cyc_per_um=orientation_frequency_range_cyc_per_um,
            bin_edge_drift_px=orientation_bin_edge_drift_px,
            background_kernel_um=orientation_background_kernel_um,
            angle_method=orientation_angle_method,
            centroid_floor_percentile=orientation_centroid_floor_percentile,
        )
        messages.append(
            f"  → FFT peak: {peak_theta:.1f}°  spatial direction: {spatial_dir:.1f}°"
        )
        messages.append(f"  → Angular histogram bins: {_fft_dbg['nbins']}")
        messages.append(f"  → Orientation angle method: {_fft_dbg['angle_method']}")
        if _fft_dbg["angle_method"] == "centroid":
            messages.append(
                "  → Histogram peak for comparison: "
                f"{_fft_dbg['histogram_peak_theta_deg']:.2f}°; "
                f"centroid strength {_fft_dbg['centroid_strength']:.3f}"
            )
        if orientation_frequency_range_cyc_per_um is not None:
            messages.append(
                "  → Orientation frequency range: "
                f"{orientation_frequency_range_cyc_per_um[0]:.4g}–"
                f"{orientation_frequency_range_cyc_per_um[1]:.4g} cycles/µm"
            )
        elif orientation_expected_period_um is not None:
            messages.append(
                "  → Orientation target period: "
                f"{orientation_expected_period_um:.1f} µm "
                f"(±{100 * orientation_period_tolerance_fraction:.0f}% in frequency)"
            )
        if _fft_dbg["background_kernel_px"] is not None:
            messages.append(
                "  → Orientation background subtraction: "
                f"{orientation_background_kernel_um:.1f} µm "
                f"({_fft_dbg['background_kernel_px']} px uniform filter)"
            )
        scores['orientation_nbins'] = _fft_dbg['nbins']
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
            f"  → Raw rotation: {peak_theta - 90.0:.2f}°  →  applied: {rotate_angle:.2f}°"
        )
        scores['rotation_success'] = 1.0
        success = True

        if debug:
            _theta_deg = np.degrees(_fft_dbg["theta_centers"])

            fig, axes = plt.subplots(1, 3, figsize=(20, 5))

            axes[0].imshow(_fft_dbg["img_for_fft"], cmap="gray")
            image_title = "Background-subtracted image used for FFT"
            if _fft_dbg["background_kernel_px"] is None:
                image_title = "Image used for FFT"
            axes[0].set_title(
                f"{image_title}\n"
                f"stripe dir: {spatial_dir:.1f}°  →  rotation to apply: {rotate_angle:.1f}°"
            )
            axes[0].axis("off")

            axes[1].imshow(_fft_dbg["mag"], cmap="inferno", origin="upper")
            axes[1].set_title("FFT log-magnitude spectrum")
            axes[1].axis("off")

            axes[2].plot(_theta_deg, _fft_dbg["ang_mean"], color="steelblue", lw=1.5)
            axes[2].axvline(
                peak_theta, color="red", lw=1.5, ls="-",
                label=f"Selected angle  {peak_theta:.1f}°  (raw rot {peak_theta - 90:.1f}°)",
            )
            if _fft_dbg["angle_method"] == "centroid":
                axes[2].axvline(
                    _fft_dbg["histogram_peak_theta_deg"], color="0.35", lw=1.2, ls=":",
                    label=f"Histogram peak  {_fft_dbg['histogram_peak_theta_deg']:.1f}°",
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


# ─────────────────────────────────────────────────────────────────────────────
# FOURIER + DELAY-REFINED MAIN-CHANNEL ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def _normalize01(x: np.ndarray) -> np.ndarray:
    """Map an array to [0, 1] while tolerating constant/NaN-heavy inputs."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    return (x - lo) / (hi - lo + 1e-12)


def _subtract_image_background(
    img: np.ndarray,
    pixel_size_um: float,
    background_kernel_um: Optional[float],
) -> np.ndarray:
    """
    Subtract a broad 2-D background once, in image space.

    A uniform filter is used because it is fast on large microscopy frames. The
    corrected image is intended for analysis steps that need periodic structure
    to dominate over illumination gradients; it is not used as the returned
    image data.
    """
    img_float = np.asarray(img, dtype=float)
    if background_kernel_um is None or background_kernel_um <= 0:
        return img_float

    kernel_px = max(3, int(round(background_kernel_um / pixel_size_um)))
    if kernel_px % 2 == 0:
        kernel_px += 1
    background = ndimage.uniform_filter(img_float, size=kernel_px, mode="nearest")
    return img_float - background


def _periodogram_power_spectrum(
    profile: np.ndarray,
    pixel_size_um: float,
    background_sigma_um: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a detrended 1-D periodogram for an x-intensity profile.

    The caller normally passes profiles from an image that has already had its
    2-D background removed. The optional 1-D background subtraction is kept for
    backwards compatibility and ad hoc diagnostics.
    """
    from scipy.signal import detrend, periodogram

    x = np.asarray(profile, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x))

    if background_sigma_um is not None and background_sigma_um > 0:
        bg_sigma_px = max(1.0, background_sigma_um / pixel_size_um)
        if len(x) > int(6 * bg_sigma_px):
            x = x - gaussian_filter1d(x, bg_sigma_px, mode="nearest")
        else:
            x = x - np.nanmedian(x)

    x = detrend(x, type="linear")
    x = x - np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        freqs = np.fft.rfftfreq(len(x), d=pixel_size_um)
        return freqs, np.zeros_like(freqs)

    freqs, power = periodogram(
        x / x_std,
        fs=1.0 / pixel_size_um,
        window="hann",
        detrend=False,
        scaling="spectrum",
    )
    return freqs, power


def _autocorrelation_periodicity_score(
    profile: np.ndarray,
    pixel_size_um: float,
    target_period_um: float,
    relative_bandwidth: float = 0.12,
    background_sigma_um: Optional[float] = None,
) -> float:
    """
    Score whether a profile repeats at approximately `target_period_um`.

    This complements the periodogram: the periodogram checks power at the target
    frequency, while autocorrelation checks that the signal actually repeats with
    the expected lag.
    """
    from scipy.signal import detrend, find_peaks

    x = np.asarray(profile, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    if background_sigma_um is not None and background_sigma_um > 0:
        bg_sigma_px = max(1.0, background_sigma_um / pixel_size_um)
        if len(x) > int(6 * bg_sigma_px):
            x = x - gaussian_filter1d(x, bg_sigma_px, mode="nearest")
        else:
            x = x - np.nanmedian(x)

    x = detrend(x, type="linear")
    x = x - np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        return 0.0

    x = x / x_std
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] + 1e-12)

    target_lag = target_period_um / pixel_size_um
    bandwidth_lag = max(2.0, relative_bandwidth * target_lag)
    lo = max(1, int(round(target_lag - bandwidth_lag)))
    hi = min(len(ac), int(round(target_lag + bandwidth_lag)) + 1)
    if hi <= lo + 2:
        return 0.0

    region = ac[lo:hi]
    peaks, props = find_peaks(region, prominence=0.01)
    if len(peaks) == 0:
        return max(0.0, float(np.max(region) - np.median(region)))
    return max(0.0, float(np.max(props["prominences"])))


def _target_periodicity_score(
    profile: np.ndarray,
    pixel_size_um: float,
    target_period_um: float,
    relative_bandwidth: float = 0.12,
    background_sigma_um: Optional[float] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Return a robust target-period score for one horizontal image transect.

    The score is the geometric mean of a local periodogram peak ratio and an
    autocorrelation prominence near the expected period. It is intentionally
    stricter than a raw FFT value.
    """
    freqs, power = _periodogram_power_spectrum(
        profile,
        pixel_size_um=pixel_size_um,
        background_sigma_um=background_sigma_um,
    )

    target_freq = 1.0 / target_period_um
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else target_freq
    bandwidth = max(relative_bandwidth * target_freq, 2.0 * freq_resolution)
    band = (freqs > 0) & (np.abs(freqs - target_freq) <= bandwidth)
    if not np.any(band):
        return 0.0, np.nan, freqs, power

    band_power = power[band]
    band_freqs = freqs[band]
    best = int(np.argmax(band_power))
    peak_power = float(band_power[best])
    peak_freq = float(band_freqs[best])

    local = (freqs > 0) & (freqs >= 0.4 * target_freq) & (
        freqs <= 2.5 * target_freq
    ) & ~band
    noise = power[local]
    if len(noise) == 0:
        noise = power[(freqs > 0) & ~band]
    noise_floor = float(np.median(noise)) if len(noise) else 1.0
    periodogram_score = peak_power / (noise_floor + 1e-12)
    ac_score = _autocorrelation_periodicity_score(
        profile,
        pixel_size_um=pixel_size_um,
        target_period_um=target_period_um,
        relative_bandwidth=relative_bandwidth,
        background_sigma_um=background_sigma_um,
    )
    score = np.sqrt(max(0.0, periodogram_score) * max(0.0, ac_score))
    return score, peak_freq, freqs, power


def _fourier_channel_scan(
    img_rotated: np.ndarray,
    pixel_size_um: float,
    target_period_um: float = 405.0,
    band_height_um: float = 140.0,
    step_um: float = 10.0,
    score_smooth_um: float = 60.0,
    relative_bandwidth: float = 0.12,
    background_sigma_um: Optional[float] = None,
    display_period_range_um: Tuple[float, float] = (150.0, 900.0),
) -> Dict:
    """
    Scan y positions and measure pillar periodicity along x.

    Each row in the scan corresponds to a horizontal band. A high score means
    that the x-profile in that band contains a real repeating pillar pattern at
    `target_period_um`.
    """
    img = np.asarray(img_rotated, dtype=float)
    H, _W = img.shape[:2]
    half_band_px = max(2, int(round(0.5 * band_height_um / pixel_size_um)))
    step_px = max(1, int(round(step_um / pixel_size_um)))
    ys = np.arange(half_band_px, H - half_band_px, step_px)

    scores = []
    peak_freqs = []
    period_power_rows = []
    period_axis_um = None

    for y in ys:
        band = img[y - half_band_px:y + half_band_px + 1, :]
        profile = np.nanmean(band, axis=0)
        score, peak_freq, freqs, power = _target_periodicity_score(
            profile,
            pixel_size_um=pixel_size_um,
            target_period_um=target_period_um,
            relative_bandwidth=relative_bandwidth,
            background_sigma_um=background_sigma_um,
        )
        scores.append(score)
        peak_freqs.append(peak_freq)

        min_period, max_period = display_period_range_um
        display = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
        display &= freqs > 0
        if period_axis_um is None:
            period_axis_um = 1.0 / freqs[display]
        period_power_rows.append(np.log1p(power[display]))

    ys = np.asarray(ys)
    scores = np.asarray(scores, dtype=float)
    peak_freqs = np.asarray(peak_freqs, dtype=float)
    period_power = np.vstack(period_power_rows) if period_power_rows else np.empty((0, 0))

    smooth_sigma_px = max(0.0, score_smooth_um / (step_px * pixel_size_um))
    scores_smooth = gaussian_filter1d(scores, smooth_sigma_px) if smooth_sigma_px > 0 else scores

    return {
        "ys_px": ys,
        "ys_um": ys * pixel_size_um,
        "scores": scores,
        "scores_smooth": scores_smooth,
        "peak_freqs": peak_freqs,
        "target_period_um": target_period_um,
        "pixel_size_um": pixel_size_um,
        "half_band_px": half_band_px,
        "step_px": step_px,
        "period_axis_um": period_axis_um,
        "period_power": period_power,
    }


def _mask_to_regions(mask: np.ndarray) -> list:
    """Return inclusive `(start, end)` index runs for a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(np.diff(padded.astype(int)))
    starts = changes[0::2]
    ends = changes[1::2] - 1
    return list(zip(starts, ends))


def _refine_main_channel_by_box_convolution(
    scan: Dict,
    gap_y0_um: float,
    gap_y1_um: float,
    expected_main_width_um: float = 450.0,
) -> Dict:
    """
    Place a fixed-width main-channel box where target periodicity is minimal.

    The thresholded periodic pillar regions define the allowed gap. Inside that
    gap, the main channel is the `expected_main_width_um` box with the lowest
    average target-periodicity score.
    """
    ys_um = np.asarray(scan["ys_um"], dtype=float)
    scores = _normalize01(np.asarray(scan["scores_smooth"], dtype=float))
    if len(ys_um) < 2:
        raise ValueError("Need at least two y samples for box convolution.")

    step_um = float(np.median(np.diff(ys_um)))
    width_samples = max(1, int(round(expected_main_width_um / step_um)))
    if width_samples % 2 == 0:
        width_samples += 1
    kernel = np.ones(width_samples, dtype=float) / width_samples
    box_score = np.convolve(scores, kernel, mode="same")

    half_width_um = 0.5 * expected_main_width_um
    allowed = (ys_um >= gap_y0_um + half_width_um) & (ys_um <= gap_y1_um - half_width_um)
    fits_inside_gap = bool(np.any(allowed))
    if not fits_inside_gap:
        allowed = (ys_um >= gap_y0_um) & (ys_um <= gap_y1_um)
    if not np.any(allowed):
        raise ValueError("No scan samples lie inside the selected pillar-region gap.")

    allowed_idx = np.flatnonzero(allowed)
    best_idx = int(allowed_idx[np.argmin(box_score[allowed])])
    center_um = float(ys_um[best_idx])

    return {
        "box_score": box_score,
        "box_width_samples": int(width_samples),
        "box_score_min": float(box_score[best_idx]),
        "box_best_idx": best_idx,
        "box_center_um": center_um,
        "box_y0_um": center_um - half_width_um,
        "box_y1_um": center_um + half_width_um,
        "box_center_px": float(center_um / scan["pixel_size_um"]),
        "box_y0_px": float((center_um - half_width_um) / scan["pixel_size_um"]),
        "box_y1_px": float((center_um + half_width_um) / scan["pixel_size_um"]),
        "fits_inside_gap": fits_inside_gap,
    }


def _find_main_channel_between_periodic_regions(
    scan: Dict,
    expected_main_width_um: float = 450.0,
    score_threshold_fraction: float = 0.35,
    min_pillar_region_width_um: float = 80.0,
    gap_tolerance_um: float = 180.0,
) -> Dict:
    """
    Locate the main channel from two flanking thresholded pillar regions.

    Regions with high target-periodicity are treated as pillar fields. Adjacent
    pillar fields define candidate gaps; the candidate gap closest to
    `expected_main_width_um` is selected, then refined by fixed-width box
    convolution.
    """
    ys_px = np.asarray(scan["ys_px"], dtype=float)
    ys_um = np.asarray(scan["ys_um"], dtype=float)
    scores = np.asarray(scan["scores_smooth"], dtype=float)
    if len(ys_um) < 2:
        raise ValueError("Need at least two y samples to locate periodic regions.")

    step_um = float(np.median(np.diff(ys_um)))
    step_px = float(np.median(np.diff(ys_px)))
    score_norm = _normalize01(scores)
    threshold = float(score_threshold_fraction)
    raw_regions = _mask_to_regions(score_norm >= threshold)

    periodic_regions = []
    for start, end in raw_regions:
        y0_um = ys_um[start] - 0.5 * step_um
        y1_um = ys_um[end] + 0.5 * step_um
        width_um = y1_um - y0_um
        if width_um < min_pillar_region_width_um:
            continue
        periodic_regions.append(
            {
                "start_idx": int(start),
                "end_idx": int(end),
                "y0_um": float(y0_um),
                "y1_um": float(y1_um),
                "y0_px": float(ys_px[start] - 0.5 * step_px),
                "y1_px": float(ys_px[end] + 0.5 * step_px),
                "width_um": float(width_um),
                "mean_score": float(np.mean(score_norm[start:end + 1])),
                "max_score": float(np.max(score_norm[start:end + 1])),
            }
        )

    if len(periodic_regions) < 2:
        raise ValueError(
            "Could not find two thresholded pillar regions. Try lowering "
            "score_threshold_fraction or min_pillar_region_width_um."
        )

    candidates = []
    for i in range(len(periodic_regions) - 1):
        upper = periodic_regions[i]
        lower = periodic_regions[i + 1]
        gap_y0_um = upper["y1_um"]
        gap_y1_um = lower["y0_um"]
        gap_width_um = gap_y1_um - gap_y0_um
        if gap_width_um <= 0:
            continue
        gap_error_um = abs(gap_width_um - expected_main_width_um)
        support_score = 0.5 * (upper["mean_score"] + lower["mean_score"])
        candidates.append((gap_error_um, -support_score, i, gap_y0_um, gap_y1_um, gap_width_um))

    if not candidates:
        raise ValueError("Periodic regions were found, but none leave a positive gap.")

    gap_error_um, _neg_support_score, region_idx, gap_y0_um, gap_y1_um, gap_width_um = min(candidates)
    upper = periodic_regions[region_idx]
    lower = periodic_regions[region_idx + 1]
    refined = _refine_main_channel_by_box_convolution(
        scan,
        gap_y0_um=gap_y0_um,
        gap_y1_um=gap_y1_um,
        expected_main_width_um=expected_main_width_um,
    )
    main_center_um = refined["box_center_um"]

    return {
        "main_center_um": float(main_center_um),
        "main_center_px": float(main_center_um / scan["pixel_size_um"]),
        "main_width_um": float(expected_main_width_um),
        "expected_main_width_um": float(expected_main_width_um),
        "main_y0_um": float(refined["box_y0_um"]),
        "main_y1_um": float(refined["box_y1_um"]),
        "main_y0_px": float(refined["box_y0_px"]),
        "main_y1_px": float(refined["box_y1_px"]),
        "initial_gap_y0_um": float(gap_y0_um),
        "initial_gap_y1_um": float(gap_y1_um),
        "initial_gap_width_um": float(gap_width_um),
        "gap_error_um": float(gap_error_um),
        "within_gap_tolerance": bool(gap_error_um <= gap_tolerance_um),
        "box_score_min": refined["box_score_min"],
        "box_fits_inside_gap": refined["fits_inside_gap"],
        "box_convolution": refined,
        "score_threshold_fraction": threshold,
        "periodic_regions": periodic_regions,
        "flanking_regions": [upper, lower],
    }


def _channel_info_to_band_info(channel_info: Dict, pixel_size_um: float) -> Dict:
    """Convert Fourier channel geometry into the band_info schema used downstream."""
    band_top = int(round(channel_info["main_y0_px"]))
    band_bottom = int(round(channel_info["main_y1_px"]))
    band_width_px = band_bottom - band_top
    return {
        "band_top": band_top,
        "band_bottom": band_bottom,
        "band_centre_row": float(channel_info["main_center_px"]),
        "band_half_width_um": 0.5 * band_width_px * pixel_size_um,
        "band_width_um": band_width_px * pixel_size_um,
        "band_width_px": band_width_px,
        "row_projection": None,
        "source": "fourier_channel",
    }


def _fixed_channel_pillar_windows(
    channel_info: Dict,
    img_shape: Tuple[int, int],
    pixel_size_um: float,
    band_width_um: float = 200.0,
    margin_um: float = 80.0,
) -> list:
    """
    Build two local windows in the periodic pillar fields around the main channel.

    The first Fourier pass may put the main-channel edge a bit off, so a margin
    separates the refinement windows from the detected channel. Each window then
    spans `band_width_um` into the upper/lower pillar regions.
    """
    H, _W = img_shape[:2]
    band_px = max(3, int(round(band_width_um / pixel_size_um)))
    margin_px = max(0, int(round(margin_um / pixel_size_um)))

    main_y0 = int(round(channel_info["main_y0_px"]))
    main_y1 = int(round(channel_info["main_y1_px"]))

    upper_y1 = main_y0 - margin_px
    upper_y0 = upper_y1 - band_px
    lower_y0 = main_y1 + margin_px
    lower_y1 = lower_y0 + band_px

    windows = [
        {"label": "upper", "y0": upper_y0, "y1": upper_y1},
        {"label": "lower", "y0": lower_y0, "y1": lower_y1},
    ]
    for win in windows:
        win["valid"] = win["y1"] > 0 and win["y0"] < H and win["y1"] > win["y0"]
        win["y0_clip"] = int(np.clip(win["y0"], 0, H - 1))
        win["y1_clip"] = int(np.clip(win["y1"], win["y0_clip"] + 1, H))
        win["y0_um"] = win["y0_clip"] * pixel_size_um
        win["y1_um"] = win["y1_clip"] * pixel_size_um
    return windows


def _sample_x_transect_profiles(
    img: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    pixel_size_um: float,
    step_um: float = 10.0,
    height_um: float = 10.0,
    background_sigma_um: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample normalized x-profiles at fixed y positions inside one pillar window.

    Each profile is a thin horizontal strip averaged along y. The image passed
    here is normally already background-corrected in 2-D; optional 1-D
    background subtraction is retained only as a fallback.
    """
    half_height_px = max(1, int(round(0.5 * height_um / pixel_size_um)))
    step_px = max(1, int(round(step_um / pixel_size_um)))
    ys = np.arange(y0 + half_height_px, y1 - half_height_px, step_px, dtype=int)
    if len(ys) < 3:
        return ys, np.empty((0, max(0, x1 - x0)))

    profiles = []
    for y in ys:
        strip = img[y - half_height_px:y + half_height_px + 1, x0:x1]
        profiles.append(np.nanmean(strip, axis=0))
    profiles = np.asarray(profiles, dtype=float)

    if background_sigma_um is not None and background_sigma_um > 0:
        bg_sigma_px = max(1.0, background_sigma_um / pixel_size_um)
        if profiles.shape[1] > int(6 * bg_sigma_px):
            profiles = profiles - gaussian_filter1d(profiles, bg_sigma_px, axis=1, mode="nearest")
        else:
            profiles = profiles - np.nanmedian(profiles, axis=1, keepdims=True)

    profiles = profiles - np.nanmean(profiles, axis=1, keepdims=True)
    profiles = profiles / (np.nanstd(profiles, axis=1, keepdims=True) + 1e-12)
    return ys, profiles


def _reference_xcorr_lags(
    profiles: np.ndarray,
    max_lag_px: int,
    reference_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-correlate every transect to a center reference transect.

    A correct rotation makes the periodic pillar profile line up at the same x
    phase for all y positions, so the best correlation lag should be zero.
    Parabolic interpolation around the discrete maximum gives subpixel lag
    estimates without needing a slower full image registration method.
    """
    if profiles.shape[0] < 3 or profiles.shape[1] < 2 * max_lag_px + 5:
        return np.array([]), np.array([]), np.empty((0, 0)), np.array([]), np.array([])

    if reference_index is None:
        reference_index = profiles.shape[0] // 2
    reference_index = int(np.clip(reference_index, 0, profiles.shape[0] - 1))

    keep = np.arange(profiles.shape[0]) != reference_index
    moving = profiles[keep]
    ref = profiles[reference_index]
    profile_indices = np.flatnonzero(keep)
    lags = np.arange(-max_lag_px, max_lag_px + 1, dtype=int)
    corr = np.empty((len(lags), moving.shape[0]), dtype=float)

    for i, lag in enumerate(lags):
        if lag < 0:
            corr[i] = np.mean(moving[:, -lag:] * ref[:lag], axis=1)
        elif lag > 0:
            corr[i] = np.mean(moving[:, :-lag] * ref[lag:], axis=1)
        else:
            corr[i] = np.mean(moving * ref, axis=1)

    best_i = np.argmax(corr, axis=0)
    col_i = np.arange(corr.shape[1])
    best_lags = lags[best_i].astype(float)

    can_interp = (best_i > 0) & (best_i < len(lags) - 1)
    for col in np.flatnonzero(can_interp):
        i = best_i[col]
        y0, y1, y2 = corr[i - 1, col], corr[i, col], corr[i + 1, col]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            offset = 0.5 * (y0 - y2) / denom
            best_lags[col] += float(np.clip(offset, -1.0, 1.0))

    best_corr = corr[best_i, col_i]
    return best_lags, best_corr, corr, lags, profile_indices


def _score_delay_refinement_delta(
    base_rotated_img: np.ndarray,
    delta_deg: float,
    windows: list,
    pixel_size_um: float,
    x_margin_um: float = 250.0,
    max_lag_um: float = 80.0,
    transect_step_um: float = 10.0,
    transect_height_um: float = 10.0,
    background_sigma_um: Optional[float] = None,
    collect_details: bool = False,
) -> Dict:
    """
    Score one extra rotation by measuring residual x-delay in pillar windows.

    Only a local crop covering both pillar windows is rotated, making the angle
    search cheap. The score is mostly weighted RMS lag, with a small penalty for
    weak correlation so that noisy/flat profiles are not preferred.
    """
    H, W = base_rotated_img.shape[:2]
    y_pad = int(round(60 / pixel_size_um))
    y0_all = max(0, min(win["y0_clip"] for win in windows) - y_pad)
    y1_all = min(H, max(win["y1_clip"] for win in windows) + y_pad)
    x_margin_px = max(0, int(round(x_margin_um / pixel_size_um)))
    x0 = min(x_margin_px, W - 2)
    x1 = max(x0 + 2, W - x_margin_px)

    crop = base_rotated_img[y0_all:y1_all, :]
    crop_rot = ndimage.rotate(crop, delta_deg, reshape=False, order=1, mode="nearest")
    max_lag_px = max(1, int(round(max_lag_um / pixel_size_um)))

    all_lags = []
    all_corrs = []
    details = []
    for win in windows:
        y0 = win["y0_clip"] - y0_all
        y1 = win["y1_clip"] - y0_all
        ys, profiles = _sample_x_transect_profiles(
            crop_rot,
            y0,
            y1,
            x0,
            x1,
            pixel_size_um,
            step_um=transect_step_um,
            height_um=transect_height_um,
            background_sigma_um=background_sigma_um,
        )
        lags, corrs, corr_matrix, lag_axis, lag_profile_indices = _reference_xcorr_lags(
            profiles,
            max_lag_px=max_lag_px,
        )
        if len(lags):
            all_lags.append(lags.astype(float))
            all_corrs.append(corrs.astype(float))
        if collect_details:
            pair_y_px = y0_all + ys[lag_profile_indices] if len(ys) and len(lags) else np.array([])
            details.append(
                {
                    "label": win["label"],
                    "ys_px": ys + y0_all,
                    "pair_y_px": pair_y_px,
                    "profiles": profiles,
                    "lags_px": lags,
                    "corrs": corrs,
                    "corr_matrix": corr_matrix,
                    "lag_axis_px": lag_axis,
                    "window": win,
                }
            )

    if not all_lags:
        return {
            "delta_deg": float(delta_deg),
            "score": np.inf,
            "lag_rms_px": np.inf,
            "median_abs_lag_px": np.inf,
            "median_corr": np.nan,
            "n_pairs": 0,
            "details": details,
        }

    lags = np.concatenate(all_lags)
    corrs = np.concatenate(all_corrs)
    weights = np.clip(corrs, 0.0, None) + 1e-6
    lag_rms = float(np.sqrt(np.average(lags ** 2, weights=weights)))
    median_abs_lag = float(np.median(np.abs(lags)))
    median_corr = float(np.median(corrs))
    score = lag_rms + 0.05 * max_lag_px * max(0.0, 1.0 - median_corr)
    return {
        "delta_deg": float(delta_deg),
        "score": float(score),
        "lag_rms_px": lag_rms,
        "median_abs_lag_px": median_abs_lag,
        "median_corr": median_corr,
        "n_pairs": int(len(lags)),
        "details": details,
    }


def _run_delay_rotation_refinement(
    base_rotated_img: np.ndarray,
    channel_info: Dict,
    pixel_size_um: float,
    band_width_um: float = 200.0,
    margin_um: float = 80.0,
    angle_range_deg: float = 1.2,
    coarse_step_deg: float = 0.10,
    fine_half_width_deg: float = 0.15,
    fine_step_deg: float = 0.02,
    max_lag_um: Optional[float] = None,
    x_margin_um: float = 250.0,
    transect_step_um: float = 10.0,
    transect_height_um: float = 10.0,
    background_sigma_um: Optional[float] = None,
    image_background_kernel_um: Optional[float] = None,
) -> Dict:
    """
    Find the small extra rotation that minimizes pillar-profile x-delay.

    This is designed as a local refinement after a good first rotation. It is
    intentionally a two-stage grid search rather than a black-box optimizer so
    the diagnostic curves are easy to inspect. If `image_background_kernel_um`
    is provided, a broad 2-D background is subtracted once from
    `base_rotated_img` before the angle search; no per-profile background
    subtraction is needed in the normal pipeline.

    Parameters
    ----------
    base_rotated_img
        Current rotated analysis image. In the normal pipeline this has already
        had broad 2-D background removed.
    channel_info
        Main-channel geometry from `_find_main_channel_between_periodic_regions`.
        Its channel edges anchor the upper/lower pillar windows.
    pixel_size_um
        Pixel size in µm/px.
    band_width_um
        Height of each upper/lower pillar window used for lag measurement.
    margin_um
        Distance from main-channel edge to the nearest pillar-window edge.
    angle_range_deg
        Half-width of the initial coarse extra-rotation search.
    coarse_step_deg
        Angle step for the coarse search.
    fine_half_width_deg
        Half-width of the fine search around the best coarse angle.
    fine_step_deg
        Angle step for the fine search.
    max_lag_um
        Maximum x-shift considered in profile cross-correlation.
    x_margin_um
        Left/right columns excluded before cross-correlation.
    transect_step_um
        y-spacing between sampled transects inside each pillar window.
    transect_height_um
        y-height averaged to form each x-profile.
    background_sigma_um
        Optional legacy per-profile 1-D background subtraction. Normally None.
    image_background_kernel_um
        Optional 2-D background subtraction applied once to `base_rotated_img`
        before scoring all candidate angles.
    """
    if image_background_kernel_um is not None and image_background_kernel_um > 0:
        base_rotated_img = _subtract_image_background(
            base_rotated_img,
            pixel_size_um=pixel_size_um,
            background_kernel_um=image_background_kernel_um,
        )

    if max_lag_um is None:
        max_lag_um = 0.2 * channel_info.get("expected_main_width_um", 450.0)

    windows = _fixed_channel_pillar_windows(
        channel_info,
        base_rotated_img.shape,
        pixel_size_um,
        band_width_um=band_width_um,
        margin_um=margin_um,
    )

    coarse_deltas = np.arange(
        -angle_range_deg,
        angle_range_deg + 0.5 * coarse_step_deg,
        coarse_step_deg,
    )
    coarse = [
        _score_delay_refinement_delta(
            base_rotated_img,
            delta,
            windows,
            pixel_size_um,
            x_margin_um=x_margin_um,
            max_lag_um=max_lag_um,
            transect_step_um=transect_step_um,
            transect_height_um=transect_height_um,
            background_sigma_um=background_sigma_um,
        )
        for delta in coarse_deltas
    ]
    best_coarse = min(coarse, key=lambda r: r["score"])

    fine_deltas = np.arange(
        best_coarse["delta_deg"] - fine_half_width_deg,
        best_coarse["delta_deg"] + fine_half_width_deg + 0.5 * fine_step_deg,
        fine_step_deg,
    )
    fine = [
        _score_delay_refinement_delta(
            base_rotated_img,
            delta,
            windows,
            pixel_size_um,
            x_margin_um=x_margin_um,
            max_lag_um=max_lag_um,
            transect_step_um=transect_step_um,
            transect_height_um=transect_height_um,
            background_sigma_um=background_sigma_um,
        )
        for delta in fine_deltas
    ]
    best = min(fine, key=lambda r: r["score"])
    best_details = _score_delay_refinement_delta(
        base_rotated_img,
        best["delta_deg"],
        windows,
        pixel_size_um,
        x_margin_um=x_margin_um,
        max_lag_um=max_lag_um,
        transect_step_um=transect_step_um,
        transect_height_um=transect_height_um,
        background_sigma_um=background_sigma_um,
        collect_details=True,
    )
    zero_details = _score_delay_refinement_delta(
        base_rotated_img,
        0.0,
        windows,
        pixel_size_um,
        x_margin_um=x_margin_um,
        max_lag_um=max_lag_um,
        transect_step_um=transect_step_um,
        transect_height_um=transect_height_um,
        background_sigma_um=background_sigma_um,
        collect_details=True,
    )
    return {
        "windows": windows,
        "coarse": coarse,
        "fine": fine,
        "best": best_details,
        "zero": zero_details,
    }


def align_chip_to_image_fourier_channel(
    img: np.ndarray,
    pixel_size_um: float,
    geom: ChipGeometry = None,
    target_period_um: Optional[float] = None,
    expected_main_width_um: Optional[float] = None,
    n_refinement_iters: int = 2,
    debug: bool = False,
    orientation_nbins: Optional[int] = None,
    orientation_expected_period_um: Optional[float] = None,
    orientation_period_tolerance_fraction: float = 1.0,
    orientation_frequency_range_cyc_per_um: Optional[Tuple[float, float]] = None,
    orientation_bin_edge_drift_px: float = 30.0,
    analysis_background_kernel_um: Optional[float] = 300.0,
    orientation_background_kernel_um: Optional[float] = None,
    orientation_angle_method: str = "histogram_peak",
    scan_band_height_um: float = 140.0,
    scan_step_um: float = 10.0,
    scan_score_smooth_um: float = 20.0,
    scan_relative_bandwidth: float = 0.12,
    scan_background_sigma_um: Optional[float] = None,
    score_threshold_fraction: float = 0.35,
    min_pillar_region_width_um: float = 80.0,
    gap_tolerance_um: float = 180.0,
    delay_band_width_um: float = 220.0,
    delay_margin_um: float = 100.0,
    delay_angle_range_deg: float = 1.2,
    delay_coarse_step_deg: float = 0.10,
    delay_fine_half_width_deg: float = 0.15,
    delay_fine_step_deg: float = 0.01,
    delay_max_lag_um: Optional[float] = None,
    delay_x_margin_um: float = 250.0,
    delay_transect_step_um: float = 10.0,
    delay_transect_height_um: float = 10.0,
    delay_background_sigma_um: Optional[float] = None,
) -> Dict:
    """
    Align a chip and locate the main channel without a pillar U-Net.

    This is the preferred non-ML replacement for the previous pillar-mask based
    channel detection. The algorithm is intentionally decomposed into diagnostic
    steps:

    1. **Global FFT orientation.** Estimate the first rotation from the 2-D FFT,
       optionally restricted to frequencies near the expected pillar period.
    2. **Fourier channel detection.** After rotation, scan horizontal bands and
       score real x-periodicity near `target_period_um`. High-score regions are
       pillar fields. The main channel is the fixed-width low-periodicity box
       between the two flanking pillar fields.
    3. **Delay refinement.** In the pillar windows immediately above and below
       the detected channel, fixed-y transects should have the same x phase. A
       small extra rotation is chosen by minimizing cross-correlation delay. The
       channel is then re-detected and the cycle repeats.

    The returned dictionary is compatible with the older alignment API:
    `rotate_fn`, `rotate_angle_deg`, `pixel_size_um`, `bounding_box`,
    `x_middle_px`, `middle_px`, `main_px`, and `is_flipped` are populated. It
    also includes `band_info`, which can be passed directly to
    `build_cell_dataframe`.

    When `debug=True`, compact figures are returned in `result["figures"]`.

    Parameters
    ----------
    img
        Raw 2-D image used for alignment, typically a brightfield frame.
    pixel_size_um
        Pixel size in µm/px. The current implementation assumes square pixels.
    geom
        Optional `ChipGeometry`. Kept for compatibility with the older API.
    target_period_um
        Expected pillar periodicity along x. Used to score periodic pillar
        regions after rotation.
    expected_main_width_um
        Expected physical width of the main non-periodic channel. The final
        channel box is forced to this width when locating the channel.
    n_refinement_iters
        Number of delay-refinement cycles. Each cycle refines rotation, then
        re-detects the main-channel position on the updated rotated image.
    debug
        If True, return compact diagnostic figures and the final rotated image.
    orientation_nbins
        Number of angular FFT bins for the initial orientation. If None, it is
        estimated from image size by `estimate_orientation_nbins`.
    orientation_expected_period_um
        Period used to restrict the initial FFT orientation frequency band. If
        None, defaults to `target_period_um`.
    orientation_period_tolerance_fraction
        Relative frequency half-width around `orientation_expected_period_um`.
        Example: 1.0 means ±100% in frequency.
    orientation_frequency_range_cyc_per_um
        Explicit initial-orientation frequency range in cycles/µm. If provided,
        it takes precedence over `orientation_expected_period_um`.
    orientation_bin_edge_drift_px
        Image-size heuristic for automatic `orientation_nbins`: one angular bin
        corresponds approximately to this many pixels of drift across the long
        image axis.
    analysis_background_kernel_um
        Broad 2-D uniform-filter kernel used once to create the analysis image.
        The Fourier channel scan and delay refinement both use this same
        background-corrected image. Set to None or <=0 to disable.
    orientation_background_kernel_um
        Optional extra background subtraction inside the low-level FFT
        orientation routine. Normally leave as None because
        `analysis_background_kernel_um` already handles this once.
    orientation_angle_method
        Initial FFT angle estimator: "histogram_peak" reproduces the original
        binned peak method; "centroid" uses a weighted axial centroid for
        sub-bin angles.
    scan_band_height_um
        Height of each horizontal band averaged into one x-profile for Fourier
        periodicity scoring.
    scan_step_um
        Spacing between successive y positions in the Fourier channel scan.
    scan_score_smooth_um
        Gaussian smoothing scale applied to the y-profile of periodicity scores.
    scan_relative_bandwidth
        Relative frequency half-width around `target_period_um` used by the
        periodogram/autocorrelation score.
    scan_background_sigma_um
        Optional legacy 1-D x-profile background subtraction during Fourier
        scoring. Normally None because the 2-D analysis image is already
        background-corrected once.
    score_threshold_fraction
        Threshold on normalized smoothed periodicity score used to define
        periodic pillar regions.
    min_pillar_region_width_um
        Minimum physical y-width for a thresholded region to count as a pillar
        field rather than noise.
    gap_tolerance_um
        Diagnostic tolerance for how close the initial thresholded gap is to
        `expected_main_width_um`.
    delay_band_width_um
        Height of each pillar window used for cross-correlation delay
        refinement above and below the detected main channel.
    delay_margin_um
        Gap between the detected main-channel edge and each delay-refinement
        pillar window. This avoids using incorrectly detected channel pixels.
    delay_angle_range_deg
        Half-width of the coarse extra-rotation search around the current angle.
    delay_coarse_step_deg
        Step size for the coarse delay-refinement angle search.
    delay_fine_half_width_deg
        Half-width of the fine search centered on the best coarse angle.
    delay_fine_step_deg
        Step size for the fine delay-refinement angle search.
    delay_max_lag_um
        Maximum x-lag allowed in transect cross-correlations. If None, defaults
        to 20% of `target_period_um`.
    delay_x_margin_um
        Columns trimmed from left and right before cross-correlation, avoiding
        rotation padding and chip-edge artifacts.
    delay_transect_step_um
        y-spacing between transects sampled inside each pillar window.
    delay_transect_height_um
        Height of each thin horizontal transect before averaging into a profile.
    delay_background_sigma_um
        Optional legacy 1-D x-profile background subtraction during delay
        refinement. Normally None because `analysis_background_kernel_um`
        provides shared 2-D correction.
    """
    if geom is None:
        geom = ChipGeometry()
    if target_period_um is None:
        target_period_um = geom.target_period_um
    if expected_main_width_um is None:
        expected_main_width_um = geom.expected_main_width_um
    if orientation_expected_period_um is None:
        orientation_expected_period_um = target_period_um

    figures = {}
    messages = ["Initial FFT orientation..."]
    scores = {}
    iterations = []
    img_analysis = _subtract_image_background(
        img,
        pixel_size_um=pixel_size_um,
        background_kernel_um=analysis_background_kernel_um,
    )

    initial = align_chip_to_image(
        img_analysis,
        pixel_size_um=pixel_size_um,
        geom=geom,
        debug=False,
        orientation_nbins=orientation_nbins,
        orientation_expected_period_um=orientation_expected_period_um,
        orientation_period_tolerance_fraction=orientation_period_tolerance_fraction,
        orientation_frequency_range_cyc_per_um=orientation_frequency_range_cyc_per_um,
        orientation_bin_edge_drift_px=orientation_bin_edge_drift_px,
        orientation_background_kernel_um=orientation_background_kernel_um,
        orientation_angle_method=orientation_angle_method,
    )
    messages.extend(initial.get("messages", []))
    if not initial["success"]:
        result = dict(initial)
        result["method"] = "fourier_channel"
        result["messages"] = messages
        return result

    total_angle = float(initial["rotate_angle_deg"])
    img_rotated = ndimage.rotate(img, total_angle, reshape=False)
    img_analysis_rotated = ndimage.rotate(img_analysis, total_angle, reshape=False)

    scan_kwargs = dict(
        target_period_um=target_period_um,
        band_height_um=scan_band_height_um,
        step_um=scan_step_um,
        score_smooth_um=scan_score_smooth_um,
        relative_bandwidth=scan_relative_bandwidth,
        background_sigma_um=scan_background_sigma_um,
    )
    channel_kwargs = dict(
        expected_main_width_um=expected_main_width_um,
        score_threshold_fraction=score_threshold_fraction,
        min_pillar_region_width_um=min_pillar_region_width_um,
        gap_tolerance_um=gap_tolerance_um,
    )

    def _detect_channel(current_analysis_rotated: np.ndarray) -> Tuple[Dict, Dict]:
        scan = _fourier_channel_scan(
            current_analysis_rotated,
            pixel_size_um=pixel_size_um,
            **scan_kwargs,
        )
        channel_info = _find_main_channel_between_periodic_regions(scan, **channel_kwargs)
        return scan, channel_info

    scan, channel_info = _detect_channel(img_analysis_rotated)
    messages.append(
        "Initial Fourier channel: "
        f"center={channel_info['main_center_um']:.1f} µm, "
        f"width={channel_info['main_width_um']:.1f} µm"
    )

    for i in range(int(n_refinement_iters)):
        refine = _run_delay_rotation_refinement(
            img_analysis_rotated,
            channel_info,
            pixel_size_um,
            band_width_um=delay_band_width_um,
            margin_um=delay_margin_um,
            angle_range_deg=delay_angle_range_deg,
            coarse_step_deg=delay_coarse_step_deg,
            fine_half_width_deg=delay_fine_half_width_deg,
            fine_step_deg=delay_fine_step_deg,
            max_lag_um=delay_max_lag_um if delay_max_lag_um is not None else 0.2 * target_period_um,
            x_margin_um=delay_x_margin_um,
            transect_step_um=delay_transect_step_um,
            transect_height_um=delay_transect_height_um,
            background_sigma_um=delay_background_sigma_um,
            image_background_kernel_um=None,
        )
        delta = float(refine["best"]["delta_deg"])
        total_angle += delta
        img_rotated = ndimage.rotate(img, total_angle, reshape=False)
        img_analysis_rotated = ndimage.rotate(img_analysis, total_angle, reshape=False)
        scan, channel_info = _detect_channel(img_analysis_rotated)

        iteration = {
            "iteration": i + 1,
            "delta_deg": delta,
            "total_angle_deg": total_angle,
            "zero_lag_rms_px": refine["zero"]["lag_rms_px"],
            "best_lag_rms_px": refine["best"]["lag_rms_px"],
            "best_median_corr": refine["best"]["median_corr"],
            "delay_refinement": refine,
            "scan": scan,
            "channel_info": channel_info,
        }
        iterations.append(iteration)
        messages.append(
            f"Delay refinement {i + 1}: delta={delta:+.4f}°, "
            f"angle={total_angle:+.4f}°, lag RMS "
            f"{refine['zero']['lag_rms_px']:.2f}→{refine['best']['lag_rms_px']:.2f} px"
        )

    band_info = _channel_info_to_band_info(channel_info, pixel_size_um)
    H, W = img.shape[:2]
    upper_region, lower_region = channel_info["flanking_regions"]
    middle_px = lower_region["y1_px"] - upper_region["y0_px"]
    main_px = channel_info["main_width_um"] / pixel_size_um

    scores.update(
        {
            "orientation_nbins": initial.get("scores", {}).get("orientation_nbins"),
            "rotation_success": 1.0,
            "channel_box_score_min": channel_info["box_score_min"],
            "channel_gap_error_um": channel_info["gap_error_um"],
            "analysis_background_kernel_um": analysis_background_kernel_um,
        }
    )

    if debug:
        img_disp = _normalize01(np.asarray(img_rotated, dtype=float))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].imshow(img_disp, cmap="gray")
        axes[0, 0].axhspan(
            channel_info["main_y0_px"],
            channel_info["main_y1_px"],
            color="cyan",
            alpha=0.20,
            label="main channel",
        )
        for j, region in enumerate(channel_info["periodic_regions"]):
            axes[0, 0].axhspan(
                region["y0_px"],
                region["y1_px"],
                color="magenta",
                alpha=0.12,
                label="periodic pillar regions" if j == 0 else "_nolegend_",
            )
        axes[0, 0].set_title(f"Final rotated image ({total_angle:+.3f}°)")
        axes[0, 0].axis("off")
        axes[0, 0].legend(loc="upper right", fontsize=8)

        axes[0, 1].plot(scan["scores"], scan["ys_um"], color="0.7", lw=1, label="raw")
        axes[0, 1].plot(scan["scores_smooth"], scan["ys_um"], color="tab:blue", lw=1.5, label="smoothed")
        axes[0, 1].plot(
            channel_info["box_convolution"]["box_score"],
            scan["ys_um"],
            color="tab:orange",
            lw=1.5,
            label="main-width box avg.",
        )
        axes[0, 1].axhspan(channel_info["main_y0_um"], channel_info["main_y1_um"], color="cyan", alpha=0.16)
        for region in channel_info["periodic_regions"]:
            axes[0, 1].axhspan(region["y0_um"], region["y1_um"], color="magenta", alpha=0.10)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel("periodicity score")
        axes[0, 1].set_ylabel("y (µm)")
        axes[0, 1].set_title("Fourier channel score")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.25)

        if iterations:
            last = iterations[-1]["delay_refinement"]
            coarse = last["coarse"]
            fine = last["fine"]
            axes[1, 0].plot([r["delta_deg"] for r in coarse], [r["score"] for r in coarse], "o-", ms=3, label="coarse")
            axes[1, 0].plot([r["delta_deg"] for r in fine], [r["score"] for r in fine], "o-", ms=3, label="fine")
            axes[1, 0].axvline(0, color="0.5", ls=":", lw=1)
            axes[1, 0].axvline(last["best"]["delta_deg"], color="tab:red", lw=1.4, label="best")
            axes[1, 0].set_xlabel("extra rotation (deg)")
            axes[1, 0].set_ylabel("delay score")
            axes[1, 0].set_title("Last delay-refinement search")
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].grid(True, alpha=0.25)

            for details, label, color in [
                (last["zero"]["details"], "before", "tab:blue"),
                (last["best"]["details"], "after", "tab:red"),
            ]:
                for band in details:
                    if len(band["lags_px"]) == 0:
                        continue
                    axes[1, 1].scatter(
                        band["pair_y_px"] * pixel_size_um,
                        band["lags_px"],
                        s=14,
                        alpha=0.65,
                        color=color,
                        label=f"{label} {band['label']}",
                    )
            axes[1, 1].axhline(0, color="0.2", lw=1)
            axes[1, 1].set_xlabel("y (µm)")
            axes[1, 1].set_ylabel("lag to center transect (px)")
            axes[1, 1].set_title("Lag diagnostic")
            axes[1, 1].legend(fontsize=7, ncol=2)
            axes[1, 1].grid(True, alpha=0.25)
        else:
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")

        plt.tight_layout()
        figures["fourier_channel_alignment"] = fig

    def rotate_fn(other_img: np.ndarray) -> np.ndarray:
        """Apply the final Fourier/channel refined rotation to another image."""
        return ndimage.rotate(other_img, total_angle, reshape=False)

    result = {
        "success": True,
        "method": "fourier_channel",
        "rotate_angle_deg": total_angle,
        "rotate_angle_deg_initial": float(initial["rotate_angle_deg"]),
        "rotate_fn": rotate_fn,
        "pixel_size_um": pixel_size_um,
        "scores": scores,
        "messages": messages,
        "bounding_box": {"left_um": 0.0, "right_um": W * pixel_size_um},
        "x_middle_px": float(channel_info["main_center_px"]),
        "middle_px": float(middle_px),
        "main_px": float(main_px),
        "is_flipped": True,
        "band_info": band_info,
        "fourier_channel_scan": scan,
        "fourier_channel_geometry": channel_info,
        "delay_refinement_iterations": iterations,
        "analysis_background_kernel_um": analysis_background_kernel_um,
        "final_rotated_image": img_rotated if debug else None,
        "final_analysis_image": img_analysis_rotated if debug else None,
    }
    if debug:
        result["figures"] = figures
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
