"""
video_renderer.py
=================
Headless video frame rendering — cluster-safe, no display required.

Uses ``FigureCanvasAgg`` directly (never touches ``pyplot`` global state), so
it is safe to import inside notebooks that already have another backend active,
and safe to use on a headless cluster node.

Exports
-------
render_detection_frame(img_bf_raw, binary_mask, band_info, time_label, ...)
    → uint8 RGB numpy array

render_detection_video(czi, scene, timepoints, channel_detect, channel_bf,
                       rotate_fn, detect_fn, band_info, px_um, output_path, …)
    → saves mp4 via ffmpeg (writes one PNG per frame to a temp dir, then
      calls ffmpeg to assemble; memory-efficient for large/4K exports)

_time_label(t, show_real_time, times_lookup, t0_dt)
    → str time annotation
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import numpy as np
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.measure import label as sk_label, regionprops as sk_regionprops

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Time label helper
# ──────────────────────────────────────────────────────────────────────────────

def _time_label(t, show_real_time=False, times_lookup=None, t0_dt=None):
    """Build the time annotation string for frame *t*.

    Parameters
    ----------
    t              : int  – timepoint index
    show_real_time : bool – if True, format as elapsed time
    times_lookup   : dict {t_idx: datetime} or None
    t0_dt          : datetime or None – reference (t=0) timestamp
    """
    if show_real_time and times_lookup and t in times_lookup and t0_dt is not None:
        elapsed_s = (times_lookup[t] - t0_dt).total_seconds()
        if elapsed_s < 60:
            return f"{elapsed_s:.0f} s"
        elif elapsed_s < 3600:
            return f"{elapsed_s / 60:.1f} min"
        else:
            return f"{elapsed_s / 3600:.2f} h"
    return f"t = {t}"


# ──────────────────────────────────────────────────────────────────────────────
# Single-frame renderer
# ──────────────────────────────────────────────────────────────────────────────

def render_detection_frame(
    img_bf_raw,
    binary_mask,
    band_info,
    time_label,
    resize_width=4096,
    dpi=150,
    overlay_color=(1.0, 0.35, 0.0),
    overlay_alpha=0.45,
    centroid_color="yellow",
    centroid_size=6,
):
    """
    Render one video frame as a uint8 RGB numpy array.

    Fully headless — uses ``FigureCanvasAgg`` directly, never calls
    ``matplotlib.pyplot``, does not affect the notebook's backend.

    Parameters
    ----------
    img_bf_raw    : (H, W) array  – raw brightfield (any dtype / range)
    binary_mask   : (H, W) bool   – detected objects (same spatial shape)
    band_info     : dict          – main-channel geometry; usually
                    ``align_result["band_info"]`` from Fourier/delay alignment
    time_label    : str           – annotation text in the top-left corner
    resize_width  : int           – output frame width in pixels
    dpi           : int
    overlay_color : (r, g, b) ∈ [0, 1]³  – tint colour for detected regions
    overlay_alpha : float         – opacity of the tint  [0, 1]
    centroid_color: str or tuple
    centroid_size : int           – scatter marker size (points²)

    Returns
    -------
    frame : np.ndarray  uint8  shape (H', W', 3)
    """
    if not _PIL_AVAILABLE:
        raise ImportError(
            "Pillow is required for render_detection_frame.  "
            "Install with:  pip install Pillow"
        )

    H, W = img_bf_raw.shape

    # ── Normalise BF & build RGB canvas ──────────────────────────────────────
    img_n = img_bf_raw.astype(np.float32)
    img_n = (img_n - img_n.min()) / (img_n.max() - img_n.min() + 1e-8)
    canvas = np.stack([img_n, img_n, img_n], axis=-1)

    # ── Tint detected regions ─────────────────────────────────────────────────
    if binary_mask.any():
        r, g, b = overlay_color
        a = overlay_alpha
        canvas[binary_mask, 0] = canvas[binary_mask, 0] * (1 - a) + r * a
        canvas[binary_mask, 1] = canvas[binary_mask, 1] * (1 - a) + g * a
        canvas[binary_mask, 2] = canvas[binary_mask, 2] * (1 - a) + b * a

    # ── Centroids via regionprops ─────────────────────────────────────────────
    labeled  = sk_label(binary_mask)
    regions  = sk_regionprops(labeled)
    cents_rc = (
        np.array([reg.centroid for reg in regions])
        if regions else np.empty((0, 2))
    )

    # ── Headless matplotlib figure ────────────────────────────────────────────
    aspect = H / W
    fig_w  = resize_width / dpi
    fig    = Figure(figsize=(fig_w, fig_w * aspect), dpi=dpi)
    FigureCanvasAgg(fig)                       # attach Agg canvas (no pyplot)
    ax     = fig.add_axes([0, 0, 1, 1])

    ax.imshow(np.clip(canvas, 0, 1))
    ax.axhline(band_info["band_top"],    color="cyan", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(band_info["band_bottom"], color="cyan", lw=0.8, ls="--", alpha=0.6)

    if len(cents_rc):
        ax.scatter(
            cents_rc[:, 1], cents_rc[:, 0],
            c=centroid_color, s=0.1 * centroid_size ** 2,
            alpha=0.9, linewidths=0.5, edgecolors="black", zorder=5,
        )

    _ann = dict(
        fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55),
    )
    ax.text(0.01, 0.98, time_label,
            transform=ax.transAxes, va="top", ha="left",  **_ann)
    ax.text(0.99, 0.98, f"n = {len(regions)}",
            transform=ax.transAxes, va="top", ha="right", **_ann)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    frame_arr = np.array(Image.open(buf).convert("RGB"))
    fig.clf()
    return frame_arr


# ──────────────────────────────────────────────────────────────────────────────
# Full video renderer (memory-efficient: one frame at a time → ffmpeg)
# ──────────────────────────────────────────────────────────────────────────────

def render_detection_video(
    czi,
    scene,
    timepoints,
    channel_detect,
    channel_bf,
    rotate_fn,
    detect_fn,
    band_info,
    px_um,
    output_path,
    fps=1,
    resize_width=4096,
    dpi=150,
    overlay_color=(1.0, 0.35, 0.0),
    overlay_alpha=0.45,
    show_real_time=False,
    times_lookup=None,
    t0_dt=None,
    verbose=True,
):
    """
    Render an mp4 detection-overlay video for a complete time series.

    Frames are rendered one at a time and written to a temporary directory as
    PNGs, then assembled by ``ffmpeg`` (must be on ``PATH``).  This approach
    keeps peak memory usage to a single frame rather than the entire video.

    Parameters
    ----------
    czi             : CziFile
    scene           : int or None
    timepoints      : list[int]
    channel_detect  : int   – channel fed to ``detect_fn``
    channel_bf      : int   – brightfield channel (video background)
    rotate_fn       : callable  img → rotated_img
    detect_fn       : callable  (image, px_um) → bool mask
    band_info       : dict   main-channel geometry; usually
                    ``align_result["band_info"]`` from Fourier/delay alignment
    px_um           : float
    output_path     : str or Path
    fps             : int    – frames per second in the output video
    resize_width    : int    – output frame width in pixels
    dpi             : int
    overlay_color   : (r, g, b) ∈ [0, 1]³
    overlay_alpha   : float
    show_real_time  : bool   – format time label as elapsed time
    times_lookup    : dict {t_idx: datetime} or None
    t0_dt           : datetime or None
    verbose         : bool
    """
    from chipanalysis.utils.file_reader import get_frame
    from skimage.transform import resize as sk_resize

    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required: pip install Pillow")

    output_path  = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times_lookup = times_lookup or {}

    with tempfile.TemporaryDirectory(prefix="chip_dynamics_video_") as _tmp:
        tmpdir = Path(_tmp)

        for i, t in enumerate(timepoints):
            # ── Load frames ───────────────────────────────────────────────────
            img_bf_raw, _  = get_frame(czi, t, channel_bf,     scene=scene)
            img_det_raw, _ = get_frame(czi, t, channel_detect, scene=scene)
            img_bf_raw  = rotate_fn(img_bf_raw)
            img_det_raw = rotate_fn(img_det_raw)

            # ── Detect ────────────────────────────────────────────────────────
            binary = detect_fn(img_det_raw, px_um)
            if binary.shape != img_bf_raw.shape:
                binary = sk_resize(
                    binary.astype(np.float32), img_bf_raw.shape,
                    order=0, preserve_range=True, anti_aliasing=False,
                ).astype(bool)

            # ── Render & save PNG ─────────────────────────────────────────────
            label = _time_label(t, show_real_time, times_lookup, t0_dt)
            frame = render_detection_frame(
                img_bf_raw, binary, band_info, label,
                resize_width=resize_width, dpi=dpi,
                overlay_color=overlay_color, overlay_alpha=overlay_alpha,
            )
            Image.fromarray(frame).save(str(tmpdir / f"frame_{i:06d}.png"))

            if verbose and (i == 0 or (i + 1) % 20 == 0 or i == len(timepoints) - 1):
                n_obj = sk_label(binary).max()
                print(f"  [{i+1:4d}/{len(timepoints)}]  t={t}  {label!r}  n={n_obj}")

        # ── Assemble with ffmpeg ──────────────────────────────────────────────
        if verbose:
            print(f"  Assembling {len(timepoints)} frames with ffmpeg …")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-profile:v", "baseline",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}).\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    if verbose:
        print(
            f"  ✓ Video saved → {output_path.name}  "
            f"({len(timepoints)} frames @ {fps} fps  |  {resize_width}px wide)"
        )
