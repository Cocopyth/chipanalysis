"""
make_video_from_czi.py
----------------------
Convert a CZI time-lapse file into an annotated MP4 movie.

Usage
-----
    python make_video_from_czi.py /path/to/file.czi [options]

Arguments
---------
    czi_path            Path to the .czi file (required)

Options
-------
    --output PATH       Output .mp4 path  [default: <czi_dir>/<czi_stem>.mp4]
    --scale FLOAT       Time acceleration factor  [default: 6000.0]
    --fps INT           Output frames per second  [default: 10]
    --resize INT        Output width in pixels  [default: 1024]
    --stretch-min FLOAT Percentile for low contrast clip  [default: 90]
    --stretch-max FLOAT Percentile for high contrast clip [default: 99.5]
    --gamma-color FLOAT Gamma correction for colour channels [default: 0.45]
    --pad-left FLOAT    ROI left padding (µm, negative = inward)  [default: -500]
    --pad-right FLOAT   ROI right padding (µm, negative = inward) [default: -1000]
    --pad-top FLOAT     ROI top/bottom padding (µm)               [default: 2000]
    --ch-gray INT       Channel index for gray  [default: 2]
    --ch-magenta INT    Channel index for magenta  [default: 1]
    --ch-green INT      Channel index for green  [default: 0]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median

# Must be set before any matplotlib-touching import (colormaps, make_annotated…)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from aicspylibczi import CziFile
from moviepy import ImageClip, concatenate_videoclips

from chipanalysis.utils.file_reader import get_pixel_sizes_um, get_timestamps_by_T, get_frame
from chipanalysis.utils.maye_video_axio import (
    mcherry, gfp, gray_cmap, gfp,
    norm, clamp, make_annotated,
)
from chipanalysis.chip_alignment import (
    align_chip_to_image_fourier_channel,
    get_roi_from_result,
    ChipGeometry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Make an annotated MP4 from a CZI time-lapse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("czi_path", type=Path, help="Path to the input .czi file")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .mp4 path (default: same folder as input)")
    p.add_argument("--scale", type=float, default=6000.0,
                   help="Time acceleration factor")
    p.add_argument("--fps", type=int, default=10,
                   help="Output frames per second")
    p.add_argument("--resize", type=int, default=1024,
                   help="Output width in pixels")
    p.add_argument("--stretch-min", type=float, default=90.0,
                   help="Percentile for low contrast clip (colour channels)")
    p.add_argument("--stretch-max", type=float, default=99.5,
                   help="Percentile for high contrast clip (colour channels)")
    p.add_argument("--gamma-color", type=float, default=0.45,
                   help="Gamma for colour (magenta/green) channels")
    p.add_argument("--pad-left", type=float, default=-500.0,
                   help="ROI left padding in µm (negative = crop inward)")
    p.add_argument("--pad-right", type=float, default=-1000.0,
                   help="ROI right padding in µm (negative = crop inward)")
    p.add_argument("--pad-top", type=float, default=2000.0,
                   help="ROI top/bottom padding in µm")
    p.add_argument("--ch-gray", type=int, default=2,
                   help="Channel index for brightfield/gray")
    p.add_argument("--ch-magenta", type=int, default=1,
                   help="Channel index for magenta (mCherry)")
    p.add_argument("--ch-green", type=int, default=0,
                   help="Channel index for green (GFP)")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker threads for frame rendering "
                        "(frames are independent; >1 enables concurrent I/O+processing)")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    parser = build_arg_parser()
    opts = parser.parse_args(args)

    czi_path: Path = opts.czi_path.expanduser().resolve()
    if not czi_path.exists():
        print(f"ERROR: file not found: {czi_path}", file=sys.stderr)
        sys.exit(1)

    output_path: Path = opts.output / f"{czi_path.stem}_video.mp4" if opts.output else czi_path.parent / f"{czi_path.stem}_video.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input : {czi_path}")
    print(f"Output: {output_path}")

    # ── 1. Open CZI and read metadata ──────────────────────────────────────
    czi = CziFile(czi_path)
    pixel_size = get_pixel_sizes_um(czi)
    times = get_timestamps_by_T(czi, C=0, Z=0)
    print(f"Frames: {len(times)}  |  pixel size X: {pixel_size['X']:.4f} µm")

    # ── 2. Align chip on brightfield frame 0 ──────────────────────────────
    print("Aligning chip to image …")
    _, _ = get_frame(czi, 0, opts.ch_gray)        # warm-up / sanity check
    img_align, _ = get_frame(czi, 0, opts.ch_gray)
    
    geom = ChipGeometry()  # Use default geometry; customize as needed
    result = align_chip_to_image_fourier_channel(
        img_align,
        pixel_size_um=pixel_size["X"],
        geom=geom,
        debug=False,
    )
    if not result["success"]:
        print("WARNING: chip alignment failed – ROI may be incorrect.")
        for msg in result["messages"]:
            print("  ", msg)

    def select_roi(img):
        roi, _ = get_roi_from_result(
            result,
            result["rotate_fn"](img),
            pad_left_um=opts.pad_left,
            pad_right_um=opts.pad_right,
            pad_top_um=opts.pad_top,
        )
        return roi

    # ── 3. Determine contrast limits from frame 0 ─────────────────────────
    print("Computing contrast limits …")
    raw_magenta, _ = get_frame(czi, 0, opts.ch_magenta)
    raw_green, _   = get_frame(czi, 0, opts.ch_green)

    roi_magenta = select_roi(raw_magenta)
    roi_green   = select_roi(raw_green)

    lo_magenta, hi_magenta = np.percentile(roi_magenta,
                                           (opts.stretch_min, opts.stretch_max))
    lo_green, hi_green     = np.percentile(roi_green,
                                           (opts.stretch_min, opts.stretch_max))

    # ── 4. Build video duration list ──────────────────────────────────────
    ts = [s for _, s in times]
    real_deltas = [None]
    for k in range(1, len(ts)):
        dt = max((ts[k] - ts[k - 1]).total_seconds(), 0.0)
        real_deltas.append(dt)

    positive = [d for d in real_deltas[1:] if d and d > 0]
    baseline = median(positive) if positive else 1.0
    real_deltas[0] = baseline
    if len(real_deltas) > 1:
        real_deltas[-1] = real_deltas[-2] if real_deltas[-2] is not None else baseline

    video_durations = [clamp(float(d) / opts.scale, 0.0, 1e12) for d in real_deltas]

    # ── 5. Build clips (parallel frame rendering) ────────────────────────
    print(f"Rendering {len(times)} frames with {opts.workers} worker(s) …")
    
    def _render_frame(frame_i, time_i, dur):
        """Render a single frame."""
        if frame_i % max(1, len(times) // 20) == 0:
            print(f"  frame {frame_i+1}/{len(times)}")

        _, ch_magenta = get_frame(czi, time_i, opts.ch_magenta,
                                  gamma=opts.gamma_color,
                                  lo=lo_magenta, hi=hi_magenta)
        _, ch_gray    = get_frame(czi, time_i, opts.ch_gray, gamma=1)
        _, ch_green   = get_frame(czi, time_i, opts.ch_green,
                                  gamma=opts.gamma_color,
                                  lo=lo_green, hi=hi_green)

        ch_magenta = norm(select_roi(ch_magenta))
        ch_gray    = norm(select_roi(ch_gray))
        ch_green   = norm(select_roi(ch_green))

        rgb_magenta = mcherry(ch_magenta)[..., :3]
        rgb_gray    = gray_cmap(ch_gray)[..., :3]
        rgb_green   = gfp(ch_green)[..., :3]
        merged      = np.clip(rgb_magenta + rgb_green + rgb_gray, 0, 1)

        frame = make_annotated(
            merged,
            time_i,
            times,
            pixel_size["X"],
            resize_width=opts.resize,
            mode="RGB",
        )

        if frame.dtype != np.uint8:
            frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8) \
                       if frame.max() <= 1.0 else frame.astype(np.uint8)
        else:
            frame_u8 = frame

        # Release any figures created inside make_annotated to prevent leaking
        plt.close("all")

        return (frame_i, ImageClip(frame_u8, duration=dur))
    
    clips_dict = {}
    if opts.workers > 1:
        with ThreadPoolExecutor(max_workers=opts.workers) as executor:
            futures = {}
            for frame_i, ((time_i, _), dur) in enumerate(zip(times, video_durations)):
                futures[executor.submit(_render_frame, frame_i, time_i, dur)] = frame_i
            for future in as_completed(futures):
                frame_i, clip = future.result()
                clips_dict[frame_i] = clip
    else:
        for frame_i, ((time_i, _), dur) in enumerate(zip(times, video_durations)):
            frame_i_out, clip = _render_frame(frame_i, time_i, dur)
            clips_dict[frame_i] = clip
    
    # Restore frame order
    clips = [clips_dict[i] for i in range(len(times))]

    # ── 6. Write video ────────────────────────────────────────────────────
    print("Writing video …")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        str(output_path),
        fps=opts.fps,
        codec="libx264",
        audio=False,
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-profile:v", "baseline",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        ],
    )
    print(f"✅ Done: {output_path}")


if __name__ == "__main__":
    main()
