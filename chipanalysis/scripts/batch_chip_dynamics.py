#!/usr/bin/env python
"""
batch_chip_dynamics.py
======================
Batch chip dynamics analysis: per-scene object detection, CSV output, and
detection-overlay video export.

Pipeline (for each CZI × scene in the manifest)
------------------------------------------------
1. Rotation alignment + main-channel geometry (Fourier/delay refinement, BF channel)
2. Optional legacy pillar detection (U-Net) + band geometry
3. Temporal mean computation (for fluorescence threshold detector)
4. Build object detector from organism config (see YAML)
5. Run detection on ALL timepoints  →  save objects.csv
6. Render detection-overlay video   →  save scene_video.mp4

Usage — local / debug (single row)
-----------------------------------
python batch_chip_dynamics.py \\
    --excel  /path/to/manifest.xlsx \\
    --config /path/to/config_chip_dynamics.yaml \\
    --output /path/to/results \\
    [--row 3]           # process only row 3 (0-indexed)
    [--skip-video]      # skip video export (faster data-only runs)

Usage — SLURM job array (one job per Excel row)
------------------------------------------------
# Step 1: find out how many rows you have
N=$(python batch_chip_dynamics.py \\
        --excel manifest.xlsx --config config.yaml --output /tmp --count-rows)

# Step 2: submit
sbatch --array=0-${N} batch_chip_dynamics.sh \\
    --excel /path/to/manifest.xlsx \\
    --config /path/to/config.yaml \\
    --output /path/to/results

Expected Excel columns
-----------------------
czi_path        full path to the .czi file  (required)
scene           scene index; leave blank to process ALL scenes  (optional)
organism        physarum | celegans | dictyostelium  (required)
channel_bf      integer index of the brightfield channel  (required)
channel_purple  integer index of the purple/mCherry channel  (optional)
channel_green   integer index of the green channel  (optional)
notes           free text  (optional)

Output naming
--------------
{output_dir}/{czi_stem}_scene{s}_ch{c}_objects.csv
{output_dir}/{czi_stem}_scene{s}_ch{c}_video.mp4

Both outputs are skipped if they already exist (safe to re-run / resume).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(yaml_path: str | Path) -> dict:
    """Load YAML config and resolve all model_path values relative to the config file."""
    config_path = Path(yaml_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    _resolve_model_paths(cfg, config_path)
    return cfg


def _resolve_model_paths(cfg: dict, config_path: Path) -> None:
    """Resolve all model_path values in-place, relative to the config file's directory."""
    config_dir = config_path.parent.resolve()

    def _resolve(p: str) -> str:
        p = Path(p)
        if p.is_absolute():
            resolved = p
        else:
            resolved = (config_dir / p).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Model not found: {resolved}\n"
                f"  (config dir: {config_dir}, original path: '{p}')"
            )
        return str(resolved)

    pillar_cfg = cfg.get("pillar", {})
    if pillar_cfg.get("method", "fourier_channel") in ("ml", "unet") and "model_path" in pillar_cfg:
        cfg["pillar"]["model_path"] = _resolve(cfg["pillar"]["model_path"])

    for org_cfg in cfg.get("organisms", {}).values():
        if "model_path" in org_cfg:
            org_cfg["model_path"] = _resolve(org_cfg["model_path"])


def _require(d: dict, *keys: str) -> object:
    """Return nested value or raise a helpful KeyError."""
    path = []
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            raise KeyError(
                f"Required config key missing: {' → '.join(path + [k])}"
            )
        d = d[k]
        path.append(k)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Detector builder
# ──────────────────────────────────────────────────────────────────────────────

def build_detector(org_cfg, czi, scene, detect_channel, rotate_fn, px_um,
                   n_frames_mean):
    """
    Instantiate the object detector described by *org_cfg*.

    Returns
    -------
    detect_fn     : callable  (image, px_um) -> bool mask
    temporal_mean : np.ndarray or None
    """
    from chipanalysis.functions.detectors import (
        make_fluo_detector,
        calibrate_fluo_threshold,
        make_unet_detector,
    )
    from chipanalysis.functions.chip_dynamics import (
        compute_temporal_mean,
        get_rotated_frame,
    )

    detector_type = _require(org_cfg, "detector")

    if detector_type == "unet":
        detect_fn = make_unet_detector(
            model_path=_require(org_cfg, "model_path"),
            threshold=org_cfg.get("unet_threshold", 0.5),
            patch_size=org_cfg.get("patch_size", 256),
            patch_stride=org_cfg.get("patch_stride", 128),
            min_obj_um2=org_cfg.get("min_obj_um2", 0.0),
            normalization=org_cfg.get("normalization", "percentile"),
            norm_percentiles=tuple(org_cfg.get("norm_percentiles", (0.0, 99.5))),
        )
        return detect_fn, None   # no temporal mean needed

    elif detector_type == "fluo_threshold":
        temporal_mean = compute_temporal_mean(
            czi, channel=detect_channel, scene=scene,
            rotate_fn=rotate_fn, n_frames=n_frames_mean,
        )
        cal_t    = org_cfg.get("calibrate_timepoint", 0)
        img_ref, _ = get_rotated_frame(czi, cal_t, detect_channel, scene, rotate_fn)
        bg_sigma   = org_cfg.get("bg_sigma_um", 1000.0)
        t_factor   = org_cfg.get("thresh_factor",  0.2)
        fixed_thr  = calibrate_fluo_threshold(
            img_ref,
            temporal_mean=temporal_mean,
            bg_sigma_um=bg_sigma,
            px_um=px_um,
        ) * t_factor
        detect_fn = make_fluo_detector(
            temporal_mean=temporal_mean,
            bg_sigma_um=bg_sigma,
            min_obj_um2=org_cfg.get("min_obj_um2",  25.0),
            max_hole_um2=org_cfg.get("max_hole_um2",  5.0),
            thresh_factor=t_factor,
            fixed_threshold=fixed_thr,
        )
        return detect_fn, temporal_mean

    else:
        raise ValueError(
            f"Unknown detector type {detector_type!r}. "
            f"Supported: 'unet', 'fluo_threshold'."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-scene processing
# ──────────────────────────────────────────────────────────────────────────────

def process_scene(row, scene: int, cfg: dict, output_dir: Path,
                  skip_video: bool) -> None:
    """Run the full pipeline for one (CZI file, scene) pair."""
    from chipanalysis.functions.chip_dynamics import (
        load_czi, get_rotation_fn, get_rotated_frame,
        detect_channel_from_mask, build_cell_dataframe,
    )
    czi_path = Path(row.czi_path)
    organism  = str(row.organism).strip().lower()
    stem      = f"{czi_path.stem}_scene{scene}"

    print(f"\n{'='*64}")
    print(f"  CZI      : {czi_path.name}")
    print(f"  Scene    : {scene}")
    print(f"  Organism : {organism}")

    # ── Organism config ───────────────────────────────────────────────────────
    org_cfg = cfg.get("organisms", {}).get(organism)
    if org_cfg is None:
        print(f"  [WARN] No config for organism {organism!r}.  Skipping.")
        return

    # ── Resolve channels ──────────────────────────────────────────────────────
    channel_bf  = int(row.channel_bf)
    channel_key = _require(org_cfg, "channel_key")
    raw_ch = getattr(row, channel_key, None)
    if raw_ch is None or (isinstance(raw_ch, float) and np.isnan(raw_ch)) \
            or pd.isna(raw_ch):
        print(
            f"  [WARN] Channel column {channel_key!r} is missing for this "
            f"row.  Skipping."
        )
        return
    detect_channel = int(raw_ch)

    print(f"  Channels : BF={channel_bf}  detect={detect_channel} ({channel_key})")

    # ── Check output already done ─────────────────────────────────────────────
    csv_path   = output_dir / f"{stem}_ch{detect_channel}_objects.csv"
    video_path = output_dir / f"{stem}_ch{detect_channel}_video.mp4"
    if csv_path.exists() and (skip_video or video_path.exists()):
        print(f"  [SKIP] Already done: {csv_path.name}")
        return

    # ── Load CZI ──────────────────────────────────────────────────────────────
    czi, px_um, dim_sizes, _ = load_czi(czi_path)
    timepoints = list(range(dim_sizes.get("T", 1)))
    print(f"  Timepoints: {len(timepoints)}  |  px: {px_um:.4f} µm")

    # ── Alignment and channel geometry ────────────────────────────────────────
    timepoint_ref = cfg.get("timepoint_ref", 0)
    pillar_cfg = cfg.get("pillar", {})
    pillar_method = pillar_cfg.get("method", "fourier_channel")
    alignment_kwargs = {}
    alignment_kwargs.update(cfg.get("alignment", {}))
    alignment_kwargs.update(pillar_cfg.get("fourier", {}))

    rotate_fn, align_result = get_rotation_fn(
        czi, bf_channel=channel_bf, scene=scene,
        timepoint=timepoint_ref,
        px_um=px_um,
        debug=False,
        alignment_method="fft_only" if pillar_method in ("ml", "unet") else "fourier_channel",
        alignment_kwargs=alignment_kwargs,
    )
    print(f"  Rotation  : {align_result['rotate_angle_deg']:.2f}°")

    img_bf_ref, _ = get_rotated_frame(czi, timepoint_ref, channel_bf, scene, rotate_fn)

    if pillar_method in ("ml", "unet"):
        # Legacy option: detect pillar masks with the U-Net and infer the empty
        # channel from that mask. Kept for compatibility, no longer preferred.
        from chipanalysis.functions.detectors import make_unet_pillar_detector

        pillar_fn = make_unet_pillar_detector(
            model_path=_require(pillar_cfg, "model_path"),
            threshold=pillar_cfg.get("threshold",    0.5),
            patch_size=pillar_cfg.get("patch_size",  256),
            patch_stride=pillar_cfg.get("patch_stride", 128),
            min_pillar_um2=pillar_cfg.get("min_obj_um2", 1000.0),
        )
        print("  Running legacy pillar detector …", flush=True)
        pillar_mask = pillar_fn(img_bf_ref, px_um)
        crop_cols   = pillar_cfg.get("crop_cols", 500)
        band_info   = detect_channel_from_mask(pillar_mask, px_um, crop_cols=crop_cols)
        band_source = "pillar U-Net"
    else:
        band_info = align_result["band_info"]
        band_source = "Fourier channel"

    print(f"  Band      : {band_info['band_width_um']:.1f} µm wide  "
          f"(rows {band_info['band_top']}–{band_info['band_bottom']}, {band_source})")

    # ── Build object detector ─────────────────────────────────────────────────
    n_frames_mean = cfg.get("n_frames_mean", 10)
    print("  Building detector …", flush=True)
    detect_fn, _ = build_detector(
        org_cfg, czi, scene, detect_channel, rotate_fn, px_um, n_frames_mean,
    )
    print(f"  Detector  : {detect_fn.__doc__}")

    # ── Video config (needed here to record the scale factor in the CSV) ──────
    vid_cfg            = cfg.get("video", {})
    video_resize_width = vid_cfg.get("width", 4096)
    # Scale factor applied to every video frame: video_px = image_px * video_scale
    video_scale        = video_resize_width / img_bf_ref.shape[1]
    print(f"  Video scale: {video_scale:.4f}  "
          f"({img_bf_ref.shape[1]} px → {video_resize_width} px wide)")

    # ── CSV output ────────────────────────────────────────────────────────────
    if not csv_path.exists():
        print(f"  Detecting objects ({len(timepoints)} timepoints) …", flush=True)
        df = build_cell_dataframe(
            czi=czi,
            scene=scene,
            channel=detect_channel,
            timepoints=timepoints,
            rotate_fn=rotate_fn,
            detect_fn=detect_fn,
            band_info=band_info,
            px_um=px_um,
            min_obj_um2=org_cfg.get("min_obj_um2", 25.0),
            reference_shape=img_bf_ref.shape,
            verbose=True,
        )
        # Prepend metadata columns
        df.insert(0, "czi_file",      czi_path.name)
        df.insert(1, "scene",         scene)
        df.insert(2, "channel",       detect_channel)
        df.insert(3, "organism",      organism)
        df.insert(4, "band_top_px",   band_info["band_top"])
        df.insert(5, "band_bottom_px",band_info["band_bottom"])
        df.insert(6, "band_width_um", band_info["band_width_um"])
        df.insert(7, "px_um",         px_um)
        df.insert(8, "rotation_deg",  align_result["rotate_angle_deg"])
        df.insert(9, "video_scale",   video_scale)

        df.to_csv(csv_path, index=False)
        print(f"  ✓ CSV saved → {csv_path.name}  ({len(df)} objects detected)")
    else:
        print(f"  [SKIP] CSV already exists: {csv_path.name}")

    # ── Video output ──────────────────────────────────────────────────────────
    if skip_video:
        return

    if not video_path.exists():
        from chipanalysis.functions.video_renderer import (
            render_detection_video, _time_label,
        )
        # vid_cfg and video_resize_width already computed above

        # Timestamps for real-time labels
        times_lookup, t0_dt = {}, None
        try:
            from chipanalysis.utils.file_reader import get_timestamps_by_T
            times = get_timestamps_by_T(czi, C=0, Z=0)
            if times:
                times_lookup = {t_idx: t_dt for t_idx, t_dt in times}
                t0_dt = times_lookup.get(min(times_lookup))
        except Exception:
            pass

        print(f"  Rendering video ({len(timepoints)} frames) …", flush=True)
        render_detection_video(
            czi=czi,
            scene=scene,
            timepoints=timepoints,
            channel_detect=detect_channel,
            channel_bf=channel_bf,
            rotate_fn=rotate_fn,
            detect_fn=detect_fn,
            band_info=band_info,
            px_um=px_um,
            output_path=video_path,
            fps=vid_cfg.get("fps", 1),
            resize_width=video_resize_width,
            dpi=vid_cfg.get("dpi", 150),
            overlay_color=tuple(vid_cfg.get("overlay_color", [1.0, 0.35, 0.0])),
            overlay_alpha=vid_cfg.get("overlay_alpha", 0.45),
            show_real_time=vid_cfg.get("show_real_time", False),
            times_lookup=times_lookup,
            t0_dt=t0_dt,
            verbose=True,
        )
    else:
        print(f"  [SKIP] Video already exists: {video_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-row dispatcher (handles scene=NaN → all scenes)
# ──────────────────────────────────────────────────────────────────────────────

def process_row(row, cfg: dict, output_dir: Path, skip_video: bool) -> None:
    from aicspylibczi import CziFile

    czi_path = Path(row.czi_path)
    if not czi_path.exists():
        print(f"[ERROR] CZI not found: {czi_path}")
        return

    # Scene column: specific index or NaN → all scenes
    if hasattr(row, "scene") and not pd.isna(row.scene):
        scenes = [int(row.scene)]
    else:
        tmp      = CziFile(czi_path)
        tmp_dims = dict(zip(tmp.dims, tmp.size))
        scenes   = list(range(tmp_dims.get("S", 1)))
        print(f"  scene column empty → processing {len(scenes)} scene(s)")

    for scene in scenes:
        try:
            process_scene(row, scene, cfg, output_dir, skip_video)
        except Exception as e:
            print(f"[ERROR] {czi_path.name}  scene={scene}: {e}")
            traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch chip dynamics analysis — detection + video + CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--excel",       required=True,
                        help="Excel manifest (.xlsx)")
    parser.add_argument("--config",      required=True,
                        help="YAML config file")
    parser.add_argument("--output",      required=True,
                        help="Output directory")
    parser.add_argument("--row",         type=int, default=None,
                        help="Process only this 0-indexed row (for SLURM array)")
    parser.add_argument("--skip-video",  action="store_true",
                        help="Skip video export")
    parser.add_argument("--count-rows",  action="store_true",
                        help="Print (number_of_rows - 1) and exit "
                             "(used to set SLURM --array upper bound)")
    args = parser.parse_args()

    # ── Load manifest ─────────────────────────────────────────────────────────
    df_manifest = pd.read_excel(
        args.excel,
        dtype={
            "channel_bf":     "Int64",
            "channel_purple": "Int64",
            "channel_green":  "Int64",
            "scene":          "Int64",
        },
    )
    # Normalise column names: strip, lower, spaces → underscores
    df_manifest.columns = [
        c.strip().lower().replace(" ", "_") for c in df_manifest.columns
    ]
    # Drop fully-empty rows (common in Excel files)
    df_manifest = df_manifest.dropna(subset=["czi_path"]).reset_index(drop=True)

    if args.count_rows:
        print(len(df_manifest) - 1)   # upper bound for --array=0-N
        sys.exit(0)

    cfg        = load_config(args.config)   # also resolves ../models/ paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Select rows to process ────────────────────────────────────────────────
    if args.row is not None:
        if args.row >= len(df_manifest):
            print(
                f"[ERROR] --row {args.row} is out of range "
                f"(manifest has {len(df_manifest)} data rows, "
                f"valid indices: 0–{len(df_manifest)-1})."
            )
            sys.exit(1)
        rows = [df_manifest.iloc[args.row]]
        print(f"Processing row {args.row} of {len(df_manifest)}  →  {output_dir}")
    else:
        rows = [df_manifest.iloc[i] for i in range(len(df_manifest))]
        print(f"Processing {len(rows)} row(s)  →  {output_dir}")

    for row in rows:
        process_row(row, cfg, output_dir, args.skip_video)

    print("\nDone.")


if __name__ == "__main__":
    main()
