"""
analyze_region_fluorescence.py
-------------------------------
Extract fluorescence metrics (mean intensity, cell counts, etc.) from different
ROIs (top, bottom, main channel) in a CZI time-lapse, output to CSV.

Usage
-----
    python analyze_region_fluorescence.py /path/to/file.czi [options]

Arguments
---------
    czi_path            Path to the .czi file (required)

Options
-------
    --output PATH       Output .csv path  [default: <czi_dir>/<czi_stem>_fluorescence.csv]
    --channels INTS     Channel indices to process  [default: 0,1]
    --metric STR        Which metric to compute: 'mean', 'cell_count', or 'both'
                        [default: mean]
    --pad-left FLOAT    ROI left padding (µm, negative = inward)  [default: -500]
    --pad-right FLOAT   ROI right padding (µm, negative = inward) [default: -1000]
    --pad-top FLOAT     Top region padding (µm)                   [default: 1300]
    --pad-bottom FLOAT  Bottom region padding (µm)                [default: 1300]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from aicspylibczi import CziFile

from chipanalysis.utils.file_reader import get_pixel_sizes_um, get_timestamps_by_T, get_frame
from chipanalysis.chip_alignment import align_chip_to_image, get_roi_from_result, ChipGeometry
from chipanalysis.functions.region_fluorescence import (
    profile_mean,
    count_cells,
    compute_profiles_over_time_roi,
)

# ---------------------------------------------------------------------------
# ROI selectors
# ---------------------------------------------------------------------------

def make_roi_selectors(result, pad_left_um, pad_right_um, pad_top_um, pad_bottom_um):
    """Create ROI selector functions for top, bottom, and main regions."""
    
    def select_roi_top(img):
        roi, _ = get_roi_from_result(
            result,
            result['rotate_fn'](img),
            region="top",
            pad_left_um=pad_left_um,
            pad_right_um=pad_right_um,
            pad_top_um=pad_top_um,
            pad_bottom_um=-10.0,
        )
        return roi

    def select_roi_bottom(img):
        roi, _ = get_roi_from_result(
            result,
            result['rotate_fn'](img),
            region="bottom",
            pad_left_um=pad_left_um,
            pad_right_um=pad_right_um,
            pad_top_um=-10.0,
            pad_bottom_um=pad_bottom_um,
        )
        return roi

    def select_roi_main(img):
        roi, _ = get_roi_from_result(
            result,
            result['rotate_fn'](img),
            region="main",
            pad_left_um=pad_left_um,
            pad_right_um=pad_right_um,
            pad_top_um=-20.0,
            pad_bottom_um=-20.0,
        )
        return roi

    return [select_roi_bottom, select_roi_top, select_roi_main]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract fluorescence metrics from CZI regions over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("czi_path", type=Path, help="Path to the input .czi file")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .csv path (default: same folder as input)")
    p.add_argument("--channels", type=str, default="0,1",
                   help="Channel indices to process (comma-separated, e.g. '0,1,2')")
    p.add_argument("--metric", type=str, default="mean", choices=["mean", "cell_count", "both"],
                   help="Which metric to compute")
    p.add_argument("--pad-left", type=float, default=-500.0,
                   help="ROI left padding in µm (negative = crop inward)")
    p.add_argument("--pad-right", type=float, default=-1000.0,
                   help="ROI right padding in µm (negative = crop inward)")
    p.add_argument("--pad-top", type=float, default=1300.0,
                   help="Top region padding in µm")
    p.add_argument("--pad-bottom", type=float, default=1300.0,
                   help="Bottom region padding in µm")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker threads for timestep processing "
                        "(each timestep is independent; >1 enables concurrent I/O+compute)")
    return p


def main(args=None):
    parser = build_arg_parser()
    opts = parser.parse_args(args)

    czi_path: Path = opts.czi_path.expanduser().resolve()
    
    if not czi_path.exists():
        print(f"ERROR: CZI file not found: {czi_path}", file=sys.stderr)
        return 1

    channels = [0,1]

    # Determine output path
    output_path: Path = opts.output / f"{czi_path.stem}_fluorescence.csv" if opts.output else czi_path.parent / f"{czi_path.stem}_fluorescence.csv"

    print(f"Loading CZI: {czi_path}")
    czi = CziFile(czi_path)

    # Extract metadata
    dims = czi.dims
    sizes = czi.size
    dim_sizes = dict(zip(dims, sizes))
    pixel_size_um = get_pixel_sizes_um(czi)

    print(f"  Dimensions: {dim_sizes}")
    print(f"  Pixel size: {pixel_size_um['X']:.4f} µm")

    # Align chip on the first frame of the first channel
    print("Aligning chip to image...")
    first_frame, _ = get_frame(czi, 0, 0)
    result = align_chip_to_image(first_frame, pixel_size_um=pixel_size_um["X"], debug=False, geom=ChipGeometry())


    print("  Alignment successful.")

    # Create ROI selectors
    roi_selectors = make_roi_selectors(
        result,
        pad_left_um=opts.pad_left,
        pad_right_um=opts.pad_right,
        pad_top_um=opts.pad_top,
        pad_bottom_um=opts.pad_bottom,
    )
    roi_names = ["roi_dic", "roi_bac", "roi_air"]


    metrics = {"mean": profile_mean, "cell_count": count_cells}

    # Compute all timepoints
    times = list(range(dim_sizes["T"]))
    print(f"Computing metrics for {len(times)} timepoint(s), {len(channels)} channel(s), "
          f"{len(roi_names)} region(s) using {opts.workers} worker(s)...")

    datatable = compute_profiles_over_time_roi(
        czi,
        times,
        channels,
        roi_selectors=roi_selectors,
        roi_names=roi_names,
        metrics=metrics,
        px_um=pixel_size_um["X"],
        n_workers=opts.workers,
    )

    # Save to CSV
    datatable.to_csv(output_path, index=False)
    print(f"✓ Results saved: {output_path}")
    print(f"  Shape: {datatable.shape}")
    print(f"\nDataFrame preview:")
    print(datatable.head(10))

    return 0


if __name__ == "__main__":
    sys.exit(main())
