"""
chip_dynamics_analysis.py
=========================
High-level analysis helpers for batch_chip_dynamics.py CSV output.

Public API
----------
load_results(results_dir, experiments=None, scene=None)
    Load *_objects.csv files and return a tidy DataFrame.

add_tracks(df, max_speed_um_h, area_weight, min_len)
    Run frame-by-frame Hungarian tracking and append a 'track_id' column.

fit_msd(df, min_pts, max_frac, min_r2)
    Fit MSD(τ) = 4Dτ^α per track. Returns a per-track DataFrame.

plot_tracks_on_video(df, results_dir, experiment, frame_idx, ...)
    Overlay tracked trajectories on a video frame (or µm-space if no video).

plot_summary(df, df_fits)
    Track-length histogram + α-vs-D scatter coloured by mean object size.

plot_size_vs_distance(df, df_fits, alpha_range, D_range)
    Object size (µm²) vs distance to channel centre, filtered by α/D range.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_results(
    results_dir: Union[str, Path],
    experiments: Optional[Union[str, List[str]]] = None,
    scene: Optional[int] = None,
    area_min_um2: Optional[float] = None,
    area_max_um2: Optional[float] = None,
) -> pd.DataFrame:
    """Load *_objects.csv files produced by batch_chip_dynamics.py.

    Parameters
    ----------
    results_dir:
        Directory that was passed as ``--output`` to batch_chip_dynamics.py.
    experiments:
        Substring(s) to match against ``czi_file``.
        - ``None``  → load everything
        - ``"ICP2"`` → files whose czi_file contains "ICP2"
        - ``["ICP2", "CB13"]`` → union of both matches
    scene:
        If given, keep only rows whose ``scene`` column equals this integer.
    area_min_um2 / area_max_um2:
        Optional area filters applied after loading.

    Returns
    -------
    pd.DataFrame with extra columns:
        ``experiment``  – unique key  "<czi_stem>  s<scene>  ch<channel>"
        ``x_um``        – centroid_col × px_um
        ``y_um``        – centroid_row × px_um
    """
    results_dir = Path(results_dir)
    csv_files = sorted(results_dir.glob("*_objects.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_objects.csv files found in {results_dir}.\n"
            "Check that results_dir points to the --output of batch_chip_dynamics.py."
        )

    # ── if experiments specified, pre-filter CSV files to reduce I/O ──────────
    if experiments is not None:
        if isinstance(experiments, str):
            experiments = [experiments]
        # Only load CSVs whose filename matches one of the experiment strings
        filtered_csv_files = []
        for f in csv_files:
            if any(exp.lower() in f.stem.lower() for exp in experiments):
                filtered_csv_files.append(f)
        if not filtered_csv_files:
            raise ValueError(
                f"No CSV files match experiments={experiments!r}.\n"
                f"Available files: {[f.stem for f in csv_files[:5]]}"
                + ("..." if len(csv_files) > 5 else "")
            )
        csv_files = filtered_csv_files

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["_source_file"] = f.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["organism"] = df["organism"].str.strip().str.lower()

    # ── experiment filter (within-CSV row filtering, now redundant if file-level pre-filter used)
    if experiments is not None and not filtered_csv_files:
        # This branch is now only hit if experiments is specified but file-level filter wasn't applied
        if isinstance(experiments, str):
            experiments = [experiments]
        mask = pd.Series(False, index=df.index)
        for exp in experiments:
            mask |= df["czi_file"].str.contains(exp, case=False)
        df = df[mask].copy()
        if df.empty:
            raise ValueError(
                f"No rows match experiments={experiments!r}.\n"
                f"Available czi_files: {dfs[0]['czi_file'].unique().tolist()}"
            )

    # ── scene filter ──────────────────────────────────────────────────────────
    if scene is not None:
        df = df[df["scene"] == scene].copy()
        if df.empty:
            raise ValueError(f"No rows match scene={scene}.")

    # ── area filter ───────────────────────────────────────────────────────────
    if area_min_um2 is not None:
        df = df[df["area_um2"] >= area_min_um2]
    if area_max_um2 is not None:
        df = df[df["area_um2"] <= area_max_um2]

    df = df.copy()

    # ── derived columns ───────────────────────────────────────────────────────
    df["experiment"] = (
        df["czi_file"].str.replace(r"\.czi$", "", regex=True)
        + "  s" + df["scene"].astype(str)
        + "  ch" + df["channel"].astype(str)
    )
    df["x_um"] = df["centroid_col"] * df["px_um"]
    df["y_um"] = df["centroid_row"] * df["px_um"]

    print(
        f"Loaded {len(df):,} rows | "
        f"{df['experiment'].nunique()} experiments | "
        f"timepoints {df['timepoint'].min()}–{df['timepoint'].max()}"
    )
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def _track_single_experiment(
    exp_df: pd.DataFrame,
    max_speed_um_h: float,
    px_um: float,
    area_weight: float,
) -> pd.DataFrame:
    """Hungarian tracking for one (experiment, scene, channel) group."""
    exp_df = exp_df.sort_values("timepoint").reset_index(drop=True)
    tps = sorted(exp_df["timepoint"].unique())
    ids = np.full(len(exp_df), -1, dtype=int)
    nxt = 0

    m0 = exp_df["timepoint"] == tps[0]
    ids[m0.values] = np.arange(m0.sum()) + nxt
    nxt += int(m0.sum())

    for ta, tb in zip(tps[:-1], tps[1:]):
        pi = np.where((exp_df["timepoint"] == ta).values)[0]
        ci = np.where((exp_df["timepoint"] == tb).values)[0]

        if not len(pi) or not len(ci):
            ids[ci] = np.arange(len(ci)) + nxt
            nxt += len(ci)
            continue

        pr = exp_df.iloc[pi]
        cu = exp_df.iloc[ci]

        # Actual Δt for this frame pair
        dt_h = float(cu["time_hours"].iloc[0] - pr["time_hours"].iloc[0])
        if dt_h <= 0:
            # Fallback: assume timepoint difference of 1 maps to median frame rate
            dt_h = max(1e-6, dt_h)
        max_dist_px = int(np.ceil(max_speed_um_h * dt_h / px_um))

        drow = pr["centroid_row"].values[:, None] - cu["centroid_row"].values[None, :]
        dcol = pr["centroid_col"].values[:, None] - cu["centroid_col"].values[None, :]
        dist = np.hypot(drow, dcol)

        ai = pr["area_px"].values[:, None]
        aj = cu["area_px"].values[None, :]
        adiff = np.abs(ai - aj) / np.maximum(np.maximum(ai, aj), 1.0)

        cost = np.where(
            dist <= max_dist_px,
            dist / max_dist_px + area_weight * adiff,
            1e9,
        )

        rows, cols = linear_sum_assignment(cost)
        linked: set = set()
        for r, c in zip(rows, cols):
            if cost[r, c] < 1e8:
                ids[ci[c]] = ids[pi[r]]
                linked.add(c)
        for c in range(len(ci)):
            if c not in linked:
                ids[ci[c]] = nxt
                nxt += 1

    out = exp_df.copy()
    out["track_id"] = ids
    return out


def add_tracks(
    df: pd.DataFrame,
    max_speed_um_h: float = 300.0,
    area_weight: float = 0.1,
    min_len: int = 5,
) -> pd.DataFrame:
    """Add a ``track_id`` column by running Hungarian tracking per experiment.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`load_results` (must have ``time_hours``).
    max_speed_um_h:
        Maximum object speed in µm/h. Sets the per-frame spatial gate:
        ``max_dist_px = ceil(max_speed_um_h × Δt_h / px_um)``
    area_weight:
        Weight of normalised area difference in the cost matrix.
    min_len:
        Minimum track length (timepoints); shorter tracks are removed.

    Returns
    -------
    A new DataFrame with an added ``track_id`` column (string
    ``"<experiment>|<int>"``).  Rows belonging to tracks shorter than
    *min_len* are dropped.
    """
    if "time_hours" not in df.columns or df["time_hours"].isna().mean() > 0.5:
        raise ValueError(
            "'time_hours' column is missing or mostly NaN. "
            "Re-run batch_chip_dynamics.py or set time_hours manually."
        )

    parts = []
    for exp_name, exp_df in df.groupby("experiment"):
        px_um = float(exp_df["px_um"].iloc[0])
        tr = _track_single_experiment(exp_df.copy(), max_speed_um_h, px_um, area_weight)
        tr["track_id"] = exp_name + "|" + tr["track_id"].astype(str)
        parts.append(tr)
        n_raw = tr["track_id"].nunique()
        print(f"  {exp_name}  →  {n_raw:,} raw tracks")

    df_tracked = pd.concat(parts, ignore_index=True)

    # ── minimum length filter ─────────────────────────────────────────────────
    tlen = df_tracked.groupby("track_id")["timepoint"].count()
    keep = tlen[tlen >= min_len].index
    df_tracked = df_tracked[df_tracked["track_id"].isin(keep)].copy()

    n_kept = df_tracked["track_id"].nunique()
    print(
        f"\nAfter min_len ≥ {min_len}: {n_kept:,} tracks  "
        f"({len(df_tracked):,} detections)"
    )
    return df_tracked.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MSD FITTING
# ─────────────────────────────────────────────────────────────────────────────

def _fit_msd_single(
    tdf: pd.DataFrame,
    max_frac: float,
    min_pts: int,
) -> Optional[dict]:
    """Time-averaged MSD fit for one track. Returns None if not enough lags."""
    tdf = tdf.sort_values("timepoint")
    x = tdf["x_um"].values
    y = tdf["y_um"].values
    th = tdf["time_hours"].values
    max_lag = max(1, int(np.floor(len(x) * max_frac)))

    pts = []
    for k in range(1, max_lag + 1):
        dt = np.mean(th[k:] - th[:-k])
        msd = np.mean((x[k:] - x[:-k]) ** 2 + (y[k:] - y[:-k]) ** 2)
        if dt > 0 and msd > 0:
            pts.append((dt, msd))

    if len(pts) < min_pts:
        return None

    dt_arr, msd_arr = zip(*pts)
    slope, intercept, r, *_ = linregress(np.log(dt_arr), np.log(msd_arr))
    return {
        "alpha": slope,
        "D": np.exp(intercept) / 4.0,
        "r2": r ** 2,
        "n_pts": len(pts),
    }


def fit_msd(
    df: pd.DataFrame,
    min_pts: int = 4,
    max_frac: float = 0.5,
    min_r2: float = 0.7,
) -> pd.DataFrame:
    """Fit MSD(τ) = 4Dτ^α per track.

    Parameters
    ----------
    df:
        Tracked DataFrame (output of :func:`add_tracks`).
    min_pts:
        Minimum number of lag points required for a fit.
    max_frac:
        Use only the first *max_frac* fraction of lags to avoid bias.
    min_r2:
        Minimum R² to keep a fit.

    Returns
    -------
    pd.DataFrame with columns:
        ``track_id, experiment, length_tp, duration_h, alpha, D, r2, n_pts,
        mean_area_um2``
    """
    records = []
    for tk, tdf in df.groupby("track_id"):
        if len(tdf) < min_pts + 1:
            continue
        res = _fit_msd_single(tdf, max_frac=max_frac, min_pts=min_pts)
        if res is not None:
            res["track_id"] = tk
            res["experiment"] = tdf["experiment"].iloc[0]
            res["length_tp"] = len(tdf)
            res["duration_h"] = float(tdf["time_hours"].max() - tdf["time_hours"].min())
            res["mean_area_um2"] = float(tdf["area_um2"].mean())
            records.append(res)

    if not records:
        warnings.warn("No tracks could be fitted.")
        return pd.DataFrame()

    df_fits = pd.DataFrame(records)
    before = len(df_fits)
    df_fits = df_fits[
        (df_fits["r2"] >= min_r2) & (df_fits["alpha"] > 0) & (df_fits["D"] > 0)
    ].copy()

    print(
        f"Fitted {before:,} tracks → {len(df_fits):,} pass quality filter "
        f"(R²≥{min_r2}, α>0, D>0)\n"
        f"  α  median={df_fits['alpha'].median():.2f}  "
        f"mean={df_fits['alpha'].mean():.2f}\n"
        f"  D  median={df_fits['D'].median():.4f}  "
        f"mean={df_fits['D'].mean():.4f}  µm²/h^α"
    )
    return df_fits.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VIDEO OVERLAY PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_tracks_on_video(
    df: pd.DataFrame,
    results_dir: Union[str, Path],
    experiment: Optional[str] = None,
    frame_idx: int = 0,
    max_tracks: int = 2000,
    track_lw: float = 1.2,
    alpha: float = 0.65,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Overlay tracked trajectories on a video frame.

    If the video file is not found, falls back to plotting in µm-space.

    Parameters
    ----------
    df:
        Tracked DataFrame (output of :func:`add_tracks`).
    results_dir:
        Directory containing the ``*_video.mp4`` files.
    experiment:
        Experiment key to plot. Defaults to the first experiment in *df*.
    frame_idx:
        Video frame index to use as background (0 = first).
    max_tracks:
        Cap on the number of trajectories rendered.
    track_lw / alpha:
        Line width and opacity for trajectories.
    ax:
        Existing matplotlib Axes to draw into (optional).

    Returns
    -------
    matplotlib Figure
    """
    results_dir = Path(results_dir)
    if experiment is None:
        experiment = df["experiment"].iloc[0]

    sub = df[df["experiment"] == experiment].copy()
    if sub.empty:
        raise ValueError(f"No data for experiment '{experiment}'.")

    row0 = sub.iloc[0]
    keys = sub["track_id"].unique()
    if len(keys) > max_tracks:
        keys = np.random.default_rng(42).choice(keys, max_tracks, replace=False)
    sub = sub[sub["track_id"].isin(keys)]

    t_lo = sub["time_hours"].min()
    t_hi = sub["time_hours"].max()
    norm_t = plt.Normalize(t_lo, t_hi)
    cmap_t = plt.cm.plasma

    # ── Try to load video frame ───────────────────────────────────────────────
    czi_stem = Path(row0["czi_file"]).stem
    scene_idx = int(row0["scene"])
    ch_idx = int(row0["channel"])
    video_path = results_dir / f"{czi_stem}_scene{scene_idx}_ch{ch_idx}_video.mp4"

    use_video = False
    frame_rgb = None
    vs = 1.0

    if video_path.exists():
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total_frames - 1))
            ok, frame_bgr = cap.read()
            cap.release()
            if ok:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                vid_h, vid_w = frame_rgb.shape[:2]
                if "video_scale" in row0.index and not pd.isna(row0["video_scale"]):
                    vs = float(row0["video_scale"])
                else:
                    max_col = sub["centroid_col"].max()
                    vs = vid_w / (max_col * 1.05)
                    warnings.warn(
                        "video_scale not found in CSV – using estimated value."
                    )
                use_video = True
                print(
                    f"Video: {video_path.name}  "
                    f"frame {frame_idx}/{total_frames-1}  "
                    f"{vid_w}×{vid_h} px  scale={vs:.4f}"
                )
        except ImportError:
            warnings.warn("opencv-python not installed — falling back to µm-space.")
    else:
        warnings.warn(f"Video not found: {video_path}  — plotting in µm-space.")

    # ── Figure setup ─────────────────────────────────────────────────────────
    if ax is None:
        if use_video and frame_rgb is not None:
            figw = 14
            figh = figw * frame_rgb.shape[0] / frame_rgb.shape[1]
        else:
            figw, figh = 10, 8
        fig, ax = plt.subplots(figsize=(figw, figh))
    else:
        fig = ax.figure

    if use_video and frame_rgb is not None:
        ax.imshow(frame_rgb, origin="upper", zorder=0)

        # Band boundaries in video pixels
        bt = row0["band_top_px"] * vs
        bb = row0["band_bottom_px"] * vs
        bc = (bt + bb) / 2
        ax.axhspan(bt, bb, color="gold", alpha=0.15, zorder=1)
        ax.axhline(bt, color="gold", lw=1.5, ls="--", zorder=2)
        ax.axhline(bb, color="gold", lw=1.5, ls="--", zorder=2)
        ax.axhline(bc, color="gold", lw=0.8, ls=":", zorder=2, alpha=0.7)

        for tk in keys:
            tdf = sub[sub["track_id"] == tk].sort_values("timepoint")
            col_v = tdf["centroid_col"].values * vs
            row_v = tdf["centroid_row"].values * vs
            c = cmap_t(norm_t(tdf["time_hours"].mean()))
            ax.plot(col_v, row_v, "-", color=c, lw=track_lw, alpha=alpha, zorder=3)
            ax.scatter(col_v[0], row_v[0], s=14, color=c, zorder=5, marker="o")
            ax.scatter(col_v[-1], row_v[-1], s=20, color=c, zorder=5, marker=">")

        ax.set_xlim(0, vid_w)
        ax.set_ylim(vid_h, 0)
        ax.set_xlabel("X (video px)")
        ax.set_ylabel("Y (video px)")
        title_suffix = f"frame {frame_idx}  |  scale={vs:.4f}"
    else:
        # µm-space fallback
        px = float(row0["px_um"])
        bt_um = row0["band_top_px"] * px
        bb_um = row0["band_bottom_px"] * px
        bc_um = (bt_um + bb_um) / 2
        ax.axhspan(bt_um, bb_um, color="gold", alpha=0.18, zorder=0)
        ax.axhline(bt_um, color="goldenrod", lw=1.5, ls="--", zorder=1)
        ax.axhline(bb_um, color="goldenrod", lw=1.5, ls="--", zorder=1)
        ax.axhline(bc_um, color="goldenrod", lw=0.8, ls=":", zorder=1, alpha=0.6)

        for tk in keys:
            tdf = sub[sub["track_id"] == tk].sort_values("timepoint")
            c = cmap_t(norm_t(tdf["time_hours"].mean()))
            ax.plot(tdf["x_um"], tdf["y_um"], "-", color=c, lw=track_lw, alpha=alpha, zorder=2)
            ax.scatter(tdf["x_um"].iloc[0], tdf["y_um"].iloc[0], s=14, color=c, zorder=4, marker="o")
            ax.scatter(tdf["x_um"].iloc[-1], tdf["y_um"].iloc[-1], s=20, color=c, zorder=4, marker=">")

        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        title_suffix = "µm-space (no video)"

    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_t, norm=norm_t),
        ax=ax, label="Mean track time (h)", shrink=0.6,
    )
    ax.set_title(
        f"{experiment}\n{len(keys)} tracks  |  {title_suffix}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SUMMARY PLOT  (track length + α-vs-D coloured by size)
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(
    df: pd.DataFrame,
    df_fits: pd.DataFrame,
) -> plt.Figure:
    """Summary figure: track-length histogram and α-vs-D scatter by size.

    Parameters
    ----------
    df:
        Tracked DataFrame (output of :func:`add_tracks`).
    df_fits:
        MSD fit results (output of :func:`fit_msd`).

    Returns
    -------
    matplotlib Figure  (1 row × 2 columns)
    """
    tlen_tp = df.groupby("track_id")["timepoint"].count().rename("length_tp")

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    # ── Left: track length histogram ─────────────────────────────────────────
    ax = axes[0]
    ax.hist(
        tlen_tp.values,
        bins=min(80, int(tlen_tp.max())),
        color="steelblue",
        edgecolor="none",
        log=True,
    )
    ax.axvline(
        tlen_tp.median(),
        color="red", lw=2, ls="--",
        label=f"median = {tlen_tp.median():.0f} tp",
    )
    ax.set_xlabel("Track length (timepoints)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Track length distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Right: α vs D scatter, colour = mean area ─────────────────────────────
    ax = axes[1]
    if df_fits.empty:
        ax.text(0.5, 0.5, "No MSD fits available", ha="center", va="center",
                transform=ax.transAxes)
    else:
        sc = ax.scatter(
            df_fits["alpha"],
            np.log10(df_fits["D"]),
            c=np.log10(df_fits["mean_area_um2"].clip(lower=1e-3)),
            cmap="RdYlBu_r",
            s=18,
            alpha=0.55,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="log₁₀ mean area (µm²)")
        ax.axvline(1.0, color="gray", lw=1, ls=":", alpha=0.6, label="α=1 diffusion")
        ax.axvline(2.0, color="gray", lw=1, ls="-", alpha=0.3, label="α=2 ballistic")
        ax.set_xlabel("Anomalous exponent  α")
        ax.set_ylabel("log₁₀  D  (µm²/h^α)")
        ax.set_title(
            f"α vs D  [{len(df_fits):,} tracks]  colour = log₁₀ area"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Track summary", fontsize=12)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SIZE vs DISTANCE TO CHANNEL  (filtered by α / D range)
# ─────────────────────────────────────────────────────────────────────────────

def plot_size_vs_distance(
    df: pd.DataFrame,
    df_fits: pd.DataFrame,
    alpha_range: Tuple[float, float] = (0.0, np.inf),
    D_range: Tuple[float, float] = (0.0, np.inf),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Object size (µm²) vs distance to channel centre, filtered by α/D range.

    Each point is one detection row (not per-track), restricted to tracks that
    pass the α/D filter.  Points are colour-coded by ``time_hours``.
    Vertical dashed lines mark the channel boundaries (±half-width).

    Parameters
    ----------
    df:
        Tracked DataFrame (output of :func:`add_tracks`).
    df_fits:
        MSD fit results (output of :func:`fit_msd`).
    alpha_range:
        ``(alpha_min, alpha_max)`` inclusive filter.
    D_range:
        ``(D_min, D_max)`` inclusive filter.
    ax:
        Optional existing Axes.

    Returns
    -------
    matplotlib Figure
    """
    # ── filter df_fits ────────────────────────────────────────────────────────
    mask = (
        (df_fits["alpha"] >= alpha_range[0])
        & (df_fits["alpha"] <= alpha_range[1])
        & (df_fits["D"] >= D_range[0])
        & (df_fits["D"] <= D_range[1])
    )
    good_tracks = df_fits.loc[mask, "track_id"]
    if good_tracks.empty:
        raise ValueError(
            f"No tracks pass α∈{alpha_range}, D∈{D_range}. "
            "Relax the filter ranges."
        )

    sub = df[df["track_id"].isin(good_tracks)].copy()
    print(
        f"Plotting {len(sub):,} detections from {sub['track_id'].nunique():,} "
        f"tracks (α∈{alpha_range}, D∈{D_range})"
    )

    # ── band half-width ───────────────────────────────────────────────────────
    band_half_um = float(sub["band_width_um"].iloc[0]) / 2.0

    # ── figure ────────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    norm_t = plt.Normalize(sub["time_hours"].min(), sub["time_hours"].max())
    sub["dist_to_band_um_absolute"] = np.abs(sub["dist_to_band_um"])
    sc = ax.scatter(
        sub["dist_to_band_um_absolute"],
        sub["area_um2"],
        c=sub["time_hours"],
        cmap="plasma",
        s=8,
        alpha=0.4,
        edgecolors="none",
        norm=norm_t,
    )
    plt.colorbar(sc, ax=ax, label="Time (h)")

    ax.axvline(band_half_um, color="gold", lw=1.5, ls="--")
    ax.axvline(0, color="goldenrod", lw=0.8, ls=":", alpha=0.7, label="channel centre")

    ax.set_xlabel("Distance to channel centre (µm)")
    ax.set_ylabel("Object area (µm²)")
    ax.set_title(
        f"Size vs channel distance  [{sub['track_id'].nunique():,} tracks]\n"
        f"α∈{alpha_range}  D∈{D_range}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig,ax
