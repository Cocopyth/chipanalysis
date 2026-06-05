"""
pillar_detection.py
====================
Utility functions for detecting pillar bounding-box rectangles from a binary
segmentation mask produced by the pillar U-Net.

Public API
----------
clean_pillar_mask       -- morphological clean-up of the raw binary mask
detect_pillars          -- contour-based minAreaRect fitting + quality filter
pillars_to_dataframe    -- convert pillar_list to a pandas DataFrame
estimate_grid_rotation  -- infer grid tilt angle from pillar orientations
rotate_and_redetect     -- rotate image+mask and re-run detection
draw_pillar_boxes       -- draw oriented boxes on a matplotlib Axes
"""

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import rotate as nd_rotate
from scipy.stats import circmean
from skimage.morphology import remove_small_objects, remove_small_holes


# ─────────────────────────────────────────────────────────────────────────────
# Mask cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_pillar_mask(pred_binary, px_um, min_area_um2=100.0):
    """
    Morphologically clean a raw binary pillar mask.

    Parameters
    ----------
    pred_binary  : np.ndarray (H, W) bool — raw network output mask
    px_um        : float — pixel size in µm/px
    min_area_um2 : float — blobs smaller than this (µm²) are removed

    Returns
    -------
    clean : np.ndarray (H, W) bool
    """
    min_px = max(1, int(np.round(min_area_um2 / px_um ** 2)))
    clean = remove_small_objects(pred_binary.copy(), max_size=min_px)
    clean = remove_small_holes(clean, max_size=min_px)
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Rectangle detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_pillars(pillar_clean, px_um, max_black_frac=0.35, min_area_um2=100.0):
    """
    Fit a minimum-area rotated rectangle to every connected component of
    `pillar_clean` using cv2.minAreaRect.

    Quality filter: rectangles whose interior contains more than
    `max_black_frac` fraction of background pixels are discarded.

    Parameters
    ----------
    pillar_clean  : np.ndarray (H, W) bool — cleaned binary mask
    px_um         : float — pixel size in µm/px
    max_black_frac: float — max allowed background fraction inside the box
    min_area_um2  : float — contours smaller than this (µm²) are skipped

    Returns
    -------
    pillar_list : list of dict, each with keys:
        id, centroid_col, centroid_row, width_px, height_px,
        width_um, height_um, area_um2, angle_deg, black_frac,
        corners_xy (4×2, col/row), corners_rc (4×2, row/col)
    """
    min_px  = max(1, int(np.round(min_area_um2 / px_um ** 2)))
    mask_u8 = pillar_clean.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    pillar_list = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_px:
            continue

        rect   = cv2.minAreaRect(cnt)            # ((cx, cy), (w, h), angle)
        box_xy = cv2.boxPoints(rect).astype(float)  # 4 × (x, y) = (col, row)

        cx, cy = rect[0]
        w,  h  = rect[1]
        angle  = rect[2]

        # ── Quality filter ────────────────────────────────────────────────────
        canvas = np.zeros_like(mask_u8)
        cv2.fillPoly(canvas, [box_xy.astype(np.int32)], 255)
        box_area  = canvas.sum() / 255.0
        inside_fg = np.logical_and(canvas > 0, pillar_clean).sum()
        black_frac = 1.0 - inside_fg / (box_area + 1e-6)

        if black_frac > max_black_frac:
            continue

        pillar_list.append({
            "id":           len(pillar_list) + 1,
            "centroid_col": float(cx),
            "centroid_row": float(cy),
            "width_px":     float(w),
            "height_px":    float(h),
            "width_um":     float(w)  * px_um,
            "height_um":    float(h)  * px_um,
            "area_um2":     area_px * px_um ** 2,
            "angle_deg":    float(angle),
            "black_frac":   float(black_frac),
            "corners_xy":   box_xy,            # (4, 2)  (x=col, y=row)
            "corners_rc":   box_xy[:, ::-1],   # (4, 2)  (row, col)
        })

    return pillar_list


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame helper
# ─────────────────────────────────────────────────────────────────────────────

def pillars_to_dataframe(pillar_list):
    """
    Convert a pillar_list to a pandas DataFrame (corner arrays excluded).
    """
    return pd.DataFrame([
        {k: v for k, v in p.items() if k not in ("corners_xy", "corners_rc")}
        for p in pillar_list
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Grid rotation estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_grid_rotation(pillar_list):
    """
    Estimate the dominant grid tilt from minAreaRect pillar angles.

    Because a rectangular grid has 90° periodicity, all pillar angles are
    folded into [0°, 90°) before averaging.  The result is returned in
    [-45°, 45°] (the smallest equivalent correction rotation).

    Parameters
    ----------
    pillar_list : list of dicts as returned by detect_pillars()

    Returns
    -------
    rotation_deg : float  — estimated grid tilt in [-45°, 45°]
    """
    raw  = np.array([p["angle_deg"] for p in pillar_list])
    fold = raw % 90.0                          # [0°, 90°)

    # Circular mean via doubling trick
    rad      = np.deg2rad(fold * 2.0)
    mean_rad = circmean(rad, low=0.0, high=2.0 * np.pi)
    grid_ang = np.degrees(mean_rad) / 2.0      # [0°, 90°)

    rotation_deg = grid_ang if grid_ang <= 45.0 else grid_ang - 90.0
    return float(rotation_deg)


# ─────────────────────────────────────────────────────────────────────────────
# Rotate → re-detect pipeline
# ─────────────────────────────────────────────────────────────────────────────

def rotate_and_redetect(img_norm, pred_binary, rotation_deg, px_um,
                        min_area_um2=100.0, max_black_frac=0.35):
    """
    Apply a correction rotation to the image and mask, then re-run detection.

    The image is rotated by `-rotation_deg` (i.e. counter-clockwise to undo a
    clockwise tilt).

    Parameters
    ----------
    img_norm     : np.ndarray (H, W) float32 — normalised image
    pred_binary  : np.ndarray (H, W) bool    — raw network mask
    rotation_deg : float — grid tilt to correct (as from estimate_grid_rotation)
    px_um        : float — pixel size µm/px
    min_area_um2 : float — passed to clean_pillar_mask / detect_pillars
    max_black_frac: float — passed to detect_pillars

    Returns
    -------
    img_rot         : (H, W) float32 — rotated image
    pillar_clean_rot: (H, W) bool    — cleaned rotated mask
    pillar_list_rot : list of dicts  — detected pillars in rotated frame
    """
    img_rot = nd_rotate(img_norm, angle=-rotation_deg, reshape=False, order=1)
    mask_rot = nd_rotate(
        pred_binary.astype(np.float32), angle=-rotation_deg,
        reshape=False, order=0,
    ) > 0.5

    pillar_clean_rot = clean_pillar_mask(mask_rot, px_um, min_area_um2)
    pillar_list_rot  = detect_pillars(
        pillar_clean_rot, px_um, max_black_frac, min_area_um2
    )
    return img_rot, pillar_clean_rot, pillar_list_rot


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helper
# ─────────────────────────────────────────────────────────────────────────────

def draw_pillar_boxes(ax, pillar_list, color="cyan", fontsize=5, lw=1.2,
                      show_id=True):
    """
    Draw oriented bounding-box outlines on a matplotlib Axes.

    Parameters
    ----------
    ax          : matplotlib.axes.Axes
    pillar_list : list of dicts as returned by detect_pillars()
    color       : edge colour string or RGB tuple
    fontsize    : label font size (0 to suppress labels)
    lw          : line width
    show_id     : whether to annotate each box with its id
    """
    import matplotlib.pyplot as plt

    for p in pillar_list:
        ax.add_patch(
            plt.Polygon(p["corners_xy"], closed=True,
                        lw=lw, edgecolor=color, facecolor="none")
        )
        if show_id and fontsize > 0:
            ax.text(p["centroid_col"], p["centroid_row"], str(p["id"]),
                    color=color, fontsize=fontsize, ha="center", va="center")
