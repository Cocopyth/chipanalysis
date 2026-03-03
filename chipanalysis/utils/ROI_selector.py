import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from chipanalysis.utils.file_reader import stretch_contrast

# img_disp should be a 2D array you display (e.g. contrast-stretched)
# If you already have `img` from the CZI loader, do:
def ROI_selector(img):
    img_disp = stretch_contrast(img, 1, 99)

    roi = {"x0": None, "y0": None, "x1": None, "y1": None}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_disp, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Drag to select ROI. Press Enter to accept, Esc to clear.")
    ax.set_axis_off()

    def onselect(eclick, erelease):
        x0, y0 = float(eclick.xdata), float(eclick.ydata)
        x1, y1 = float(erelease.xdata), float(erelease.ydata)

        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        roi.update({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
        print("ROI:", roi)

    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],          # left mouse button
        interactive=True,    # allow resizing/moving after draw
        minspanx=5,
        minspany=5,
        spancoords="pixels",
    )

    def on_key(event):
        if event.key == "enter":
            print("Accepted ROI:", roi)
        elif event.key == "escape":
            rect_selector.set_visible(False)
            rect_selector.set_visible(True)  # quick clear/redraw
            roi.update({"x0": None, "y0": None, "x1": None, "y1": None})
            fig.canvas.draw_idle()
            print("Cleared ROI")

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return(roi)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from chipanalysis.utils.file_reader import stretch_contrast


def ROI_selector_down(img, downsample=1, snap_to_int=True):
    """
    Select an ROI interactively.

    Parameters
    ----------
    img : 2D ndarray
        Full-resolution image.
    downsample : int
        Downsampling factor for DISPLAY ONLY. ROI is returned in full-res coords.
        Example: downsample=4 displays every 4th pixel in x/y.
        Use 1 to disable downsampling.
    snap_to_int : bool
        If True, return integer pixel indices (recommended for array slicing).

    Returns
    -------
    roi : dict with keys x0,y0,x1,y1 (in full-resolution coordinates)
    """
    if downsample is None:
        downsample = 1
    downsample = int(max(1, downsample))

    H, W = img.shape[:2]

    # Contrast stretch on the full image, then downsample for display
    img_disp = stretch_contrast(img, 1, 99)

    if downsample == 1:
        img_disp_ds = img_disp
        ds_H, ds_W = H, W
        scale_x = 1.0
        scale_y = 1.0
    else:
        # Simple decimation for display (fast). ROI mapping remains correct.
        img_disp_ds = img_disp[::downsample, ::downsample]
        ds_H, ds_W = img_disp_ds.shape[:2]

        # Map from display coords (0..ds_W-1) to full-res coords (0..W-1)
        # Use ratio of sizes rather than assuming perfect divisibility.
        scale_x = W / float(ds_W)
        scale_y = H / float(ds_H)

    roi = {"x0": None, "y0": None, "x1": None, "y1": None}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_disp_ds, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(
        f"Drag to select ROI (display downsample={downsample}x). "
        "Press Enter to accept, Esc to clear."
    )
    ax.set_axis_off()

    def _clip(v, lo, hi):
        return max(lo, min(hi, v))

    def _to_fullres(x_ds, y_ds):
        # Convert display coordinates to full-res coordinates
        x_full = x_ds * scale_x
        y_full = y_ds * scale_y

        # Clip to valid full-res pixel coordinate range
        x_full = _clip(x_full, 0, W - 1)
        y_full = _clip(y_full, 0, H - 1)
        return x_full, y_full

    def onselect(eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return

        x0_ds, y0_ds = float(eclick.xdata), float(eclick.ydata)
        x1_ds, y1_ds = float(erelease.xdata), float(erelease.ydata)

        # Sort in display coords
        x0_ds, x1_ds = sorted([x0_ds, x1_ds])
        y0_ds, y1_ds = sorted([y0_ds, y1_ds])

        # Convert to full-res coords
        x0, y0 = _to_fullres(x0_ds, y0_ds)
        x1, y1 = _to_fullres(x1_ds, y1_ds)

        # Sort again (in case clipping swapped edges)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        if snap_to_int:
            # For slicing, it's often best to use int indices.
            # Use floor for start and ceil for end to include the selected area.
            x0i = int(np.floor(x0))
            y0i = int(np.floor(y0))
            x1i = int(np.ceil(x1))
            y1i = int(np.ceil(y1))

            # Clip to bounds
            x0i = _clip(x0i, 0, W - 1)
            x1i = _clip(x1i, 0, W - 1)
            y0i = _clip(y0i, 0, H - 1)
            y1i = _clip(y1i, 0, H - 1)

            roi.update({"x0": x0i, "y0": y0i, "x1": x1i, "y1": y1i})
        else:
            roi.update({"x0": x0, "y0": y0, "x1": x1, "y1": y1})

        print("ROI (full-res):", roi)

    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        interactive=True,
        minspanx=5,
        minspany=5,
        spancoords="pixels",
    )

    def on_key(event):
        if event.key == "enter":
            print("Accepted ROI (full-res):", roi)
            plt.close(fig)
        elif event.key == "escape":
            # Clear the selection and reset ROI
            rect_selector.set_visible(False)
            rect_selector.set_visible(True)
            roi.update({"x0": None, "y0": None, "x1": None, "y1": None})
            fig.canvas.draw_idle()
            print("Cleared ROI")

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return roi