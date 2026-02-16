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
