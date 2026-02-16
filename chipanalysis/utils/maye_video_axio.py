import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from statistics import median
from moviepy import ImageClip, concatenate_videoclips
from matplotlib import cm
from datetime import datetime

mcherry = LinearSegmentedColormap.from_list(
    "mcherry",
    [
        (0.00, "#000000"),  # black
        (0.25, "#2a001f"),  # very dark purple
        (0.50, "#5c005c"),  # purple
        (0.75, "#a100a1"),  # strong magenta
        (1.00, "#ff33ff"),  # bright magenta
    ]
)


gray_cmap = cm.get_cmap("gray")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))
def norm(x):
    x = x.astype(float)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

from chipanalysis.scripts.make_video import make_frame_array_from_image, CFG
from datetime import timedelta
from dataclasses import replace
import cv2



def get_datetime(isoformat_str):
    dt = datetime.fromisoformat(
        isoformat_str.replace("Z", "+00:00")[:26] + "+00:00"
    )
    return(dt)

def make_annotated(im,time,times,units_per_pixel,resize_width,mode = "RGB"):
    CFG_A = replace(CFG,
                add_scale_bar=True,
                box_alpha=30,
                timestamp_position="br",
                bar_position="bl",
                units_per_pixel = units_per_pixel,
                bar_length_units =500,
                resize_width = resize_width)
    np_img_u8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img_u8, mode=mode)
    cfg = CFG_A
    first_ts = times[0][1]
    ts = times[time][1]

    frame = make_frame_array_from_image(pil_img, ts, first_ts, CFG_A)
    return(frame)