#!/usr/bin/env python3
"""
Image → time-proportional video builder
- Asks only: folder, output, scale_factor
- Timestamp font size, margins etc. are fractions of final image HEIGHT
- Optional scale bar from units-per-pixel
- Semi-transparent background boxes (set alpha in CONFIG; 0 = invisible)
"""

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# MoviePy v1/v2 compatible imports
try:
    from moviepy import ImageClip, concatenate_videoclips  # v2 style
except Exception:
    from moviepy.editor import ImageClip, concatenate_videoclips  # v1 style


# =========================
# CONFIG (edit these)
# =========================
@dataclass
class Config:
    # Filename timestamp parsing
    timestamp_regex: str = r"-(\d{14})-"
    timestamp_dt_format: str = "%m%d%Y%H%M%S"

    # Encoding
    fps: int = 5
    # Resize images to this width BEFORE stamping (None = keep original)
    resize_width: int | None = 512

    # Image file extensions to include
    exts: str = ".jpg,.jpeg,.png,.bmp,.tif,.tiff"

    # Timestamp overlay
    add_timestamp: bool = True
    hours_decimals: int = 2
    timestamp_position: str = "br"  # tl,tr,bl,br
    timestamp_font_h_frac: float = 0.055
    timestamp_margin_h_frac: float = 0.02
    timestamp_boxpad_h_frac: float = 0.012
    # Optional font path (recommended on Windows):
    # e.g. r"C:\Windows\Fonts\arial.ttf" or r"C:\Windows\Fonts\segoeui.ttf"
    timestamp_font_path: str | None = None

    # Scale bar overlay
    add_scale_bar: bool = True
    # units per pixel for the ORIGINAL images (before resize)
    units_per_pixel: float = 0.48
    units_label: str = "µm"
    bar_length_units: float = 200.0
    bar_position: str = "bl"  # tl,tr,bl,br
    bar_height_h_frac: float = 0.012
    bar_font_h_frac: float = 0.045
    bar_margin_h_frac: float = 0.025
    bar_boxpad_h_frac: float = 0.012
    bar_font_path: str | None = None

    # Box + text styling
    # 0 = fully transparent (box invisible), 255 = opaque
    box_alpha: int = 30
    text_rgba: tuple[int, int, int, int] = (255, 255, 255, 255)

    # Stroke for readability (helps a lot if box_alpha is low)
    stroke_width_px: int = 2
    stroke_fill_rgba: tuple[int, int, int, int] = (0, 0, 0, 255)


CFG = Config()
# =========================


def ask(prompt, default=None, cast=None):
    while True:
        txt = input(f"{prompt} [{default if default is not None else ''}]: ").strip()
        if not txt and default is not None:
            return default
        if cast:
            try:
                return cast(txt)
            except Exception:
                print("❌ Invalid input, try again.")
        else:
            return txt


def find_images(folder: Path, exts: str) -> list[Path]:
    extset = {e.lower().strip() if e.startswith(".") else "." + e.lower().strip()
              for e in exts.split(",")}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extset])


def extract_timestamp(path: Path, regex: str, dt_fmt: str) -> datetime | None:
    m = re.search(regex, path.name, flags=re.IGNORECASE)
    if not m:
        return None
    digits = m.group(1)
    try:
        return datetime.strptime(digits, dt_fmt)
    except ValueError:
        return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def humanize_seconds(total: float) -> str:
    total = int(round(total))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def set_clip_duration(clip, seconds: float):
    return clip.with_duration(seconds) if hasattr(clip, "with_duration") else clip.set_duration(seconds)


def _resize_pil_to_width(img: Image.Image, width: int | None) -> Image.Image:
    if not width:
        return img
    w, h = img.size
    if w == width:
        return img
    new_h = int(round(h * (width / w)))
    return img.resize((width, new_h), Image.Resampling.LANCZOS)


def _load_font(size: int, font_path: str | None):
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size=size)

    candidates: list[str] = []
    if os.name == "nt":
        win_dir = os.environ.get("WINDIR", r"C:\Windows")
        fonts_dir = os.path.join(win_dir, "Fonts")
        candidates += [
            os.path.join(fonts_dir, "segoeui.ttf"),
            os.path.join(fonts_dir, "arial.ttf"),
            os.path.join(fonts_dir, "calibri.ttf"),
            os.path.join(fonts_dir, "tahoma.ttf"),
            os.path.join(fonts_dir, "verdana.ttf"),
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]

    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)

    print("⚠️  WARNING: No TrueType font found; using PIL default (may be small). "
          "Set CFG.timestamp_font_path / CFG.bar_font_path to a .ttf for consistent sizing.")
    return ImageFont.load_default()


def _anchor_xy(W: int, H: int, tw: int, th: int, margin: int, position: str):
    pos = position.lower()
    if pos == "tl":
        return margin, margin
    if pos == "tr":
        return W - margin - tw, margin
    if pos == "bl":
        return margin, H - margin - th
    return W - margin - tw, H - margin - th  # br


def draw_timestamp(im_rgba: Image.Image, text: str, *,
                   position: str,
                   font_px: int,
                   font_path: str | None,
                   margin_px: int,
                   boxpad_px: int,
                   box_alpha: int,
                   text_rgba,
                   stroke_width_px: int,
                   stroke_fill_rgba):
    draw = ImageDraw.Draw(im_rgba)
    font = _load_font(font_px, font_path)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    W, H = im_rgba.size
    x, y = _anchor_xy(W, H, tw, th, margin_px, position)

    if box_alpha > 0:
        draw.rectangle([x - boxpad_px, y - boxpad_px, x + tw + boxpad_px, y + th + boxpad_px],
                       fill=(0, 0, 0, box_alpha))

    draw.text((x, y), text, font=font, fill=text_rgba,
              stroke_width=stroke_width_px, stroke_fill=stroke_fill_rgba)


def draw_scale_bar(im_rgba: Image.Image, *,
                   units_per_pixel_effective: float,
                   bar_length_units: float,
                   units_label: str,
                   position: str,
                   margin_px: int,
                   bar_height_px: int,
                   font_px: int,
                   font_path: str | None,
                   boxpad_px: int,
                   box_alpha: int,
                   text_rgba,
                   stroke_width_px: int,
                   stroke_fill_rgba):
    if units_per_pixel_effective <= 0:
        return

    W, H = im_rgba.size
    draw = ImageDraw.Draw(im_rgba)
    font = _load_font(font_px, font_path)

    bar_len_px = int(round(bar_length_units / units_per_pixel_effective))
    bar_len_px = max(1, min(bar_len_px, W - 2 * margin_px))

    label = f"{bar_length_units:g} {units_label}"
    lb = draw.textbbox((0, 0), label, font=font)
    tw, th = lb[2] - lb[0], lb[3] - lb[1]

    gap = max(4, int(round(0.01 * H)))
    content_w = max(tw, bar_len_px)
    content_h = th + gap + bar_height_px

    pos = position.lower()
    if pos == "tl":
        x0, y0 = margin_px, margin_px
    elif pos == "tr":
        x0, y0 = W - margin_px - content_w, margin_px
    elif pos == "br":
        x0, y0 = W - margin_px - content_w, H - margin_px - content_h
    else:  # bl
        x0, y0 = margin_px, H - margin_px - content_h

    if box_alpha > 0:
        draw.rectangle([x0 - boxpad_px, y0 - boxpad_px,
                        x0 + content_w + boxpad_px, y0 + content_h + boxpad_px],
                       fill=(0, 0, 0, box_alpha))

    draw.text((x0, y0), label, font=font, fill=text_rgba,
              stroke_width=stroke_width_px, stroke_fill=stroke_fill_rgba)

    bar_y = y0 + th + gap
    draw.rectangle([x0, bar_y, x0 + bar_len_px, bar_y + bar_height_px],
                   fill=text_rgba)


def make_frame_array(image_path: Path, ts: datetime, first_ts: datetime, cfg: Config) -> np.ndarray:
    img = Image.open(image_path)
    orig_w, _ = img.size

    img = _resize_pil_to_width(img, cfg.resize_width)
    W, H = img.size

    def frac_px(frac: float, min_px: int = 1) -> int:
        return max(min_px, int(round(frac * H)))

    ts_font_px = frac_px(cfg.timestamp_font_h_frac, 8)
    ts_margin_px = frac_px(cfg.timestamp_margin_h_frac, 2)
    ts_boxpad_px = frac_px(cfg.timestamp_boxpad_h_frac, 2)

    bar_height_px = frac_px(cfg.bar_height_h_frac, 2)
    bar_font_px = frac_px(cfg.bar_font_h_frac, 8)
    bar_margin_px = frac_px(cfg.bar_margin_h_frac, 2)
    bar_boxpad_px = frac_px(cfg.bar_boxpad_h_frac, 2)

    units_per_pixel_effective = cfg.units_per_pixel
    if cfg.resize_width and orig_w != W and cfg.units_per_pixel > 0:
        units_per_pixel_effective = cfg.units_per_pixel * (orig_w / W)

    im_rgba = img.convert("RGBA")

    if cfg.add_timestamp:
        elapsed_hours = (ts - first_ts).total_seconds() / 3600.0
        draw_timestamp(
            im_rgba,
            text=f"{elapsed_hours:.{cfg.hours_decimals}f} h",
            position=cfg.timestamp_position,
            font_px=ts_font_px,
            font_path=cfg.timestamp_font_path,
            margin_px=ts_margin_px,
            boxpad_px=ts_boxpad_px,
            box_alpha=cfg.box_alpha,
            text_rgba=cfg.text_rgba,
            stroke_width_px=cfg.stroke_width_px,
            stroke_fill_rgba=cfg.stroke_fill_rgba,
        )

    if cfg.add_scale_bar:
        draw_scale_bar(
            im_rgba,
            units_per_pixel_effective=units_per_pixel_effective,
            bar_length_units=cfg.bar_length_units,
            units_label=cfg.units_label,
            position=cfg.bar_position,
            margin_px=bar_margin_px,
            bar_height_px=bar_height_px,
            font_px=bar_font_px,
            font_path=cfg.bar_font_path,
            boxpad_px=bar_boxpad_px,
            box_alpha=cfg.box_alpha,
            text_rgba=cfg.text_rgba,
            stroke_width_px=cfg.stroke_width_px,
            stroke_fill_rgba=cfg.stroke_fill_rgba,
        )

    return np.array(im_rgba.convert("RGB"), dtype=np.uint8)


def main():
    print("=== Image → Time-proportional Video Builder ===")
    folder = Path(ask("Folder containing images", ".", str))
    output = ask("Output video filename", "out.mp4", str)
    scale_factor = ask("Scale factor (10 = 10× faster, 1 = real-time)", 10.0, float)

    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    files = find_images(folder, CFG.exts)
    if not files:
        print("❌ No images found.")
        sys.exit(1)

    items: list[tuple[Path, datetime]] = []
    for f in files:
        ts = extract_timestamp(f, CFG.timestamp_regex, CFG.timestamp_dt_format)
        if ts is not None:
            items.append((f, ts))

    if not items:
        print("❌ No valid timestamps found in filenames.")
        sys.exit(1)

    items.sort(key=lambda x: (x[1], x[0].name))

    first_ts = items[0][1]
    last_ts = items[-1][1]
    real_total_seconds = max((last_ts - first_ts).total_seconds(), 0.0)
    print(f"🕒 Real elapsed time across images: {real_total_seconds:.3f} s ({humanize_seconds(real_total_seconds)})")

    # Per-frame real-time deltas
    times = [t for _, t in items]
    real_deltas: list[float | None] = [None]
    for i in range(1, len(times)):
        dt = max((times[i] - times[i - 1]).total_seconds(), 0.0)
        real_deltas.append(dt)

    positive = [d for d in real_deltas[1:] if d and d > 0]
    baseline = median(positive) if positive else 1.0
    real_deltas[0] = baseline
    if len(real_deltas) > 1:
        real_deltas[-1] = real_deltas[-2] if real_deltas[-2] is not None else baseline

    # Convert real deltas to video durations
    video_durations = [clamp(float(d) / scale_factor, 0.0, 1e12) for d in real_deltas]  # type: ignore[arg-type]

    clips = []
    for (path, ts), dur in zip(items, video_durations):
        frame = make_frame_array(path, ts, first_ts, CFG)
        clip = ImageClip(frame)
        clip = set_clip_duration(clip, dur)
        clips.append(clip)

    approx_video_len = sum(video_durations)
    print(f"📸 {len(clips)} frames")
    print(f"🎞️  Approx. output video length: {approx_video_len:.3f} s ({humanize_seconds(approx_video_len)}) at {CFG.fps} FPS")

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        output,
        fps=CFG.fps,
        codec="libx264",
        audio=False,
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-profile:v", "baseline",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        ],
    )
    print("✅ Done!")


if __name__ == "__main__":
    main()
