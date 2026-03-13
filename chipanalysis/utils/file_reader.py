from xml.etree import ElementTree as ET
from datetime import datetime

import numpy as np
from aicspylibczi import CziFile

def _squeeze_to_2d(arr):
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    # If it’s still not 2D, take the last two dims as Y,X
    return arr.reshape(arr.shape[-2], arr.shape[-1])

def load_czi_2d(path, channel=0, z=0, time=0, try_mosaic=True):
    """
    Returns a 2D image (Y, X) float32.
    For mosaic CZIs, do NOT set S in read_mosaic().
    """
    czi = CziFile(path)

    if try_mosaic:
        try:
            # Key fix: no S here
            mosaic = czi.read_mosaic(C=channel, Z=z, T=time, scale_factor=1.0)
            return _squeeze_to_2d(mosaic).astype(np.float32)
        except Exception as e:
            print("Mosaic read failed, falling back to single-plane read.")
            print("Reason:", repr(e))

    # Fallback: single-plane read. Some files may require fewer/more dims.
    img, _ = czi.read_image(C=channel, Z=z, T=time)
    return _squeeze_to_2d(img).astype(np.float32)

import numpy as np

def get_frame(
    czi,
    time,
    channel,
    gamma=1.0,
    roi=None,
    scale_factor=1.0
):
    """
    Load a frame from a CZI mosaic.
    
    Parameters
    ----------
    roi : dict or None
        {"x0","y0","x1","y1"} in FULL-RES pixel coordinates
    scale_factor : float
        Same scale_factor used in read_mosaic()
    """

    mosaic = czi.read_mosaic(
        C=channel,
        T=time,
        scale_factor=scale_factor
    )

    img = _squeeze_to_2d(mosaic).astype(np.float32)

    # --- Apply ROI if provided ---
    if roi is not None:
        # Convert ROI from full-res to current scale
        x0 = int(round(roi["x0"] * scale_factor))
        x1 = int(round(roi["x1"] * scale_factor))
        y0 = int(round(roi["y0"] * scale_factor))
        y1 = int(round(roi["y1"] * scale_factor))

        # Clamp to image bounds
        h, w = img.shape
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)

        img = img[y0:y1, x0:x1]

    # --- Display-only contrast ---
    img_disp = stretch_contrast(img, 1, 99)

    if gamma != 1.0:
        img_disp = np.clip(img_disp, 0, 1) ** gamma

    return img, img_disp


def stretch_contrast(img, p_low=1, p_high=99,lo=None,hi=None):
    """
    Contrast stretch using percentiles.
    Does NOT modify original image values.
    """
    if lo is None or hi is None:
        lo, hi = np.percentile(img, (p_low, p_high))
    img2 = np.clip(img, lo, hi)
    img2 = (img2 - lo) / (hi - lo + 1e-8)
    return img2

def _strip_ns_inplace(root: ET.Element) -> None:
    """Remove namespaces to make .find/.findall simpler."""
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

def get_pixel_sizes_um(czi) -> dict:
    """
    Returns dict like {"X": um_per_px, "Y": um_per_px, "Z": um_per_step} when available.
    """
    root = czi.meta  # ET.Element
    _strip_ns_inplace(root)

    out = {}

    # Common pattern: Scaling/Items/Distance with Id in {"X","Y","Z"}
    for axis in ("X", "Y", "Z"):
        # Try a few plausible XPath-ish searches
        candidates = [
            f".//Scaling//Distance[@Id='{axis}']//Value",
            f".//Scaling//Items//Distance[@Id='{axis}']//Value",
            f".//Scaling//Distance[@Id='{axis}']",
        ]
        val = None
        for path in candidates:
            node = root.find(path)
            if node is not None:
                text = (node.text or "").strip()
                if not text and node.get("Value"):
                    text = node.get("Value").strip()
                if text:
                    print(path)
                    val = float(text)
                    break

        if val is None:
            continue

        # Heuristic: many CZIs store distances in meters (e.g. 1.083e-07 m/px == 0.1083 µm/px)
        # Convert meters -> µm
        out[axis] = val * 1e6

    return out




def _try_parse_datetime(s: str):
    s = s.strip()
    # common: ISO 8601 (sometimes with Z)
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None

def get_timestamps_by_T(czi, S=0, C=0, Z=0):
    """
    Returns list of (t_index, datetime_or_string_or_None)
    """
    # find T size from binary dims
    dims_shapes = czi.get_dims_shape()
    # pick first shape dict; if scenes inconsistent you may want to choose by S
    shape0 = dims_shapes[0]
    if "T" not in shape0:
        return []

    t_start, t_size = shape0["T"]
    results = []

    for t in range(t_start, t_start + t_size):
        pairs = czi.read_subblock_metadata(T=t, S=S, C=C, Z=Z, unified_xml=False)
        # pairs is [(dims_dict, xml_string), ...]
        if not pairs:
            results.append((t, None))
            continue

        # take the first matching subblock (you can also choose by dims_dict)
        xml_str = pairs[0][1]
        root = ET.fromstring(xml_str)
        _strip_ns_inplace(root)

        # try likely tag names (varies by dataset / ZEN version)
        tag_candidates = [
            ".//AcquisitionTime",
            ".//AcquisitionDateAndTime",
            ".//TimeStamp",
            ".//DateTime",
        ]

        ts = None
        for path in tag_candidates:
            node = root.find(path)
            if node is not None and (node.text or "").strip():
                ts = node.text.strip()
                break

        # parse if possible
        dt = _try_parse_datetime(ts) if ts else None
        results.append((t, dt if dt is not None else ts))

    return results

def get_frame(
    czi,
    time,
    channel,
    gamma=1.0,
    roi=None,
    scale_factor=1.0,
    stretch_min = 1,
    stretch_max = 99,
    lo=None,
    hi=None
):
    """
    Load a frame from a CZI mosaic.
    
    Parameters
    ----------
    roi : dict or None
        {"x0","y0","x1","y1"} in FULL-RES pixel coordinates
    scale_factor : float
        Same scale_factor used in read_mosaic()
    """

    mosaic = czi.read_mosaic(
        C=channel,
        T=time,
        scale_factor=scale_factor
    )

    img = _squeeze_to_2d(mosaic).astype(np.float32)

    # --- Apply ROI if provided ---
    if roi is not None:
        # Convert ROI from full-res to current scale
        x0 = int(round(roi["x0"] * scale_factor))
        x1 = int(round(roi["x1"] * scale_factor))
        y0 = int(round(roi["y0"] * scale_factor))
        y1 = int(round(roi["y1"] * scale_factor))

        # Clamp to image bounds
        h, w = img.shape
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)

        img = img[y0:y1, x0:x1]

    # --- Display-only contrast ---
    img_disp = stretch_contrast(img, stretch_min, stretch_max,lo,hi)

    if gamma != 1.0:
        img_disp = np.clip(img_disp, 0, 1) ** gamma

    return img, img_disp