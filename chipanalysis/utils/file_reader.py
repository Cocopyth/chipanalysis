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

def stretch_contrast(img, p_low=1, p_high=99):
    """
    Contrast stretch using percentiles.
    Does NOT modify original image values.
    """
    lo, hi = np.percentile(img, (p_low, p_high))
    img2 = np.clip(img, lo, hi)
    img2 = (img2 - lo) / (hi - lo + 1e-8)
    return img2
