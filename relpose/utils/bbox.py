import numpy as np


def mask_to_bbox(mask, thresh=0.4):
    """
    xyxy format
    """
    mask = mask > thresh
    if not np.any(mask):
        return []
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1]


def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.

    Args:
        bbox: Bounding box in xyxy format (4,).

    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )
    return square_bbox
