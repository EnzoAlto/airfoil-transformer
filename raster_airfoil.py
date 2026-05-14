import numpy as np
from skimage import measure


def raster_to_xy(mask_2d, level=0.5, min_points=80):
    """
    Convert a binary raster airfoil image into an ordered XY contour list.

    Returns XY in normalized coordinates:
    - x in [0, 1]
    - y in [-0.5, 0.5], with positive y upward
    """
    img = (mask_2d > 0).astype(np.float32)

    contours = measure.find_contours(img, level=level)
    if not contours:
        raise ValueError("No contour found in raster image.")

    contour = max(contours, key=len)
    if len(contour) < min_points:
        raise ValueError(f"Contour too short: {len(contour)} points")

    rows, cols = contour[:, 0], contour[:, 1]
    height, width = img.shape

    x = cols / (width - 1)
    y = 0.5 - (rows / (height - 1))

    xy = np.column_stack([x, y]).astype(np.float64)

    trailing_edge_idx = np.argmax(xy[:, 0])
    return np.roll(xy, -trailing_edge_idx, axis=0)


def orient_airfoil_left_to_right(xy, flip_y=False, flip_x=False):
    """
    Orient airfoil coordinates so airflow comes from the left.

    Convention:
    - leading edge = x_min
    - trailing edge = x_max
    - air flows from x_min to x_max
    - positive y is upward
    """
    xy = np.asarray(xy, dtype=float).copy()

    x = xy[:, 0]
    y = xy[:, 1]

    if flip_x:
        x = -x

    if flip_y:
        y = -y

    x_min = np.min(x)
    x_max = np.max(x)
    chord = x_max - x_min

    if chord <= 0:
        raise ValueError("Invalid chord length.")

    x_norm = (x - x_min) / chord
    y_norm = y / chord

    return np.column_stack([x_norm, y_norm])
