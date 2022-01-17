import numpy as np


def angle_between_2d(a, b):
    angle = np.arctan2(a[..., 1], a[..., 0]) - np.arctan2(b[..., 1], b[..., 0])
    return angle