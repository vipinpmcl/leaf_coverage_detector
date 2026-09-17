import cv2
import numpy as np


def _masked_values(values, mask):
    if mask is None:
        return values.reshape(-1)
    return values[mask > 0]


def laplacian_variance(gray, mask=None):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    values = _masked_values(lap, mask)

    if values.size == 0:
        return float("nan")

    return float(np.var(values))


def tenengrad(gray, mask=None):
    gray = gray.astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    values = _masked_values(gx * gx + gy * gy, mask)

    if values.size == 0:
        return float("nan")

    return float(np.mean(values))


def mean_gradient_magnitude(gray, mask=None):
    gray = gray.astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx * gx + gy * gy)
    values = _masked_values(magnitude, mask)

    if values.size == 0:
        return float("nan")

    return float(np.mean(values))
