"""Shared center-and-scale alignment for spiral images and kiosk drawings."""

import cv2
import numpy as np

OUTPUT_SIZE = 224
TARGET_RADIUS = OUTPUT_SIZE * 0.44


def _foreground_mask(rgb):
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 8:
            cleaned[labels == index] = 255
    return cleaned


def _skeleton(mask):
    remaining = mask.copy()
    skeleton = np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(remaining):
        opened = cv2.morphologyEx(remaining, cv2.MORPH_OPEN, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(remaining, opened))
        remaining = cv2.erode(remaining, kernel)
    return skeleton


def _inner_endpoint(mask):
    ys, xs = np.nonzero(mask)
    fallback = np.array([np.median(xs), np.median(ys)], dtype=np.float32)
    skeleton = (_skeleton(mask) > 0).astype(np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, np.ones((3, 3), np.uint8))
    end_y, end_x = np.nonzero((skeleton == 1) & (neighbors == 2))
    if not len(end_x):
        return fallback
    endpoints = np.column_stack((end_x, end_y)).astype(np.float32)
    return endpoints[np.argmin(np.linalg.norm(endpoints - fallback, axis=1))]


def align_spiral_rgb(image, output_size=OUTPUT_SIZE):
    """Place the endpoint nearest the spiral center at canvas center and normalize size."""
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    if rgb.shape[-1] == 4:
        alpha = rgb[..., 3:4].astype(np.float32) / 255.0
        rgb = rgb[..., :3] * alpha + 255.0 * (1.0 - alpha)
    rgb = np.clip(rgb[..., :3], 0, 255).astype(np.uint8)
    mask = _foreground_mask(rgb)
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return cv2.resize(rgb, (output_size, output_size), interpolation=cv2.INTER_AREA)
    center = _inner_endpoint(mask)
    distances = np.hypot(xs - center[0], ys - center[1])
    radius = max(float(np.percentile(distances, 99.0)), 1.0)
    scale = (output_size * 0.44) / radius
    target = (output_size - 1) / 2.0
    matrix = np.array([[scale, 0.0, target - scale * center[0]],
                       [0.0, scale, target - scale * center[1]]], dtype=np.float32)
    return cv2.warpAffine(rgb, matrix, (output_size, output_size),
                          flags=cv2.INTER_AREA, borderValue=(255, 255, 255))


def aligned_grayscale_3ch(image):
    aligned = align_spiral_rgb(image)
    gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)


def _guide_corridor(rgb, width_ratio=0.05):
    """Return a generous ROI around the central black printed spiral."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    dark = ((hsv[..., 2] < 150) & (hsv[..., 1] < 100)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    height, width = dark.shape
    guide = np.zeros_like(dark)
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if area >= 30:
            guide[labels == index] = 255
    if not cv2.countNonZero(guide):
        return np.full_like(dark, 255)
    radius = max(1, int(round(min(height, width) * width_ratio)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(guide, kernel)


def extract_blue_trace_rgb(image, corridor_ratio=None):
    """Remove the black guide while preserving the full HandPD blue trace."""
    rgb = np.asarray(image, dtype=np.uint8)[..., :3]
    red, green, blue = [rgb[..., index].astype(np.int16) for index in range(3)]
    dominance = blue - np.maximum(red, green)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # JPEG color fringing around the black guide can look weakly blue. Requiring
    # clearer blue dominance removes that fringe without a spatial crop.
    raw_blue = ((dominance >= 20) & (hsv[..., 1] >= 50) & (blue <= 245)).astype(np.uint8) * 255
    if corridor_ratio is None:
        mask = raw_blue.copy()
    elif corridor_ratio < 0:
        raise ValueError("corridor_ratio must be non-negative or None")
    else:
        mask = cv2.bitwise_and(raw_blue, _guide_corridor(rgb, corridor_ratio))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    output = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    output[mask > 0] = (25, 25, 25)
    return output, mask, raw_blue
