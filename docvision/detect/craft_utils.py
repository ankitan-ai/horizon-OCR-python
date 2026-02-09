"""
CRAFT post-processing utilities.

Based on the official CRAFT-pytorch implementation by NAVER Corp.
License: MIT (https://github.com/clovaai/CRAFT-pytorch)

Functions for extracting text detection boxes from CRAFT score maps.
"""

import math
import numpy as np
import cv2


def get_det_boxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    """
    Core detection box extraction from text and link score maps.

    Args:
        textmap: Character region score map (H, W)
        linkmap: Affinity (link) score map (H, W)
        text_threshold: Text confidence threshold
        link_threshold: Link confidence threshold
        low_text: Low text threshold for initial binarization

    Returns:
        det: List of detected box coordinates
        labels: Connected component labels
        mapper: Label to detection mapping
    """
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    # Binarize and combine
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []
    mapper = []

    for k in range(1, n_labels):
        # Size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # Thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # Make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # Remove link area

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)

        sx, ex = max(0, x - niter), min(img_w, x + w + niter + 1)
        sy, ey = max(0, y - niter), min(img_h, y + h + niter + 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # Make box
        np_contours = (
            np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
            .transpose()
            .reshape(-1, 2)
        )
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # Align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l_val, r_val = min(np_contours[:, 0]), max(np_contours[:, 0])
            t_val, b_val = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array(
                [[l_val, t_val], [r_val, t_val], [r_val, b_val], [l_val, b_val]],
                dtype=np.float32,
            )

        # Clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def get_det_boxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    """
    Get detection boxes from score maps.

    Args:
        textmap: Character region score map
        linkmap: Affinity score map
        text_threshold: Text confidence threshold
        link_threshold: Link confidence threshold
        low_text: Low text threshold
        poly: Whether to return polygon results

    Returns:
        boxes: List of detected bounding boxes
        polys: List of polygons (same as boxes if poly=False)
    """
    boxes, labels, mapper = get_det_boxes_core(
        textmap, linkmap, text_threshold, link_threshold, low_text
    )

    if poly:
        polys = _get_poly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys


def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    """
    Adjust detection coordinates back to original image scale.

    Args:
        polys: List of polygon/box coordinates
        ratio_w: Width ratio
        ratio_h: Height ratio
        ratio_net: Network output ratio (default 2 for CRAFT)

    Returns:
        Adjusted polygon coordinates
    """
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """
    Normalize image with ImageNet mean and variance.

    Args:
        in_img: Input RGB image (H, W, 3), uint8 or float
        mean: Channel means
        variance: Channel standard deviations

    Returns:
        Normalized float32 image
    """
    img = in_img.copy().astype(np.float32)
    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    """
    Resize image maintaining aspect ratio, padded to multiple of 32.

    Args:
        img: Input image (H, W, C)
        square_size: Maximum canvas size
        interpolation: OpenCV interpolation method
        mag_ratio: Image magnification ratio

    Returns:
        resized: Padded resized image
        ratio: Resize ratio
        size_heatmap: Size of output heatmap (w, h)
    """
    height, width, channel = img.shape

    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)

    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # Pad to multiple of 32
    target_h32 = target_h + (32 - target_h % 32) if target_h % 32 != 0 else target_h
    target_w32 = target_w + (32 - target_w % 32) if target_w % 32 != 0 else target_w

    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def _get_poly_core(boxes, labels, mapper, linkmap):
    """
    Get polygon boundaries for detected text regions.

    Simplified version that returns oriented rectangles.
    """
    polys = []
    for box in boxes:
        polys.append(box)  # Use oriented bounding box as polygon
    return polys
