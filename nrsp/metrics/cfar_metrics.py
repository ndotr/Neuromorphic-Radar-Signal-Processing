import numpy as np
import cupy as cp
from cupyx.scipy.signal import convolve2d

# https://datascience.stackexchange.com/questions/16179/what-is-the-correct-way-to-compute-mean-f1-score
# http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html
# https://dl.acm.org/doi/10.1145/1882471.1882479


def apply_detection_area(detections, detection_area):
    # cupy can't convolve bool array
    return (
        convolve2d(
            cp.array(detections, dtype=int),
            cp.array(detection_area, dtype=int),
            mode="same",
        )
        .get()
        .astype(bool)
    )


def calc_metrics_extended(detections, targets, detection_area, targets_ext=None):
    detections_ext = apply_detection_area(detections, detection_area)

    if targets_ext is None:
        targets_ext = apply_detection_area(targets, detection_area)

    # compute IoU
    sum_intersection_ext = np.sum(np.logical_and(detections_ext, targets_ext))
    sum_union_ext = np.sum(np.logical_or(detections_ext, targets_ext))

    if sum_intersection_ext == 0 and sum_union_ext == 0:
        iou_score_ext = 1
    else:
        iou_score_ext = sum_intersection_ext / sum_union_ext

    # compute precision and recall for extended masks
    recall_ext, precision_ext = _calc_recall_precision(sum_intersection_ext, np.sum(targets_ext), np.sum(detections_ext))

    return recall_ext, precision_ext, iou_score_ext


def calc_metrics(detections, targets):
    tp = np.sum(np.logical_and(detections, targets))
    t = np.sum(targets)
    d = np.sum(detections)

    recall, precision = _calc_recall_precision(tp, t, d)
    return recall, precision


def _calc_recall_precision(tp, t, d):
    """
    Args:
        tp : total number of true positives
        t : total number of targets
        d : total number of detections
    """

    recall = 0
    precision = 0

    if t == 0:  # no targets
        recall = 1
    else:  # some targets
        recall = tp / t

    if d == 0 and t == 0:  # no detections and no targets
        precision = 1
    elif d == 0 and t != 0:  # no detections but some targets
        precision = 0
    else:  # some detections
        precision = tp / d

    return recall, precision
