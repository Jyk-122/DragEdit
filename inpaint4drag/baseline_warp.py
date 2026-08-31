"""Strict Inpaint4Drag bi_warp baseline.

The baseline calls the reference implementation directly. No control points,
mapping steps, sampling rules or masks are changed here.
"""

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference.Inpaint4Drag.utils.drag import bi_warp


def hole_placeholder(image, size=10):
    """Create a Photoshop-style gray-white transparency grid."""
    y, x = np.indices(image.shape[:2])
    value = np.where(((x // size + y // size) & 1) == 0, 240, 200).astype(np.uint8)
    return np.repeat(value[..., None], 3, axis=2)


def _warp_result(image, mask, point_pairs, kernel_size=5):
    """Return the clean warp composite together with the styled preview."""
    if not point_pairs:
        clean = image.copy()
        return clean, clean.copy(), np.zeros(mask.shape, np.uint8), mask.copy()

    # Gradio SelectData supplies integer pixel indices. Browser coordinates are
    # floats after CSS scaling, so restore the reference UI's input semantics
    # before calling the unchanged bi_warp algorithm.
    control_points = np.rint(
        [point for pair in point_pairs for point in pair]
    ).astype(np.int32)
    source, target, inpaint_mask = bi_warp(mask, control_points, kernel_size)
    clean = image.copy()
    clean[target[:, 1], target[:, 0]] = image[source[:, 1], source[:, 0]]
    preview = clean.copy()
    preview = np.where(
        inpaint_mask[..., None] == 1, hole_placeholder(image), preview
    )
    target_mask = np.zeros(mask.shape, np.uint8)
    target_mask[target[:, 1], target[:, 0]] = 1
    return clean, preview, inpaint_mask, target_mask


def warp_preview(image, mask, point_pairs, kernel_size=5):
    """Apply the reference warp with the demo's current preview styling."""
    _, preview, inpaint_mask, target_mask = _warp_result(
        image, mask, point_pairs, kernel_size
    )
    return preview, inpaint_mask, target_mask


class BaselineWarpSession:
    def __init__(self):
        self.image = None
        self.mask = None
        self.inpaint_mask = None
        self.target_mask = None
        self.warped_image = None

    def set_image(self, image):
        self.image = image
        self.mask = np.zeros(image.shape[:2], np.uint8)
        self.inpaint_mask = np.zeros(image.shape[:2], np.uint8)
        self.target_mask = self.mask.copy()
        self.warped_image = image.copy()

    def set_mask(self, mask):
        self.mask = (mask > 0).astype(np.uint8)

    def preview(self, point_pairs, kernel_size=5):
        self.warped_image, preview, self.inpaint_mask, self.target_mask = _warp_result(
            self.image, self.mask, point_pairs, kernel_size
        )
        return preview, self.inpaint_mask, self.target_mask
