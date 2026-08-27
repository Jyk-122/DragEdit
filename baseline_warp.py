"""Strict Inpaint4Drag bi_warp baseline.

The baseline calls the reference implementation directly. No control points,
mapping steps, sampling rules or masks are changed here.
"""

import numpy as np

from reference.Inpaint4Drag.utils.drag import bi_warp


def checkerboard(image, size=10):
    background = np.ones(image.shape[:2], np.uint8) * 255
    background[::size] = 0
    background[:, ::size] = 0
    return np.repeat(background[..., None], 3, axis=2)


def warp_preview(image, mask, point_pairs, kernel_size=5):
    """Apply the same preview composition used by Inpaint4Drag's UI."""
    if not point_pairs:
        return image.copy(), np.zeros(mask.shape, np.uint8), mask.copy()

    # Gradio SelectData supplies integer pixel indices. Browser coordinates are
    # floats after CSS scaling, so restore the reference UI's input semantics
    # before calling the unchanged bi_warp algorithm.
    control_points = np.rint(
        [point for pair in point_pairs for point in pair]
    ).astype(np.int32)
    source, target, inpaint_mask = bi_warp(mask, control_points, kernel_size)
    preview = image.copy()
    preview[target[:, 1], target[:, 0]] = image[source[:, 1], source[:, 0]]
    preview = np.where(
        inpaint_mask[..., None] == 1, checkerboard(image), preview
    )
    target_mask = np.zeros(mask.shape, np.uint8)
    target_mask[target[:, 1], target[:, 0]] = 1
    return preview, inpaint_mask, target_mask


class BaselineWarpSession:
    def __init__(self):
        self.image = None
        self.mask = None
        self.inpaint_mask = None
        self.target_mask = None

    def set_image(self, image):
        self.image = image
        self.mask = np.zeros(image.shape[:2], np.uint8)

    def set_mask(self, mask):
        self.mask = (mask > 0).astype(np.uint8)

    def preview(self, point_pairs):
        preview, self.inpaint_mask, self.target_mask = warp_preview(
            self.image, self.mask, point_pairs
        )
        return preview, self.inpaint_mask, self.target_mask
