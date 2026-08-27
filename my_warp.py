"""Experimental warp implementation derived from baseline_warp.

OPTIMIZATION 1: build a dense Numba-compiled global displacement field.
OPTIMIZATION 2: add automatic zero-displacement boundary anchors.
OPTIMIZATION 3: use fixed-point inversion to construct the backward map.
OPTIMIZATION 4: derive the target mask from the inverse-warped source mask.
OPTIMIZATION 5: use floating-point maps and bilinear image sampling.
OPTIMIZATION 6: cache the checker pattern and source-hole preview image.

The matching inline comments identify the exact implementation locations so
each change can be evaluated independently against baseline_warp.
"""

import cv2
import numpy as np
from numba import njit, prange


def idw_displacement(points, source_points, target_points, neighbors=4):
    """Interpolate point-pair directions with four-neighbor inverse-distance weights."""
    points = np.asarray(points, np.float32)
    source_points = np.asarray(source_points, np.float32)
    target_points = np.asarray(target_points, np.float32)
    directions = target_points - source_points

    if len(source_points) == 1:
        return np.broadcast_to(directions, points.shape).copy()

    result = np.empty_like(points)
    for start in range(0, len(points), 65536):
        query = points[start:start + 65536]
        difference = query[:, None] - source_points[None]
        distance_squared = np.sum(difference * difference, axis=2)
        count = min(neighbors, len(source_points))
        indices = np.argpartition(distance_squared, count - 1, axis=1)[:, :count]
        distances = np.take_along_axis(distance_squared, indices, axis=1)
        weights = 1.0 / (np.sqrt(distances) + 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        result[start:start + len(query)] = np.sum(
            directions[indices] * weights[..., None], axis=1
        )

    return result


@njit(cache=True, parallel=True)
def _build_displacement_field(height, width, source_points, directions):
    # OPTIMIZATION 1: build one dense global forward field with Numba instead
    # of processing mask contours and region pixels separately.
    field = np.empty((height, width, 2), np.float32)
    count = min(4, len(source_points))
    for y in prange(height):
        for x in range(width):
            distances = np.empty(4, np.float32)
            indices = np.empty(4, np.int32)
            for slot in range(4):
                distances[slot] = np.inf
                indices[slot] = -1

            exact = -1
            for index in range(len(source_points)):
                dx = x - source_points[index, 0]
                dy = y - source_points[index, 1]
                distance = dx * dx + dy * dy
                if distance < 1e-12:
                    exact = index
                    break
                if distance >= distances[count - 1]:
                    continue
                slot = count - 1
                while slot > 0 and distance < distances[slot - 1]:
                    distances[slot] = distances[slot - 1]
                    indices[slot] = indices[slot - 1]
                    slot -= 1
                distances[slot] = distance
                indices[slot] = index

            if exact >= 0:
                field[y, x] = directions[exact]
                continue

            weight_sum = 0.0
            direction_x = 0.0
            direction_y = 0.0
            for slot in range(count):
                weight = 1.0 / (np.sqrt(distances[slot]) + 1e-6)
                weight_sum += weight
                direction_x += weight * directions[indices[slot], 0]
                direction_y += weight * directions[indices[slot], 1]
            field[y, x, 0] = direction_x / weight_sum
            field[y, x, 1] = direction_y / weight_sum
    return field


def build_displacement_field(shape, source_points, target_points):
    """Return an H x W x 2 forward displacement field in (dx, dy) order."""
    height, width = shape
    source_points = np.asarray(source_points, np.float32)
    target_points = np.asarray(target_points, np.float32)
    return _build_displacement_field(
        height, width, source_points, target_points - source_points
    )


def sample_boundary_anchors(mask, count=12):
    """Sample zero-displacement anchors from the mask boundary."""
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    y, x = np.where((mask > 0) & (eroded == 0))
    boundary = np.column_stack((x, y)).astype(np.float32)
    if len(boundary) <= count:
        return boundary
    indices = np.linspace(0, len(boundary) - 1, count).astype(np.int32)
    return boundary[indices]


def add_boundary_anchors(mask, source_points, target_points, count=12):
    """Keep distant mask boundaries fixed so one user pair remains local."""
    # OPTIMIZATION 2: add invisible zero-displacement constraints. The
    # baseline has no automatic anchors, so a single point pair is translation.
    anchors = sample_boundary_anchors(mask, count)
    if len(anchors) == 0:
        return source_points, target_points

    y, x = np.where(mask > 0)
    radius = 0.16 * np.hypot(x.max() - x.min(), y.max() - y.min())
    distance_squared = np.sum(
        (anchors[:, None] - source_points[None]) ** 2, axis=2
    )
    anchors = anchors[np.all(distance_squared > radius * radius, axis=1)]
    return (
        np.concatenate((source_points, anchors)),
        np.concatenate((target_points, anchors)),
    )


def backward_map(displacement, iterations=3):
    """Solve target = source + displacement(source) for every target pixel."""
    # OPTIMIZATION 3: invert the dense forward field by fixed-point iteration.
    # Baseline bi_warp instead interpolates reverse directions from up to 100
    # forward-warped source-region pixels.
    height, width = displacement.shape[:2]
    target_y, target_x = np.indices((height, width), np.float32)
    source_x = target_x.copy()
    source_y = target_y.copy()
    for _ in range(iterations):
        dx = cv2.remap(displacement[..., 0], source_x, source_y, cv2.INTER_LINEAR)
        dy = cv2.remap(displacement[..., 1], source_x, source_y, cv2.INTER_LINEAR)
        source_x = target_x - dx
        source_y = target_y - dy
    return source_x, source_y


def build_inpaint_mask(source_mask, target_mask, kernel_size=5):
    """Combine the revealed source region with the target boundary."""
    # OPTIMIZATION 4: target_mask comes from the inverse-warped source mask.
    # Baseline target_mask is a polygon filled from the warped source contour.
    revealed = source_mask & ~target_mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    revealed = cv2.dilate(revealed.astype(np.uint8), kernel)
    boundary = np.zeros_like(source_mask, np.uint8)
    contours = cv2.findContours(
        target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    cv2.drawContours(boundary, contours, -1, 1, kernel_size)
    return np.maximum(revealed, boundary)


def checkerboard(image, size=10):
    y, x = np.indices(image.shape[:2])
    values = np.where(((x // size + y // size) & 1) == 0, 235, 202).astype(np.uint8)
    return np.repeat(values[..., None], 3, axis=2)


def warp_preview(
    image, mask, point_pairs, keep_boundary=True, kernel_size=5,
    pattern=None, base=None
):
    """Return the warped RGB preview, inpainting mask and dense displacement field."""
    if not point_pairs:
        empty = np.zeros(mask.shape, np.uint8)
        field = np.zeros((*mask.shape, 2), np.float32)
        return image.copy(), empty, field, mask.copy()

    source_points = np.array([pair[0] for pair in point_pairs], np.float32)
    target_points = np.array([pair[1] for pair in point_pairs], np.float32)
    if keep_boundary:
        source_points, target_points = add_boundary_anchors(
            mask, source_points, target_points
        )

    displacement = build_displacement_field(mask.shape, source_points, target_points)
    source_x, source_y = backward_map(displacement)
    # OPTIMIZATION 5: float backward maps and bilinear sampling replace the
    # baseline's rounded integer source/target pixel assignment.
    warped = cv2.remap(image, source_x, source_y, cv2.INTER_LINEAR)
    alpha = cv2.remap(mask.astype(np.float32), source_x, source_y, cv2.INTER_LINEAR)

    pattern = checkerboard(image) if pattern is None else pattern
    base = np.where(mask[..., None] > 0, pattern, image) if base is None else base
    target_mask = alpha >= 0.5
    preview = base.copy()
    preview[target_mask] = warped[target_mask]
    inpaint_mask = build_inpaint_mask(mask > 0, target_mask, kernel_size)
    preview[inpaint_mask > 0] = pattern[inpaint_mask > 0]
    return preview, inpaint_mask, displacement, target_mask.astype(np.uint8)


class MyWarpSession:
    def __init__(self):
        self.image = None
        self.mask = None
        self.inpaint_mask = None
        self.target_mask = None
        self.pattern = None
        self.base = None
        build_displacement_field(
            (1, 1), np.array([[0, 0]], np.float32), np.array([[0, 0]], np.float32)
        )

    def set_image(self, image):
        self.image = image
        self.mask = np.zeros(image.shape[:2], np.uint8)
        # OPTIMIZATION 6: cache preview-only arrays outside the drag loop.
        self.pattern = checkerboard(image)
        self.base = image.copy()

    def set_mask(self, mask):
        self.mask = (mask > 0).astype(np.uint8)
        self.base = np.where(self.mask[..., None] > 0, self.pattern, self.image)

    def preview(self, point_pairs, keep_boundary=True):
        preview, self.inpaint_mask, displacement, self.target_mask = warp_preview(
            self.image, self.mask, point_pairs, keep_boundary,
            pattern=self.pattern, base=self.base
        )
        return preview, self.inpaint_mask, displacement, self.target_mask
