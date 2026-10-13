"""
Which part of a frame actually shows sample.

Micrographs routinely contain large areas that are not specimen, and every one
of them is high-contrast enough to dominate a whole-frame mask:

* the black corners left by a circular aperture on an SEM frame,
* the specimen grid bar blocking the beam across a TEM frame,
* a scale bar and its label burned into the image itself.

Measured on the NIOSH set, these produced the pipeline's worst results — five
"particles" of 119 µm on an SEM frame that were the corner wedges, and 221
spurious particles on a TEM frame that were the grid bar's texture. Excluding
them costs nothing when they are absent and is the difference between a usable
count and a meaningless one when they are present.

The first two turn out to be one thing: a large, near-black, featureless region
touching the frame edge. A genuine object running off the edge is textured, so a
single rule serves both.

What is excluded must never reach a particle. The background of an SEM filter is
often as dark as a grid bar, and the first version of this rule took that
background, filled in every particle sitting on it, and grew the result into the
particles' edges: on 300 test images it removed real particles from analysis on
dozens of SEM frames. So the region is grown only into dark pixels, and only
pinholes in it are filled.
"""

import numpy as np
from skimage import measure, morphology

from scipy import ndimage as ndi

from ._compat import binary_closing

# A blocked region reads as near-black: not merely darker than its surroundings
# but close to the floor of the frame's own range. Held as a fraction of that
# range so it survives differences in exposure and bit depth.
BLOCKED_LEVEL_FRACTION = 0.12

# ...and is large. Below this it is far more likely to be a particle, an
# agglomerate, or a fibre crossing the edge.
BLOCKED_MIN_FRACTION = 0.004

# Blocked regions are contiguous with the frame edge — an aperture vignette
# reaches the corners, a grid bar crosses the field. A dark object floating in
# the middle of the field is sample.
EDGE_TOUCH_PX = 2

# Grown by this much before exclusion, to take in the soft shoulder where a
# vignette or grid bar fades into the image. Grown only into pixels at most
# SHOULDER_LEVEL_FRACTION up the frame's range, so never into a particle.
BLOCKED_MARGIN_PX = 6
SHOULDER_LEVEL_FRACTION = 0.3

# A beam-blocked area carries nothing but detector noise. Grid bars and aperture
# corners on the 286 test images measured 1-10 grey levels of standard deviation
# inside; dark specimen regions mostly 10-55.
BLOCKED_MAX_STD = 12.0

# Holes in a blocked region up to this size are noise and are filled. Anything
# larger is something visible in it, so it is not blocked there.
BLOCKED_PINHOLE_PX = 64


def _grey(image):
    return image[..., 0] if getattr(image, "ndim", 2) == 3 else image


def blocked_area(image, level_fraction=BLOCKED_LEVEL_FRACTION,
                 min_fraction=BLOCKED_MIN_FRACTION, margin=BLOCKED_MARGIN_PX):
    """
    Large, near-black, featureless regions reaching the frame edge: beam-blocked,
    not sample.

    Args:
        image (np.ndarray): Greyscale or RGB frame.
        level_fraction (float): "Near black" as a fraction of the frame's range.
        min_fraction (float): Smallest share of the frame worth excluding.
        margin (int): Pixels to grow the region by, covering its soft edge.

    Returns:
        np.ndarray: Boolean mask, True where the frame is blocked.
    """
    grey = _grey(image).astype(np.float64)
    blocked = np.zeros(grey.shape, dtype=bool)

    low, high = float(grey.min()), float(grey.max())
    if high - low < 1e-9:
        return blocked

    dark = grey <= low + level_fraction * (high - low)
    if not dark.any():
        return blocked

    # Close pinholes so a speckled vignette is one region rather than thousands,
    # but leave anything bigger: a hole in a dark region is something visible.
    shoulder = grey <= low + SHOULDER_LEVEL_FRACTION * (high - low)
    dark = binary_closing(dark, morphology.disk(3)) & shoulder
    holes, count = ndi.label(ndi.binary_fill_holes(dark) & ~dark)
    if count:
        sizes = np.bincount(holes.ravel())
        pinholes = np.flatnonzero(sizes <= BLOCKED_PINHOLE_PX)
        dark |= np.isin(holes, pinholes[pinholes > 0])

    edge = np.zeros(grey.shape, dtype=bool)
    edge[:EDGE_TOUCH_PX, :] = edge[-EDGE_TOUCH_PX:, :] = True
    edge[:, :EDGE_TOUCH_PX] = edge[:, -EDGE_TOUCH_PX:] = True

    minimum_area = min_fraction * grey.size
    labels = measure.label(dark, connectivity=2)
    for region in measure.regionprops(labels):
        if region.area < minimum_area:
            continue
        component = labels == region.label
        if not (component & edge).any():
            continue          # floating in the field: that is sample
        inner = ndi.binary_erosion(component, morphology.disk(4))
        values = grey[inner] if inner.any() else grey[component]
        if values.std() * 255.0 / (high - low) > BLOCKED_MAX_STD:
            continue          # textured: an object or background, not a blocked beam
        blocked |= component

    if blocked.any() and margin > 0:
        # Grow into the soft shoulder only: dark pixels next to the region.
        grown = blocked.copy()
        step = morphology.disk(1)
        for _ in range(margin):
            grown = ndi.binary_dilation(grown, step) & (shoulder | grown)
        blocked = grown
    return blocked


def analysable_region(image, exclude_boxes=(), blocked=None, margin=BLOCKED_MARGIN_PX):
    """
    The part of ``image`` worth measuring.

    Args:
        image (np.ndarray): The frame that will be segmented, already cropped of
            any databar.
        exclude_boxes: Iterable of (x0, y0, width, height) to blank out — a scale
            bar burned into the frame, an instrument watermark.
        blocked (np.ndarray, optional): Precomputed blocked-area mask, to avoid
            recomputing it.
        margin (int): Padding applied around each excluded box.

    Returns:
        tuple: ``(region, info)``. ``region`` is a boolean mask, True where
        measurement is valid. ``info`` records what was removed and why, for the
        provenance file.
    """
    grey = _grey(image)
    region = np.ones(grey.shape, dtype=bool)

    if blocked is None:
        blocked = blocked_area(image)
    if blocked.any():
        region &= ~blocked

    height, width = grey.shape
    boxes_applied = 0
    for box in exclude_boxes or ():
        if box is None or len(box) != 4:
            continue
        x0, y0, box_w, box_h = (int(round(v)) for v in box)
        # The far edge comes from the box itself, before the near edge is
        # clamped to the frame. Computed from the clamped edge, a box near the
        # left or top lost up to a margin's worth of specimen on its far side.
        x1 = min(width, x0 + box_w + margin)
        y1 = min(height, y0 + box_h + margin)
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        if x1 > x0 and y1 > y0:
            region[y0:y1, x0:x1] = False
            boxes_applied += 1

    info = {
        "blocked_fraction": round(float(blocked.mean()), 5),
        "excluded_boxes": boxes_applied,
        "analysable_fraction": round(float(region.mean()), 5),
    }
    return region, info


def usable(region, minimum=0.2):
    """
    Whether enough of the frame survived to be worth measuring.

    A frame that is almost entirely grid bar carries too little sample for a
    count to mean anything, and saying so beats reporting a number from the
    sliver that is left.
    """
    return region is None or float(region.mean()) >= minimum
