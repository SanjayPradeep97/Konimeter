"""prompting.py -- how the demo turns an image, or a click, into a mask.

Two small, explainable rules sit between SAM and the classifier, tuned for
TEM micrographs, where the particle is darker than the support film.  Both are
evaluated in demo/README.md.  (Moving a click to the darkest nearby pixel was
also tried and measured no benefit, so clicks are used exactly as made.)

auto_point(gray)
    The first guess, made when an image is opened, before any click: the
    deepest interior point of the darkest large object away from the frame
    edge, which is used as if it were a click.  The edge is avoided because
    grid bars and the neighbouring particles cut off by the frame are dark too.

choose_mask(masks, scores, gray, mode)
    SAM returns three candidate masks per prompt with a confidence score.
    The score often prefers a single fibre inside a bundle, or a large patch
    of film.  The default rule keeps the LARGEST candidate that covers less
    than 60 % of the frame and is at least 30 % as dark, relative to its
    surroundings, as the darkest candidate.  MODE_SCORE keeps SAM's own
    highest-scoring candidate instead.
"""
from __future__ import annotations
import numpy as np

MODE_RULE = "largest dark candidate (default)"
MODE_SCORE = "SAM's highest score"


def choose_mask(masks, scores, gray, mode=MODE_RULE, area_cap=0.6, rel_contrast=0.3):
    """Pick one of SAM's candidate masks; returns a boolean array."""
    masks = [np.asarray(m).astype(bool) for m in masks]
    if mode == MODE_SCORE or len(masks) == 1:
        return masks[int(np.argmax(scores))]
    stats = []
    for m in masks:
        area = float(m.mean())
        contrast = (float(gray[~m].mean() - gray[m].mean())
                    if m.any() and (~m).any() else -1e9)
        stats.append((m, area, contrast))
    mx = max(s[2] for s in stats)
    valid = [s for s in stats if s[1] < area_cap and (mx <= 0 or s[2] >= rel_contrast * mx)]
    pool = valid or [s for s in stats if s[1] < area_cap] or stats
    return max(pool, key=lambda s: s[1])[0]


def auto_point(gray, sigma=4.0, min_area=0.001):
    """
    Where to put the first prompt when nobody has clicked yet; returns (x, y).

    The frame is smoothed and split at Otsu's threshold into dark material and
    film.  Each dark object is scored by its area times how much darker than
    the film it is, reduced the more of the frame's edge it runs along, and
    the best one's deepest interior point is returned.  The middle of the
    frame if nothing dark is found.
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import disk

    gray = np.asarray(gray, dtype=float)
    h, w = gray.shape
    smooth = ndi.gaussian_filter(gray, sigma)
    if smooth.max() - smooth.min() < 1e-6:
        return w // 2, h // 2
    dark = ndi.binary_opening(smooth < threshold_otsu(smooth), structure=disk(3))
    if not dark.any() or dark.all():
        return w // 2, h // 2
    film = float(smooth[~dark].mean())
    labels = label(dark)

    best, best_score = None, 0.0
    for region in regionprops(labels):
        if region.area < min_area * h * w:
            continue
        mask = labels == region.label
        on_edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean()
        contrast = film - float(smooth[mask].mean())
        score = region.area * contrast * max(0.0, 1 - 4 * on_edge)
        if score > best_score:
            best, best_score = mask, score
    if best is None:
        return w // 2, h // 2
    depth = ndi.distance_transform_edt(best)
    y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return int(x), int(y)
