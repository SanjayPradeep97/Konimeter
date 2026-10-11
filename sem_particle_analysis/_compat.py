"""
scikit-image compatibility shims.

scikit-image 0.26 deprecated three functions this pipeline leans on, and one of
them changes meaning rather than merely changing name. Measurements must not
shift because of a routine dependency upgrade, so the intended semantics are
pinned here in one place instead of being spread across call sites.

Deliberately free of heavy imports (no matplotlib, no cv2) so the measurement
path can import it cheaply.
"""

from skimage import morphology


def binary_opening(mask, footprint):
    """
    Morphological opening of a boolean mask.

    ``morphology.binary_opening`` is deprecated since 0.26 and removed in 0.28;
    ``morphology.opening`` returns an identical boolean result for boolean input.
    """
    try:
        return morphology.opening(mask, footprint).astype(bool)
    except AttributeError:  # pragma: no cover - very old scikit-image
        return morphology.binary_opening(mask, footprint)


def binary_closing(mask, footprint):
    """
    Morphological closing of a boolean mask. See :func:`binary_opening`.
    """
    try:
        return morphology.closing(mask, footprint).astype(bool)
    except AttributeError:  # pragma: no cover - very old scikit-image
        return morphology.binary_closing(mask, footprint)


def remove_objects_smaller_than(mask, min_size):
    """
    Drop connected components whose area is strictly below ``min_size``.

    A component of area exactly ``min_size`` is **kept**. That is what the rest
    of the pipeline promises — ``analyze_mask`` filters with ``area >= min_size``
    and ``sem-analyze --min-size N`` is documented as ignoring particles *below*
    N — but it is not what the library does any more.

    ``remove_small_objects(min_size=N)`` was documented as removing components
    smaller than N. In scikit-image 0.26 the parameter is deprecated in favour of
    ``max_size``, which removes components smaller than *or equal to* its value,
    and the deprecated path now follows the new inclusive rule. A component of
    area exactly N is therefore destroyed by the morphology step even though the
    size filter immediately after it would have kept it.

    Passing ``max_size=min_size - 1`` restores the documented boundary and keeps
    it stable across versions.

    Args:
        mask (np.ndarray): Boolean mask.
        min_size (int): Smallest component area to keep, inclusive.

    Returns:
        np.ndarray: Boolean mask with the small components removed.
    """
    mask = mask.astype(bool)
    if min_size <= 1:
        return mask

    try:
        return morphology.remove_small_objects(mask, max_size=min_size - 1)
    except TypeError:  # scikit-image < 0.26 has no max_size
        return morphology.remove_small_objects(mask, min_size=min_size)


def binary_dilation(mask, footprint):
    """
    Morphological dilation of a boolean mask. See :func:`binary_opening`.

    Note that scikit-image's replacement does not mirror non-symmetric
    footprints; every footprint used here is symmetric, so the two agree.
    """
    try:
        return morphology.dilation(mask, footprint).astype(bool)
    except AttributeError:  # pragma: no cover - very old scikit-image
        return morphology.binary_dilation(mask, footprint)


def fill_holes(mask):
    """Fill enclosed background regions in a boolean mask."""
    from scipy import ndimage as ndi

    return ndi.binary_fill_holes(mask.astype(bool))
