"""
Particle Analysis and Measurements

Analyzes segmented particles and calculates size measurements.
"""

import numpy as np
from skimage import measure, morphology
from skimage.segmentation import clear_border

from ._compat import binary_closing, binary_opening, remove_objects_smaller_than


def equivalent_diameter(region):
    """
    Diameter of the circle with the same area as ``region``, in pixels.

    scikit-image 0.26 deprecated ``equivalent_diameter`` in favour of
    ``equivalent_diameter_area`` and will drop it in 2.0. The two return the same
    number, so prefer the new spelling and fall back to the old one — this is the
    measurement every reported size derives from, and it should not start warning
    or break on a routine scikit-image upgrade.
    """
    try:
        return region.equivalent_diameter_area
    except AttributeError:  # scikit-image < 0.26
        return region.equivalent_diameter


class ParticleAnalyzer:
    """
    Analyzes segmented particles and computes measurements.

    Attributes:
        mask: Current binary mask
        labeled_mask: Labeled regions
        regions: List of region properties
        conversion: Conversion factor (nm/pixel)
    """

    def __init__(self, conversion_factor=None, min_size=30):
        """
        Initialize the particle analyzer.

        Args:
            conversion_factor (float, optional): nm/pixel conversion factor
            min_size (int): Minimum particle size in pixels for filtering
        """
        self.conversion = conversion_factor
        self.min_size = min_size
        self.mask = None
        self.labeled_mask = None
        self.regions = []
        # Set by merge_particles: closing can fail to join distant particles.
        self.last_merge_succeeded = True

    def set_conversion_factor(self, conversion_factor):
        """
        Set or update the pixel-to-nanometer conversion factor.

        Args:
            conversion_factor (float): nm/pixel conversion
        """
        self.conversion = conversion_factor

    @staticmethod
    def _clean_mask(mask, min_size):
        """
        Morphological cleanup applied to every mask before measurement.

        Both the initial analysis and each refinement operation run this, so a
        given particle measures the same regardless of how many times the mask
        has been edited. The opening alone shifts the area of a ragged SAM mask
        by a few percent, so applying it on one path but not the other would make
        reported areas depend on whether the user happened to touch refinement.

        Args:
            mask (np.ndarray): Binary mask
            min_size (int): Minimum particle size in pixels

        Returns:
            np.ndarray: Cleaned boolean mask
        """
        cleaned = binary_opening(mask.astype(bool), morphology.disk(1))
        return remove_objects_smaller_than(cleaned, min_size)

    def _relabel_and_filter(self):
        """
        Relabel mask and filter by size consistently to prevent fake particles.

        This method applies morphological cleanup and size filtering to remove
        noise and small artifacts that can appear after particle operations.

        Returns:
            int: Number of particles after filtering
        """
        self.mask = self._clean_mask(self.mask, self.min_size)

        # Relabel
        self.labeled_mask = measure.label(self.mask, connectivity=2)

        # Get all regions and filter by minimum size
        all_regions = measure.regionprops(self.labeled_mask)
        self.regions = [r for r in all_regions if r.area >= self.min_size]

        return len(self.regions)

    def analyze_mask(self, mask, min_size=None, remove_border=True,
                    border_buffer=4):
        """
        Analyze a binary mask to identify and measure particles.

        Args:
            mask (np.ndarray): Binary mask (particles as True/1)
            min_size (int, optional): Minimum particle size in pixels. Uses instance default if None.
            remove_border (bool): Whether to remove border-touching particles
            border_buffer (int): Buffer width for border removal (pixels)

        Returns:
            tuple: (num_particles, regions)
                - num_particles: Number of detected particles
                - regions: List of RegionProperties objects
        """
        # Use instance min_size if not provided
        if min_size is None:
            min_size = self.min_size

        # Same cleanup the refinement operations use, so measurements don't shift
        # the first time a mask is edited.
        clean_mask = self._clean_mask(mask, min_size)

        # Remove border artifacts
        if remove_border:
            clean_mask = self._remove_border_artifacts(clean_mask, border_buffer)
            clean_mask = self._remove_edge_slivers(clean_mask)

        # Label connected components
        self.labeled_mask = measure.label(clean_mask, connectivity=2)

        # Get region properties
        all_regions = measure.regionprops(self.labeled_mask)

        # Filter by minimum size
        self.regions = [r for r in all_regions if r.area >= min_size]
        self.mask = clean_mask

        num_particles = len(self.regions)
        print(f"Detected {num_particles} particles")

        return num_particles, self.regions

    def _remove_border_artifacts(self, mask, border_width=4):
        """
        Remove pixels directly on the image border.

        Args:
            mask (np.ndarray): Boolean mask
            border_width (int): Width of border to clear (pixels). Zero clears
                nothing.

        Returns:
            np.ndarray: Cleaned boolean mask
        """
        cleaned = mask.copy()
        if border_width <= 0:
            # Guarded rather than left to the slicing below, which reads
            # mask[-0:] as the whole array and so erases every pixel instead of
            # none. --border-buffer 0 asked for no border to be cleared and got
            # an empty mask and zero particles, with nothing to say why.
            return cleaned
        H, W = mask.shape

        # Clear border strips
        cleaned[:border_width, :] = False      # Top
        cleaned[-border_width:, :] = False     # Bottom
        cleaned[:, :border_width] = False      # Left
        cleaned[:, -border_width:] = False     # Right

        return cleaned

    # A region lying almost entirely within this many pixels of the frame edge is
    # a mask-boundary artefact, not an object.
    EDGE_BAND_PX = 10
    EDGE_SLIVER_FRACTION = 0.85

    @classmethod
    def _remove_edge_slivers(cls, mask, band=None, fraction=None):
        """
        Drop thin strips that hug the frame edge.

        SAM masks often end in a two-pixel ribbon running along the border, which
        survives size filtering and gets counted as a particle. Simply discarding
        anything that touches the edge would also discard genuine objects running
        off the frame — common for CNTs — so the test is how much of the region
        sits in the edge band: an artefact is almost entirely inside it, whereas a
        real object crossing the border extends well into the image.

        Args:
            mask (np.ndarray): Boolean mask.
            band (int): Width of the edge band in pixels.
            fraction (float): Drop a region when at least this share of it lies
                within the band.

        Returns:
            np.ndarray: Boolean mask with edge slivers removed.
        """
        band = cls.EDGE_BAND_PX if band is None else band
        fraction = cls.EDGE_SLIVER_FRACTION if fraction is None else fraction

        mask = mask.astype(bool)
        if not mask.any() or band <= 0:
            return mask

        in_band = np.zeros(mask.shape, dtype=bool)
        in_band[:band, :] = in_band[-band:, :] = True
        in_band[:, :band] = in_band[:, -band:] = True

        labels = measure.label(mask, connectivity=2)
        cleaned = mask.copy()
        for region in measure.regionprops(labels):
            pixels = labels == region.label
            if in_band[pixels].mean() >= fraction:
                cleaned[pixels] = False
        return cleaned

    def clear_edge_particles(self, buffer_size=0):
        """
        Remove particles touching the image edges.

        Args:
            buffer_size (int): Buffer distance from edge (pixels)

        Returns:
            int: Number of particles removed
        """
        if self.mask is None:
            raise RuntimeError("No mask to process. Run analyze_mask() first.")

        n_before = len(self.regions)

        # Clear border-touching components
        cleaned = clear_border(self.mask.astype(bool), buffer_size=buffer_size)
        self.mask = cleaned.astype(bool)

        # Relabel with filtering to prevent fake particles
        self._relabel_and_filter()

        n_removed = max(0, n_before - len(self.regions))
        print(f"Removed {n_removed} edge-touching particles")

        return n_removed

    def get_measurements(self, in_nm=True):
        """
        Get particle measurements.

        Args:
            in_nm (bool): If True and conversion is set, return measurements in nm.
                         Otherwise return in pixels.

        Returns:
            dict: Dictionary containing:
                - 'num_particles': Number of particles
                - 'areas': List of particle areas
                - 'diameters': List of equivalent diameters
                - 'centroids': List of (x, y) centroid coordinates
                - 'bboxes': List of bounding boxes (min_row, min_col, max_row, max_col)
                - 'unit': Measurement unit ('nm' or 'pixels')
        """
        if not self.regions:
            return {
                'num_particles': 0,
                'areas': [],
                'diameters': [],
                'centroids': [],
                'bboxes': [],
                'unit': 'nm' if in_nm and self.conversion else 'pixels',
                'nm_per_px': self.conversion,
                'areas_px': [],
                'diameters_px': [],
            }

        # Extract measurements in pixels
        areas_px = [r.area for r in self.regions]
        diams_px = [equivalent_diameter(r) for r in self.regions]
        centroids = [(r.centroid[1], r.centroid[0]) for r in self.regions]  # (x, y)
        bboxes = [r.bbox for r in self.regions]

        # Convert to nm if requested and conversion is available
        if in_nm and self.conversion is not None:
            areas = [a * (self.conversion ** 2) for a in areas_px]
            diameters = [d * self.conversion for d in diams_px]
            unit = 'nm'
        else:
            areas = areas_px
            diameters = diams_px
            unit = 'pixels'

        return {
            'num_particles': len(self.regions),
            'areas': areas,
            'diameters': diameters,
            'centroids': centroids,
            'bboxes': bboxes,
            'unit': unit,
            # ResultsManager writes this to the nm_per_px column so a nm value can
            # be traced back to the scale that produced it. The no-particles
            # branch above always returned it; omitting it here left the column
            # empty on precisely the rows that have measurements in them.
            'nm_per_px': self.conversion,
            'areas_px': areas_px,
            'diameters_px': diams_px
        }

    def delete_particles(self, labels_to_delete):
        """
        Delete specific particles by their labels.

        Args:
            labels_to_delete (list): List of region labels to remove

        Returns:
            int: Number of particles removed
        """
        if self.labeled_mask is None:
            raise RuntimeError("No labeled mask available. Run analyze_mask() first.")

        # Create deletion mask
        delete_mask = np.isin(self.labeled_mask, labels_to_delete)
        self.mask = self.mask & (~delete_mask)

        # Relabel with filtering to prevent fake particles
        self._relabel_and_filter()

        return len(labels_to_delete)

    def merge_particles(self, labels_to_merge):
        """
        Merge multiple particles into a single particle.

        Args:
            labels_to_merge (list): List of region labels to merge

        Returns:
            int: New number of particles after merge
        """
        if self.labeled_mask is None:
            raise RuntimeError("No labeled mask available. Run analyze_mask() first.")

        if len(labels_to_merge) < 2:
            print("Warning: Need at least 2 particles to merge")
            return len(self.regions)

        # Create merge mask
        merge_mask = np.isin(self.labeled_mask, labels_to_merge)

        # Apply morphological closing to bridge gaps
        merge_mask_closed = binary_closing(merge_mask.astype(bool), morphology.disk(1))

        # Update main mask
        self.mask = (self.mask & (~merge_mask)) | merge_mask_closed

        # Relabel with filtering to prevent fake particles
        before = len(self.regions)
        self._relabel_and_filter()

        # Closing only bridges gaps of a pixel or two. Particles further apart
        # come back as separate components, so the merge silently did nothing —
        # report that rather than claiming success.
        expected = before - (len(labels_to_merge) - 1)
        self.last_merge_succeeded = len(self.regions) <= expected
        if self.last_merge_succeeded:
            print(f"Merged {len(labels_to_merge)} particles. New count: {len(self.regions)}")
        else:
            print(f"Could not merge: the selected particles are too far apart to "
                  f"join. Count unchanged at {len(self.regions)}.")
        return len(self.regions)

    def add_particle_from_sam(self, sam_mask, largest_only=False):
        """
        Add a new particle from a SAM-generated mask.

        Args:
            sam_mask (np.ndarray): Boolean mask from SAM
            largest_only (bool): Keep only the biggest connected component of the
                mask. Use when the mask is meant to be one particle — refining a
                single particle otherwise multiplies the count, because SAM often
                returns several disconnected blobs and every one becomes a
                particle of its own.

        Returns:
            int: New particle count
        """
        sam_mask = sam_mask.astype(bool)

        if largest_only and sam_mask.any():
            components = measure.label(sam_mask, connectivity=2)
            regions = measure.regionprops(components)
            if len(regions) > 1:
                biggest = max(regions, key=lambda r: r.area)
                sam_mask = components == biggest.label

        if self.mask is None:
            self.mask = sam_mask
        else:
            # Union with existing mask
            self.mask = self.mask | sam_mask

        # Relabel with filtering to prevent fake particles
        self._relabel_and_filter()

        print(f"Added particle. New count: {len(self.regions)}")
        return len(self.regions)

    # How far from a particle a click may land and still count as hitting it.
    # Generous enough to forgive aiming at a thin object on a scaled-down view,
    # tight enough that a click in open background hits nothing.
    CLICK_TOLERANCE_PX = 12

    def find_particle_at_point(self, x, y, tolerance=None):
        """
        Find the particle the user clicked on.

        The pixel under the cursor decides, and only if nothing is there does
        this fall back to the nearest particle within ``tolerance``. A click in
        open background selects nothing at all.

        This used to return the nearest centroid unconditionally, which had two
        consequences on a dense frame. A click in empty space still selected a
        particle, so a misclick silently queued a real one for deletion with no
        way to notice. And a centroid is not necessarily inside its own particle
        — for a C-shaped or elongated region, which is what SAM produces on
        entangled objects, it often is not — so clicking such a particle could
        select a smaller neighbour whose centroid happened to be closer.

        Args:
            x (float): X coordinate, in image pixels.
            y (float): Y coordinate, in image pixels.
            tolerance (float, optional): Search radius in pixels for the
                near-miss fallback. Defaults to ``CLICK_TOLERANCE_PX``; pass 0 to
                require an exact hit.

        Returns:
            tuple: (region, index, label), or (None, None, None) when the click
            is not on or near any particle.
        """
        if not self.regions or self.labeled_mask is None:
            return None, None, None

        tolerance = self.CLICK_TOLERANCE_PX if tolerance is None else tolerance
        height, width = self.labeled_mask.shape
        column, row = int(round(x)), int(round(y))
        if not (0 <= column < width and 0 <= row < height):
            return None, None, None

        by_label = {region.label: (index, region)
                    for index, region in enumerate(self.regions)}

        # Directly on a particle: that is the one, whatever else is nearby.
        hit = int(self.labeled_mask[row, column])
        if hit in by_label:
            index, region = by_label[hit]
            return region, index, region.label

        if tolerance <= 0:
            return None, None, None

        # Near miss: the closest particle whose own pixels come within tolerance,
        # measured to the particle itself rather than to its centroid.
        radius = int(np.ceil(tolerance))
        top, bottom = max(0, row - radius), min(height, row + radius + 1)
        left, right = max(0, column - radius), min(width, column + radius + 1)
        window = self.labeled_mask[top:bottom, left:right]

        best = None
        for label in np.unique(window):
            label = int(label)
            if label not in by_label:
                continue
            ys, xs = np.where(window == label)
            distance = np.min(np.hypot((ys + top) - row, (xs + left) - column))
            if distance <= tolerance and (best is None or distance < best[0]):
                best = (distance, label)

        if best is None:
            return None, None, None
        index, region = by_label[best[1]]
        return region, index, region.label

    def get_summary_statistics(self):
        """
        Calculate summary statistics for all particles.

        Returns:
            dict: Statistics including mean, median, std, min, max for areas and diameters
        """
        measurements = self.get_measurements(in_nm=bool(self.conversion))

        if measurements['num_particles'] == 0:
            return {'num_particles': 0}

        areas = np.array(measurements['areas'])
        diameters = np.array(measurements['diameters'])

        return {
            'num_particles': measurements['num_particles'],
            'unit': measurements['unit'],
            'area_mean': float(np.mean(areas)),
            'area_median': float(np.median(areas)),
            'area_std': float(np.std(areas)),
            'area_min': float(np.min(areas)),
            'area_max': float(np.max(areas)),
            'diameter_mean': float(np.mean(diameters)),
            'diameter_median': float(np.median(diameters)),
            'diameter_std': float(np.std(diameters)),
            'diameter_min': float(np.min(diameters)),
            'diameter_max': float(np.max(diameters))
        }
