"""
Particle Analysis and Measurements

Analyzes segmented particles and calculates size measurements.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import draw, measure, morphology

from ._compat import (binary_closing, binary_dilation, binary_opening,
                      remove_objects_smaller_than)


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
        # How analyze_mask treated the frame edge, so that a particle added or
        # redrawn later is held to the same rule and measures as it would have
        # done had SAM found it in the first place.
        self.remove_border = False
        self.border_buffer = 4
        # Set by merge_particles: whether the last merge was carried out, and how
        # many pixels it added to close the gaps between the pieces.
        self.last_merge_succeeded = True
        self.last_merge_added_px = 0

    @staticmethod
    def _clean_mask(mask, min_size):
        """
        Morphological cleanup: a radius-1 opening, then the size filter.

        Applied to everything that enters the mask — the initial analysis and
        each particle added or redrawn — but never again to particles already
        measured. The opening alone shifts the area of a ragged SAM mask by a few
        percent, so a particle has to have been through it exactly once, however
        it got into the mask.

        Args:
            mask (np.ndarray): Binary mask
            min_size (int): Minimum particle size in pixels

        Returns:
            np.ndarray: Cleaned boolean mask
        """
        cleaned = binary_opening(mask.astype(bool), morphology.disk(1))
        return remove_objects_smaller_than(cleaned, min_size)

    def _prepare(self, mask, min_size, remove_border, border_buffer):
        """
        Everything a mask goes through before its pixels may be measured.

        The cleanup runs again after the border strip is cut. Cutting a ragged
        particle off at the strip leaves shreds the opening would have removed,
        and while they stayed the result was not a fixed point of its own
        cleanup — re-cleaning the mask, which every edit used to do, changed the
        areas of particles nobody had touched. Running it again makes the output
        a fixed point: preparing it a second time changes nothing.

        Returns:
            np.ndarray: Boolean mask in which every connected component is a
            particle.
        """
        cleaned = self._clean_mask(mask, min_size)
        if remove_border:
            cleaned = self._remove_border_artifacts(cleaned, border_buffer)
            cleaned = self._clean_mask(cleaned, min_size)
            cleaned = self._remove_edge_slivers(cleaned)
        return cleaned

    def _relabel(self):
        """
        Relabel the mask and measure every connected component in it.

        Nothing is cleaned or filtered here. Everything that enters the mask has
        already been prepared, so every component is a particle, and cleaning
        the whole mask again after each edit is what used to change the areas
        of particles the edit never touched — and made Undo come back with
        different numbers from the ones it was undoing to.

        Returns:
            int: Number of particles.
        """
        self.labeled_mask = measure.label(self.mask, connectivity=2)
        self.regions = measure.regionprops(self.labeled_mask)
        return len(self.regions)

    def analyze_mask(self, mask, min_size=None, remove_border=True,
                    border_buffer=4):
        """
        Analyze a binary mask to identify and measure particles.

        Args:
            mask (np.ndarray): Binary mask (particles as True/1)
            min_size (int, optional): Minimum particle size in pixels. Uses instance default if None.
            remove_border (bool): Clear a ``border_buffer``-wide strip round the
                frame and drop slivers hugging the edge. A particle crossing the
                edge is kept, clipped at the strip, not removed.
            border_buffer (int): Width of that strip (pixels)

        Returns:
            tuple: (num_particles, regions)
                - num_particles: Number of detected particles
                - regions: List of RegionProperties objects
        """
        # Use instance min_size if not provided
        if min_size is None:
            min_size = self.min_size

        self.remove_border, self.border_buffer = remove_border, border_buffer
        self.mask = self._prepare(mask, min_size, remove_border, border_buffer)
        num_particles = self._relabel()
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

        # Removing whole components leaves every other particle exactly as it was.
        delete_mask = np.isin(self.labeled_mask, labels_to_delete)
        self.mask = self.mask & (~delete_mask)
        self._relabel()

        return len(labels_to_delete)

    def merge_particles(self, labels_to_merge):
        """
        Merge several particles into one.

        Particles a pixel or two apart are joined along their seam by a radius-1
        closing, as they always were. Pieces still apart after that — the
        fragments of one broken particle usually are — are joined by straight
        bridges one or two pixels wide between their closest points, chained so
        that every piece ends up in one connected particle. The bridges are as
        thin as a join allows, but they are part of the merged particle and
        their pixels count towards its area; ``last_merge_added_px`` says how
        many pixels the closing and the bridges added.

        Neither may take in a particle that was not selected. The closing is
        kept off the pixels next to one; a bridge that would run into one makes
        the merge refused instead, the mask left as it was and
        ``last_merge_succeeded`` False.

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

        selected = np.isin(self.labeled_mask, labels_to_merge)
        others = self.mask & ~selected
        closed = (binary_closing(selected, morphology.disk(1))
                  & ~binary_dilation(others, np.ones((3, 3), dtype=bool)))
        bridges = self._bridges(closed)
        merged = self.mask | closed | bridges

        components = measure.label(merged, connectivity=2)
        pieces = np.unique(components[selected])
        self.last_merge_succeeded = (
            len(pieces) == 1 and not (components[others] == pieces[0]).any())
        if not self.last_merge_succeeded:
            self.last_merge_added_px = 0
            print("Could not merge: joining the selected particles would take in "
                  "a particle that was not selected.")
            return len(self.regions)

        self.last_merge_added_px = int((merged & ~self.mask).sum())
        self.mask = merged
        self._relabel()
        print(f"Merged {len(labels_to_merge)} particles, adding "
              f"{self.last_merge_added_px} px. New count: {len(self.regions)}")
        return len(self.regions)

    @staticmethod
    def _bridges(pieces):
        """
        Thin straight bridges joining every component of ``pieces`` into one.

        Grown like a minimum spanning tree: the piece nearest to everything
        joined so far is linked to it by a line between its closest pixel and
        the joined pixel nearest that, until no piece is left. Each line is
        thickened at its diagonal steps so that it is 4-connected, and so reads
        as one piece whatever connectivity it is later read with.

        Args:
            pieces (np.ndarray): Boolean mask of the pieces to join.

        Returns:
            np.ndarray: Boolean mask of the bridge pixels only.
        """
        bridges = np.zeros(pieces.shape, dtype=bool)
        rows, cols = np.nonzero(pieces)
        if not len(rows):
            return bridges
        # Every closest pair lies inside the pieces' bounding box, so the
        # distance transforms need not run over the whole frame.
        box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        components = measure.label(pieces[box], connectivity=2)
        remaining = set(range(2, components.max() + 1))
        joined = components == 1
        lines = np.zeros(joined.shape, dtype=bool)
        while remaining:
            distance, (near_rows, near_cols) = ndi.distance_transform_edt(
                ~joined, return_indices=True)
            candidates = np.isin(components, list(remaining))
            row, col = np.unravel_index(
                np.where(candidates, distance, np.inf).argmin(), distance.shape)
            rr, cc = draw.line(row, col, near_rows[row, col], near_cols[row, col])
            lines[rr, cc] = True
            lines[rr[1:], cc[:-1]] = True
            piece = components[row, col]
            joined |= (components == piece) | lines
            remaining.discard(piece)
        bridges[box] = lines & (components == 0)
        return bridges

    def prepare_addition(self, mask, largest_only=False, region=None):
        """
        The pixels that adding ``mask`` would actually contribute.

        Put through what the initial analysis applied: restricted to the
        analysable region, cleaned, cut at the same border strip. A particle
        added or redrawn by hand used to skip all of it, so a burned-in scale
        bar could be added and measured, and a particle re-added at the frame
        edge measured larger than the same particle had the first time.

        The Redraw preview is checked with this too, so a preview that passes
        is exactly what Apply will add.

        Args:
            mask (np.ndarray): Boolean mask, usually from SAM.
            largest_only (bool): Keep only the biggest connected component. Use
                when the mask is meant to be one particle — refining a single
                particle otherwise multiplies the count, because SAM often
                returns several disconnected blobs and every one becomes a
                particle of its own.
            region (np.ndarray, optional): Analysable region; pixels outside it
                are dropped. Ignored when its shape does not match the mask.

        Returns:
            np.ndarray: Boolean mask, empty when nothing survives.
        """
        mask = np.asarray(mask).astype(bool)
        if region is not None and region.shape == mask.shape:
            mask = mask & region
        mask = self._prepare(mask, self.min_size, self.remove_border,
                             self.border_buffer)
        if largest_only and mask.any():
            components = measure.label(mask, connectivity=2)
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            mask = components == int(sizes.argmax())
        return mask

    def add_particle_from_sam(self, sam_mask, largest_only=False, region=None):
        """
        Add a new particle from a SAM-generated mask.

        Only the mask's own pixels are cleaned (see ``prepare_addition``); the
        particles already measured are left exactly as they were.

        Args:
            sam_mask (np.ndarray): Boolean mask from SAM
            largest_only (bool): See ``prepare_addition``.
            region (np.ndarray, optional): See ``prepare_addition``.

        Returns:
            int: New particle count
        """
        added = self.prepare_addition(sam_mask, largest_only, region)
        self.mask = added if self.mask is None else (self.mask | added)
        self._relabel()

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

