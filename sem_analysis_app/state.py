"""
Application state.

One process-wide instance, shared by every callback. This mirrors the
original single-file design: the app is intended for one analyst working
locally, and two browser tabs pointed at the same server share this object.
"""

import os

import numpy as np


class AppState:
    """Enhanced application state with image processing tracking."""

    def __init__(self):
        # Model components
        self.sam_model = None
        self.scale_detector = None
        self.segmenter = None
        self.results_manager = None

        # Image management
        self.image_paths = []
        self.current_index = 0
        self.processed_images = {}  # {index: {"status": "completed", "num_particles": 23}}

        # Current processing state
        self.current_image = None
        self.cropped_image = None
        self.scale_info = None
        # The resolved ScaleCalibration for the image on screen. Declared here
        # rather than being attached on first use, so that reset_image_state can
        # clear it: it used to outlive the image it belonged to, and since the
        # header and the frame panel read it, opening a new image could show the
        # previous one's nm/px.
        self.scale_calibration = None
        self.masks = None
        self.scores = None
        # Ranked candidates from ParticleSegmenter.rank_candidates, each carrying
        # the polarity that produced it so selection can't pick the wrong side.
        self.candidates = []
        # (x0, y0, w, h) of a scale bar printed inside the frame, excluded from
        # segmentation so it isn't measured as a particle.
        self.scale_bar_region = None
        # Which instrument produced the current frame. Decides particle polarity
        # and whether a databar is expected; see sem_particle_analysis.modality.
        self.modality = None
        # Boolean mask of the part of the frame that is specimen — beam-blocked
        # area and burned-in annotations removed. Everything outside it would
        # otherwise be measured as particles.
        self.analysable_region = None
        self.region_info = {}
        # "SEM" | "TEM" | None: the microscope the folder came from, chosen on
        # the Setup tab (or read from the files). Sets which way round the
        # particles are expected to be.
        self.microscope = None
        # "auto" | "bright" | "dark": which way the particles differ from the
        # background, when the analyst has said. Kept across images in a session
        # because a folder is normally one kind of sample on one instrument.
        self.particle_choice = "auto"
        # Whether saving a frame also keeps its image and mask as training data
        # (sem_particle_analysis.dataset). Session-wide, set on the Setup tab.
        self.save_dataset = True
        # Percent of frame height to trim, overriding the measured databar. None
        # means measure it, which is right whenever the instrument recorded a
        # scan height. An escape hatch, not a routine control.
        self.crop_override = None
        # Scale already established, keyed by image path: {path: ScaleCalibration}.
        # Session-wide, so it survives moving between images. Without it, coming
        # back to a frame whose bar was read by hand meant doing that work again,
        # and there was no way to tell a restored calibration from a fresh guess.
        self.scale_by_image = {}
        # nm/px most recently accepted by the analyst in this session. A reading
        # that agrees with it is almost certainly the same magnification read the
        # same way, so it is let through without a stop; anything else is worth a
        # look. Used only to decide whether to interrupt, never to set a scale.
        self.scale_baseline = None
        # Every scale accepted this run, so a folder at several magnifications
        # asks once for each rather than every time it changes.
        self.scale_baselines = []
        # {(value read, bar length in px): value printed}, for OCR misreadings
        # the analyst has corrected. Kept for the whole session.
        self.label_corrections = {}
        self.selected_mask_index = None
        self.analyzer = None
        self.min_particle_size = 30  # Minimum particle size in pixels for filtering
        self.show_particle_numbers = True  # Whether to show particle numbers in visualization
        self.crop_percent = 7.0  # Percentage to crop from bottom (for scale bar removal)

        # Scale detection mode state
        self.scale_mode = "Metadata"       # "Metadata", "OCR", or "Manual"
        self.ocr_click_points = []         # [(x1,y1), (x2,y2)] for OCR box corners
        self.manual_click_points = []      # [(x1,y1), (x2,y2)] for manual endpoints

        # Refinement state
        self.click_mode = "delete"  # "delete", "add", "merge", "point_refine"
        self.pending_deletes = []  # List of particle labels to delete
        # A pixel inside each selected particle, keyed by the label it had when
        # it was clicked. Labels are not stable: every edit relabels the mask
        # from scratch, so a label recorded at click time can name a different
        # particle by the time it is used. A pixel still names the same object.
        self.pending_anchors = {}
        self.pending_add_points = []  # List of (x, y) click points for addition
        self.pending_add_masks = []  # List of SAM-generated masks for preview
        self.pending_merge = []  # List of particle labels to merge
        self.point_refine_particle = None  # Particle label being refined in point_refine mode
        self.point_refine_base_mask = None  # Base mask of selected particle for ROI/IoU
        self.point_refine_points = []  # List of (x, y) points for refinement
        self.point_refine_labels = []  # List of point labels (1=positive, 0=negative)
        self.point_refine_preview_mask = None  # Live preview mask from SAM
        self.point_refine_logits = None  # SAM logits for iterative refinement
        self.point_type = "positive"  # "positive" or "negative" for point_refine mode

        # Undo history for individual clicks, before they are applied.
        self.undo_history = []
        # Undo history for applied edits: a stack of mask snapshots taken just
        # before each Apply. Without it an Apply is a point of no return, and
        # since the analyst applies constantly to see the result, one wrong
        # Apply would mean re-segmenting the image from scratch.
        self.mask_history = []

    def reset_image_state(self):
        """Reset processing state for new image."""
        self.current_image = None
        self.cropped_image = None
        self.scale_info = None
        # Cleared with everything else that belongs to one image. A calibration
        # that survived into the next image would be silently wrong.
        self.scale_calibration = None
        self.ocr_click_points = []
        self.manual_click_points = []
        self.masks = None
        self.scores = None
        # Ranked candidates from ParticleSegmenter.rank_candidates, each carrying
        # the polarity that produced it so selection can't pick the wrong side.
        self.candidates = []
        # (x0, y0, w, h) of a scale bar printed inside the frame, excluded from
        # segmentation so it isn't measured as a particle.
        self.scale_bar_region = None
        # Per-image, unlike particle_choice, which the analyst sets once for a
        # folder.
        self.modality = None
        self.analysable_region = None
        self.region_info = {}
        self.selected_mask_index = None
        # Set by ranking when the best candidate should be put to use next.
        self.use_best_pending = False
        self.analyzer = None
        self.click_mode = "delete"
        self.pending_deletes = []
        self.pending_anchors = {}
        self.pending_add_points = []
        self.pending_add_masks = []
        self.pending_merge = []
        self.point_refine_particle = None
        self.point_refine_base_mask = None
        self.point_refine_points = []
        self.point_refine_labels = []
        self.point_refine_preview_mask = None
        self.point_refine_logits = None
        self.point_type = "positive"
        self.undo_history = []
        self.mask_history = []

    # Applied edits kept for undo. Each snapshot is a full-frame boolean mask, so
    # a deep stack costs real memory on 2048px frames; this is plenty to recover
    # from a mistake without holding the whole session.
    MAX_MASK_HISTORY = 20

    def snapshot_mask(self):
        """
        Record the current mask so an applied edit can be undone.

        Called immediately before an edit is committed. Does nothing when there
        is no analysis yet.
        """
        if self.analyzer is None or self.analyzer.mask is None:
            return
        self.mask_history.append(self.analyzer.mask.copy())
        if len(self.mask_history) > self.MAX_MASK_HISTORY:
            self.mask_history.pop(0)

    def restore_mask(self):
        """
        Roll back to the mask as it was before the last applied edit.

        Relabelled only, not cleaned again: the snapshot is the mask exactly as
        it was measured, and cleaning it a second time used to bring Undo back
        to slightly different areas from the ones it was undoing to.

        Returns:
            bool: True if a snapshot was restored, False if there was none.
        """
        if self.analyzer is None or not self.mask_history:
            return False
        self.analyzer.mask = self.mask_history.pop()
        self.analyzer._relabel()
        return True

    def microscope_kind(self):
        """
        SEM or TEM for the image on screen, or None: the analyst's choice on
        Setup, or else what the file says.
        """
        if self.microscope in ("SEM", "TEM"):
            return self.microscope
        kind = getattr(self.modality, "kind", None)
        return kind if kind in ("SEM", "TEM") else None

    def accept_scale(self, calibration):
        """
        Record a scale the analyst has accepted, as the yardstick for the next.

        A folder is normally shot at one or two magnifications, so the first
        accepted reading vouches for every later one that matches it — and a
        change of magnification stops for a fresh look exactly once. Not cleared
        by reset_image_state: it belongs to the run, not to one image.
        """
        if calibration is not None:
            self.scale_baseline = float(calibration.nm_per_px)
            if self.scale_baseline not in self.scale_baselines:
                self.scale_baselines.append(self.scale_baseline)

    def remember_correction(self, read_nm, pixel_length, corrected_nm):
        """
        A label OCR misread, and what it really says. The same misreading of a bar
        the same length is the same label on the same instrument, so it is
        corrected the same way next time.
        """
        if read_nm and pixel_length:
            key = (round(read_nm, 3), round(pixel_length))
            if abs(corrected_nm - read_nm) <= 1e-9 * read_nm:
                self.label_corrections.pop(key, None)       # set back to what was read
            else:
                self.label_corrections[key] = corrected_nm

    def known_correction(self, read_nm, pixel_length):
        """The corrected value for a reading misread before, or None."""
        if not read_nm or not pixel_length:
            return None
        return self.label_corrections.get((round(read_nm, 3), round(pixel_length)))

    def current_path(self):
        """Path of the image on screen, or None."""
        if not self.image_paths or not 0 <= self.current_index < len(self.image_paths):
            return None
        return str(self.image_paths[self.current_index])

    def remember_scale(self, calibration):
        """Keep this image's calibration so returning to it does not redo the work."""
        path = self.current_path()
        if path and calibration is not None:
            self.scale_by_image[path] = calibration

    def recall_scale(self):
        """The calibration established for the image on screen, or None."""
        path = self.current_path()
        return self.scale_by_image.get(path) if path else None

    def forget_scale(self):
        """Drop the stored calibration, so the next visit detects afresh."""
        path = self.current_path()
        if path:
            self.scale_by_image.pop(path, None)

    # --- selections that have to survive a relabel ---------------------------

    def anchor_selection(self, label, point, labeled_mask=None):
        """
        Remember a pixel inside a particle that has been selected.

        A click may land a few pixels outside the particle it selects (see
        ParticleAnalyzer.CLICK_TOLERANCE_PX), so the clicked pixel is not
        necessarily inside it. Anchored there, the selection resolved to
        background at Apply and silently vanished: a near-miss Remove did
        nothing and a near-miss Merge said it had too few particles. Given the
        labelled mask, the anchor moves to the particle's pixel nearest the click.

        Args:
            label (int): The particle's label at the moment it was clicked.
            point (tuple): (x, y) in image coordinates, where it was clicked.
            labeled_mask (np.ndarray, optional): The labelling ``label`` is from.
        """
        if label is None:
            return
        x, y = int(point[0]), int(point[1])
        if labeled_mask is not None:
            height, width = labeled_mask.shape
            inside = 0 <= y < height and 0 <= x < width and labeled_mask[y, x] == label
            if not inside:
                rows, cols = np.nonzero(labeled_mask == label)
                if len(rows):
                    nearest = int(np.argmin(np.hypot(rows - y, cols - x)))
                    x, y = int(cols[nearest]), int(rows[nearest])
        self.pending_anchors[int(label)] = (x, y)

    def resolve_label(self, label, labeled_mask):
        """
        The label a selected particle carries now.

        Returns:
            int or None: Its current label, or None if it no longer exists —
            deleted by an earlier stage of the same Apply, or dropped by an edge
            clean-up since it was clicked.
        """
        if label is None or labeled_mask is None:
            return None
        point = self.pending_anchors.get(int(label))
        if point is None:
            # Never anchored: the best that can be done is to take it at face
            # value, which is what the code did for everything before.
            return int(label)
        x, y = point
        height, width = labeled_mask.shape
        if not (0 <= y < height and 0 <= x < width):
            return None
        return int(labeled_mask[y, x]) or None

    def reindex_pending(self, labeled_mask):
        """
        Rewrite pending selections after something has relabelled the mask.

        Called by any operation that edits the mask while clicks are still
        queued. Without it those clicks go on naming labels that now belong to
        different particles, and applying them edits the wrong ones.
        """
        anchors = {}

        def carry(labels):
            kept = []
            for label in labels:
                current = self.resolve_label(label, labeled_mask)
                if current is None:
                    continue
                anchor = self.pending_anchors.get(int(label))
                if anchor is not None:
                    anchors[current] = anchor
                kept.append(current)
            return kept

        deletes = carry(self.pending_deletes)
        merges = carry(self.pending_merge)
        refine = carry([self.point_refine_particle]
                       if self.point_refine_particle is not None else [])

        self.pending_deletes = deletes
        self.pending_merge = merges
        self.point_refine_particle = refine[0] if refine else None
        self.pending_anchors = anchors
        # The per-click history still holds the old labels, and undoing into it
        # outlined and edited whichever particles had inherited them. Dropped,
        # as Apply drops it; the edit that renumbered the mask is itself undone
        # from its mask snapshot.
        self.undo_history = []

    def mark_processed(self, index, num_particles):
        """Mark an image as processed."""
        self.processed_images[index] = {
            "status": "completed",
            "num_particles": num_particles
        }

    def is_processed(self, index):
        """Check if an image has been processed."""
        return index in self.processed_images

    def save_pending_state(self):
        """Save current pending changes state to undo history (for undoing individual clicks)."""
        self.undo_history.append({
            'mode': self.click_mode,
            'pending_deletes': self.pending_deletes.copy(),
            'pending_anchors': self.pending_anchors.copy(),
            'pending_add_points': self.pending_add_points.copy(),
            'pending_add_masks': [mask.copy() for mask in self.pending_add_masks],  # Deep copy numpy arrays
            'pending_merge': self.pending_merge.copy(),
            'point_refine_particle': self.point_refine_particle,
            'point_refine_base_mask': self.point_refine_base_mask.copy() if self.point_refine_base_mask is not None else None,
            'point_refine_points': self.point_refine_points.copy(),
            'point_refine_labels': self.point_refine_labels.copy(),
            'point_refine_preview_mask': self.point_refine_preview_mask.copy() if self.point_refine_preview_mask is not None else None,
            'point_refine_logits': self.point_refine_logits.copy() if self.point_refine_logits is not None else None
        })

    def has_corrections(self):
        """
        True once the analyst has changed the mask in use, applied or queued.

        Re-ranking or re-segmenting may then offer new candidates, but must not
        swap one in on its own: that would throw the corrections away.
        """
        return bool(self.mask_history or self.pending_deletes or self.pending_add_masks
                    or self.pending_merge or self.point_refine_points)

    def clear_pending_edits(self):
        """
        Forget every queued click, and the per-click undo history with them.

        Applied edits are left alone: the mask and mask_history are untouched.
        """
        self.pending_deletes = []
        self.pending_anchors = {}
        self.pending_add_points = []
        self.pending_add_masks = []
        self.pending_merge = []
        self.point_refine_particle = None
        self.point_refine_base_mask = None
        self.point_refine_points = []
        self.point_refine_labels = []
        self.point_refine_preview_mask = None
        self.point_refine_logits = None
        self.undo_history = []

    def sync_processed_from_csv(self):
        """Cross-reference loaded images with results CSV to set checkmarks.

        Returns:
            tuple: (matched_count, unmatched_count)
                matched_count: images in both CSV and loaded images
                unmatched_count: CSV entries with no matching loaded image
        """
        if self.results_manager is None or not self.image_paths:
            return 0, 0

        results_df = self.results_manager.get_results()
        if len(results_df) == 0:
            return 0, 0

        # Matched on the name without its extension, so a frame recorded
        # under its TIFF name is still found when a PNG copy is open.
        by_stem = {}
        for _, row in results_df.iterrows():
            stem = os.path.splitext(str(row['file_name']))[0]
            by_stem[stem] = int(row['num_particles'])

        matched = 0
        for idx, img_path in enumerate(self.image_paths):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if stem in by_stem:
                self.mark_processed(idx, by_stem[stem])
                matched += 1

        loaded = {os.path.splitext(os.path.basename(p))[0] for p in self.image_paths}
        unmatched = sum(1 for stem in by_stem if stem not in loaded)

        return matched, unmatched


# Global state instance
state = AppState()
