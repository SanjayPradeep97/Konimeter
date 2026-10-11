"""
Refinement tab: click-driven add/delete/merge/point-refine.
"""
import gradio as gr
import numpy as np

from ..visualization import (
    create_particle_visualization,
    create_point_refine_visualization,
    create_results_dataframe,
    create_summary_statistics_table,
)
from ..state import state
from sem_particle_analysis._compat import remove_objects_smaller_than

# How much room to leave round the region SAM is asked to answer in.
ROI_PAD_PX = 10


def roi_box(base_mask, points, shape, pad=ROI_PAD_PX):
    """
    The rectangle SAM is allowed to answer in, or None for the whole frame.

    Grown to contain every point clicked, not just the particle the refinement
    started from. SAM cannot return anything outside the box it is given, so a
    box drawn round the starting particle alone silently ignored every click
    that landed beyond its edge — which is exactly the click someone makes when
    the mask has missed part of an object. The refinement simply did nothing.

    Args:
        base_mask (np.ndarray or None): Mask of the particle being refined.
        points (list): [(x, y), ...] clicked so far.
        shape (tuple): (height, width) of the image.
        pad (int): Margin in pixels.

    Returns:
        np.ndarray or None: A 1x4 array of [x0, y0, x1, y1], or None when there
        is nothing to constrain to.
    """
    xs, ys = [], []
    if base_mask is not None and base_mask.any():
        rows, cols = np.where(base_mask)
        xs += [int(cols.min()), int(cols.max())]
        ys += [int(rows.min()), int(rows.max())]
    if base_mask is None or not base_mask.any():
        # Nothing to grow from: let SAM see the whole frame, as before.
        return None
    for x, y in points:
        xs.append(int(x))
        ys.append(int(y))

    height, width = shape[:2]
    return np.array([[
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(width - 1, max(xs) + pad),
        min(height - 1, max(ys) + pad),
    ]])


def get_current_visualization():
    """Get current particle visualization with all pending changes."""
    if state.analyzer is None:
        return None
    return create_particle_visualization(
        state.cropped_image,
        state.analyzer.labeled_mask,
        state.analyzer.regions,
        show_labels=state.show_particle_numbers,
        pending_deletes=state.pending_deletes,
        pending_add_masks=state.pending_add_masks,
        pending_merge=state.pending_merge
    )


def handle_image_click(evt: gr.SelectData):
    """Handle clicks on particle visualization for all refinement modes."""
    try:
        if state.analyzer is None:
            return get_current_visualization(), "❌ No analysis available"

        x, y = evt.index[0], evt.index[1]

        if state.click_mode in ("add", "point_refine") and state.segmenter is None:
            # These two ask SAM what is under the click; the other two only
            # read the mask. Said plainly, because the alternative was an
            # AttributeError on None reported as "SAM refinement failed".
            return (get_current_visualization(),
                    "⚠️ This tool needs a model — load a SAM checkpoint on "
                    "the Setup tab. Remove and Merge work without one.")

        if state.click_mode == "delete":
            # DELETE MODE: Click particles to remove them
            region, idx, label = state.analyzer.find_particle_at_point(x, y)
            if region is not None:
                if label not in state.pending_deletes:
                    # Save pending state before modification (for undo of this click)
                    state.save_pending_state()
                    state.pending_deletes.append(label)
                    # Anchored to the pixel clicked, so this stays the same
                    # particle even after an earlier edit renumbers the mask.
                    state.anchor_selection(label, (x, y))

                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    show_labels=state.show_particle_numbers,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks
                )
                return particle_viz, f"🟡 Queued particle #{idx+1} for deletion (yellow outline)"
            else:
                return get_current_visualization(), "No particle found at this location"

        elif state.click_mode == "add":
            # ADD MODE: Click to add a single particle at this location
            try:
                # Save pending state before modification (for undo of this click)
                state.save_pending_state()

                # Encoding is skipped when this image is already the encoded
                # one, so asking costs nothing and saves the caller tracking it.
                state.segmenter.encode_image(state.cropped_image)

                # Use single positive point WITHOUT base_mask to segment just the clicked particle
                refined_mask, score = state.segmenter.refine_with_sam(
                    state.cropped_image,
                    [[x, y]],
                    [1],
                    base_mask=None,  # Don't constrain to existing mask - segment the clicked object
                    multimask_output=True,
                    image_already_encoded=True
                )

                state.pending_add_points.append((x, y))
                state.pending_add_masks.append(refined_mask)

                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    show_labels=state.show_particle_numbers,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks
                )
                return particle_viz, f"🟢 Added particle preview (green outline) - score: {score:.3f}"
            except Exception as e:
                return get_current_visualization(), f"❌ SAM refinement failed: {str(e)}"

        elif state.click_mode == "merge":
            # MERGE MODE: Click multiple particles to merge them
            region, idx, label = state.analyzer.find_particle_at_point(x, y)
            if region is not None:
                if label not in state.pending_merge:
                    # Save pending state before modification (for undo of this click)
                    state.save_pending_state()
                    state.pending_merge.append(label)
                    state.anchor_selection(label, (x, y))

                # Visualization will show selected particles in different color
                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks,
                    pending_merge=state.pending_merge
                )
                return particle_viz, f"🔵 Selected {len(state.pending_merge)} particles for merging"
            else:
                return get_current_visualization(), "No particle found at this location"

        elif state.click_mode == "point_refine":
            # POINT REFINE MODE: Click anywhere to add positive/negative points with live preview
            # Users can refine existing particles OR create new ones from scratch

            # Save pending state before adding point (for undo of this click)
            state.save_pending_state()
            state.segmenter.encode_image(state.cropped_image)

            # Check if user clicked on an existing particle (only on first click)
            if len(state.point_refine_points) == 0:
                region, idx, label = state.analyzer.find_particle_at_point(x, y)
                if region is not None:
                    # User clicked on existing particle - use it as base mask for IoU selection
                    state.point_refine_particle = label
                    state.anchor_selection(label, (x, y))
                    state.point_refine_base_mask = (state.analyzer.labeled_mask == label).astype(bool)
                # If no particle found, that's OK - user is creating a new particle from scratch

            # Add the point
            point_label = 1 if state.point_type == "positive" else 0
            state.point_refine_points.append((x, y))
            state.point_refine_labels.append(point_label)

            # Generate live preview with SAM using all accumulated points
            try:
                box = roi_box(state.point_refine_base_mask,
                              state.point_refine_points,
                              state.cropped_image.shape)

                # Call SAM with all points using iterative refinement
                if state.point_refine_logits is not None:
                    # Feed back the previous logits, as SAM's predictor example does
                    masks_out, scores, logits_out = state.segmenter.sam_model.predictor.predict(
                        point_coords=np.array(state.point_refine_points, dtype=float),
                        point_labels=np.array(state.point_refine_labels, dtype=int),
                        box=box,
                        mask_input=state.point_refine_logits[None, ...],  # Use previous mask logits
                        multimask_output=False  # Single mask output for iterative refinement
                    )
                    refined_mask = masks_out[0].astype(bool)
                    state.point_refine_logits = logits_out[0]  # Store for next iteration
                else:
                    # First point: get initial masks and select best one
                    masks_out, scores, logits_out = state.segmenter.sam_model.predictor.predict(
                        point_coords=np.array(state.point_refine_points, dtype=float),
                        point_labels=np.array(state.point_refine_labels, dtype=int),
                        box=box,
                        multimask_output=True  # Multiple masks for initial selection
                    )

                    # Select best mask by IoU with base mask
                    if state.point_refine_base_mask is not None:
                        ious = []
                        for mask in masks_out:
                            intersection = np.logical_and(mask, state.point_refine_base_mask).sum()
                            union = np.logical_or(mask, state.point_refine_base_mask).sum()
                            iou = intersection / union if union > 0 else 0
                            ious.append(iou)
                        best_idx = int(np.argmax(ious))
                    else:
                        best_idx = int(np.argmax(scores))

                    refined_mask = masks_out[best_idx].astype(bool)
                    state.point_refine_logits = logits_out[best_idx]  # Store for next iteration

                # remove_objects_smaller_than, not skimage's remove_small_objects:
                # since 0.26 that one drops objects smaller than *or equal to*
                # min_size, so the boundary case silently lost a particle of
                # exactly the minimum size.
                raw = refined_mask
                refined_mask = remove_objects_smaller_than(
                    refined_mask, state.min_particle_size)

                if not refined_mask.any():
                    # Do not store an empty preview. Apply would take it as a
                    # refinement, delete the particle being refined and add
                    # nothing back — the particle would simply disappear.
                    state.point_refine_preview_mask = None
                    if raw.any():
                        reason = (f"the region drawn is under the "
                                  f"{state.min_particle_size} px minimum size")
                    else:
                        reason = "SAM found nothing for these points"
                    return (create_point_refine_visualization(
                        state.cropped_image, raw, state.point_refine_points,
                        state.point_refine_labels),
                        f"⚠️ No preview — {reason}. Add another point, or "
                        f"lower the minimum size.")

                # Store as preview (will be applied when user clicks Apply)
                state.point_refine_preview_mask = refined_mask

                # Create visualization with point markers overlaid
                particle_viz = create_point_refine_visualization(
                    state.cropped_image,
                    refined_mask,
                    state.point_refine_points,
                    state.point_refine_labels
                )

                point_type_str = "positive ✓" if point_label == 1 else "negative ✗"
                return particle_viz, (
                    f"➕ Added {point_type_str} point "
                    f"({len(state.point_refine_points)} total)")

            except Exception as e:
                point_type_str = "positive ✓" if point_label == 1 else "negative ✗"
                return get_current_visualization(), f"➕ Added {point_type_str} point - Preview update failed: {str(e)}"

        return get_current_visualization(), "Click registered"

    except Exception as e:
        return get_current_visualization(), f"❌ Error: {str(e)}"


def set_min_particle_size(size):
    """Set minimum particle size for filtering."""
    state.min_particle_size = int(size)
    return f"✓ Minimum particle size set to {int(size)} pixels"


def toggle_particle_numbers(show_numbers):
    """Toggle visibility of particle numbers in visualization."""
    state.show_particle_numbers = show_numbers
    # Return updated visualization
    return get_current_visualization()


def set_point_type(point_type):
    """Set point type for point refine mode."""
    state.point_type = point_type
    return f"✓ Point type: {point_type.upper()}"


def reset_point_refine():
    """Reset point refine state and redraw visualization."""
    try:
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None

        # Redraw regular visualization
        if state.analyzer is not None:
            particle_viz = create_particle_visualization(
                state.cropped_image,
                state.analyzer.labeled_mask,
                state.analyzer.regions,
                show_labels=state.show_particle_numbers,
                pending_deletes=state.pending_deletes,
                pending_add_masks=state.pending_add_masks,
                pending_merge=state.pending_merge
            )
            return particle_viz, "✅ Reset point refinement"
        else:
            return None, "✅ Reset point refinement"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def set_click_mode(mode):
    """Set click mode and return status message."""
    state.click_mode = mode

    # Reset mode-specific state
    if mode != "merge":
        state.pending_merge = []
    if mode != "point_refine":
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None

    # Return appropriate status message and visibility for point refine controls
    mode_messages = {
        "delete": "🚫 Remove — click any particle that isn't one. "
                  "Clicking background does nothing.",
        "add": "➕ Add — click something SAM missed; it segments what you clicked.",
        "merge": "🔗 Merge — click two or more pieces that are really one particle.",
        "point_refine": "🎯 Redraw — click to include, switch to exclude to cut "
                        "away. Start on a particle to fix it, on background to "
                        "draw a new one.",
    }

    # Show point refine controls only in point_refine mode
    show_controls = (mode == "point_refine")

    return mode_messages.get(mode, "Unknown mode"), gr.update(visible=show_controls)


def apply_refinement_changes(progress=gr.Progress()):
    """Apply all pending refinement changes (delete, add, merge, point_refine)."""
    try:
        if state.analyzer is None:
            return gr.update(), gr.update(), "❌ No analysis available", gr.update(), gr.update()

        changes_made = False
        status_messages = []

        # Snapshot the mask first, so this Apply can be undone. Applying is the
        # only irreversible step in the loop otherwise, and it happens often.
        state.snapshot_mask()

        # Each stage below relabels the mask from scratch, so a label recorded
        # when the particle was clicked can name a different particle by the
        # time the next stage runs. Every selection is therefore resolved
        # through the pixel it was clicked on, immediately before it is used.
        def selected(labels):
            resolved = (state.resolve_label(label, state.analyzer.labeled_mask)
                        for label in labels)
            return [label for label in resolved if label is not None]

        # Apply deletions
        if state.pending_deletes:
            progress(0.2, desc=f"Deleting {len(state.pending_deletes)} particles...")
            targets = selected(state.pending_deletes)
            if targets:
                state.analyzer.delete_particles(targets)
                status_messages.append(f"Deleted {len(targets)} particles")
                changes_made = True
            state.pending_deletes = []

        # Apply additions using pre-generated masks
        if state.pending_add_masks:
            progress(0.4, desc=f"Adding {len(state.pending_add_masks)} particles...")
            for add_mask in state.pending_add_masks:
                state.analyzer.add_particle_from_sam(add_mask)
            status_messages.append(f"Added {len(state.pending_add_masks)} particles")
            state.pending_add_points = []
            state.pending_add_masks = []
            changes_made = True

        # Apply merge
        if state.pending_merge:
            progress(0.6, desc=f"Merging {len(state.pending_merge)} particles...")
            targets = selected(state.pending_merge)
            state.pending_merge = []
            if len(targets) >= 2:
                state.analyzer.merge_particles(targets)
                if getattr(state.analyzer, "last_merge_succeeded", True):
                    status_messages.append(f"Merged {len(targets)} particles")
                else:
                    status_messages.append(
                        f"⚠️ Could not merge {len(targets)} particles — they "
                        f"are too far apart to join"
                    )
                changes_made = True
            else:
                status_messages.append("⚠️ Need at least 2 particles to merge")

        # Apply point refinement
        preview = state.point_refine_preview_mask
        if preview is not None and preview.any():
            progress(0.7, desc="Applying point refinement...")

            # Resolved here, not at click time: deletes and merges above have
            # renumbered the mask, and the old label would name someone else.
            target = (state.resolve_label(state.point_refine_particle,
                                          state.analyzer.labeled_mask)
                      if state.point_refine_particle is not None else None)
            refining_existing = target is not None
            if refining_existing:
                # Refining existing particle - delete old and add refined one
                state.analyzer.delete_particles([target])
                status_messages.append(f"Refined particle with {len(state.point_refine_points)} points")
            else:
                # Creating new particle from scratch
                status_messages.append(f"Created new particle with {len(state.point_refine_points)} points")

            # Refining one particle must yield one particle. SAM's mask often has
            # stray disconnected blobs, and unioning those in turns a single
            # refinement into several new particles.
            state.analyzer.add_particle_from_sam(preview, largest_only=refining_existing)

            state.point_refine_particle = None
            state.point_refine_base_mask = None
            state.point_refine_points = []
            state.point_refine_labels = []
            state.point_refine_preview_mask = None
            state.point_refine_logits = None
            changes_made = True
        elif preview is not None:
            # An empty preview is not a refinement. Applying it used to delete
            # the particle being refined and add nothing back.
            status_messages.append(
                "⚠️ Nothing to redraw — the preview was empty, so the "
                "particle was left as it was")
            state.point_refine_preview_mask = None

        state.pending_anchors = {}

        if not changes_made:
            # Nothing happened, so the snapshot taken above is dead weight.
            if state.mask_history:
                state.mask_history.pop()
            # Say *why* nothing happened when there is a reason. Reporting a bare
            # "No changes to apply" over the top of it is what made a redraw that
            # produced nothing look like a button that does nothing.
            return (gr.update(), gr.update(),
                    " | ".join(status_messages) or "No changes to apply",
                    gr.update(), gr.update())

        # The per-click history described edits that are now committed; undo from
        # here on rolls back the whole Apply, using the snapshot above.
        state.undo_history = []

        progress(0.9, desc="Updating visualization...")

        # Update visualization (no pending changes now)
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Update measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        progress(1.0, desc="Complete!")

        num_particles = len(state.analyzer.regions)
        status = f"✅ {' | '.join(status_messages)} | Total: {num_particles} particles"

        # Return: refine_viz, refine_results, refine_status, current_results, current_stats
        return (
            particle_viz,
            results_df,
            status,
            results_df,
            stats_df
        )

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()


def undo_last_action():
    """Undo the last click (removes last item from pending changes)."""
    try:
        if state.analyzer is None:
            return gr.update(), gr.update(), "❌ No analysis available"

        if not state.undo_history:
            # No un-applied clicks left, so step back over the last Apply
            # instead. Without this, applying was a point of no return.
            if state.restore_mask():
                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    show_labels=state.show_particle_numbers,
                )
                measurements = state.analyzer.get_measurements(in_nm=True)
                return (particle_viz, create_results_dataframe(measurements),
                        f"↩️ Undid the last applied change — back to "
                        f"{len(state.analyzer.regions)} particles "
                        f"({len(state.mask_history)} further steps available)")
            return gr.update(), gr.update(), "❌ Nothing left to undo"

        # Restore previous pending state (before last click)
        previous_state = state.undo_history.pop()

        # Restore only the pending changes (not the mask itself)
        state.pending_deletes = previous_state['pending_deletes']
        state.pending_anchors = previous_state.get('pending_anchors', {})
        state.pending_add_points = previous_state['pending_add_points']
        state.pending_add_masks = previous_state['pending_add_masks']
        state.pending_merge = previous_state['pending_merge']
        state.point_refine_particle = previous_state['point_refine_particle']
        state.point_refine_base_mask = previous_state['point_refine_base_mask']
        state.point_refine_points = previous_state['point_refine_points']
        state.point_refine_labels = previous_state['point_refine_labels']
        state.point_refine_preview_mask = previous_state['point_refine_preview_mask']
        state.point_refine_logits = previous_state['point_refine_logits']

        # Update visualization based on mode
        if state.click_mode == "point_refine" and state.point_refine_preview_mask is not None:
            # Show point refine visualization with restored points
            particle_viz = create_point_refine_visualization(
                state.cropped_image,
                state.point_refine_preview_mask,
                state.point_refine_points,
                state.point_refine_labels
            )
        else:
            # Show regular visualization with restored pending changes
            particle_viz = create_particle_visualization(
                state.cropped_image,
                state.analyzer.labeled_mask,
                state.analyzer.regions,
                show_labels=state.show_particle_numbers,
                pending_deletes=state.pending_deletes,
                pending_add_masks=state.pending_add_masks,
                pending_merge=state.pending_merge
            )

        # Measurements don't change (we're only undoing pending changes)
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)

        num_particles = len(state.analyzer.regions)
        status = f"↩️ Undone last click! {num_particles} particles | {len(state.undo_history)} undo steps remaining"

        return particle_viz, results_df, status

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}"


def clear_edge_particles(buffer_size):
    """Clear particles whose centroid is within buffer distance from edges."""
    try:
        if state.analyzer is None:
            return gr.update(), gr.update(), "❌ No analysis available", gr.update(), gr.update()

        buffer = int(buffer_size)
        H, W = state.analyzer.mask.shape

        # Identify particles to remove based on centroid position
        labels_to_remove = []
        for region in state.analyzer.regions:
            y, x = region.centroid
            # Check if centroid is within buffer distance from any edge
            if (x < buffer or x > W - buffer or
                y < buffer or y > H - buffer):
                labels_to_remove.append(region.label)

        n_removed = len(labels_to_remove)

        if n_removed > 0:
            state.snapshot_mask()
            state.analyzer.delete_particles(labels_to_remove)
            # That renumbered every remaining particle, so any click still
            # queued now names the wrong one. Point them back at what was
            # actually clicked, and drop the ones that were just removed.
            state.reindex_pending(state.analyzer.labeled_mask)

        # Update visualization
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Update measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        num_particles = len(state.analyzer.regions)
        status = f"✅ Removed {n_removed} edge particles (centroid within {buffer}px of edge) - Now {num_particles} total"

        # Return: refine_viz, refine_results, refine_status, current_results, current_stats
        return (
            particle_viz,
            results_df,
            status,
            results_df,
            stats_df
        )

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()


def clear_all_particles():
    """Clear ALL particles from the mask, giving user a blank canvas to add particles manually."""
    try:
        if state.analyzer is None:
            return gr.update(), gr.update(), "❌ No analysis available", gr.update(), gr.update()

        # Get all current labels and delete them all
        all_labels = [r.label for r in state.analyzer.regions]
        n_removed = len(all_labels)

        if n_removed > 0:
            # The most destructive action in the app; it must be recoverable.
            state.snapshot_mask()
            state.analyzer.delete_particles(all_labels)

        # Clear all pending state too
        state.pending_deletes = []
        state.pending_anchors = {}
        state.pending_add_points = []
        state.pending_add_masks = []
        state.pending_merge = []
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None
        state.undo_history = []

        # Redraw (empty image, no particles)
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Empty measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        status = f"🗑️ Cleared all {n_removed} particles. Use 'add' mode to start segmenting."

        return particle_viz, results_df, status, results_df, stats_df

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()


def clear_pending_changes():
    """Clear all pending changes and redraw visualization."""
    try:
        state.pending_deletes = []
        state.pending_anchors = {}
        state.pending_add_points = []
        state.pending_add_masks = []
        state.pending_merge = []
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None
        state.undo_history = []  # Clear undo history since all pending changes are cleared

        # Redraw visualization without pending changes
        if state.analyzer is not None:
            particle_viz = create_particle_visualization(
                state.cropped_image,
                state.analyzer.labeled_mask,
                state.analyzer.regions,
                pending_deletes=[],
                pending_add_masks=[],
                pending_merge=[]
            )
            return particle_viz, "✅ Cleared all pending changes"
        else:
            return gr.update(), "✅ Cleared all pending changes"
    except Exception as e:
        return gr.update(), f"❌ Error: {str(e)}"
