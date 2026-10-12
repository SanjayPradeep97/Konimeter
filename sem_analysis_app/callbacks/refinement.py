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

# How much room to leave round the region SAM is asked to answer in.
ROI_PAD_PX = 10


def _plural(count, noun):
    """'1 particle', '2 particles'."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _analysable():
    """The specimen part of the frame; anything drawn outside it is dropped."""
    return getattr(state, "analysable_region", None)


def _nothing_left_reason(raw):
    """Why a mask from SAM contributed nothing once prepared for measurement."""
    if not raw.any():
        return "SAM found nothing there"
    return (f"nothing of it survives the cleanup every particle gets (at least "
            f"{state.analyzer.min_size} px and 3 px wide, inside the analysable "
            f"area, clear of the frame-edge strip)")


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
                    # Anchored to a pixel of the particle, so this stays the
                    # same particle even after an earlier edit renumbers the
                    # mask — and a near miss is not lost at Apply.
                    state.anchor_selection(label, (x, y),
                                           state.analyzer.labeled_mask)

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

                # Queued as it will be measured: restricted to the specimen and
                # cleaned like every other particle. SAM is happy to segment a
                # burned-in scale bar, and that must not become a particle.
                added = state.analyzer.prepare_addition(refined_mask,
                                                        region=_analysable())
                if not added.any():
                    return (get_current_visualization(),
                            f"⚠️ Nothing to add here — "
                            f"{_nothing_left_reason(refined_mask)}.")

                # Save pending state before modification (for undo of this click)
                state.save_pending_state()
                state.pending_add_points.append((x, y))
                state.pending_add_masks.append(added)

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
                    state.anchor_selection(label, (x, y),
                                           state.analyzer.labeled_mask)

                # Visualization will show selected particles in different color
                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks,
                    pending_merge=state.pending_merge
                )
                return particle_viz, (f"🔵 Selected "
                                      f"{_plural(len(state.pending_merge), 'particle')} "
                                      f"for merging")
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
                    state.anchor_selection(label, (x, y),
                                           state.analyzer.labeled_mask)
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

                # The preview is what Apply will add, prepared exactly as Apply
                # prepares it. It used to be checked against the slider's size
                # alone while Apply also cleaned and used the size fixed when
                # the candidate was chosen, so a preview that passed could
                # still come to nothing — and Apply deleted the particle being
                # redrawn and added nothing back.
                raw = refined_mask
                refined_mask = state.analyzer.prepare_addition(
                    refined_mask,
                    largest_only=state.point_refine_particle is not None,
                    region=_analysable())

                if not refined_mask.any():
                    # Do not store an empty preview: there is nothing to apply.
                    state.point_refine_preview_mask = None
                    return (create_point_refine_visualization(
                        state.cropped_image, raw, state.point_refine_points,
                        state.point_refine_labels),
                        f"⚠️ No preview — {_nothing_left_reason(raw)}. Add "
                        f"another point, or lower the smallest particle size.")

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
    """
    Set the smallest particle, for the next analysis and for edits from now on.

    Kept in step with the analyzer, which is what Apply cleans with. The two
    used to drift apart: the slider never reached the analyzer, so a Redraw
    preview checked against the slider could pass and then come to nothing at
    Apply. Particles already measured are not refiltered — an edit never
    changes a particle it did not touch — so the new size applies to what is
    added or redrawn, and to the next candidate chosen.
    """
    size = int(size)
    state.min_particle_size = size
    if state.analyzer is None:
        return f"✓ Smallest particle set to {size} px"
    state.analyzer.min_size = size
    return (f"✓ Smallest particle set to {size} px for particles added or "
            f"redrawn from now on — choose the candidate again to refilter "
            f"the ones already measured")


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
    # The per-click history describes the tool just left: undoing into it
    # brought an abandoned Redraw or Merge back as a hidden edit that the next
    # Apply then carried out. Dropped, as Apply drops it.
    state.undo_history = []

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
            progress(0.2, desc=f"Deleting {_plural(len(state.pending_deletes), 'particle')}...")
            targets = selected(state.pending_deletes)
            if targets:
                state.analyzer.delete_particles(targets)
                status_messages.append(f"Deleted {_plural(len(targets), 'particle')}")
                changes_made = True
            state.pending_deletes = []

        # Apply additions. Prepared again rather than trusted as queued: the
        # smallest-particle size may have changed since the click.
        if state.pending_add_masks:
            progress(0.4, desc=f"Adding {_plural(len(state.pending_add_masks), 'particle')}...")
            added = 0
            for add_mask in state.pending_add_masks:
                pixels = state.analyzer.prepare_addition(add_mask, region=_analysable())
                if pixels.any():
                    state.analyzer.add_particle_from_sam(pixels)
                    added += 1
            skipped = len(state.pending_add_masks) - added
            if added:
                status_messages.append(f"Added {_plural(added, 'particle')}")
                changes_made = True
            if skipped:
                status_messages.append(
                    f"⚠️ {_plural(skipped, 'addition')} left out — nothing "
                    f"survived the cleanup every particle gets")
            state.pending_add_points = []
            state.pending_add_masks = []

        # Apply merge
        if state.pending_merge:
            progress(0.6, desc=f"Merging {_plural(len(state.pending_merge), 'particle')}...")
            targets = selected(state.pending_merge)
            state.pending_merge = []
            if len(targets) >= 2:
                state.analyzer.merge_particles(targets)
                if state.analyzer.last_merge_succeeded:
                    status_messages.append(
                        f"Merged {_plural(len(targets), 'particle')} "
                        f"(+{state.analyzer.last_merge_added_px} px joining them)")
                    changes_made = True
                else:
                    status_messages.append(
                        f"⚠️ Did not merge {_plural(len(targets), 'particle')} — "
                        f"joining them would take in a particle you did not "
                        f"select. Select it too, or Redraw instead")
            else:
                status_messages.append("⚠️ Need at least 2 particles to merge")

        # Apply point refinement
        preview = state.point_refine_preview_mask
        if preview is not None:
            progress(0.7, desc="Applying point refinement...")

            # Resolved here, not at click time: deletes and merges above have
            # renumbered the mask, and the old label would name someone else.
            target = (state.resolve_label(state.point_refine_particle,
                                          state.analyzer.labeled_mask)
                      if state.point_refine_particle is not None else None)
            refining_existing = target is not None

            # Prepared before anything is deleted, so that a redraw which comes
            # to nothing is refused instead of removing the particle and adding
            # nothing back. Refining one particle must yield one particle:
            # SAM's mask often has stray disconnected blobs, and unioning those
            # in turns a single refinement into several new particles.
            replacement = state.analyzer.prepare_addition(
                preview, largest_only=refining_existing, region=_analysable())
            points = _plural(len(state.point_refine_points), 'point')
            if not preview.any():
                status_messages.append(
                    "⚠️ Nothing to redraw — the preview was empty, so the "
                    "particle was left as it was")
            elif not replacement.any():
                status_messages.append(
                    f"⚠️ Redraw not applied — {_nothing_left_reason(preview)}. "
                    f"The particle was left as it was")
            else:
                if refining_existing:
                    state.analyzer.delete_particles([target])
                    status_messages.append(f"Redrew a particle with {points}")
                else:
                    status_messages.append(f"Drew a new particle with {points}")
                state.analyzer.add_particle_from_sam(replacement)
                changes_made = True

            state.point_refine_particle = None
            state.point_refine_base_mask = None
            state.point_refine_points = []
            state.point_refine_labels = []
            state.point_refine_preview_mask = None
            state.point_refine_logits = None

        if state.point_refine_particle is None:
            state.pending_anchors = {}
        elif changes_made:
            # A redraw still being drawn, with no preview yet, outlives this
            # Apply. The edits above renumbered the mask, so point it at its
            # particle again.
            state.reindex_pending(state.analyzer.labeled_mask)

        if not changes_made:
            # Nothing happened, so the snapshot taken above is dead weight.
            if state.mask_history:
                state.mask_history.pop()
            # Say *why* nothing happened when there is a reason. Reporting a bare
            # "No changes to apply" over the top of it is what made a redraw that
            # produced nothing look like a button that does nothing. The view is
            # redrawn so outlines of selections just used up go away, unless a
            # redraw is still in progress and showing its own points.
            viz = gr.update() if state.point_refine_points else get_current_visualization()
            return (viz, gr.update(),
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
        status = (f"✅ {' | '.join(status_messages)} | "
                  f"Total: {_plural(num_particles, 'particle')}")

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
    """
    Undo the last click, or the last Apply when no clicks are queued.

    Returns:
        tuple: (refine_viz, refine_results, refine_status, current_results,
        current_stats), as apply_refinement_changes does — undoing an Apply
        changes the measurements, and the Results tab has to follow.
    """
    try:
        if state.analyzer is None:
            return (gr.update(), gr.update(), "❌ No analysis available",
                    gr.update(), gr.update())

        if not state.undo_history:
            # No un-applied clicks left, so step back over the last Apply
            # instead. Without this, applying was a point of no return.
            if state.restore_mask():
                # Clicks still queued (a tool switch keeps deletes and adds)
                # name labels from before the restore.
                state.reindex_pending(state.analyzer.labeled_mask)
                particle_viz = get_current_visualization()
                measurements = state.analyzer.get_measurements(in_nm=True)
                results_df = create_results_dataframe(measurements)
                steps = _plural(len(state.mask_history), 'further step')
                return (particle_viz, results_df,
                        f"↩️ Undid the last applied change — back to "
                        f"{_plural(len(state.analyzer.regions), 'particle')} "
                        f"({steps} available)",
                        results_df, create_summary_statistics_table(measurements))
            return (gr.update(), gr.update(), "❌ Nothing left to undo",
                    gr.update(), gr.update())

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

        # Measurements don't change: only queued clicks were undone.
        num_particles = len(state.analyzer.regions)
        status = (f"↩️ Undone last click! {_plural(num_particles, 'particle')} | "
                  f"{_plural(len(state.undo_history), 'undo step')} remaining")

        return particle_viz, gr.update(), status, gr.update(), gr.update()

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()


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
        status = (f"✅ Removed {_plural(n_removed, 'edge particle')} (centroid within "
                  f"{buffer}px of edge) - Now {num_particles} total")

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
        state.clear_pending_edits()

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

        status = (f"🗑️ Cleared {_plural(n_removed, 'particle')}. "
                  f"Use 'add' mode to start segmenting.")

        return particle_viz, results_df, status, results_df, stats_df

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()


def clear_pending_changes():
    """Clear all pending changes and redraw visualization."""
    try:
        # The per-click undo history goes too, since every click it describes is gone.
        state.clear_pending_edits()

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
