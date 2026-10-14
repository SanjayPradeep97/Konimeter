"""
Processing tab: SAM segmentation and initial particle analysis.
"""
import gradio as gr

from sem_particle_analysis import ParticleAnalyzer
from ..visualization import (
    candidate_thumbnails,
    create_particle_visualization,
    create_results_dataframe,
    create_summary_statistics_table,
)
from ..state import state

# The analyst's polarity choice, as the ``dark_features`` the segmenter takes.
POLARITY_CHOICES = {"auto": None, "bright": False, "dark": True}


def _particle_polarity():
    """
    Whether particles are expected to be the darker side of the frame, or None.

    An explicit choice wins; otherwise the microscope decides, since
    electron-dense material in TEM bright field reads darker than the support
    film and material on an SEM substrate usually reads brighter. It orders the
    candidates and never hides any, so a wrong expectation costs a click.
    """
    chosen = getattr(state, "particle_choice", "auto") or "auto"
    if chosen != "auto":
        return POLARITY_CHOICES[chosen]
    return {"SEM": False, "TEM": True}.get(state.microscope_kind())


def candidate_gallery(selected=None):
    """
    The ranked candidates as thumbnails, ready for the gallery.

    Args:
        selected: Index of the candidate in use, highlighted; None for none.
    """
    if not state.masks:
        return gr.update(value=[], selected_index=None)
    thumbs = candidate_thumbnails(state.cropped_image, state.masks)
    items = [(thumb, "Best" if i == 0 else f"Option {i + 1}")
             for i, thumb in enumerate(thumbs)]
    return gr.update(value=items, selected_index=selected)


def _rank_and_show():
    """
    Rank the outlines SAM last returned and offer them as the candidates.

    The best one is then put to use by use_best_candidate, unless the analyst
    has already corrected the mask in use: swapping in a new candidate would
    throw that work away, so then they are only offered.

    Returns:
        tuple: (candidate gallery update, status)
    """
    # Everything that is not specimen: beam-blocked area — an SEM aperture
    # vignette, a TEM grid bar — and a scale bar printed inside the frame.
    # All of them out-contrast the particles, so leaving them in means
    # measuring them instead.
    analysable = getattr(state, "analysable_region", None)
    exclude = ~analysable if analysable is not None else None

    # Rank by contrast rather than showing SAM's raw output in its own order.
    # Either polarity of any candidate may be the one holding the objects, and
    # the highest-confidence mask is frequently the background.
    state.candidates = state.segmenter.rank_candidates(
        state.cropped_image, state.segmenter.masks, top_k=3,
        dark_features=_particle_polarity(), exclude=exclude,
    )

    if not state.candidates:
        state.masks = None
        state.use_best_pending = False
        return (candidate_gallery(),
                "SAM found nothing that stands out from the background. Draw the "
                "particles with the Add tool, or lower Smallest particle.")

    state.masks = [c["mask"] for c in state.candidates]
    keep = state.analyzer is not None and state.has_corrections()
    state.use_best_pending = not keep
    if keep:
        return (candidate_gallery(),
                "New candidates. Your corrections are kept; click a candidate to "
                "start again from it instead.")
    return candidate_gallery(selected=0), ""


def use_best_candidate(progress=gr.Progress()):
    """
    Put the best candidate to use, when ranking has just said to.

    Returns:
        tuple: as select_mask_and_analyze, or no change at all.
    """
    if not getattr(state, "use_best_pending", False) or not state.masks:
        return (gr.update(),) * 5
    state.use_best_pending = False
    return select_mask_and_analyze("Option 1", progress)


def choose_candidate(evt: gr.SelectData, progress=gr.Progress()):
    """Use the candidate whose thumbnail was clicked."""
    index = evt.index if isinstance(evt.index, int) else (evt.index or [0])[0]
    return select_mask_and_analyze(f"Option {index + 1}", progress)


def segment_with_sam(progress=gr.Progress()):
    """Segment image with SAM."""
    try:
        if state.cropped_image is None:
            return candidate_gallery(), "No image open."

        if state.segmenter is None:
            return candidate_gallery(), "Load the SAM model on the Setup tab first."

        progress(0.3, desc="Running SAM...")
        raw_masks, state.scores = state.segmenter.segment_image(
            state.cropped_image,
            multimask_output=True
        )

        progress(0.7, desc="Ranking candidates...")
        shown = _rank_and_show()
        progress(1.0, desc="Complete!")
        return shown

    except Exception as e:
        return candidate_gallery(), f"❌ Error: {str(e)}"


def set_particle_polarity(choice):
    """
    Say which way the particles differ from the background, and re-sort the
    candidates to match. SAM's outlines are kept, so this takes no time.

    Returns:
        tuple: (candidate gallery update, status)
    """
    state.particle_choice = choice or "auto"
    return rerank()


def rerank():
    """
    Re-sort the candidates SAM already returned, after the expectation of which
    way round the particles are has changed.

    Returns:
        tuple: (candidate gallery update, status)
    """
    segmenter = state.segmenter
    if (segmenter is None or getattr(segmenter, "masks", None) is None
            or state.cropped_image is None
            or segmenter.masks[0].shape != state.cropped_image.shape[:2]):
        return gr.update(), gr.update()
    try:
        return _rank_and_show()
    except Exception as e:
        return gr.update(), f"❌ Error: {str(e)}"


def select_mask_and_analyze(mask_choice, progress=gr.Progress()):
    """Select mask and run initial analysis."""
    try:
        if state.masks is None:
            return (gr.update(), gr.update(),
                    "No candidate to use. Press Segment again, or draw the "
                    "particles with the Add tool.", gr.update(), gr.update())

        # Parse mask index
        mask_index = int(mask_choice.split()[1]) - 1
        if not 0 <= mask_index < len(state.masks):
            return gr.update(), gr.update(), "❌ That option isn't available", gr.update(), gr.update()
        state.selected_mask_index = mask_index

        progress(0.3, desc="Analyzing particles...")

        # Scale may be unavailable — some images have no readable bar. Measure in
        # pixels rather than refusing to analyse; the unit is reported alongside.
        conversion = (state.scale_info or {}).get('conversion')
        state.analyzer = ParticleAnalyzer(
            conversion_factor=conversion,
            min_size=state.min_particle_size
        )
        # A new analysis starts with a clean slate. Queued clicks and both undo
        # stacks belonged to the previous candidate's mask: kept, a queued click
        # was applied to this one, and Undo stepped back into the old candidate.
        state.clear_pending_edits()
        state.mask_history = []

        # Candidates already carry the polarity that made them win the ranking.
        # Re-deriving it here with a hardcoded invert=True silently analysed the
        # background whenever the objects were the brighter side.
        candidate = state.candidates[mask_index]
        binary_mask = candidate["mask"]
        # Restrict to specimen before measuring, so nothing outside the
        # analysable region is counted even if it survived into the mask.
        analysable = getattr(state, "analysable_region", None)
        if analysable is not None and analysable.shape == binary_mask.shape:
            binary_mask = binary_mask & analysable
        state.segmenter.selected_mask = binary_mask
        state.segmenter.selected_mask_index = candidate["mask_index"]

        # Analyze (use single min_size parameter, remove redundant min_area)
        num_particles, regions = state.analyzer.analyze_mask(
            binary_mask,
            min_size=state.min_particle_size,
            remove_border=True,
            border_buffer=4
        )

        progress(0.7, desc="Creating visualization...")

        # Create visualization
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Get measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        progress(1.0, desc="Complete!")

        noun = "particle" if num_particles == 1 else "particles"
        which = "the best candidate" if mask_index == 0 else f"option {mask_index + 1}"
        status = f"{num_particles} {noun} from {which}."
        if conversion is None:
            status += " No scale yet, so sizes are in pixels."

        # Return: refine_viz, refine_results, analysis_status, current_results, current_stats
        return particle_viz, results_df, status, results_df, stats_df

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()
