"""
Processing tab: SAM segmentation and initial particle analysis.
"""
import gradio as gr

from sem_particle_analysis import ParticleAnalyzer
from ..visualization import (
    create_particle_visualization,
    visualize_three_masks,
    create_results_dataframe,
    create_summary_statistics_table,
)
from ..state import state

# The analyst's polarity choice, as the ``dark_features`` the segmenter takes.
POLARITY_CHOICES = {"auto": None, "bright": False, "dark": True}


def _particle_polarity():
    """
    Whether particles are the darker side of the frame, or None to use contrast.

    An explicit choice wins; otherwise the detected modality decides, since
    material on an SEM filter reads brighter than the membrane and
    electron-dense material in TEM reads darker than the support film.
    """
    chosen = getattr(state, "particle_choice", "auto") or "auto"
    if chosen != "auto":
        return POLARITY_CHOICES[chosen]
    return state.modality.dark_particles if state.modality else None


def segment_with_sam(progress=gr.Progress()):
    """Segment image with SAM."""
    try:
        if state.cropped_image is None:
            return None, "❌ No image to segment (detect scale first)", gr.update()

        if state.segmenter is None:
            return None, "❌ SAM not initialized", gr.update()

        progress(0.3, desc="Running SAM...")
        raw_masks, state.scores = state.segmenter.segment_image(
            state.cropped_image,
            multimask_output=True
        )

        progress(0.7, desc="Ranking candidates...")
        # Everything that is not specimen: beam-blocked area — an SEM aperture
        # vignette, a TEM grid bar — and a scale bar printed inside the frame.
        # All of them out-contrast the particles, so leaving them in means
        # measuring them instead.
        analysable = getattr(state, "analysable_region", None)
        exclude = ~analysable if analysable is not None else None

        # Which side of the frame holds the particles. Left to contrast, this
        # picks the vignette or the grid bar; the modality knows the answer.
        dark_particles = _particle_polarity()

        # Rank by contrast rather than showing SAM's raw output in its own order.
        # Either polarity of any candidate may be the one holding the objects, and
        # the highest-confidence mask is frequently the background.
        state.candidates = state.segmenter.rank_candidates(
            state.cropped_image, raw_masks, top_k=3,
            dark_features=dark_particles, exclude=exclude,
        )

        if not state.candidates:
            state.masks = None
            return (None,
                    "❌ No candidate isolated anything convincing. The features may be "
                    "too faint — try Point Refine, or adjust the crop.",
                    gr.update(choices=[], value=None))

        state.masks = [c["mask"] for c in state.candidates]
        labels = [f"Option {i + 1}" for i in range(len(state.candidates))]

        progress(0.9, desc="Creating visualization...")
        mask_viz = visualize_three_masks(state.cropped_image, state.masks, labels)

        progress(1.0, desc="Complete!")

        # Offer exactly the candidates that survived ranking; a fixed three-way
        # choice lets the analyst pick an option that does not exist.
        choices = [f"Option {i + 1}" for i in range(len(state.candidates))]
        return (mask_viz,
                f"✅ {len(state.masks)} candidate(s), best first. "
                f"Pick one, then refine with clicks.",
                gr.update(choices=choices, value=choices[0]))

    except Exception as e:
        return None, f"❌ Error: {str(e)}", gr.update()


def select_mask_and_analyze(mask_choice, progress=gr.Progress()):
    """Select mask and run initial analysis."""
    try:
        if state.masks is None:
            return gr.update(), gr.update(), "❌ No masks available", gr.update(), gr.update()

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
        status = (f"✅ Detected {num_particles} {noun} ({measurements['unit']}) "
                  f"- Ready for refinement")
        if conversion is None:
            status += " ⚠️ no scale, sizes are in pixels"

        # Return: refine_viz, refine_results, analysis_status, current_results, current_stats
        return particle_viz, results_df, status, results_df, stats_df

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()
