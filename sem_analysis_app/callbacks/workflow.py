"""
Semi-automatic workflow: prepare on load, then save-and-advance.

The loop is: pick an image once, let the app settle its scale and produce ranked
mask candidates unattended, choose the best candidate, correct it with clicks,
then save and move to the next image without going back to the gallery.

Scale is not touched here. The Scale tab establishes it when the image opens and
is the only place that sets it — this module used to re-detect it on every
image, which could silently replace a calibration the analyst had confirmed.
"""
import os

import gradio as gr

from ..state import state
from .segmentation import segment_with_sam
from .results import save_current_results
from .scale_tab import frame_header, prepare_scale_tab
from ..visualization import for_display


# How many outputs open_current_image fills (ui.OPEN_OUTPUTS).
OPEN_COUNT = 12


def _load_index(index):
    """
    Load image at ``index`` into state.

    Returns:
        tuple: (image, info), or (None, message) — a message starting "❌" when
        the image exists but could not be read, rather than when the folder ran out.
    """
    from sem_particle_analysis.utils import load_image

    if not state.image_paths:
        return None, "No images loaded"
    if not 0 <= index < len(state.image_paths):
        return None, "No more images in this folder"

    state.current_index = index
    state.reset_image_state()
    try:
        state.current_image = load_image(state.image_paths[index])
    except Exception as exc:
        name = os.path.basename(state.image_paths[index])
        return None, (f"❌ {name} could not be opened ({exc}). Press Skip to go on "
                      "to the next image, or pick one in the Gallery.")

    filename = os.path.basename(state.image_paths[index])
    return state.current_image, f"Image {index + 1} / {len(state.image_paths)}: {filename}"


def auto_process_current_image(progress=gr.Progress()):
    """
    Produce ranked mask candidates for the image already on screen.

    Runs unattended so the analyst lands on candidates to judge rather than on a
    button to press. Scale and frame geometry are already settled by the time
    this is called.

    Returns:
        tuple: (mask_viz, segment_status, mask_choice)
    """
    if state.current_image is None:
        return None, "No image loaded", gr.update()
    if state.segmenter is None:
        return (None, "⚠️ SAM is not initialised — load a checkpoint on the Setup tab",
                gr.update())

    progress(0.4, desc="Segmenting...")
    try:
        return segment_with_sam(progress)
    except Exception as exc:
        return None, f"❌ Segmentation error: {exc}", gr.update()


def open_current_image(progress=gr.Progress()):
    """
    Everything that should happen when an image becomes the current one.

    Settles scale and frame geometry, then segments. Bundled so the gallery, the
    Next button and a session restore all take exactly the same path.

    Returns:
        tuple: (canvas_payload, tier1_status, scale_summary, point_readout,
                frame_status, header, mask_viz, segment_status, mask_choice,
                tool, point_type, working view)
    """
    payload, tier1, summary, hint, frame = prepare_scale_tab()
    mask_viz, segment_status, choice = auto_process_current_image(progress)
    # The header is built last, so it can report the particle count. The two
    # tool controls are set to what reset_image_state left in state, so the
    # screen cannot show Merge while clicks queue deletions. The working view
    # shows the new frame plain: left alone, it kept the previous image's
    # outlines under the new image's candidates until one was chosen.
    frame_view = state.cropped_image if state.cropped_image is not None else state.current_image
    return (payload, tier1, summary, hint, frame, frame_header(),
            mask_viz, segment_status, choice, state.click_mode, state.point_type,
            for_display(frame_view))


def restore_working_tab():
    """
    The Segment & Refine tab as the session has it, for a browser connecting
    mid-image. A refresh used to show it empty, with three candidates on offer
    whatever SAM had actually returned.

    Returns:
        tuple: (mask_viz, segment_status, mask_choice, working view, measurements,
                header, candidate panel)
    """
    from .refinement import get_current_visualization
    from ..visualization import create_results_dataframe, visualize_three_masks

    if state.current_image is None:
        return (None, "", gr.update(choices=[], value=None), None, None,
                frame_header(), gr.update())

    frame = state.cropped_image if state.cropped_image is not None else state.current_image
    if state.masks:
        labels = [f"Option {i + 1}" for i in range(len(state.masks))]
        chosen = state.selected_mask_index
        value = labels[chosen] if chosen is not None and chosen < len(labels) else labels[0]
        candidates = visualize_three_masks(frame, state.masks, labels)
        choice = gr.update(choices=labels, value=value)
        status = f"↻ Reconnected — {len(labels)} candidate(s), best first."
    else:
        candidates, choice = None, gr.update(choices=[], value=None)
        status = "↻ Reconnected — press Segment to get candidates."

    if state.analyzer is not None:
        view = get_current_visualization()
        table = create_results_dataframe(state.analyzer.get_measurements(in_nm=True))
    else:
        view, table = for_display(frame), None
    # Candidates open until one is chosen, as when the image was first opened.
    return (candidates, status, choice, view, table, frame_header(),
            gr.update(open=state.analyzer is None))


def save_and_next(progress=gr.Progress()):
    """
    Save the current image's measurements, then load and prepare the next.

    The next image is calibrated from scratch — nothing is inherited — and the
    analyst stays on the working tab. A forced detour through Scale on every
    frame costs more than it saves over a folder of hundreds, so the interruption
    is made conditional instead: scale_tab.check_outputs, wired after this,
    switches to the Scale tab when the new reading has nothing vouching for it,
    and otherwise offers it for a glance without moving.

    Returns:
        tuple: (save_status, gallery, image_info, canvas_payload, tier1_status,
                scale_summary, point_readout, frame_status, header, mask_viz,
                segment_status, mask_choice)
    """
    save_status, gallery_data = save_current_results()
    if not save_status.startswith("✅"):
        # Nothing was saved — stay put rather than advancing past unsaved work.
        return (save_status,) + (gr.update(),) * (OPEN_COUNT + 2)

    image, info = _load_index(state.current_index + 1)
    if image is None:
        ending = info if info.startswith("❌") else "That was the last image. All done."
        return ((f"{save_status}. {ending}", gallery_data, info)
                + (gr.update(),) * OPEN_COUNT)

    return (save_status, gallery_data, info) + open_current_image(progress)


def skip_to_next(progress=gr.Progress()):
    """
    Advance without saving, for images that aren't worth measuring.

    Returns:
        tuple: (image_info, canvas_payload, tier1_status, scale_summary,
                point_readout, frame_status, header, mask_viz, segment_status,
                mask_choice)
    """
    image, info = _load_index(state.current_index + 1)
    if image is None:
        return (info,) + (gr.update(),) * OPEN_COUNT
    return (info,) + open_current_image(progress)
