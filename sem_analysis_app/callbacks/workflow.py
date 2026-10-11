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


def _load_index(index):
    """Load image at ``index`` into state. Returns (image, info) or (None, error)."""
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
        return None, f"❌ Error loading image: {exc}"

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
                frame_status, header, mask_viz, segment_status, mask_choice)
    """
    payload, tier1, summary, hint, frame = prepare_scale_tab()
    mask_viz, segment_status, choice = auto_process_current_image(progress)
    # The header is built last, so it can report the particle count.
    return (payload, tier1, summary, hint, frame, frame_header(),
            mask_viz, segment_status, choice)


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
    blank = (gr.update(),) * 11
    save_status, gallery_data = save_current_results()
    if gallery_data is None:
        # Nothing was saved — stay put rather than advancing past unsaved work.
        return (save_status,) + blank

    image, info = _load_index(state.current_index + 1)
    if image is None:
        return ((f"{save_status} — {info}. All done.", gallery_data, info)
                + (gr.update(),) * 9)

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
        return (info,) + (gr.update(),) * 9
    return (info,) + open_current_image(progress)
