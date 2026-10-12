"""
Setup tab: SAM initialisation, and opening a folder of images with its results.
"""
import os
import gradio as gr

from sem_particle_analysis import (SAMModel, ScaleDetector, ParticleSegmenter,
                                   ResultsManager, discover_checkpoints,
                                   infer_model_type)
from sem_particle_analysis import weights
from sem_particle_analysis.utils import find_images_in_folder
from ..state import state

# What the checkpoint list says, depending on whether it found any.
CHECKPOINTS_FOUND = "Found in sam_weights/. Paste a path to use one from elsewhere."
CHECKPOINTS_MISSING = ("None yet: download the model with the button below, or paste "
                       "the path of a checkpoint you already have.")
from .gallery import create_image_gallery

def resolve_checkpoint(value):
    """
    Turn whatever the checkpoint dropdown hands back into a real path.

    Depending on version and settings, a Gradio dropdown may return the value, a
    single-item list, or the displayed label. Rather than depend on which, fall
    back to matching the text against the checkpoints actually on disk.

    Returns:
        str or None: An existing path, or None if nothing matches.
    """
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None

    text = os.path.expanduser(str(value)).strip()
    if os.path.exists(text):
        return text
    return next((str(p) for p in discover_checkpoints() if p.name in text), None)


def initialize_sam(sam_path, progress=gr.Progress()):
    """Load the SAM checkpoint chosen on the Setup tab."""
    try:
        sam_path = resolve_checkpoint(sam_path)
        if sam_path is None:
            return ("❌ No checkpoint at that path. Pick one from the list, or "
                    "download the model with the button below."), gr.update(interactive=False)

        # Derived from the filename rather than chosen separately, so the
        # architecture can never disagree with the weights being loaded.
        model_type = infer_model_type(sam_path)

        progress(0.3, desc="Loading SAM model...")
        state.sam_model = SAMModel(sam_path, model_type=model_type)
        state.segmenter = ParticleSegmenter(state.sam_model)

        progress(0.7, desc="Initializing scale detector...")
        state.scale_detector = ScaleDetector(use_gpu=False)

        progress(1.0, desc="Complete!")

        # Named, because falling back to the CPU is otherwise silent and only
        # shows as segmentation taking ten times longer.
        device = state.sam_model.device.type
        status = f"✅ SAM model loaded ({model_type}, on {device})"
        if device == "cpu":
            status += " — no GPU found, so segmenting will be slow"
        return status, gr.update(interactive=True)

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


RESULTS_FILENAME = "analysis_results.csv"


def clean_path(text):
    """
    A path as the analyst pasted it: absolute, with ``~`` expanded.

    Explorer's "Copy as path" wraps the path in double quotes and Finder's
    Option-Command-C does not, so surrounding quotes and whitespace are dropped.

    Returns:
        str: The path, or "" for empty input.
    """
    text = str(text or "").strip().strip('"').strip("'").strip()
    return os.path.abspath(os.path.expanduser(text)) if text else ""


def download_model(progress=gr.Progress()):
    """
    Download SAM ViT-B into Konimeter's weights folder, then load it.

    For a first start, when there is nothing to load. The file goes where the
    app looks every time it starts, so this is needed once.

    Returns:
        tuple: (status, checkpoint list update, download button update,
        Load images button update, Load model button update)
    """
    def report(done, total):
        if total:
            progress(done / total, desc=f"Downloading SAM ViT-B: {done / 2**20:.0f} "
                                        f"of {total / 2**20:.0f} MB")

    try:
        path = weights.download("vit_b", progress=report)
    except Exception as problem:
        return (f"❌ The download failed ({problem}). Check the internet connection "
                "and try again.", gr.update(), gr.update(visible=True), gr.update(),
                gr.update())

    status, load_images = initialize_sam(str(path), progress=progress)
    return (status,
            gr.update(choices=[str(p) for p in discover_checkpoints()], value=str(path),
                      info=CHECKPOINTS_FOUND),
            gr.update(visible=False),
            load_images,
            gr.update(variant="primary"))


def set_save_dataset(enabled):
    """Turn keeping each saved frame as training data on or off."""
    state.save_dataset = bool(enabled)
    return state.save_dataset


def current_save_dataset():
    """The setting, for a browser connecting to a session already running."""
    return state.save_dataset


def load_images_from_folder(folder, results_file="", progress=gr.Progress()):
    """
    Open every image in one folder, and the results file that belongs to it.

    The images are read where they are, by path. A browser upload would hand the
    app a copy in Gradio's temporary cache instead, which is where the results
    file would then have been written — somewhere the operating system clears on
    its own schedule, and nowhere the analyst would look for it.

    Results go to ``analysis_results.csv`` in the folder unless ``results_file``
    names another file (or a folder to put it in). An existing file is carried
    on with, so opening a folder again resumes it.

    Returns:
        tuple: (status text, gallery items or None)
    """
    try:
        folder = clean_path(folder)
        if not folder:
            return "❌ Paste the path of a folder of images.", None
        if not os.path.isdir(folder):
            return f"❌ No folder at {folder}", None

        progress(0.3, desc="Finding images...")
        image_paths = find_images_in_folder(folder)
        if not image_paths:
            return (f"❌ No images in {folder}. Looking for .tif, .tiff, .png, "
                    ".jpg and .jpeg files directly inside it."), None

        csv_path = clean_path(results_file) or os.path.join(folder, RESULTS_FILENAME)
        if os.path.isdir(csv_path):
            csv_path = os.path.join(csv_path, RESULTS_FILENAME)
        try:
            results_manager = ResultsManager(csv_file=csv_path)
        except OSError as e:
            return (f"❌ Cannot write results to {csv_path} ({e.strerror or e}). "
                    "Give a results file somewhere writable under "
                    "'Save results somewhere else'."), None

        state.image_paths = image_paths
        state.current_index = 0
        state.processed_images = {}
        state.results_manager = results_manager

        # Ticks come from the results file, matched by name, so a frame saved
        # in an earlier session shows as done.
        state.load_state()
        matched, unmatched = state.sync_processed_from_csv()

        progress(0.7, desc="Creating gallery...")
        gallery_data = create_image_gallery()

        progress(1.0, desc="Complete!")

        status_parts = [f"✅ {len(image_paths)} images from {folder}",
                        f"results in {csv_path}"]
        if matched > 0:
            status_parts.append(f"{matched} already analysed")
        if unmatched > 0:
            status_parts.append(f"⚠️ {unmatched} rows in the results file match "
                                "no image in this folder")

        return " | ".join(status_parts), gallery_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", None
