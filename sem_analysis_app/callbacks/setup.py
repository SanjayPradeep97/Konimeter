"""
Setup tab: SAM initialisation, and opening a folder of images with its results.
"""
import os
import tempfile
import urllib.error

import gradio as gr

from sem_particle_analysis import (SAMModel, ScaleDetector, ParticleSegmenter,
                                   ResultsManager, discover_checkpoints,
                                   infer_model_type)
from sem_particle_analysis import weights
from sem_particle_analysis.model import weights_dir
from sem_particle_analysis.utils import find_images_in_folder
from ..state import state
from .gallery import create_image_gallery

RESULTS_FILENAME = "analysis_results.csv"

# What the checkpoint list says, depending on whether it found any.
CHECKPOINTS_FOUND = "Found in sam_weights/. Paste a path to use one from elsewhere."
CHECKPOINTS_MISSING = ("None yet: download the model with the button below, or paste "
                       "the path of a checkpoint you already have.")


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

    text = clean_path(value)
    if os.path.isfile(text):
        return text
    return next((str(p) for p in discover_checkpoints() if p.name in str(value)), None)


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
    except urllib.error.URLError as problem:   # before OSError, which it is too
        return (f"❌ The download failed ({problem.reason}). "
                "Check the internet connection and try again.",
                gr.update(), gr.update(visible=True), gr.update(), gr.update())
    except OSError as problem:
        return (f"❌ The model could not be saved in {weights_dir()} "
                f"({problem.strerror or problem}). Check that folder can be written to.",
                gr.update(), gr.update(visible=True), gr.update(), gr.update())
    except Exception as problem:
        return (f"❌ The download failed ({problem}). Try again.",
                gr.update(), gr.update(visible=True), gr.update(), gr.update())

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


def restore_setup():
    """
    The Setup tab as the session left it, for a browser connecting to a running
    app — after a refresh, say. The page itself is built once, at start-up, so
    without this it shows the start-up state: no model, Load images disabled,
    and the download button back even after a download.

    Returns:
        tuple: (checkpoint list, download button, Load model button, model status,
        Load images button)
    """
    found = [str(p) for p in discover_checkpoints()]
    loaded = state.sam_model is not None
    current = str(state.sam_model.checkpoint) if loaded else (found[0] if found else None)
    if current and current not in found:
        found.append(current)
    status = ""
    if loaded:
        status = (f"✅ SAM model loaded ({state.sam_model.model_type}, on "
                  f"{state.sam_model.device.type})")
    return (gr.update(choices=found, value=current,
                      info=CHECKPOINTS_FOUND if found else CHECKPOINTS_MISSING),
            gr.update(visible=not found),
            gr.update(variant="primary" if found else "secondary"),
            status,
            gr.update(interactive=loaded))


def load_images_from_folder(folder, progress=gr.Progress()):
    """
    Open every image in one folder, and the results file that belongs to it.

    The images are read where they are, by path. A browser upload would hand the
    app a copy in Gradio's temporary cache instead, which is where the results
    file would then have been written — somewhere the operating system clears on
    its own schedule, and nowhere the analyst would look for it.

    Results always go to ``analysis_results.csv`` in the folder, and an existing
    one is carried on with, so opening a folder again resumes it. One folder,
    one results file: a file shared between folders would mix up images that
    happen to have the same names.

    Returns:
        tuple: (status text, gallery update, tabs update)
    """
    unchanged = (gr.update(), gr.update())
    try:
        folder = clean_path(folder)
        if not folder:
            return ("❌ Paste the path of a folder of images.",) + unchanged
        if not os.path.isdir(folder):
            return (f"❌ No folder at {folder}",) + unchanged

        progress(0.3, desc="Finding images...")
        image_paths = find_images_in_folder(folder)
        if not image_paths:
            return ((f"❌ No images in {folder}. Looking for .tif, .tiff, .png, "
                     ".jpg and .jpeg files directly inside it."),) + unchanged

        # Checked before anything changes: a folder that cannot be written to
        # would otherwise fail on the first save, after the analyst's work.
        csv_path = os.path.join(folder, RESULTS_FILENAME)
        try:
            with tempfile.TemporaryFile(dir=folder):
                pass
            if os.path.exists(csv_path):
                with open(csv_path, "a"):
                    pass
            results_manager = ResultsManager(csv_file=csv_path)
        except OSError as e:
            return ((f"❌ Results cannot be saved in {folder} ({e.strerror or e}). "
                     "Copy the images to a folder you can write to, and open that "
                     "one."),) + unchanged

        state.image_paths = image_paths
        state.current_index = 0
        # The image on screen belongs to the folder that was open before;
        # saving it now would file it under this folder's first image.
        state.reset_image_state()
        state.scale_baseline = None
        state.processed_images = {}
        state.results_manager = results_manager

        # Ticks come from the results file, matched by name, so a frame saved
        # in an earlier session shows as done.
        matched, unmatched = state.sync_processed_from_csv()

        progress(0.7, desc="Creating gallery...")
        gallery_data = create_image_gallery()

        count = len(image_paths)
        status_parts = [f"✅ {count} image{'' if count == 1 else 's'} in {folder}. "
                        f"Results are saved to {RESULTS_FILENAME} there."]
        if matched > 0:
            status_parts.append(f"{matched} already analysed.")
        if unmatched > 0:
            status_parts.append(f"⚠️ {unmatched} row{'' if unmatched == 1 else 's'} in "
                                "the results file match no image in this folder.")

        return " ".join(status_parts), gallery_data, gr.Tabs(selected=1)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (f"❌ Error: {str(e)}",) + unchanged
