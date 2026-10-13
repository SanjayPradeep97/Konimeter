"""
Image gallery tab: thumbnails and selection.
"""
import os
import gradio as gr
from PIL import Image

from sem_particle_analysis.utils import load_image
from ..state import state
from ..visualization import for_display

# Gallery tiles are ~200px on screen; anything larger is wasted bytes.
THUMBNAIL_PX = (320, 320)

# Thumbnails by path, kept while the file is unchanged. The gallery is rebuilt
# after every save, and decoding every frame in a folder of hundreds each time
# made saving slower the further through a folder the analyst got.
_thumbnails = {}


def _thumbnail(path):
    """A small copy of one image, or a grey tile saying it could not be read."""
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        stamp = None
    cached = _thumbnails.get(path)
    if cached is not None and cached[0] == stamp and stamp is not None:
        return cached[1], True
    try:
        # load_image, so a 16-bit frame looks as it will when opened.
        image = Image.fromarray(load_image(path))
        image.thumbnail(THUMBNAIL_PX, Image.LANCZOS)
        _thumbnails[path] = (stamp, image)
        return image, True
    except Exception as problem:
        print(f"Could not read {path}: {problem}")
        return Image.new("RGB", (THUMBNAIL_PX[0], THUMBNAIL_PX[0] * 3 // 4), (90, 90, 90)), False


def create_image_gallery():
    """
    One tile per image, in the order of ``state.image_paths``, ticked once saved.

    An image that cannot be read still gets a tile, a grey one, rather than being
    skipped: tiles are opened by position, so skipping one made every later tile
    open the image after it.
    """
    gallery_items = []
    for idx, img_path in enumerate(state.image_paths):
        image, readable = _thumbnail(img_path)
        filename = os.path.basename(img_path)
        if not readable:
            label = f"⚠️ {filename} (cannot be read)"
        elif state.is_processed(idx):
            particles = state.processed_images[idx].get("num_particles", "?")
            label = f"✅ {filename} ({particles} particle{'' if particles == 1 else 's'})"
        else:
            label = f"⚪ {filename}"
        gallery_items.append((image, label))
    return gallery_items


def select_image_from_gallery(evt: gr.SelectData):
    """
    Open the clicked image and go to Scale & Frame.

    Scale first, because every measurement is a pixel count times that number
    and it is worth a glance before anything is measured with it. It also puts
    the canvas on screen at the moment the image arrives, which is when it can
    be drawn — landing on Segment & Refine instead left the scale tab's canvas
    unmounted, so it was blank and could not be dragged on when finally opened.
    """
    if not state.image_paths:
        return None, "No images loaded", gr.Tabs(selected=1), "delete"

    state.current_index = evt.index
    state.reset_image_state()

    try:
        state.current_image = load_image(state.image_paths[state.current_index])
        filename = os.path.basename(state.image_paths[state.current_index])
        info = f"Image {state.current_index + 1} / {len(state.image_paths)}: {filename}"
        return for_display(state.current_image), info, gr.Tabs(selected=2), "delete"
    except Exception as e:
        return None, f"❌ Error loading image: {str(e)}", gr.Tabs(selected=1), "delete"


def restore_session():
    """
    Repopulate the gallery when a browser connects to an already-loaded session.

    The app keeps one process-wide state, so images loaded earlier are still
    there after a refresh — but the gallery component is per-connection and comes
    up empty, which looks like the work was lost.

    Returns:
        tuple: (gallery_items, status_text)
    """
    if not state.image_paths:
        return [], ""

    done = sum(1 for i in range(len(state.image_paths)) if state.is_processed(i))
    count = len(state.image_paths)
    status = f"✅ {count} image{'' if count == 1 else 's'} open"
    if done:
        status += f", {done} already analysed"
    if state.results_manager is not None:
        status += f". Results are saved to {state.results_manager.csv_file}"
    return create_image_gallery(), status
