"""
Image gallery tab: thumbnails and selection.
"""
import os
import gradio as gr
from PIL import Image

from sem_particle_analysis.utils import load_image
from ..state import state

# Gallery tiles are ~200px on screen; anything larger is wasted bytes.
THUMBNAIL_PX = (320, 320)

def create_image_gallery():
    """Create gallery view with processing status."""
    if not state.image_paths:
        return []

    gallery_items = []
    for idx, img_path in enumerate(state.image_paths):
        # Create thumbnail
        try:
            # Verify file exists and is readable
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue

            # Load image as PIL Image for Gradio Gallery
            # This ensures compatibility across Windows and macOS
            img = Image.open(img_path)

            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Actually make a thumbnail. Handing Gradio the full frame means
            # serialising four megapixels per tile — a folder of 2048px
            # micrographs then takes so long the gallery looks hung.
            img.thumbnail(THUMBNAIL_PX, Image.LANCZOS)

            filename = os.path.basename(img_path)

            # Add status indicator to filename
            if state.is_processed(idx):
                particles = state.processed_images[idx].get("num_particles", "?")
                label = f"✅ {filename} ({particles} particles)"
            else:
                label = f"⚪ {filename}"

            # Gradio Gallery expects (image, caption) tuples where image is PIL Image or path
            gallery_items.append((img, label))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

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
        return state.current_image, info, gr.Tabs(selected=2), "delete"
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
    status = f"✅ {len(state.image_paths)} images in this session"
    if done:
        status += f", {done} already analysed"
    if state.results_manager is not None:
        status += f" | results in {state.results_manager.csv_file}"
    return create_image_gallery(), status
