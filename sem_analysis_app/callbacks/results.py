"""
Results tab: saving, summarising and exporting measurements.
"""
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np

from sem_particle_analysis import __version__ as KONIMETER_VERSION
from sem_particle_analysis.cli import sha256_file
from sem_particle_analysis.data_manager import parse_measurement_list
from sem_particle_analysis.dataset import DIRNAME, TrainingSet, instance_labels

from ..state import state
from .gallery import create_image_gallery
from .segmentation import _particle_polarity


def save_training_frame(filename, measurements, scale_method):
    """
    Keep the frame and its corrected mask as training data, beside the results.

    Every saved frame has been checked by the analyst, so its mask is a label;
    see sem_particle_analysis.dataset for the layout. A failure here is
    reported but does not undo the results row, which is what the analyst
    asked for.

    Returns:
        str: A note for the status line; empty when saving training data is off.
    """
    if not getattr(state, "save_dataset", True):
        return ""
    image = state.cropped_image
    if image is None or state.analyzer is None:
        return " · ⚠️ training data not saved: no segmented frame"
    try:
        source = Path(state.image_paths[state.current_index])
        valid = getattr(state, "analysable_region", None)
        if valid is not None and valid.shape != image.shape[:2]:
            valid = None
        model = getattr(state, "sam_model", None)
        full = getattr(state, "current_image", None)
        TrainingSet.beside(state.results_manager.csv_file).save(
            os.path.splitext(filename)[0],
            image,
            instance_labels(state.analyzer.regions, image.shape),
            valid,
            # Names, not paths: a dataset may be shared, and a path names the
            # analyst's machine. The hash identifies the original.
            source_file=source.name,
            source_sha256=sha256_file(source) if source.is_file() else "",
            rows_trimmed=full.shape[0] - image.shape[0] if full is not None else 0,
            nm_per_px=measurements.get("nm_per_px"),
            scale_method=scale_method,
            modality=state.modality.kind if state.modality is not None else "",
            particles={True: "dark", False: "bright"}.get(_particle_polarity(), "contrast"),
            model=getattr(model, "model_type", ""),
            checkpoint=Path(model.checkpoint).name if model is not None else "",
            saved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            konimeter_version=KONIMETER_VERSION,
        )
        return f" · image and mask added to {DIRNAME}/"
    except Exception as problem:
        return f" · ⚠️ training data not saved: {problem}"

def save_current_results(file_name=None):
    """
    Save current image results and mark as processed.

    Args:
        file_name (str, optional): What to record the frame as. Defaults to the
            basename of the open file. A caller working from converted
            copies passes the original's name, so its rows line up with
            results recorded from the originals.

    Saving a frame that has already been saved replaces its row. Going back to
    a frame to correct it is normal work, and it used to leave two rows behind:
    the same frame counted twice in every total drawn from the file, with only
    the Remove duplicates button — which nothing prompts anyone to press — to
    catch it.
    """
    try:
        if state.analyzer is None:
            return "❌ No analysis to save", None

        if state.results_manager is None:
            return "❌ Results manager not initialized", None

        # Get measurements
        measurements = state.analyzer.get_measurements(in_nm=True)

        # How the scale was arrived at goes in with the numbers it produced. An
        # OCR reading nobody checked and a pixel size out of the file are the
        # same float in the nm_per_px column, and only one of them can be wrong
        # by a factor of seven.
        calibration = getattr(state, "scale_calibration", None)
        scale_method = calibration.provenance if calibration is not None else "none"

        # Save to CSV
        filename = file_name or os.path.basename(
            state.image_paths[state.current_index])
        state.results_manager.add_result(filename, measurements,
                                         scale_method=scale_method,
                                         replace=True)

        # Mark as processed
        state.mark_processed(state.current_index, measurements['num_particles'])
        state.save_state()

        note = save_training_frame(filename, measurements, scale_method)

        # Update gallery
        gallery_data = create_image_gallery()

        return (
            f"✅ Saved results for {filename}{note}",
            gallery_data
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", None


def get_session_summary():
    """Get summary of all processed images with aggregate particle statistics."""
    try:
        if state.results_manager is None:
            return None, "No results available", "No particle statistics available", gr.update(choices=[], value=None)

        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, "No images processed yet", "No particle statistics available", gr.update(choices=[], value=None)

        summary = state.results_manager.get_summary()

        # Progress summary (left column)
        progress_text = f"""
### Session Progress

- **Images Processed:** {summary['total_images']} / {len(state.image_paths)}
- **Total Particles:** {summary['total_particles']}
- **Average Particles/Image:** {summary['avg_particles_per_image']:.1f}
- **Min Particles/Image:** {summary['min_particles']}
- **Max Particles/Image:** {summary['max_particles']}
        """

        # Aggregate particle statistics (right column)
        # Collect all particle diameters from all images
        all_diameters = []
        for idx, row in results_df.iterrows():
            all_diameters.extend(parse_measurement_list(row['equiv_diameters_nm']))

        if len(all_diameters) > 0:
            mean_diam = np.mean(all_diameters) / 1000  # Convert to μm
            median_diam = np.median(all_diameters) / 1000
            std_diam = np.std(all_diameters) / 1000
            min_diam = np.min(all_diameters) / 1000
            max_diam = np.max(all_diameters) / 1000

            particle_stats_text = f"""
### Aggregate Particle Statistics

- **Mean Diameter:** {mean_diam:.3f} μm
- **Median Diameter:** {median_diam:.3f} μm
- **Std Deviation:** {std_diam:.3f} μm
- **Min Diameter:** {min_diam:.3f} μm
- **Max Diameter:** {max_diam:.3f} μm
            """
        else:
            particle_stats_text = "### Aggregate Particle Statistics\n\nNo particle data available"

        display_df = results_df[['file_name', 'num_particles']].copy()
        display_df.columns = ['Filename', 'Particle Count']

        # Create dropdown choices for delete functionality
        # Format: "index: filename" so user can see both
        dropdown_choices = [f"{i}: {row['file_name']}" for i, row in results_df.iterrows()]

        return display_df, progress_text, particle_stats_text, gr.update(choices=dropdown_choices, value=None)

    except Exception as e:
        return None, f"❌ Error: {str(e)}", f"❌ Error: {str(e)}", gr.update(choices=[], value=None)


def delete_result_row(selected_item):
    """Delete a specific row from the results table."""
    try:
        if state.results_manager is None:
            return None, None, None, gr.update(), "❌ No results available"

        if selected_item is None:
            return None, None, None, gr.update(), "❌ Please select a row to delete"

        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, None, None, gr.update(), "❌ No results to delete"

        # Parse the selection format "index: filename"
        idx_str = selected_item.split(":")[0].strip()
        idx = int(idx_str)

        # Get filename for confirmation message
        filename = results_df.loc[idx, 'file_name']

        # Delete the row, and the frame's training data with it, so the dataset
        # holds the frames the results file does.
        state.results_manager.delete_result(idx)
        TrainingSet.beside(state.results_manager.csv_file).remove(
            os.path.splitext(str(filename))[0])

        # Get updated summary (includes new dropdown choices)
        display_df, progress_text, particle_stats_text, dropdown_update = get_session_summary()

        return display_df, progress_text, particle_stats_text, dropdown_update, f"✅ Deleted: {filename}"

    except IndexError as e:
        return None, None, None, gr.update(), f"❌ Index error: {str(e)}"
    except Exception as e:
        return None, None, None, gr.update(), f"❌ Error: {str(e)}"


def export_results():
    """
    Hand over a copy of the results to download.

    A copy, in a temporary directory, rather than the results file itself.
    Gradio serves only files under paths it has been allowed, and the results
    normally sit beside the images on another drive — so returning the live file
    made the download button do nothing at all, with no error anywhere the
    analyst could see it.

    Returns:
        tuple: (file to download or None, what happened)
    """
    try:
        if state.results_manager is None:
            return None, "❌ Nothing to export — no results file is open."
        table = state.results_manager.get_results()
        if len(table) == 0:
            return None, "❌ Nothing to export yet — save a frame first."

        source = Path(state.results_manager.csv_file)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        copy = Path(tempfile.mkdtemp(prefix="export-")) / f"{source.stem}_{stamp}.csv"
        table.to_csv(copy, index=False)

        return str(copy), (f"✅ {len(table)} rows ready to download. The working "
                           f"copy stays at {source}")
    except Exception as problem:
        return None, f"❌ Export failed: {problem}"


def check_and_remove_duplicates():
    """Check for duplicate entries and remove them."""
    try:
        if state.results_manager is None:
            return None, "❌ No results manager initialized"

        # Find duplicates
        duplicates = state.results_manager.find_duplicates()

        if len(duplicates) == 0:
            return None, "✅ No duplicate entries found"

        # Delete duplicates (keep last occurrence)
        deleted_count = state.results_manager.delete_duplicates(keep='last')

        # Get updated summary
        results_df = state.results_manager.get_results()
        if len(results_df) == 0:
            display_df = None
        else:
            display_df = results_df[['file_name', 'num_particles']].copy()
            display_df.columns = ['Filename', 'Particle Count']

        return display_df, f"✅ Removed {deleted_count} duplicate entries (kept last occurrence)"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"
