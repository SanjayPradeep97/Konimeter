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
from sem_particle_analysis.data_manager import parse_measurement_list
from sem_particle_analysis.dataset import DIRNAME, TrainingSet, instance_labels, sha256_file

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
        # An earlier copy of this frame would now disagree with its row.
        TrainingSet.beside(state.results_manager.csv_file).remove(
            os.path.splitext(filename)[0])
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
            modality=state.microscope_kind() or "",
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
        # gr.update() rather than None on every failure: None would empty the
        # gallery, which looks like the folder's work was lost.
        if state.analyzer is None:
            return "❌ Nothing to save yet: choose a candidate first.", gr.update()

        if state.results_manager is None:
            return "❌ No folder is open. Open one on the Setup tab.", gr.update()

        # Measured with the scale in force now, whatever it was when the
        # candidate was chosen, so the row's numbers and its scale_method agree.
        state.analyzer.conversion = (state.scale_info or {}).get("conversion")
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

        note = save_training_frame(filename, measurements, scale_method)

        # Update gallery
        gallery_data = create_image_gallery()

        return (
            f"✅ Saved results for {filename}{note}",
            gallery_data
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update()


def get_session_summary():
    """
    The whole folder's results: progress, particle statistics, and the list of
    saved frames to delete from.

    Returns:
        tuple: (table, progress markdown, statistics markdown, delete-list update)
    """
    no_choices = gr.update(choices=[], value=None)
    try:
        if state.results_manager is None:
            return gr.update(), "No folder is open.", "No particle statistics yet.", no_choices

        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            empty = results_df[['file_name', 'num_particles']].set_axis(
                ['Filename', 'Particle Count'], axis=1)
            return (empty, "Nothing saved in this folder yet.",
                    "No particle statistics yet.", no_choices)

        summary = state.results_manager.get_summary()
        progress_text = f"""
### Progress

- **Images saved:** {summary['total_images']} of {len(state.image_paths)}
- **Particles:** {summary['total_particles']}
- **Particles per image:** {summary['avg_particles_per_image']:.1f} on average, {summary['min_particles']} to {summary['max_particles']}
        """

        # Only frames with a scale have sizes in nm; the rest were measured in
        # pixels and cannot be pooled with them. Say how many are left out
        # rather than leaving them out silently.
        all_diameters = []
        for _, row in results_df.iterrows():
            all_diameters.extend(parse_measurement_list(row['equiv_diameters_nm']))
        total = int(summary['total_particles'])

        if all_diameters:
            um = np.asarray(all_diameters) / 1000
            particle_stats_text = f"""
### Particle sizes (equivalent diameter)

- **Mean:** {um.mean():.3f} μm
- **Median:** {np.median(um):.3f} μm
- **Standard deviation:** {um.std():.3f} μm
- **Range:** {um.min():.3f} to {um.max():.3f} μm
            """
            if len(all_diameters) < total:
                particle_stats_text += (
                    f"\nFrom {len(all_diameters)} of {total} particles: the rest are "
                    "in frames saved without a scale, measured in pixels.")
        else:
            particle_stats_text = ("### Particle sizes\n\nNo frame has been saved with a "
                                   "scale yet, so there are no sizes in nm.")

        display_df = results_df[['file_name', 'num_particles']].copy()
        display_df.columns = ['Filename', 'Particle Count']

        # By name: positions change whenever the file is rewritten.
        names = [str(name) for name in results_df['file_name']]
        return (display_df, progress_text, particle_stats_text,
                gr.update(choices=names, value=None))

    except Exception as e:
        return gr.update(), f"❌ Error: {str(e)}", "", gr.update()


def delete_result_row(selected_item):
    """
    Delete one image's results, and its training data, by file name.

    Returns:
        tuple: (table, progress, statistics, delete-list update, status, gallery)
    """
    unchanged = (gr.update(),) * 4
    try:
        if state.results_manager is None:
            return unchanged + ("❌ No folder is open.", gr.update())
        if not selected_item:
            return unchanged + ("❌ Choose an image to delete first.", gr.update())

        name = str(selected_item)
        if not state.results_manager.delete_frame(name):
            return unchanged + (f"❌ {name} is no longer in the results file. "
                                "Press Refresh.", gr.update())
        # The dataset holds the frames the results file does.
        TrainingSet.beside(state.results_manager.csv_file).remove(os.path.splitext(name)[0])

        # Ticks are rebuilt from the file, so the deleted frame loses its tick.
        state.processed_images = {}
        state.sync_processed_from_csv()

        return (get_session_summary() + (f"✅ Deleted the results for {name}",
                                         create_image_gallery()))
    except Exception as e:
        return unchanged + (f"❌ Error: {str(e)}", gr.update())


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

        rows = f"{len(table)} row{'' if len(table) == 1 else 's'}"
        return str(copy), (f"✅ {rows} ready to download. The working copy stays "
                           f"at {source}")
    except Exception as problem:
        return None, f"❌ Export failed: {problem}"


def check_and_remove_duplicates():
    """Check for duplicate entries and remove them."""
    try:
        if state.results_manager is None:
            return gr.update(), "❌ No folder is open."

        # Find duplicates
        duplicates = state.results_manager.find_duplicates()

        if len(duplicates) == 0:
            return gr.update(), "✅ No duplicate entries found"

        # Delete duplicates (keep last occurrence)
        deleted_count = state.results_manager.delete_duplicates(keep='last')

        # Get updated summary
        results_df = state.results_manager.get_results()
        if len(results_df) == 0:
            display_df = None
        else:
            display_df = results_df[['file_name', 'num_particles']].copy()
            display_df.columns = ['Filename', 'Particle Count']

        entries = f"{deleted_count} duplicate entr{'y' if deleted_count == 1 else 'ies'}"
        return display_df, f"✅ Removed {entries}, keeping the latest of each"

    except Exception as e:
        return gr.update(), f"❌ Error: {str(e)}"
