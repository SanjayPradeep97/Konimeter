"""
Plotting tab: session-wide histograms.
"""

from sem_particle_analysis.data_manager import parse_measurement_list

from ..state import state

# The choices on the Plots tab, and whether each draws the size axis in log.
AXIS_CHOICES = {"Log": True, "Linear": False}


def update_histogram_plots(axis="Log"):
    """
    Generate histogram plots for all particles from all images in the session.

    Args:
        axis (str): "Log" or "Linear", for the size axis.
    """
    try:
        if state.results_manager is None:
            return None, "❌ No results manager initialized. Save some image results first."

        # Get all results
        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, "❌ No saved results to plot. Save at least one image's results first."

        # Only frames saved with a scale have sizes in nm. Those saved without
        # one are in pixels and cannot share an axis with them, so they are left
        # out, and the status says how many.
        all_areas, all_diameters = [], []
        for _, row in results_df.iterrows():
            all_areas.extend(parse_measurement_list(row['particle_areas_nm2']))
            all_diameters.extend(parse_measurement_list(row['equiv_diameters_nm']))
        total = int(results_df['num_particles'].sum())

        if not all_areas:
            return None, ("❌ No frame has been saved with a scale yet, so there "
                          "are no sizes in nm to plot.")

        combined_measurements = {
            'num_particles': len(all_areas),
            'areas': all_areas,
            'diameters': all_diameters,
            'unit': 'nm'
        }

        # Relative, like every other import in this package. As an absolute
        # import this resolved only if the working directory happened to be
        # sem_analysis_app, so the Plots tab reported "No module named
        # 'visualization'" wherever the app is actually launched from.
        from ..visualization import create_histogram_plots
        histogram_img = create_histogram_plots(combined_measurements,
                                               log_scale=AXIS_CHOICES.get(axis, True))

        images = len(results_df)
        status = (f"✅ {len(all_areas)} particle{'' if len(all_areas) == 1 else 's'} "
                  f"from {images} image{'' if images == 1 else 's'}")
        if len(all_areas) < total:
            status += (f". {total - len(all_areas)} more are in frames saved without "
                       "a scale, measured in pixels, and are not plotted.")
        return histogram_img, status

    except Exception as e:
        return None, f"❌ Error: {str(e)}"
