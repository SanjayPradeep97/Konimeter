"""
Gradio interface definition: tabs, controls and event wiring.

Six tabs, in the order the work happens: set up, choose an image, settle the
frame, find the particles, read the numbers, plot them.

Segmentation and refinement share a tab because they are one task — you judge a
mask by correcting it, and splitting them would mean hopping between tabs for
every image in a folder of hundreds. Scale lives on exactly one tab, because two
places to set it would mean two answers.
"""
import gradio as gr

from sem_particle_analysis import discover_checkpoints
from .callbacks.refinement import TOOL_HINTS
from .callbacks.scale_tab import POINTS_HINT, UNITS

from .callbacks import (
    # setup + gallery
    browse_for_folder,
    initialize_sam,
    restore_session,
    restore_scale_tab,
    restore_setup,
    set_microscope,
    current_save_dataset,
    set_save_dataset,
    download_model,
    CHECKPOINTS_FOUND,
    CHECKPOINTS_MISSING,
    current_frame_controls,
    select_image_from_gallery,
    # scale and frame
    accept_from_work_tab,
    apply_two_points,
    check_outputs,
    clear_crop_override,
    confirm_scale,
    detect_scale_again,
    frame_header,
    frame_summary,
    measurement_summary,
    work_status,
    live_point_readout,
    read_box_scale,
    set_canvas_mode,
    set_crop_override,
    set_particle_polarity,
    rerank,
    # segmentation + refinement
    choose_candidate,
    segment_with_sam,
    use_best_candidate,
    apply_refinement_changes,
    clear_all_particles,
    clear_edge_particles,
    clear_pending_changes,
    handle_image_click,
    reset_point_refine,
    set_click_mode,
    set_min_particle_size,
    set_point_type,
    toggle_particle_numbers,
    undo_last_action,
    # workflow
    open_current_image,
    restore_working_tab,
    save_and_next,
    skip_to_next,
    # results
    check_and_remove_duplicates,
    delete_result_row,
    export_results,
    get_session_summary,
    save_current_results,
    update_histogram_plots,
)

# The refinement tools, as (value, label). The value is what the callbacks
# already understand; the label is what a person reads. Kept to one word so they
# fit the rail on one line — what each does is spelled out in the status line
# when it is picked, where it is read at the moment it matters.
REFINE_MODES = [
    ("delete", "Remove"),
    ("add", "Add"),
    ("merge", "Merge"),
    ("point_refine", "Redraw"),
]

APP_CSS = """
/* ---- layout ------------------------------------------------------------ */
.tabs {font-size: 15px;}
.tab-nav button {padding: 10px 20px; font-weight: 500;}

/* Data channels between the scale canvas and Python. They must exist in the
   DOM for client code to write to them, so they are hidden rather than absent. */
.scale-channel {display: none !important;}

/* ---- Segment & Refine -------------------------------------------------- */
/* The bar across the top: which image, at what scale, and the way on. */
.work-bar {
    align-items: center;
    padding: 8px 12px;
    border-radius: 10px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
}
.work-head {display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px;}
.wh-name {font-weight: 600; font-size: 15px; margin-right: 4px; word-break: break-all;}
.wh-chip {
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 999px;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    color: var(--body-text-color-subdued);
    white-space: nowrap;
}
.wh-chip.warn {background: rgba(217, 119, 6, .12); border-color: transparent; color: #b45309;}
.wh-chip.strong {font-weight: 600; color: var(--body-text-color);}
.dark .wh-chip.warn {color: #fbbf24;}
.wh-warning {font-size: 13px; margin-top: 6px; color: #b45309;}
.dark .wh-warning {color: #fbbf24;}
.save-status p {font-size: 13px; margin: 0; color: var(--body-text-color-subdued);}
.work-bar button {white-space: nowrap;}

/* The scale reading to look at before measuring, when one needs a look. */
.check-banner {
    align-items: center;
    padding: 8px 12px;
    border-radius: 10px;
    background: rgba(217, 119, 6, .08);
    border: 1px solid rgba(217, 119, 6, .35);
}
.check-banner p {margin: 0; font-size: 14px;}
.check-banner button {white-space: nowrap;}

/* SAM's candidates, small enough to sit three across beside the image. */
.candidates .thumbnail-item {border-radius: 6px;}
.candidates .grid-wrap {overflow: hidden !important; max-height: none !important;}

/* The image being worked on is the point of the page. */
.work-image img {
    border-radius: 8px;
    background: #0d0d0d;
}

/* Numbers read better tabular. */
.gr-dataframe table {font-variant-numeric: tabular-nums; font-size: 13px;}

/* The app's name, on one line, so the work gets the height. */
.app-header {display: flex; align-items: baseline; gap: 12px; padding: 2px 2px 0;}
.app-name {font-size: 22px; font-weight: 700; letter-spacing: -0.01em;}
.app-tagline {font-size: 14px; color: var(--body-text-color-subdued);}

/* A field and its button on one line, the field without a label box. */
.folder-row, .inline-row {align-items: center;}
.folder-row button, .inline-row button {white-space: nowrap;}

/* A label beside a field rather than above it. */
.field-caption {font-size: 13px; color: var(--body-text-color-subdued); white-space: nowrap;}

/* Short guidance under a control. */
.hint, .hint p {font-size: 13px; color: var(--body-text-color-subdued); margin: 0;}

/* ---- Scale & Frame ------------------------------------------------------ */
/* The three panels beside the image: tighter than Gradio's default spacing,
   so all three fit beside the image without scrolling. */
.inspector {gap: 10px !important;}
.inspector > .column {gap: 8px !important;}
.inspector button {white-space: nowrap;}
.panel-head {align-items: center;}
.panel-head button {flex-grow: 0 !important;}
.panel-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--body-text-color-subdued);
}
.scale-stage {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 420px;
    background: #0d0d0d;
    border-radius: 8px;
    overflow: hidden;
}
.scale-stage canvas {display: block; max-width: 100%;}
.scale-head {display: flex; align-items: center; justify-content: space-between; gap: 8px;}
.scale-value {font-size: 26px; font-weight: 600; line-height: 1.2;
              font-variant-numeric: tabular-nums;}
.scale-unit {font-size: 15px; font-weight: 500; color: var(--body-text-color-subdued);}
.scale-source {font-size: 13px; color: var(--body-text-color-subdued); margin-top: 2px;}
.scale-warning {
    font-size: 13px;
    margin-top: 8px;
    padding: 6px 10px;
    border-radius: 6px;
    background: rgba(217, 119, 6, .10);
    color: #92400e;
}
.scale-message p {font-size: 13px; margin: 0;}
.badge {font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px;
        white-space: nowrap;}
.badge-ok {background: rgba(22, 163, 74, .12); color: #15803d;}
.badge-warn {background: rgba(217, 119, 6, .14); color: #b45309;}
.badge-bad {background: rgba(220, 38, 38, .12); color: #b91c1c;}
.dark .badge-ok {color: #4ade80;}
.dark .badge-warn, .dark .kv-warn {color: #fbbf24;}
.dark .badge-bad {color: #f87171;}
.dark .scale-warning {color: #fcd34d;}

/* A list of name / value pairs. Gradio styles tables in an HTML block as a
   grid, so its borders and padding are overridden here. */
.kv {width: 100%; margin: 0 !important; border-collapse: collapse; font-size: 14px;}
.kv, .kv tbody, .kv tr, .kv th, .kv td {border: none !important;
                                        background: transparent !important;}
.kv th, .kv td {padding: 5px 0 !important; text-align: left; vertical-align: top;}
.kv th {
    width: 34%;
    padding-right: 8px !important;
    font-weight: 500 !important;
    color: var(--body-text-color-subdued) !important;
}
.kv tr + tr th, .kv tr + tr td {border-top: 1px solid var(--border-color-primary) !important;}
.kv-note {margin-left: 4px; font-size: 12px; color: var(--body-text-color-subdued);}
.kv-warn {margin-top: 2px; font-size: 12px; color: #b45309;}
.kv-empty {font-size: 13px; color: var(--body-text-color-subdued);}

/* A step heading inside a rail. */
.step-title {
    font-weight: 600;
    font-size: 13px;
    letter-spacing: .04em;
    text-transform: uppercase;
    opacity: .65;
    margin: 2px 0 6px;
}
"""


def create_interface():
    """
    Build the tabbed Gradio interface.

    Theme and CSS are applied at launch (see __main__), which is where Gradio 6
    expects them; passing them to Blocks is deprecated.
    """
    # Gradio keeps a copy of every image it shows in the system temp folder.
    # These are micrographs, often unpublished: delete the copies an hour after
    # they are made (checked hourly), and any left over when the app next starts.
    with gr.Blocks(title="Konimeter", delete_cache=(3600, 3600)) as app:

        gr.HTML('<div class="app-header"><span class="app-name">Konimeter</span>'
                '<span class="app-tagline">Particle analysis for SEM and TEM '
                'micrographs</span></div>', padding=False)

        with gr.Tabs() as tabs:

            # ============================================================
            # TAB 0: Setup
            # ============================================================
            with gr.Tab("⚙️ Setup", id=0):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1 · Model", elem_classes=["step-title"])
                        # A dropdown of checkpoints already on disk, rather than an
                        # upload widget: Gradio copies uploaded files into its
                        # cache, which meant shuffling 2.4 GB on every start.
                        # The architecture is implied by the checkpoint, so there
                        # is no separate Model Type control to contradict it — a
                        # mismatched pair fails deep inside torch with an
                        # unhelpful shape error.
                        _checkpoints = discover_checkpoints()
                        sam_file = gr.Dropdown(
                            label="SAM checkpoint",
                            # Plain path strings, not (label, value) pairs: with
                            # allow_custom_value the dropdown hands back what is
                            # displayed, so a pretty label would arrive at the
                            # callback instead of the path.
                            choices=[str(p) for p in _checkpoints],
                            value=str(_checkpoints[0]) if _checkpoints else None,
                            allow_custom_value=True,
                            info=CHECKPOINTS_FOUND if _checkpoints else CHECKPOINTS_MISSING,
                        )
                        # Offered only while there is nothing to load, and then it
                        # is the thing to press. Saved where the app looks on every
                        # start, so it is needed once.
                        download_btn = gr.Button(
                            "⬇️ Download the SAM model (ViT-B, 358 MB)",
                            variant="primary", size="lg", visible=not _checkpoints)
                        init_sam_btn = gr.Button(
                            "⚡ Load model", size="lg",
                            variant="primary" if _checkpoints else "secondary")
                        init_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column():
                        gr.Markdown("### 2 · Images", elem_classes=["step-title"])
                        microscope_choice = gr.Radio(
                            choices=["SEM", "TEM"], value=None, label="Microscope",
                            info="Particles are expected brighter than the "
                                 "background in SEM, darker in TEM. Filled in "
                                 "from the files when they say.",
                        )
                        # Chosen with the system's own folder dialog, never typed:
                        # the images are read where they are, and the results
                        # saved beside them, which an upload into the browser
                        # cannot do.
                        with gr.Row(equal_height=True, elem_classes=["folder-row"]):
                            folder_shown = gr.Textbox(
                                show_label=False, container=False, interactive=False,
                                max_lines=1, placeholder="No folder chosen", scale=4)
                            browse_btn = gr.Button("Choose folder…", variant="primary",
                                                   scale=0, min_width=160,
                                                   interactive=False)
                        gr.Markdown(
                            "Results are saved in the folder as "
                            "`analysis_results.csv`. Choose the same folder again "
                            "to carry on where you left off.",
                            elem_classes=["hint"])
                        load_status = gr.Textbox(label="Status", interactive=False,
                                                 lines=2)
                        save_dataset_box = gr.Checkbox(
                            value=True,
                            label="Also keep each saved image and its mask as training data",
                            info="Saved to konimeter_dataset/ in the image folder. "
                                 "Nothing leaves this computer.",
                        )

            # ============================================================
            # TAB 1: Gallery
            # ============================================================
            with gr.Tab("🖼️ Gallery", id=1):
                gr.Markdown("Click an image to start on it. ✅ done · ⚪ not yet")
                gallery = gr.Gallery(
                    label="Images",
                    columns=6, rows=3, height=760,
                    object_fit="contain", show_label=False,
                    # The preview overlay has no reliable way out and blocks the
                    # workflow; clicking a thumbnail should select it.
                    allow_preview=False,
                    type="pil",
                )
                selected_image_info = gr.Textbox(label="Selected", interactive=False)

            # ============================================================
            # TAB 2: Scale & Frame
            # ============================================================
            with gr.Tab("📏 Scale & Frame", id=2):
                with gr.Row(equal_height=False):
                    # ---- the image -------------------------------------------
                    with gr.Column(scale=3, min_width=480):
                        # Channels the canvas JS reads from and writes into.
                        # Hidden with CSS rather than visible=False: Gradio 6
                        # omits invisible components from the DOM entirely, and
                        # client code cannot write to an element that isn't there.
                        _chan = dict(elem_classes=["scale-channel"], show_label=False,
                                     container=False)
                        scale_image_in = gr.Textbox(elem_id="scale_image_in", **_chan)
                        scale_box_out = gr.Textbox(elem_id="scale_box_out", **_chan)
                        scale_points_out = gr.Textbox(elem_id="scale_points_out", **_chan)
                        # Its own channel: the measured span changes every time
                        # the scale is re-read, while the image stays the same.
                        scale_span_in = gr.Textbox(elem_id="scale_span_in", **_chan)

                        gr.HTML('<div class="scale-stage">'
                                '<canvas id="scale_canvas"></canvas></div>',
                                padding=False)

                    # ---- what was decided, and the means to change it ----------
                    with gr.Column(scale=2, min_width=340, elem_classes=["inspector"]):
                        with gr.Column(variant="panel"):
                            with gr.Row(elem_classes=["panel-head"]):
                                gr.HTML('<div class="panel-title">Scale</div>',
                                        padding=False)
                                detect_again_btn = gr.Button("Detect again", size="sm",
                                                             scale=0, min_width=110)
                            scale_card = gr.HTML(padding=False)
                            # The number is only half the reading: the other half
                            # is which pixels were counted, and that can only be
                            # judged by looking. Ticks should sit on the ends of
                            # the printed bar — not inside it, not past it.
                            scale_preview = gr.Image(show_label=False, height=104,
                                                     interactive=False, visible=False,
                                                     container=False, buttons=[])
                            # The value read off the bar, to correct before
                            # confirming: OCR misreads the number far more often
                            # than it mismeasures the bar.
                            with gr.Row(equal_height=True, visible=False,
                                        elem_classes=["inline-row"]) as confirm_row:
                                gr.HTML('<div class="field-caption">Printed value</div>',
                                        padding=False, scale=0, min_width=96)
                                printed_value = gr.Number(label="Printed value",
                                                          show_label=False, container=False,
                                                          min_width=80)
                                printed_unit = gr.Dropdown(choices=UNITS, label="Unit",
                                                           show_label=False, container=False,
                                                           min_width=76)
                                confirm_scale_btn = gr.Button("Confirm", variant="primary",
                                                              min_width=90)
                            confirm_message = gr.Markdown("", elem_classes=["scale-message"])

                        with gr.Column(variant="panel"):
                            gr.HTML('<div class="panel-title">Measure the scale bar</div>',
                                    padding=False)
                            canvas_mode = gr.Radio(
                                choices=[("Draw a box", "box"),
                                         ("Click both ends", "points")],
                                value="box", show_label=False, container=False,
                            )
                            with gr.Column(visible=True) as box_controls:
                                gr.Markdown("Drag a box around the scale bar and its "
                                            "label on the image.", elem_classes=["hint"])
                                with gr.Row():
                                    read_box_btn = gr.Button("Read scale bar",
                                                             variant="primary")
                                    reset_box_btn = gr.Button("Reset box")
                            with gr.Column(visible=False) as point_controls:
                                point_readout = gr.Markdown(POINTS_HINT,
                                                            elem_classes=["hint"])
                                with gr.Row(equal_height=True, elem_classes=["inline-row"]):
                                    # A text field: an empty number field shows 0,
                                    # which reads as a value already entered.
                                    bar_value = gr.Textbox(label="Length on the bar",
                                                           show_label=False,
                                                           container=False, max_lines=1,
                                                           placeholder="Length on the bar",
                                                           min_width=110)
                                    bar_unit = gr.Dropdown(choices=UNITS, value="µm",
                                                           label="Unit", show_label=False,
                                                           container=False, min_width=80)
                                with gr.Row():
                                    apply_points_btn = gr.Button("Set scale",
                                                                 variant="primary")
                                    reset_points_canvas_btn = gr.Button("Clear points")
                            scale_message = gr.Markdown("", elem_classes=["scale-message"])

                        with gr.Column(variant="panel"):
                            gr.HTML('<div class="panel-title">Frame</div>', padding=False)
                            frame_status = gr.HTML('<div class="kv-empty">No image open.</div>',
                                                   padding=False)
                            # Off by default and hiding its input: an always-on
                            # number field shows 0, and 0 means "keep the whole
                            # frame" — one absent-minded Apply would leave an SEM
                            # databar in shot to be measured as particles.
                            crop_manual = gr.Checkbox(label="Set the databar crop by hand",
                                                      value=False)
                            with gr.Row(visible=False, equal_height=True) as crop_controls:
                                crop_override = gr.Number(
                                    label="Crop from bottom (%)", value=0,
                                    minimum=0, maximum=40, scale=2,
                                )
                                crop_apply_btn = gr.Button("Apply", scale=1)

            # ============================================================
            # TAB 3: Segment & Refine
            # ============================================================
            with gr.Tab("🔬 Segment & Refine", id=3):
                # Which image, at what scale, and the way on: one bar that stays
                # in view, so saving never needs a scroll past the image.
                with gr.Row(equal_height=True, elem_classes=["work-bar"]):
                    header = gr.HTML(frame_header(), padding=False, scale=4)
                    skip_btn = gr.Button("Skip", scale=0, min_width=80)
                    save_btn = gr.Button("Save", scale=0, min_width=80)
                    save_next_btn = gr.Button("Save & next", variant="primary",
                                              scale=0, min_width=130)
                save_status = gr.Markdown("", elem_classes=["save-status"])

                # Shown only for a reading nothing has vouched for yet. A scale
                # that agrees with one already accepted never raises this, so
                # working through a folder at one magnification is uninterrupted;
                # a scale with a warning does not raise it either, because that
                # one is worth going to the Scale tab for.
                with gr.Row(visible=False, equal_height=True,
                            elem_classes=["check-banner"]) as scale_check_row:
                    scale_check_img = gr.Image(show_label=False, height=72,
                                               interactive=False, container=False,
                                               buttons=[], scale=2, min_width=180)
                    scale_check_msg = gr.Markdown("")
                    check_value = gr.Number(label="Printed value", show_label=False,
                                            container=False, visible=False,
                                            min_width=80, scale=0)
                    check_unit = gr.Dropdown(choices=UNITS, label="Unit",
                                             show_label=False, container=False,
                                             visible=False, min_width=76, scale=0)
                    scale_check_btn = gr.Button("Confirm scale", variant="primary",
                                                scale=0, min_width=140)

                with gr.Row(equal_height=False):
                    # ---- the image --------------------------------------
                    with gr.Column(scale=3, min_width=480):
                        refine_viz = gr.Image(type="numpy", show_label=False,
                                              elem_classes=["work-image"],
                                              height="calc(100vh - 230px)",
                                              buttons=["fullscreen"])

                    # ---- outline, correct, measure ----------------------
                    with gr.Column(scale=2, min_width=340, elem_classes=["inspector"]):
                        with gr.Column(variant="panel"):
                            with gr.Row(elem_classes=["panel-head"]):
                                gr.HTML('<div class="panel-title">Outline</div>',
                                        padding=False)
                                segment_btn = gr.Button("Segment again", size="sm",
                                                        scale=0, min_width=120)
                            # SAM's candidates, best first. Clicking one puts it
                            # to use; the best is in use from the start.
                            candidates = gr.Gallery(show_label=False, container=False,
                                                    columns=3, rows=1, height=152,
                                                    allow_preview=False, buttons=[],
                                                    object_fit="contain",
                                                    interactive=False,
                                                    elem_classes=["candidates"])
                            segment_status = gr.Markdown("", elem_classes=["hint"])
                            particle_choice = gr.Radio(
                                choices=[("Auto", "auto"), ("Darker", "dark"),
                                         ("Brighter", "bright")],
                                value="auto", label="Particles are",
                            )
                            min_particle_size_slider = gr.Slider(
                                minimum=5, maximum=200, value=30, step=1,
                                label="Smallest particle (px)",
                            )

                        with gr.Column(variant="panel"):
                            gr.HTML('<div class="panel-title">Correct</div>',
                                    padding=False)
                            click_mode_radio = gr.Radio(
                                choices=[(label, value) for value, label in REFINE_MODES],
                                value="delete", show_label=False, container=False,
                            )
                            refine_status = gr.Markdown(TOOL_HINTS["delete"],
                                                        elem_classes=["hint"])
                            with gr.Row(visible=False, equal_height=True) as point_refine_controls:
                                point_type_radio = gr.Radio(
                                    choices=[("Include", "positive"),
                                             ("Exclude", "negative")],
                                    value="positive", show_label=False, container=False,
                                )
                                reset_points_btn = gr.Button("Reset points", size="sm",
                                                             scale=0, min_width=110)
                            with gr.Row():
                                apply_btn = gr.Button("Apply", variant="primary",
                                                      min_width=80)
                                undo_btn = gr.Button("Undo", min_width=80)
                                clear_pending_btn = gr.Button("Discard clicks",
                                                              min_width=110)
                            with gr.Accordion("Clean up", open=False):
                                edge_buffer = gr.Slider(
                                    minimum=0, maximum=50, value=10, step=1,
                                    label="Edge margin (px)",
                                )
                                with gr.Row():
                                    clear_edges_btn = gr.Button("Remove edge particles",
                                                                size="sm", min_width=150)
                                    clear_all_btn = gr.Button("Clear all", variant="stop",
                                                              size="sm", min_width=90)

                        with gr.Column(variant="panel"):
                            gr.HTML('<div class="panel-title">Measurements</div>',
                                    padding=False)
                            measure_html = gr.HTML(measurement_summary(), padding=False)
                            show_numbers_checkbox = gr.Checkbox(
                                label="Number the particles on the image", value=True,
                            )
                            with gr.Accordion("Every particle", open=False):
                                refine_results = gr.Dataframe(
                                    show_label=False,
                                    headers=["Particle ID", "Area", "Diameter"])

            # ============================================================
            # TAB 4: Results
            # ============================================================
            with gr.Tab("💾 Results", id=4):
                gr.Markdown("### This image", elem_classes=["step-title"])
                with gr.Row():
                    current_results = gr.Dataframe(label="Particles",
                                                   headers=["Particle ID", "Area", "Diameter"])
                    current_stats = gr.Dataframe(label="Summary",
                                                 headers=["Metric", "Area", "Diameter"])

                gr.Markdown("---")
                gr.Markdown("### Whole session", elem_classes=["step-title"])
                refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                with gr.Row():
                    summary_progress = gr.Markdown("No results yet")
                    summary_particle_stats = gr.Markdown("No particle statistics yet")
                session_table = gr.Dataframe(label="Processed images",
                                             headers=["Filename", "Particle Count"])

                with gr.Accordion("Fix up the results file", open=False):
                    with gr.Row():
                        delete_row_dropdown = gr.Dropdown(
                            label="Row to delete", choices=[], value=None,
                            interactive=True,
                        )
                        delete_row_btn = gr.Button("❌ Delete row", variant="stop")
                    delete_row_status = gr.Textbox(label="", interactive=False,
                                                   show_label=False)
                    with gr.Row():
                        remove_duplicates_btn = gr.Button("🧹 Remove duplicates",
                                                          variant="secondary")
                        duplicates_status = gr.Textbox(label="", interactive=False,
                                                       show_label=False)

                gr.Markdown("---")
                with gr.Row():
                    export_btn = gr.Button("📥 Export all results", variant="primary")
                    export_file = gr.File(label="Download")
                # What Export did, or why it could not: it has nothing to hand
                # over before the first frame is saved, and an empty download
                # box says nothing about that.
                export_status = gr.Textbox(label="", interactive=False,
                                           show_label=False)

            # ============================================================
            # TAB 5: Plots
            # ============================================================
            with gr.Tab("📊 Plots", id=5):
                gr.Markdown("Size and shape of every particle saved in this folder.")
                with gr.Row():
                    update_plots_btn = gr.Button("🔄 Update", variant="primary")
                    plot_axis = gr.Radio(["Log", "Linear"], value="Log", label="Axes")
                    plot_status = gr.Textbox(label="", interactive=False,
                                             show_label=False)
                histogram_plot = gr.Image(label="", type="numpy", show_label=False)

        # ================================================================
        # Event wiring
        # ================================================================

        # Opening an image does the same thing however it was opened: settle the
        # scale and frame, then segment. Kept as one list so the three entry
        # points cannot drift apart.
        OPEN_OUTPUTS = [scale_image_in, scale_message, scale_card, point_readout,
                        frame_status, header, candidates, segment_status,
                        click_mode_radio, point_type_radio, refine_viz]
        # Putting a candidate to use: its outlines, its numbers, what was done.
        USE_OUTPUTS = [refine_viz, refine_results, segment_status, current_results,
                       current_stats]
        # The bar across the working tab and the measurements beside the image,
        # which every edit changes together.
        WORK_STATUS = [header, measure_html]
        RELOAD_CANVAS = "() => { window.SCALE && window.SCALE.load(); }"

        # Everything that shows what the scale is and how much it should be
        # trusted. Run after anything that can change it, so the canvas overlay,
        # the zoomed check, the prompt on the working tab and the tab the analyst
        # is left on cannot disagree about the same reading.
        CHECK_OUTPUTS = [scale_span_in, scale_preview, scale_check_row,
                         scale_check_msg, scale_check_img, printed_value, printed_unit,
                         confirm_row, check_value, check_unit, tabs]

        # Nothing here re-draws the scale canvas on a tab change. Gradio unmounts
        # an inactive tab's panel and builds a fresh canvas when it is shown
        # again, and a Tab's select event does not fire in a way this can hook —
        # measured, not assumed. scale_canvas.js watches for a canvas it has not
        # wired and for a payload it has not drawn, so it recovers by itself. The
        # RELOAD_CANVAS calls below remain because they are free when the tab
        # happens to be mounted; they are not what makes it work.

        # ---- Setup ----
        init_sam_btn.click(initialize_sam, inputs=[sam_file],
                           outputs=[init_status, browse_btn])
        download_btn.click(download_model,
                           outputs=[init_status, sam_file, download_btn, browse_btn,
                                    init_sam_btn])
        browse_btn.click(browse_for_folder, inputs=[microscope_choice],
                         outputs=[folder_shown, load_status, gallery, tabs,
                                  microscope_choice])
        save_dataset_box.change(set_save_dataset, inputs=[save_dataset_box],
                                outputs=[save_dataset_box])

        # A refresh reconnects to the same process-wide session, but the page is
        # built once, at start-up: bring back what the session has rather than
        # showing start-up defaults over it.
        app.load(restore_session, outputs=[gallery, load_status])
        app.load(current_save_dataset, outputs=[save_dataset_box])
        app.load(restore_setup,
                 outputs=[sam_file, download_btn, init_sam_btn, init_status, browse_btn,
                          microscope_choice, folder_shown])
        app.load(current_frame_controls,
                 outputs=[particle_choice, crop_manual, crop_override,
                          min_particle_size_slider])
        # The image being worked on, as it was: its scale and canvas, its
        # candidates, its outlines. Nothing is detected or segmented again.
        app.load(restore_scale_tab,
                 outputs=[scale_image_in, scale_message, scale_card, point_readout,
                          frame_status]
                 ).then(lambda: check_outputs()[:-1] + (gr.update(),),
                        outputs=CHECK_OUTPUTS).then(None, js=RELOAD_CANVAS)
        app.load(restore_working_tab,
                 outputs=[candidates, segment_status, refine_viz, refine_results,
                          header]).then(work_status, outputs=WORK_STATUS)

        # ---- Gallery ----
        gallery.select(
            select_image_from_gallery,
            outputs=[refine_viz, selected_image_info, tabs, click_mode_radio],
        ).then(
            open_current_image, outputs=OPEN_OUTPUTS,
        ).then(use_best_candidate, outputs=USE_OUTPUTS).then(
            work_status, outputs=WORK_STATUS,
        ).then(check_outputs, outputs=CHECK_OUTPUTS).then(None, js=RELOAD_CANVAS)

        # ---- Scale & Frame ----
        canvas_mode.change(
            set_canvas_mode, inputs=[canvas_mode],
            outputs=[box_controls, point_controls],
        ).then(
            None, inputs=[canvas_mode],
            js="(m) => { window.SCALE && window.SCALE.setMode(m); }",
        )

        read_box_btn.click(read_box_scale, inputs=[scale_box_out],
                           outputs=[scale_message, scale_card]).then(
            frame_summary, outputs=[frame_status]).then(
            work_status, outputs=WORK_STATUS).then(
            check_outputs, outputs=CHECK_OUTPUTS)

        # Live feedback as the two points are placed.
        scale_points_out.change(live_point_readout, inputs=[scale_points_out],
                                outputs=[point_readout])

        apply_points_btn.click(
            apply_two_points, inputs=[scale_points_out, bar_value, bar_unit],
            outputs=[scale_message, scale_card]).then(
            frame_summary, outputs=[frame_status]).then(
            work_status, outputs=WORK_STATUS).then(
            check_outputs, outputs=CHECK_OUTPUTS)

        for reset in (reset_box_btn, reset_points_canvas_btn):
            reset.click(None, js="() => { window.SCALE && window.SCALE.reset(); }")

        confirm_scale_btn.click(confirm_scale, inputs=[printed_value, printed_unit],
                                outputs=[confirm_message, scale_card]).then(
            work_status, outputs=WORK_STATUS).then(
            check_outputs, outputs=CHECK_OUTPUTS)
        detect_again_btn.click(
            detect_scale_again,
            outputs=[scale_image_in, scale_message, scale_card, point_readout,
                     frame_status]).then(
            work_status, outputs=WORK_STATUS).then(
            check_outputs, outputs=CHECK_OUTPUTS).then(None, js=RELOAD_CANVAS)

        # Accepting from the working tab, so checking a reading costs a glance
        # and a click rather than a trip to another tab and back.
        scale_check_btn.click(
            accept_from_work_tab, inputs=[check_value, check_unit],
            outputs=CHECK_OUTPUTS + [header, scale_card])

        def _toggle_crop(manual):
            """Show the trim field only while the override is on; off returns to
            measuring the databar straight away rather than waiting for Apply."""
            if manual:
                return gr.update(visible=True), gr.update(), gr.update()
            summary, head = clear_crop_override()
            return gr.update(visible=False), summary, head

        crop_manual.change(_toggle_crop, inputs=[crop_manual],
                           outputs=[crop_controls, frame_status, header])
        crop_apply_btn.click(set_crop_override, inputs=[crop_override],
                             outputs=[frame_status, header])

        # ---- Segment & Refine ----
        min_particle_size_slider.change(set_min_particle_size,
                                        inputs=[min_particle_size_slider],
                                        outputs=[segment_status])
        # The microscope sets which way round particles are expected: re-sort
        # the candidates to match, and use the new best one unless the mask has
        # been corrected. .input, not .change, here and below: restoring a
        # choice on a page load must not re-sort.
        microscope_choice.input(set_microscope, inputs=[microscope_choice],
                                outputs=[microscope_choice]).then(
            rerank, outputs=[candidates, segment_status]).then(
            use_best_candidate, outputs=USE_OUTPUTS).then(
            frame_summary, outputs=[frame_status]).then(
            work_status, outputs=WORK_STATUS)

        # Re-sorts SAM's existing outlines; nothing is re-run.
        particle_choice.input(set_particle_polarity, inputs=[particle_choice],
                              outputs=[candidates, segment_status]).then(
            use_best_candidate, outputs=USE_OUTPUTS).then(
            frame_summary, outputs=[frame_status]).then(
            work_status, outputs=WORK_STATUS)
        segment_btn.click(segment_with_sam, outputs=[candidates, segment_status]).then(
            use_best_candidate, outputs=USE_OUTPUTS).then(
            work_status, outputs=WORK_STATUS)
        candidates.select(choose_candidate, outputs=USE_OUTPUTS).then(
            work_status, outputs=WORK_STATUS)

        click_mode_radio.change(set_click_mode, inputs=[click_mode_radio],
                                outputs=[refine_status, point_refine_controls])
        point_type_radio.change(set_point_type, inputs=[point_type_radio],
                                outputs=[refine_status])
        reset_points_btn.click(reset_point_refine,
                               outputs=[refine_viz, refine_status])
        show_numbers_checkbox.change(toggle_particle_numbers,
                                     inputs=[show_numbers_checkbox],
                                     outputs=[refine_viz])
        refine_viz.select(handle_image_click, outputs=[refine_viz, refine_status])

        apply_btn.click(
            apply_refinement_changes,
            outputs=[refine_viz, refine_results, refine_status,
                     current_results, current_stats],
        ).then(work_status, outputs=WORK_STATUS)

        clear_pending_btn.click(clear_pending_changes,
                                outputs=[refine_viz, refine_status])
        undo_btn.click(undo_last_action,
                       outputs=[refine_viz, refine_results, refine_status,
                                current_results, current_stats]).then(
            work_status, outputs=WORK_STATUS)

        clear_edges_btn.click(
            clear_edge_particles, inputs=[edge_buffer],
            outputs=[refine_viz, refine_results, refine_status,
                     current_results, current_stats],
        ).then(work_status, outputs=WORK_STATUS)
        clear_all_btn.click(
            clear_all_particles,
            outputs=[refine_viz, refine_results, refine_status,
                     current_results, current_stats],
        ).then(work_status, outputs=WORK_STATUS)

        # ---- Save and advance ----
        save_btn.click(save_current_results, outputs=[save_status, gallery])

        save_next_btn.click(
            save_and_next,
            outputs=[save_status, gallery, selected_image_info] + OPEN_OUTPUTS,
        ).then(use_best_candidate, outputs=USE_OUTPUTS).then(
            work_status, outputs=WORK_STATUS,
        ).then(check_outputs, outputs=CHECK_OUTPUTS).then(None, js=RELOAD_CANVAS)

        skip_btn.click(
            skip_to_next, outputs=[selected_image_info] + OPEN_OUTPUTS,
        ).then(use_best_candidate, outputs=USE_OUTPUTS).then(
            work_status, outputs=WORK_STATUS,
        ).then(check_outputs, outputs=CHECK_OUTPUTS).then(None, js=RELOAD_CANVAS)

        # ---- Results ----
        refresh_btn.click(get_session_summary,
                          outputs=[session_table, summary_progress,
                                   summary_particle_stats, delete_row_dropdown])
        delete_row_btn.click(delete_result_row, inputs=[delete_row_dropdown],
                             outputs=[session_table, summary_progress,
                                      summary_particle_stats, delete_row_dropdown,
                                      delete_row_status, gallery])
        remove_duplicates_btn.click(check_and_remove_duplicates,
                                    outputs=[session_table, duplicates_status])
        export_btn.click(export_results, outputs=[export_file, export_status])

        # ---- Plots ----
        update_plots_btn.click(update_histogram_plots, inputs=[plot_axis],
                               outputs=[histogram_plot, plot_status])
        plot_axis.change(update_histogram_plots, inputs=[plot_axis],
                         outputs=[histogram_plot, plot_status])

    # Every callback reads and writes the one process-wide state, so they run one
    # at a time. In separate queues, a gallery click during Save & next's
    # segmentation could finish in between and file one image's mask under
    # another's name.
    for event in app.fns.values():
        event.concurrency_id = "konimeter"
        event.concurrency_limit = 1

    return app
