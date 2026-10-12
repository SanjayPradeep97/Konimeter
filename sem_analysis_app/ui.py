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
from .callbacks.scale_tab import UNITS

from .callbacks import (
    # setup + gallery
    initialize_sam,
    load_images_from_folder,
    restore_session,
    current_save_dataset,
    set_save_dataset,
    download_model,
    CHECKPOINTS_FOUND,
    CHECKPOINTS_MISSING,
    select_image_from_gallery,
    # scale and frame
    accept_from_work_tab,
    apply_two_points,
    check_outputs,
    clear_crop_override,
    clear_scale,
    confirm_scale,
    frame_header,
    live_point_readout,
    read_box_scale,
    set_canvas_mode,
    set_crop_override,
    set_modality,
    set_particle_polarity,
    # segmentation + refinement
    segment_with_sam,
    select_mask_and_analyze,
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
    ("delete", "🚫  Remove"),
    ("add", "➕  Add"),
    ("merge", "🔗  Merge"),
    ("point_refine", "🎯  Redraw"),
]

APP_CSS = """
/* ---- layout ------------------------------------------------------------ */
.tabs {font-size: 15px;}
.tab-nav button {padding: 10px 20px; font-weight: 500;}

/* Data channels between the scale canvas and Python. They must exist in the
   DOM for client code to write to them, so they are hidden rather than absent. */
.scale-channel {display: none !important;}

/* The frame header: one line that says where you are, always in view. */
.frame-header {
    padding: 10px 14px;
    border-radius: 8px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    font-size: 14px;
    line-height: 1.5;
}
.frame-header p {margin: 0;}

/* Tool rail: keep it compact so the image gets the room. */
.tool-rail .gr-form, .tool-rail .gr-box {border: none; background: transparent;}
.tool-rail label {font-size: 13px;}
.tool-rail .gr-button {width: 100%;}

/* The image being worked on is the point of the page. */
.work-image img {
    border-radius: 8px;
    background: #0d0d0d;
}

/* Numbers read better tabular. */
.gr-dataframe table {font-variant-numeric: tabular-nums; font-size: 13px;}

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
    with gr.Blocks(title="Konimeter") as app:

        gr.Markdown("# 🔬 Konimeter · particle analysis for SEM & TEM")
        gr.Markdown(
            "Size and count particles in electron micrographs. Scale, instrument "
            "and analysable area are worked out per image; you judge the mask and "
            "correct it."
        )

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
                        gr.Markdown(
                            "Paste the path of **one sample folder**. Every image "
                            "in it is opened, and results are saved to "
                            "`analysis_results.csv` in that folder. Open the "
                            "folder again later to carry on where you left off."
                        )
                        # A path rather than an upload: an uploaded file reaches
                        # the app as a copy in Gradio's temporary cache, so the
                        # results would have been saved there too.
                        folder_input = gr.Textbox(
                            label="Image folder",
                            max_lines=1,
                            placeholder=r"C:\Data\Sample A   or   /Users/me/Data/Sample A",
                            info="To copy a folder's path: in File Explorer, "
                                 "Shift + right-click it → Copy as path; in "
                                 "Finder, select it and press ⌥⌘C.",
                        )
                        with gr.Accordion("Save results somewhere else", open=False):
                            results_file_input = gr.Textbox(
                                label="Results file",
                                max_lines=1,
                                placeholder="Leave empty to use analysis_results.csv "
                                            "in the image folder",
                                info="For a folder you cannot write to, or to keep "
                                     "adding to a results file from an earlier "
                                     "session. A folder here gets its own "
                                     "analysis_results.csv.",
                            )
                        load_btn = gr.Button("📁 Load images", variant="primary",
                                             size="lg", interactive=False)
                        load_status = gr.Textbox(label="Status", interactive=False)
                        save_dataset_box = gr.Checkbox(
                            value=True,
                            label="Also keep each saved image and its mask as training data",
                            info="Written to konimeter_dataset/ beside the results "
                                 "file: the frame, its corrected mask and a record "
                                 "of how it was made, ready for training a "
                                 "classifier or fine-tuning segmentation. Nothing "
                                 "leaves this computer.",
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
                gr.Markdown(
                    "Every measurement is a pixel count times the scale, so this "
                    "is the one place it is set. Tier 1 runs by itself; fall "
                    "through to 2 or 3 only when it fails."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        tier1_status = gr.Textbox(
                            label="Tier 1 — file metadata", interactive=False, lines=3
                        )
                    with gr.Column(scale=1):
                        scale_summary = gr.Textbox(
                            label="Scale in force", interactive=False, lines=3
                        )
                with gr.Row():
                    confirm_scale_btn = gr.Button("✔️ Confirm scale", variant="primary")
                    clear_scale_btn = gr.Button("✖️ Clear", variant="secondary")

                with gr.Accordion("🔬 Frame — instrument, polarity, analysable area",
                                  open=True):
                    gr.Markdown(
                        "SEM and TEM need opposite handling. The instrument is "
                        "read from the file where it says so. Beam-blocked area — "
                        "an aperture vignette, a grid bar — and any burned-in "
                        "scale bar are left out of measurement, because they "
                        "out-contrast the particles and would be counted instead."
                    )
                    with gr.Row():
                        modality_choice = gr.Dropdown(
                            choices=["auto", "SEM", "TEM"], value="auto",
                            label="Instrument", scale=1,
                        )
                        particle_choice = gr.Dropdown(
                            choices=["auto", "bright", "dark"], value="auto",
                            label="Particles are", scale=1,
                            info="auto follows the instrument",
                        )
                        with gr.Column(scale=1):
                            # Off by default and hiding its input: an always-on
                            # number field shows 0, and 0 means "keep the whole
                            # frame" — one absent-minded Apply would leave an SEM
                            # databar in shot to be measured as particles.
                            crop_manual = gr.Checkbox(
                                label="Override the databar trim", value=False,
                                info="Off = measure it, which is exact whenever "
                                     "the instrument recorded a scan height",
                            )
                            with gr.Row(visible=False) as crop_controls:
                                crop_override = gr.Number(
                                    label="Trim from bottom (%)", value=0,
                                    minimum=0, maximum=40, scale=2,
                                )
                                crop_apply_btn = gr.Button("Apply", size="sm", scale=1)
                    frame_status = gr.Markdown("Load an image to detect the instrument.")

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=3):
                        canvas_mode = gr.Radio(
                            choices=["Tier 2 — draw a box and read it",
                                     "Tier 3 — click both ends of the bar"],
                            value="Tier 2 — draw a box and read it",
                            label="If tier 1 failed, pick a method",
                        )
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

                        gr.HTML(
                            '<div style="width:100%">'
                            '<canvas id="scale_canvas" '
                            'style="max-width:100%;border-radius:8px;'
                            'border:1px solid rgba(128,128,128,0.35)"></canvas>'
                            '</div>'
                        )

                    with gr.Column(scale=2):
                        with gr.Group(visible=True) as tier2_controls:
                            gr.Markdown(
                                "**Tier 2.** Drag on the image to draw a box around "
                                "the scale bar *and* its label. Drag a corner to "
                                "resize, drag the middle to move."
                            )
                            read_box_btn = gr.Button("🔍 Read scale in box",
                                                     variant="primary")

                        with gr.Group(visible=False) as tier3_controls:
                            gr.Markdown(
                                "**Tier 3.** Click each end of the bar. A magnifier "
                                "follows the cursor so you can land on the exact "
                                "pixel, then type the printed length."
                            )
                            point_readout = gr.Textbox(label="Points", interactive=False)
                            with gr.Row():
                                bar_value = gr.Number(label="Printed length", value=None)
                                bar_unit = gr.Dropdown(choices=UNITS, value="µm",
                                                       label="Unit")
                            apply_points_btn = gr.Button("📏 Apply two-point scale",
                                                         variant="primary")

                        reset_canvas_btn = gr.Button("↺ Reset box / points", size="sm")
                        tier_status = gr.Textbox(label="Result", interactive=False,
                                                 lines=3)
                        # The number is only half the reading: the other half is
                        # which pixels were counted, and that can only be judged
                        # by looking. Ticks should sit on the ends of the printed
                        # bar — not inside it, not past it.
                        scale_preview = gr.Image(
                            label="What was measured", height=190,
                            interactive=False,
                        )

            # ============================================================
            # TAB 3: Segment & Refine
            # ============================================================
            with gr.Tab("🔬 Segment & Refine", id=3):
                header = gr.Markdown("### No image loaded\nPick one from the Gallery.",
                                     elem_classes=["frame-header"])

                # Shown only for a reading nothing has vouched for yet. A scale
                # that agrees with one already accepted never raises this, so
                # working through a folder at one magnification is uninterrupted;
                # a scale with a warning does not raise it either, because that
                # one is worth going to the Scale tab for.
                with gr.Row(visible=False) as scale_check_row:
                    with gr.Column(scale=2):
                        scale_check_msg = gr.Markdown("")
                        scale_check_btn = gr.Button("✔️ Scale is right — carry on",
                                                    variant="primary", size="sm")
                    with gr.Column(scale=3):
                        scale_check_img = gr.Image(
                            label="", show_label=False, height=150,
                            interactive=False,
                        )

                with gr.Row():
                    # ---- tool rail --------------------------------------
                    with gr.Column(scale=1, min_width=260,
                                   elem_classes=["tool-rail"]):

                        gr.Markdown("Segment", elem_classes=["step-title"])
                        min_particle_size_slider = gr.Slider(
                            minimum=5, maximum=200, value=30, step=1,
                            label="Smallest particle (px)",
                            info="Anything below this is noise",
                        )
                        segment_btn = gr.Button("🤖 Segment", variant="primary")
                        mask_choice = gr.Radio(
                            choices=["Option 1", "Option 2", "Option 3"],
                            value="Option 1", label="Candidate",
                            info="Ranked best-first by contrast",
                        )
                        analyze_btn = gr.Button("✓ Use this candidate",
                                                variant="primary")
                        segment_status = gr.Textbox(label="", interactive=False,
                                                    lines=2, show_label=False)

                        gr.Markdown("---")
                        gr.Markdown("Correct", elem_classes=["step-title"])
                        click_mode_radio = gr.Radio(
                            choices=[(label, value) for value, label in REFINE_MODES],
                            value="delete", label="Tool",
                            info="Clicks queue up; nothing changes until Apply",
                        )

                        with gr.Group(visible=False) as point_refine_controls:
                            point_type_radio = gr.Radio(
                                choices=[("✓ include", "positive"),
                                         ("✗ exclude", "negative")],
                                value="positive", label="Next point",
                            )
                            reset_points_btn = gr.Button("Reset points", size="sm")

                        with gr.Row():
                            apply_btn = gr.Button("✓ Apply", variant="primary")
                            undo_btn = gr.Button("↩️ Undo", variant="secondary")
                        clear_pending_btn = gr.Button("Discard pending clicks",
                                                      variant="secondary", size="sm")

                        with gr.Accordion("Bulk cleanup", open=False):
                            edge_buffer = gr.Slider(
                                minimum=0, maximum=50, value=10, step=1,
                                label="Edge buffer (px)",
                            )
                            clear_edges_btn = gr.Button("🧹 Drop edge particles",
                                                        variant="secondary", size="sm")
                            clear_all_btn = gr.Button("🗑️ Clear all — start blank",
                                                      variant="stop", size="sm")

                        show_numbers_checkbox = gr.Checkbox(
                            label="Show particle numbers", value=True,
                        )

                    # ---- the image --------------------------------------
                    with gr.Column(scale=4):
                        # Open by default: choosing between the candidates is the
                        # first thing that happens on this tab, and it cannot be
                        # done without seeing them. Collapsed once one is chosen,
                        # so the rest of the work gets the height.
                        with gr.Accordion("Candidates — pick one, then Use it",
                                          open=True) as candidate_panel:
                            mask_viz = gr.Image(label="", show_label=False, height=300)
                            analysis_status = gr.Textbox(label="", interactive=False,
                                                         show_label=False, lines=1)

                        refine_viz = gr.Image(label="", type="numpy", show_label=False,
                                              elem_classes=["work-image"], height=760)
                        refine_status = gr.Textbox(label="", interactive=False,
                                                   show_label=False, lines=1)

                gr.Markdown("---")
                with gr.Row():
                    save_next_btn = gr.Button("✅ Save & next", variant="primary",
                                              size="lg", scale=2)
                    save_btn = gr.Button("💾 Save, stay here", variant="secondary")
                    skip_btn = gr.Button("⏭️ Skip", variant="secondary")
                save_status = gr.Textbox(label="", interactive=False, show_label=False)

                with gr.Accordion("Measurements for this image", open=False):
                    refine_results = gr.Dataframe(label="", show_label=False)

            # ============================================================
            # TAB 4: Results
            # ============================================================
            with gr.Tab("💾 Results", id=4):
                gr.Markdown("### This image", elem_classes=["step-title"])
                with gr.Row():
                    current_results = gr.Dataframe(label="Particles")
                    current_stats = gr.Dataframe(label="Summary")

                gr.Markdown("---")
                gr.Markdown("### Whole session", elem_classes=["step-title"])
                refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                with gr.Row():
                    summary_progress = gr.Markdown("No results yet")
                    summary_particle_stats = gr.Markdown("No particle statistics yet")
                session_table = gr.Dataframe(label="Processed images")

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
                gr.Markdown("Size distribution across everything saved this session.")
                with gr.Row():
                    update_plots_btn = gr.Button("🔄 Update", variant="primary")
                    plot_status = gr.Textbox(label="", interactive=False,
                                             show_label=False)
                histogram_plot = gr.Image(label="", type="numpy", show_label=False)

        # ================================================================
        # Event wiring
        # ================================================================

        # Opening an image does the same thing however it was opened: settle the
        # scale and frame, then segment. Kept as one list so the three entry
        # points cannot drift apart.
        OPEN_OUTPUTS = [scale_image_in, tier1_status, scale_summary, point_readout,
                        frame_status, header, mask_viz, segment_status, mask_choice]
        RELOAD_CANVAS = "() => { window.SCALE && window.SCALE.load(); }"

        # Everything that shows what the scale is and how much it should be
        # trusted. Run after anything that can change it, so the canvas overlay,
        # the zoomed check, the prompt on the working tab and the tab the analyst
        # is left on cannot disagree about the same reading.
        CHECK_OUTPUTS = [scale_span_in, scale_preview, scale_check_row,
                         scale_check_msg, scale_check_img, tabs]

        # Nothing here re-draws the scale canvas on a tab change. Gradio unmounts
        # an inactive tab's panel and builds a fresh canvas when it is shown
        # again, and a Tab's select event does not fire in a way this can hook —
        # measured, not assumed. scale_canvas.js watches for a canvas it has not
        # wired and for a payload it has not drawn, so it recovers by itself. The
        # RELOAD_CANVAS calls below remain because they are free when the tab
        # happens to be mounted; they are not what makes it work.

        # ---- Setup ----
        init_sam_btn.click(initialize_sam, inputs=[sam_file],
                           outputs=[init_status, load_btn])
        download_btn.click(download_model,
                           outputs=[init_status, sam_file, download_btn, load_btn,
                                    init_sam_btn])
        load_btn.click(load_images_from_folder,
                       inputs=[folder_input, results_file_input],
                       outputs=[load_status, gallery])
        save_dataset_box.change(set_save_dataset, inputs=[save_dataset_box],
                                outputs=[save_dataset_box])

        # A refresh reconnects to the same process-wide session, so bring the
        # gallery back rather than showing an empty one.
        app.load(restore_session, outputs=[gallery, load_status])
        app.load(current_save_dataset, outputs=[save_dataset_box])

        # ---- Gallery ----
        gallery.select(
            select_image_from_gallery,
            outputs=[refine_viz, selected_image_info, tabs, click_mode_radio],
        ).then(
            open_current_image, outputs=OPEN_OUTPUTS,
        ).then(  # a new image means new candidates to choose between
            lambda: gr.update(open=True), outputs=[candidate_panel]
        ).then(check_outputs, outputs=CHECK_OUTPUTS).then(None, js=RELOAD_CANVAS)

        # ---- Scale & Frame ----
        canvas_mode.change(
            set_canvas_mode, inputs=[canvas_mode],
            outputs=[tier2_controls, tier3_controls],
        ).then(
            None, inputs=[canvas_mode],
            js="(m) => { window.SCALE && window.SCALE.setMode("
               "m.startsWith('Tier 3') ? 'points' : 'box'); }",
        )

        read_box_btn.click(read_box_scale, inputs=[scale_box_out],
                           outputs=[tier_status, scale_summary]).then(
            frame_header, outputs=[header]).then(
            check_outputs, outputs=CHECK_OUTPUTS)

        # Live feedback as the two points are placed.
        scale_points_out.change(live_point_readout, inputs=[scale_points_out],
                                outputs=[point_readout])

        apply_points_btn.click(
            apply_two_points, inputs=[scale_points_out, bar_value, bar_unit],
            outputs=[tier_status, scale_summary]).then(
            frame_header, outputs=[header]).then(
            check_outputs, outputs=CHECK_OUTPUTS)

        reset_canvas_btn.click(None, js="() => { window.SCALE && window.SCALE.reset(); }")

        confirm_scale_btn.click(confirm_scale,
                                outputs=[tier_status, scale_summary]).then(
            frame_header, outputs=[header]).then(
            check_outputs, outputs=CHECK_OUTPUTS)
        clear_scale_btn.click(clear_scale,
                              outputs=[tier_status, scale_summary]).then(
            frame_header, outputs=[header]).then(
            check_outputs, outputs=CHECK_OUTPUTS)

        # Accepting from the working tab, so checking a reading costs a glance
        # and a click rather than a trip to another tab and back.
        scale_check_btn.click(
            accept_from_work_tab,
            outputs=CHECK_OUTPUTS + [header, scale_summary])

        # An override re-derives the frame's geometry, since the instrument
        # decides whether a databar is expected and which way round particles are.
        modality_choice.change(set_modality, inputs=[modality_choice],
                               outputs=[frame_status]).then(frame_header,
                                                            outputs=[header])
        particle_choice.change(set_particle_polarity, inputs=[particle_choice],
                               outputs=[frame_status]).then(frame_header,
                                                            outputs=[header])
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
        segment_btn.click(
            segment_with_sam, outputs=[mask_viz, segment_status, mask_choice],
        ).then(  # new candidates to look at, so open the panel again
            lambda: gr.update(open=True), outputs=[candidate_panel])

        analyze_btn.click(
            select_mask_and_analyze, inputs=[mask_choice],
            outputs=[refine_viz, refine_results, analysis_status,
                     current_results, current_stats],
        ).then(frame_header, outputs=[header]).then(
            # Chosen: give the height back to the frame being corrected.
            lambda: gr.update(open=False), outputs=[candidate_panel])

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
        ).then(frame_header, outputs=[header])

        clear_pending_btn.click(clear_pending_changes,
                                outputs=[refine_viz, refine_status])
        undo_btn.click(undo_last_action,
                       outputs=[refine_viz, refine_results, refine_status]).then(
            frame_header, outputs=[header])

        clear_edges_btn.click(
            clear_edge_particles, inputs=[edge_buffer],
            outputs=[refine_viz, refine_results, refine_status,
                     current_results, current_stats],
        ).then(frame_header, outputs=[header])
        clear_all_btn.click(
            clear_all_particles,
            outputs=[refine_viz, refine_results, refine_status,
                     current_results, current_stats],
        ).then(frame_header, outputs=[header])

        # ---- Save and advance ----
        save_btn.click(save_current_results, outputs=[save_status, gallery])

        save_next_btn.click(
            save_and_next,
            outputs=[save_status, gallery, selected_image_info] + OPEN_OUTPUTS,
        ).then(  # a new image means new candidates to choose between
            lambda: gr.update(open=True), outputs=[candidate_panel]
        ).then(check_outputs, outputs=CHECK_OUTPUTS).then(None, js=RELOAD_CANVAS)

        skip_btn.click(
            skip_to_next, outputs=[selected_image_info] + OPEN_OUTPUTS,
        ).then(  # a new image means new candidates to choose between
            lambda: gr.update(open=True), outputs=[candidate_panel]
        ).then(check_outputs, outputs=CHECK_OUTPUTS).then(None, js=RELOAD_CANVAS)

        # ---- Results ----
        refresh_btn.click(get_session_summary,
                          outputs=[session_table, summary_progress,
                                   summary_particle_stats, delete_row_dropdown])
        delete_row_btn.click(delete_result_row, inputs=[delete_row_dropdown],
                             outputs=[session_table, summary_progress,
                                      summary_particle_stats, delete_row_dropdown,
                                      delete_row_status])
        remove_duplicates_btn.click(check_and_remove_duplicates,
                                    outputs=[session_table, duplicates_status])
        export_btn.click(export_results, outputs=[export_file, export_status])

        # ---- Plots ----
        update_plots_btn.click(update_histogram_plots,
                               outputs=[histogram_plot, plot_status])

    return app
