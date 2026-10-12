"""
Scale tab: establish nm/pixel before anything is measured.

Three tiers, tried in order of trustworthiness. Tier 1 runs by itself when an
image is opened; tiers 2 and 3 are driven from the interactive canvas and are
only needed when the one above them fails.
"""

import base64
import io
import json
import os

import gradio as gr
import numpy as np
from PIL import Image

from sem_particle_analysis import modality, region
from sem_particle_analysis import scale_calibration as sc
from ..state import state
from ..visualization import render_scale_check

UNITS = ["nm", "µm", "mm", "Å"]

# Particle polarity choices offered in the UI, as the segmenter's dark_features.
POLARITY_CHOICES = {"auto": None, "bright": False, "dark": True}

# Anything larger is downscaled for transport to the browser. Well above the
# 2048px micrographs in use, so the magnifier still shows real pixels.
MAX_CANVAS_PX = 2600


def _detector():
    """
    The scale detector, created on first use.

    Calibration needs OCR, not segmentation, so it must not depend on a 2.4 GB
    SAM checkpoint having been loaded first — an analyst should be able to sort
    out scale before choosing a model.
    """
    if state.scale_detector is None:
        from sem_particle_analysis import ScaleDetector

        state.scale_detector = ScaleDetector(use_gpu=False)
    return state.scale_detector


def _image_payload(image, box=None):
    """
    Encode the current image as a data URL for the canvas.

    A ``box`` is included when automatic detection already found the bar, so the
    canvas opens with the region it used and the analyst can see what was read
    rather than having to guess where to look.

    ``px_scale`` is how much the frame was shrunk to get here. The canvas works
    in the coordinates of the image it was sent, so without it a frame larger
    than MAX_CANVAS_PX would have every box and click committed in the wrong
    space — an oversized frame's bar would be measured somewhere else entirely.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3]
    pil = Image.fromarray(array.astype(np.uint8))
    if max(pil.size) > MAX_CANVAS_PX:
        ratio = MAX_CANVAS_PX / max(pil.size)
        pil = pil.resize((int(pil.width * ratio), int(pil.height * ratio)),
                         Image.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG", optimize=False, compress_level=1)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = {"img": f"data:image/png;base64,{encoded}",
               "w": pil.width, "h": pil.height,
               "px_scale": pil.width / max(1, array.shape[1])}
    if box:
        payload["box"] = {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]}
    return json.dumps(payload)


def _raw_metadata():
    """TIFF tags for the current image, or None. Never fatal."""
    if not state.image_paths or state.current_index >= len(state.image_paths):
        return None
    try:
        from sem_particle_analysis.utils import extract_tiff_metadata

        return extract_tiff_metadata(str(state.image_paths[state.current_index]))
    except Exception:
        return None


def _bar_region(calibration):
    """
    Where the scale bar sits in the frame, as (x0, y0, w, h), or None.

    Only meaningful when the bar was found inside the image itself.
    """
    # Not named `region`: that is the module handling analysable area, imported
    # above and used a few lines further down.
    found = sc.search_box(calibration)
    if found is None:
        return None
    x0, y0, x1, y1 = found
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _apply_frame_geometry(calibration):
    """
    Decide which part of the frame gets segmented, and what to ignore in it.

    Two situations, and confusing them costs real data either way:

    - An SEM frame carries a databar below the micrograph. It has to be cropped
      off, or its text and rules are segmented and measured as particles.
    - A TEM frame has no databar; the scale bar is burned into the micrograph
      itself. There is nothing to crop — trimming a fixed percentage would throw
      away image — so the bar's own patch is excluded from segmentation instead.

    Deciding this here means every tier gets it, rather than only the paths that
    happen to remember.
    """
    image = state.current_image
    if image is None:
        return

    height = image.shape[0]
    override = getattr(state, "crop_override", None)

    if override is not None:
        # An explicit choice, including 0 meaning "keep the whole frame".
        databar = {"has_databar": override > 0}
        bar_height = int(round(height * override / 100.0))
    else:
        try:
            databar = _detector().detect_databar(image, _raw_metadata()) or {}
        except Exception:
            databar = {}
        bar_height = int(databar.get("databar_height") or 0)

    if databar.get("has_databar") and 0 < bar_height < height:
        state.cropped_image = image[: height - bar_height].copy()
        state.crop_percent = round(100 * bar_height / height, 2)
        # The printed bar went with the databar, so there is nothing left to
        # exclude inside the micrograph.
        state.scale_bar_region = None
    else:
        state.cropped_image = image
        state.crop_percent = 0.0
        state.scale_bar_region = _bar_region(calibration)

    # Which instrument this is decides particle polarity, and an analyst's
    # explicit choice always wins over what the file says.
    state.modality = (modality.resolve(state.modality_choice)
                      or modality.detect(image, metadata=_raw_metadata(),
                                         databar_height=bar_height))

    # Everything in the frame that is not specimen — beam-blocked area, and the
    # scale bar when it is printed inside the image. Left out of segmentation, so
    # an aperture vignette or a grid bar is never measured as a particle.
    boxes = [state.scale_bar_region] if state.scale_bar_region else []
    state.analysable_region, state.region_info = region.analysable_region(
        state.cropped_image, exclude_boxes=boxes)


def _adopt(calibration):
    """
    Make a calibration the authoritative scale for the current image.

    The rest of the app reads state.scale_info['conversion'], so keep that in
    step rather than teaching every caller about the new object.
    """
    state.scale_calibration = calibration
    state.scale_info = calibration.to_dict()
    # A scale set after a candidate was chosen has to reach its measurements;
    # otherwise the results row says one scale and was computed with another.
    if state.analyzer is not None:
        state.analyzer.conversion = calibration.nm_per_px
    # Kept against this image, so returning to it does not repeat the work — and
    # in particular does not throw away a bar the analyst read by hand.
    state.remember_scale(calibration)
    _apply_frame_geometry(calibration)
    return calibration


def frame_summary():
    """
    What the app has decided about the current frame, beyond its scale.

    Shown because both decisions change the numbers and neither is visible in
    the result: the modality sets which side of the frame the particles are on,
    and the analysable region is what is left after beam-blocked area and any
    burned-in scale bar are taken out.
    """
    kind = getattr(state, "modality", None)
    if kind is None:
        return "Load an image to detect the instrument."

    source = "from the file" if kind.from_metadata else "inferred"
    if state.modality_choice != "auto":
        source = "set by hand"
    lines = [f"**{kind.label}** ({source}) — {kind.detail}"]

    polarity = {None: "brighter or darker, decided by contrast",
                False: "brighter than the background",
                True: "darker than the background"}
    choice = state.particle_choice
    resolved = (POLARITY_CHOICES[choice] if choice != "auto" else kind.dark_particles)
    lines.append(f"Particles are **{polarity[resolved]}**"
                 + ("" if choice == "auto" else " (set by hand)"))

    info = getattr(state, "region_info", None) or {}
    blocked = 1 - info.get("analysable_fraction", 1.0)
    if blocked > 0.005:
        parts = [f"{100 * blocked:.1f}% of the frame excluded"]
        if info.get("blocked_fraction", 0) > 0.005:
            parts.append("beam-blocked area")
        if info.get("excluded_boxes"):
            parts.append("burned-in scale bar")
        lines.append(" — ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1
                     else parts[0])
    else:
        lines.append("Whole frame is analysable.")

    if state.crop_percent:
        lines.append(f"Databar trimmed: {state.crop_percent:.2f}% off the bottom.")
    return "\n\n".join(lines)


def frame_header():
    """
    One-line summary of the current frame, for the working tab's header.

    Everything an analyst needs to know they are on the right image, working at
    the right scale, without leaving the tab they are clicking in.
    """
    if state.current_image is None:
        return "### No image loaded\nPick one from the Gallery."

    total = len(state.image_paths) or 1
    name = (os.path.basename(str(state.image_paths[state.current_index]))
            if state.image_paths else "image")

    cal = getattr(state, "scale_calibration", None)
    if cal is None:
        scale = "⚠️ no scale — sizes in pixels"
    else:
        scale = f"{cal.nm_per_px:.4g} nm/px"
        if not cal.trustworthy:
            scale += " (unconfirmed)"

    kind = getattr(state, "modality", None)
    bits = [f"**{name}**", f"{state.current_index + 1} of {total}", scale]
    if kind is not None:
        bits.append(kind.label)

    info = getattr(state, "region_info", None) or {}
    blocked = 1 - info.get("analysable_fraction", 1.0)
    if blocked > 0.005:
        bits.append(f"{100 * blocked:.0f}% excluded")

    count = len(state.analyzer.regions) if state.analyzer is not None else None
    if count is not None:
        bits.append(f"**{count} particles**")

    warning = cal.warning if cal is not None else None
    line = "  ·  ".join(bits)
    return f"{line}\n\n⚠️ {warning}" if warning else line


def set_crop_override(percent):
    """
    Override how much is trimmed off the bottom of the frame.

    The databar height is measured automatically and is exact when the
    instrument recorded it, so this is an escape hatch rather than a routine
    control: use it when a frame's databar is not detected, or is detected on a
    frame that has none.

    Args:
        percent: Percent of frame height to trim, or 0 to keep the whole frame,
            or None to go back to measuring it.
    """
    state.crop_override = None if percent is None or percent < 0 else float(percent)
    if state.current_image is not None:
        _apply_frame_geometry(getattr(state, "scale_calibration", None))
    return frame_summary(), frame_header()


def clear_crop_override():
    """Go back to measuring the databar."""
    return set_crop_override(None)


def set_modality(choice):
    """Override which instrument the frame is treated as, and re-derive geometry."""
    state.modality_choice = choice or "auto"
    if state.current_image is not None:
        _apply_frame_geometry(getattr(state, "scale_calibration", None))
    return frame_summary()


def set_particle_polarity(choice):
    """Override which side of the frame holds the particles."""
    state.particle_choice = choice or "auto"
    return frame_summary()


def _status_lines():
    """Current calibration rendered for the summary box."""
    cal = getattr(state, "scale_calibration", None)
    if cal is None:
        return ("⚠️ No scale yet — the measurements will be in pixels.\n"
                "Try tier 2 (draw a box) or tier 3 (click both ends of the bar).")
    lines = [f"✅ {cal.summary()}"]
    if cal.detail:
        lines.append(f"   {cal.detail}")
    if cal.warning:
        lines.append(f"⚠️ {cal.warning}")
    if not cal.trustworthy:
        lines.append("   Not yet confirmed — check the overlay, then press "
                     "“Confirm scale”.")
    return "\n".join(lines)


def prepare_scale_tab():
    """
    Open an image on the Scale tab and run tier 1.

    Returns:
        tuple: (canvas_payload, tier1_status, summary, points_hint, frame_summary)
    """
    if state.current_image is None:
        # Nothing to calibrate — and drop any calibration left over from the
        # image before, which would otherwise still be reported as this one's.
        state.scale_calibration = None
        state.scale_info = None
        return "", "No image loaded", _status_lines(), "", frame_summary()

    state.scale_calibration = None
    state.scale_info = None

    payload = _image_payload(state.current_image)
    path = state.current_path()

    # Already settled on an earlier visit: restore it rather than detecting
    # again. Re-running detection would quietly discard a bar that was read by
    # hand, which is the one case where the stored answer is the only good one.
    remembered = state.recall_scale()
    if remembered is not None:
        _adopt(remembered)
        name = os.path.basename(str(path)) if path else "image"
        return (payload,
                f"✅ Scale already established for {name} "
                f"({remembered.method_label}) — press Clear to redo it.",
                _status_lines(), "", frame_summary())

    try:
        calibration = sc.from_metadata(_detector(), path)
    except sc.ScaleError as exc:
        tier1 = f"❌ Tier 1 — no pixel size in the file ({exc})."
        return _try_automatic_bar(payload, tier1)

    _adopt(calibration)
    name = os.path.basename(str(path)) if path else "image"
    if not calibration.trustworthy:
        # Only the TIFF's generic resolution tag, which is usually a print
        # setting: offered, but left for the analyst to confirm.
        return (payload,
                f"⚠️ Tier 1 — {name} gives a pixel size only in its TIFF "
                "resolution tag, which is usually a print setting. Check it "
                "against the scale bar, then press Confirm scale, or measure the "
                "bar (tier 2 or 3).",
                _status_lines(), "", frame_summary())
    # Exact, and therefore the best possible yardstick for the readings that
    # follow it in this folder.
    state.accept_scale(calibration)
    return (payload,
            f"✅ Tier 1 — pixel size read from {name}'s metadata. Nothing else to do.",
            _status_lines(), "", frame_summary())


def _try_automatic_bar(payload, tier1_message):
    """
    Attempt tier 2 without being asked, by looking where bars usually are.

    This is the same measurement the analyst would get by drawing the box by
    hand, so offering it saves a step on the majority of images — but it is
    reported as unconfirmed, and the box it used is shown on the canvas so it can
    be checked or corrected rather than taken on trust.
    """
    try:
        found = _detector().detect_scale_bar_anywhere(state.current_image)
    except Exception:
        # Scale failed, but the frame's geometry and instrument are still worth
        # working out — measurements can proceed in pixels.
        _apply_frame_geometry(None)
        return (payload,
                tier1_message + " Draw a box below (tier 2), or click both ends of "
                "the bar (tier 3).",
                _status_lines(), "", frame_summary())

    calibration = sc.ScaleCalibration(
        nm_per_px=float(found["conversion"]),
        method="box_ocr",
        scale_nm=float(found["scale_nm"]),
        pixel_length=float(found["pixel_length"]),
        detail=f"found automatically in the {found.get('region_name', 'image')} "
               f"corner: “{found.get('ocr_text', '').strip()}”",
        warning=found.get("warning"),
        # Recorded so that, on a frame with no databar to crop, segmentation can
        # exclude the patch the bar occupies instead of measuring it.
        # line_coords say which pixels were counted, so the span can be drawn
        # on the bar rather than only reported as a number.
        extra={"region": found.get("region"),
               "region_name": found.get("region_name"),
               "line_coords": found.get("line_coords")},
    )
    _adopt(calibration)

    box = found.get("region")
    if box:
        x0, y0, w, h = box
        payload = _image_payload(state.current_image, (x0, y0, x0 + w, y0 + h))

    return (payload,
            tier1_message + " Tier 2 ran automatically — check the box on the image.",
            _status_lines(), "", frame_summary())


def read_box_scale(box_json, progress=gr.Progress()):
    """Tier 2: OCR the scale bar inside the drawn rectangle."""
    if state.current_image is None:
        return "❌ No image loaded", _status_lines()

    box = _parse_box(box_json)
    if box is None:
        return ("❌ Draw a box around the scale bar and its label first — drag on "
                "the image, then drag the corners to adjust."), _status_lines()

    progress(0.4, desc="Reading the scale bar...")
    before = _analysable_fraction()
    try:
        calibration = sc.from_box_ocr(_detector(), state.current_image, box)
    except sc.ScaleError as exc:
        return (f"❌ Tier 2 — {exc}\nAdjust the box so it contains the whole bar "
                f"and its label, or use tier 3."), _status_lines()

    _adopt(calibration)
    note = f"  ⚠️ {calibration.warning}" if calibration.warning else ""
    return (f"✅ Tier 2 — {calibration.summary()}{note}"
            f"{_geometry_note(before)}", _status_lines())


def apply_two_points(points_json, value, unit):
    """Tier 3: the user clicked both ends of the bar and typed its length."""
    points = _parse_points(points_json)
    if len(points) != 2:
        return ("❌ Click both ends of the scale bar first — the magnifier shows "
                "exactly which pixel you are on."), _status_lines()
    if value is None:
        return "❌ Type the length printed next to the bar.", _status_lines()

    before = _analysable_fraction()
    try:
        calibration = sc.from_two_points(points[0], points[1], float(value), unit)
    except sc.ScaleError as exc:
        return f"❌ Tier 3 — {exc}", _status_lines()

    _adopt(calibration)
    state.accept_scale(calibration)
    return (f"✅ Tier 3 — {calibration.summary()}{_geometry_note(before)}",
            _status_lines())


def confirm_scale():
    """Mark the current calibration as checked by a human."""
    cal = getattr(state, "scale_calibration", None)
    if cal is None:
        return "❌ Nothing to confirm yet", _status_lines()
    cal.confirmed = True
    cal.warning = None
    state.scale_info = cal.to_dict()
    # From here on, a reading that agrees with this one needs no second look.
    state.accept_scale(cal)
    return "✅ Scale confirmed", _status_lines()


def current_frame_controls():
    """
    The frame and segmentation settings as the session has them, for a browser
    connecting to a running app. The page shows start-up defaults otherwise,
    while the session goes on using what the analyst set.

    Returns:
        tuple: (instrument, particle polarity, trim override on, trim percent,
        smallest particle)
    """
    override = state.crop_override
    return (state.modality_choice or "auto", state.particle_choice or "auto",
            override is not None, override or 0, state.min_particle_size)


def clear_scale():
    """
    Discard the calibration and start again.

    Forgets the stored one too, so the next visit to this image detects afresh
    rather than restoring the reading that was just rejected.
    """
    state.scale_calibration = None
    state.scale_info = None
    if state.analyzer is not None:
        state.analyzer.conversion = None
    state.forget_scale()
    return "Scale cleared", _status_lines()


def set_canvas_mode(mode):
    """Show the controls belonging to the selected tier."""
    points = mode.startswith("Tier 3")
    return (gr.update(visible=not points),   # tier 2 controls
            gr.update(visible=points))       # tier 3 controls


def live_point_readout(points_json):
    """Report the pixel distance between the two clicked points."""
    points = _parse_points(points_json)
    if len(points) == 0:
        return "Click the left end of the scale bar."
    if len(points) == 1:
        return f"Point 1 at ({points[0][0]:.0f}, {points[0][1]:.0f}). Now click the other end."
    length = ((points[1][0] - points[0][0]) ** 2
              + (points[1][1] - points[0][1]) ** 2) ** 0.5
    return (f"Bar spans {length:.1f} px. Enter its printed length below, "
            f"then press Apply.")


def _parse_box(raw):
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        box = (data["x0"], data["y0"], data["x1"], data["y1"])
    except (TypeError, ValueError, KeyError):
        return None
    return box if (box[2] - box[0]) > 2 and (box[3] - box[1]) > 2 else None


def _parse_points(raw):
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return [tuple(p) for p in data.get("points", [])][:2]
    except (TypeError, ValueError, AttributeError):
        return []


# ---------------------------------------------------------------------------
# Checking a reading, and deciding when it is worth interrupting for
# ---------------------------------------------------------------------------

# Two readings this close are the same magnification measured the same way. A
# bar is a whole number of pixels long, so identical setups still differ by one
# or two; 2% covers that without letting a real change of magnification past.
SAME_SCALE_TOLERANCE = 0.02


def _analysable_fraction():
    """How much of the frame is currently set to be measured, 0-1, or None."""
    info = getattr(state, "region_info", None) or {}
    return info.get("analysable_fraction")


def _geometry_note(before):
    """
    Say so when settling the scale also changed what will be segmented.

    Finding a burned-in bar excludes the patch it occupies, so a mask made
    before the bar was found still has the bar in shot — and it out-contrasts
    the particles, so it gets measured as one.
    """
    after = _analysable_fraction()
    if before is None or after is None or abs(after - before) < 0.001:
        return ""
    return "\n   Frame changed — press Segment again so the new area is used."


def review():
    """
    Whether the scale on screen should be looked at before anything is measured.

    The point of the working loop is to move through a folder quickly, so an
    interruption has to earn itself. Three cases do:

    - There is no scale at all. Everything measured would be in pixels.
    - The reading came with a warning — a non-standard bar value, a guessed
      unit, a bar running out of the search box. These are the readings that are
      wrong often enough to be worth stopping for.
    - It is an OCR reading that neither a human nor a matching earlier reading
      has vouched for.

    Anything that agrees with a scale already accepted in this session is let
    through, which is what makes the loop fast: a folder shot at one
    magnification asks once and then stays out of the way.

    Returns:
        tuple: ``(level, message)``. Level is "stop" when there is nothing
        trustworthy to measure with, "check" when a glance would settle it, and
        None when it can be taken as read.
    """
    cal = getattr(state, "scale_calibration", None)
    if cal is None:
        return "stop", ("No scale for this image — anything measured would be in "
                        "pixels. Draw a box around the bar and read it, or click "
                        "its two ends.")
    if cal.warning:
        return "stop", cal.warning
    if cal.trustworthy:
        return None, ""

    baseline = getattr(state, "scale_baseline", None)
    if baseline and abs(cal.nm_per_px - baseline) <= SAME_SCALE_TOLERANCE * baseline:
        return None, ""
    if baseline:
        return "check", (f"**{cal.nm_per_px:.4g} nm/px** — a different "
                         f"magnification from the last scale you accepted "
                         f"({baseline:.4g} nm/px). Check the bar below.")
    return "check", (f"**{cal.nm_per_px:.4g} nm/px**, read off the bar. "
                     f"Check the measured span below.")


def scale_span_payload():
    """
    The measured span, for the canvas to draw over the image.

    In full-frame coordinates; the canvas maps them into whatever size it was
    sent. Empty when there is nothing measured to show.
    """
    cal = getattr(state, "scale_calibration", None)
    span = sc.measured_span(cal) if cal is not None else None
    if span is None:
        return json.dumps({})
    (ax, ay), (bx, by) = span
    label = f"{cal.pixel_length or abs(bx - ax):.0f} px"
    if cal.scale_nm:
        label += f" = {sc.format_length(cal.scale_nm)}"
    return json.dumps({"span": [[ax, ay], [bx, by]], "label": label})


def scale_check_view():
    """The measured bar, zoomed, with the span drawn across it. None if none."""
    return render_scale_check(state.current_image,
                              getattr(state, "scale_calibration", None))


def check_outputs():
    """
    Everything the interface needs to show a reading and let it be judged.

    Returned by every path that can change the scale, so the canvas overlay, the
    zoomed check, the prompt on the working tab and the tab the analyst ends up
    on can never disagree about what the current scale is.

    Returns:
        tuple: (span_payload, scale_preview, check_row, check_message,
                check_image, tab)
    """
    level, message = review()
    view = scale_check_view()
    return (scale_span_payload(),
            view,                                  # beside the canvas
            gr.update(visible=level == "check"),
            message,
            view,                                  # again, on the working tab
            gr.Tabs(selected=2) if level == "stop" else gr.update())


def accept_from_work_tab():
    """
    Confirm the scale from the working tab, without going to look for it.

    The whole cost of checking should be one glance and one click; making the
    analyst switch tabs to accept what they have already looked at is what makes
    supervision expensive enough to skip.
    """
    confirm_scale()
    return check_outputs() + (frame_header(), _status_lines())
