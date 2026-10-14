"""
Scale tab: establish nm/pixel before anything is measured.

Three ways, tried in order of trustworthiness. The file's metadata, then the
printed scale bar, are tried by themselves when an image is opened; a box drawn
round the bar, or its two ends clicked, are for when neither works.

What the tab shows is short by design: a value, where it came from, and a
badge saying whether it needs a look. The reasoning lives here, not on screen.
"""

import base64
import html
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

# What the two-point controls say before the first click.
POINTS_HINT = "Click one end of the scale bar on the image, then the other."

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


def _image_payload(image, box=None, suggest=False):
    """
    Encode the current image as a data URL for the canvas.

    A ``box`` is included when automatic detection already found the bar, so the
    canvas opens with the region it used and the analyst can see what was read
    rather than having to guess where to look. With ``suggest``, and no box,
    the canvas offers a starting box where bars usually are; otherwise it shows
    none, since a scale from the metadata needs nothing drawn.

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
    elif suggest:
        payload["suggest"] = True
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


def _sidecar_text():
    """The .txt an instrument wrote beside the current image, or None."""
    if not state.image_paths or state.current_index >= len(state.image_paths):
        return None
    from sem_particle_analysis.sidecar import read_sidecar

    return read_sidecar(state.image_paths[state.current_index])


def _bar_region(calibration):
    """
    Where the scale bar and its label sit in the frame, as (x0, y0, w, h), or None.

    Only meaningful when the bar was found inside the image itself.
    """
    if calibration is None or state.current_image is None:
        return None
    return sc.printed_area(calibration, state.current_image.shape)


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

    # What the file says about its instrument: shown, compared with the
    # microscope chosen on Setup, and used when none was chosen.
    state.modality = modality.detect(metadata=_raw_metadata(), sidecar=_sidecar_text())

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


# Instrument makers as the metadata reader names them, as a person would.
MAKERS = {"fei": "FEI / Thermo Fisher", "zeiss": "Zeiss", "hitachi": "Hitachi",
          "tescan": "TESCAN", "jeol": "JEOL", "amt": "AMT camera"}


def _esc(text):
    return html.escape(str(text), quote=False)


def _rows(rows):
    """A two-column table of (name, value, note, warning) rows."""
    cells = []
    for name, value, note, warning in rows:
        extra = f' <span class="kv-note">{_esc(note)}</span>' if note else ""
        alert = f'<div class="kv-warn">{_esc(warning)}</div>' if warning else ""
        cells.append(f"<tr><th>{_esc(name)}</th>"
                     f"<td>{_esc(value)}{extra}{alert}</td></tr>")
    return f'<table class="kv">{"".join(cells)}</table>'


def frame_summary():
    """
    What the app has decided about the current frame, beyond its scale.

    Shown because each of these changes the numbers and none is visible in the
    result: the microscope sets which side of the frame the particles are on,
    the databar is cut off, and beam-blocked area and any burned-in scale bar
    are left out of measurement.
    """
    kind = getattr(state, "modality", None)
    if kind is None or state.current_image is None:
        return '<div class="kv-empty">No image open.</div>'

    chosen = state.microscope
    if chosen in ("SEM", "TEM"):
        disagrees = kind.kind not in (modality.UNKNOWN, chosen)
        microscope = (chosen, "set on Setup",
                      f"The file says {kind.label}." if disagrees else None)
    elif kind.kind != modality.UNKNOWN:
        microscope = (kind.label, "from the file", None)
    else:
        microscope = ("Not set", None, "Choose SEM or TEM on the Setup tab.")

    choice = state.particle_choice
    resolved = (POLARITY_CHOICES[choice] if choice != "auto"
                else {"SEM": False, "TEM": True}.get(state.microscope_kind()))
    particles = {None: "Darker or brighter", False: "Brighter than background",
                 True: "Darker than background"}[resolved]

    databar = (f"Cropped, {state.crop_percent:.1f}% of height"
               if state.crop_percent else "None")

    info = getattr(state, "region_info", None) or {}
    blocked = 1 - info.get("analysable_fraction", 1.0)
    excluded, what = "None", []
    if blocked > 0.005:
        excluded = f"{100 * blocked:.1f}% of frame"
        if info.get("blocked_fraction", 0) > 0.005:
            what.append("beam-blocked area")
        if info.get("excluded_boxes"):
            what.append("scale bar")

    return _rows([
        ("Microscope",) + microscope,
        ("Particles", particles, "set by you" if choice != "auto" else None, None),
        ("Databar", databar, None, None),
        ("Excluded", excluded, ", ".join(what) or None, None),
    ])


def frame_header():
    """
    The bar across the top of the working tab: which image, at what scale.

    Everything an analyst needs to know they are on the right image, working at
    the right scale, without leaving the tab they are clicking in.
    """
    if state.current_image is None:
        return ('<div class="work-head"><span class="wh-name">No image open</span>'
                '<span class="wh-chip">Pick one in the Gallery</span></div>')

    total = len(state.image_paths) or 1
    name = (os.path.basename(str(state.image_paths[state.current_index]))
            if state.image_paths else "image")
    chips = [(f"{state.current_index + 1} of {total}", "")]

    cal = getattr(state, "scale_calibration", None)
    if cal is None:
        chips.append(("No scale: sizes in pixels", "warn"))
    else:
        chips.append((f"{cal.nm_per_px:.4g} nm/px"
                      + ("" if cal.trustworthy else " · unconfirmed"),
                      "" if cal.trustworthy else "warn"))

    kind = state.microscope_kind()
    chips.append((kind, "") if kind else ("Microscope not set", "warn"))

    info = getattr(state, "region_info", None) or {}
    blocked = 1 - info.get("analysable_fraction", 1.0)
    if blocked > 0.005:
        chips.append((f"{100 * blocked:.0f}% excluded", ""))

    if state.analyzer is not None:
        count = len(state.analyzer.regions)
        chips.append((f"{count} particle{'' if count == 1 else 's'}", "strong"))

    spans = "".join(f'<span class="wh-chip {tone}">{_esc(text)}</span>'
                    for text, tone in chips)
    warning = cal.warning if cal is not None else None
    alert = f'<div class="wh-warning">{_esc(warning)}</div>' if warning else ""
    return (f'<div class="work-head"><span class="wh-name">{_esc(name)}</span>'
            f'{spans}</div>{alert}')


def measurement_summary():
    """The current image's particles in a few numbers, for the working tab."""
    analyzer = getattr(state, "analyzer", None)
    if analyzer is None:
        return '<div class="kv-empty">Nothing measured yet.</div>'
    m = analyzer.get_measurements(in_nm=True)
    if not m["num_particles"]:
        return _rows([("Particles", "0", None, None)])
    unit = "nm" if m["unit"] == "nm" else "px"
    d = np.asarray(m["diameters"], dtype=float)

    def size(value):          # 3 significant figures, never in e-notation
        return f"{value:,.0f}" if value >= 1000 else f"{value:.3g}"

    rows = [("Particles", str(m["num_particles"]), None, None),
            ("Median diameter", f"{size(np.median(d))} {unit}", None, None),
            ("Mean diameter", f"{size(d.mean())} {unit}", None, None),
            ("Range", f"{size(d.min())} – {size(d.max())} {unit}", None, None)]
    ratios = m.get("aspect_ratios")
    if ratios:
        rows.append(("Median aspect ratio", f"{np.median(ratios):.2f}", None, None))
    return _rows(rows)


def work_status():
    """The top bar and the measurements, which every edit changes together."""
    return frame_header(), measurement_summary()


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


def _source(cal):
    """Where a scale came from, in a few words."""
    if cal.method == "metadata":
        maker = MAKERS.get(str(cal.detail).lower())
        return f"{maker} metadata" if maker else "File metadata"
    if cal.scale_nm and cal.pixel_length:
        length = sc.format_length(cal.scale_nm)
        read = (cal.extra or {}).get("read_nm")
        if read and abs(read - cal.scale_nm) > 1e-9 * cal.scale_nm:
            length += f" (read as {sc.format_length(read)})"
        what = "Two points" if cal.method == "two_points" else "Scale bar"
        return f"{what}: {length} over {cal.pixel_length:.0f} px"
    return cal.method_label.capitalize()


def scale_card():
    """The scale in force, as the Scale panel shows it."""
    cal = getattr(state, "scale_calibration", None)
    if cal is None:
        value, badge, tone = "—", "Not set", "bad"
        source = ("No image open." if state.current_image is None
                  else "Measure the scale bar below. Until then, sizes are in pixels.")
    else:
        value = f'{cal.nm_per_px:.4g} <span class="scale-unit">nm/px</span>'
        source = _esc(_source(cal))
        if cal.method == "metadata":
            badge, tone = "From file", "ok"
        elif cal.warning:
            badge, tone = "Check", "warn"
        elif cal.trustworthy:
            badge, tone = "Confirmed", "ok"
        else:
            badge, tone = "Unconfirmed", "warn"
    warning = (f'<div class="scale-warning">{_esc(cal.warning)}</div>'
               if cal is not None and cal.warning else "")
    return ('<div class="scale-card">'
            f'<div class="scale-head"><span class="scale-value">{value}</span>'
            f'<span class="badge badge-{tone}">{badge}</span></div>'
            f'<div class="scale-source">{source}</div>{warning}</div>')


def restore_scale_tab():
    """
    The Scale tab as the session has it, for a browser connecting mid-image.

    Nothing is detected again: a page reload must not replace a scale the
    analyst set, or cost another OCR pass.

    Returns:
        tuple: (canvas_payload, message, scale_card, points_hint, frame_summary)
    """
    if state.current_image is None:
        return "", "", scale_card(), POINTS_HINT, frame_summary()
    calibration = getattr(state, "scale_calibration", None)
    box = sc.search_box(calibration) if calibration is not None else None
    payload = _image_payload(state.current_image, box, suggest=calibration is None)
    return payload, "", scale_card(), POINTS_HINT, frame_summary()


def prepare_scale_tab():
    """
    Open an image on the Scale tab and find its scale: from the metadata, else
    from the printed bar.

    Returns:
        tuple: (canvas_payload, message, scale_card, points_hint, frame_summary)
    """
    if state.current_image is None:
        # Nothing to calibrate — and drop any calibration left over from the
        # image before, which would otherwise still be reported as this one's.
        state.scale_calibration = None
        state.scale_info = None
        return "", "", scale_card(), POINTS_HINT, frame_summary()

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
        return payload, "", scale_card(), POINTS_HINT, frame_summary()

    try:
        calibration = sc.from_metadata(_detector(), path)
    except sc.ScaleError:
        return _try_automatic_bar(payload)

    _adopt(calibration)
    # Exact, and therefore the best possible yardstick for the readings that
    # follow it in this folder.
    state.accept_scale(calibration)
    return payload, "", scale_card(), POINTS_HINT, frame_summary()


def detect_scale_again():
    """
    Forget this image's scale and find it afresh, from the metadata or the bar.

    For a scale set by hand that turns out wrong. Measuring again replaces it
    anyway; this is the way back to what the file itself says.
    """
    clear_scale()
    return prepare_scale_tab()


def _try_automatic_bar(payload):
    """
    Read the printed bar without being asked, by looking where bars usually are.

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
        return (_image_payload(state.current_image, suggest=True), "", scale_card(),
                POINTS_HINT, frame_summary())

    # A misreading corrected before: the same reading of a bar the same length.
    corrected = state.known_correction(float(found["scale_nm"]), float(found["pixel_length"]))
    if corrected:
        found = dict(found, scale_nm=corrected,
                     conversion=corrected / float(found["pixel_length"]),
                     ocr_text=f"{sc.format_length(corrected)} (corrected as before)",
                     warning=None)
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
               "mark": found.get("mark"),
               "ocr_text": found.get("ocr_text", ""),
               "line_coords": found.get("line_coords")},
    )
    _adopt(calibration)

    box = found.get("region")
    if box:
        x0, y0, w, h = box
        payload = _image_payload(state.current_image, (x0, y0, x0 + w, y0 + h))

    return payload, "", scale_card(), POINTS_HINT, frame_summary()


def read_box_scale(box_json, progress=gr.Progress()):
    """Read the scale bar inside the box drawn on the canvas."""
    if state.current_image is None:
        return "Open an image first.", scale_card()

    box = _parse_box(box_json)
    if box is None:
        return "Draw a box around the scale bar and its label first.", scale_card()

    progress(0.4, desc="Reading the scale bar...")
    before = _analysable_fraction()
    try:
        calibration = sc.from_box_ocr(_detector(), state.current_image, box)
    except sc.ScaleError as exc:
        return (f"No scale bar read in the box ({exc}). Fit the box round the bar "
                "and its label, or click the bar's two ends."), scale_card()

    _adopt(calibration)
    return f"✓ Scale bar read.{_geometry_note(before)}", scale_card()


def apply_two_points(points_json, value, unit):
    """The analyst clicked both ends of the bar and typed its length."""
    points = _parse_points(points_json)
    if len(points) != 2:
        return "Click both ends of the scale bar first.", scale_card()
    try:
        length = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return "Enter the length printed on the bar.", scale_card()

    before = _analysable_fraction()
    try:
        calibration = sc.from_two_points(points[0], points[1], length, unit)
    except sc.ScaleError as exc:
        return str(exc), scale_card()

    _adopt(calibration)
    state.accept_scale(calibration)
    return f"✓ Scale set from two points.{_geometry_note(before)}", scale_card()


def confirm_scale(value=None, unit=None):
    """
    Mark the current calibration as checked by a human, with the value printed
    on the bar corrected if OCR misread it.

    Args:
        value, unit: The bar's printed length as the analyst sees it. Given for
            a bar reading; when it differs from what was read, the measured bar
            is kept and only its value replaced.
    """
    cal = getattr(state, "scale_calibration", None)
    if cal is None:
        return "No scale to confirm.", scale_card()
    if cal.method == "box_ocr" and cal.pixel_length and value not in (None, "") and unit:
        try:
            corrected = sc.relabelled(cal, float(value), unit)
        except (sc.ScaleError, ValueError) as exc:
            return str(exc), scale_card()
        if abs(corrected.scale_nm - (cal.scale_nm or 0)) > 1e-9 * corrected.scale_nm:
            # Remembered against what OCR first read; set back to that, there is
            # nothing to remember.
            read = corrected.extra["read_nm"]
            state.remember_correction(read, cal.pixel_length, corrected.scale_nm)
            _adopt(corrected)
            state.accept_scale(corrected)
            return "", scale_card()
    cal.confirmed = True
    cal.warning = None
    state.scale_info = cal.to_dict()
    # From here on, a reading that agrees with this one needs no second look.
    state.accept_scale(cal)
    return "", scale_card()


def current_frame_controls():
    """
    The frame and segmentation settings as the session has them, for a browser
    connecting to a running app. The page shows start-up defaults otherwise,
    while the session goes on using what the analyst set.

    Returns:
        tuple: (particle polarity, trim override on, trim percent,
        smallest particle)
    """
    override = state.crop_override
    return (state.particle_choice or "auto",
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
    return "", scale_card()


def set_canvas_mode(mode):
    """Show the controls for drawing a box ("box") or clicking two ends ("points")."""
    points = mode == "points"
    return (gr.update(visible=not points),   # box controls
            gr.update(visible=points))       # two-point controls


def live_point_readout(points_json):
    """Report the pixel distance between the two clicked points."""
    points = _parse_points(points_json)
    if len(points) == 0:
        return POINTS_HINT
    if len(points) == 1:
        return "Now click the other end."
    length = ((points[1][0] - points[0][0]) ** 2
              + (points[1][1] - points[0][1]) ** 2) ** 0.5
    return f"Span: **{length:.1f} px**"


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
    return " Segment again to use the new frame."


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
        return "stop", "No scale for this image, so sizes would be in pixels."
    if cal.warning:
        return "stop", cal.warning
    if cal.trustworthy:
        return None, ""

    baseline = getattr(state, "scale_baseline", None)
    accepted = getattr(state, "scale_baselines", None) or ([baseline] if baseline else [])
    if any(abs(cal.nm_per_px - b) <= SAME_SCALE_TOLERANCE * b for b in accepted):
        return None, ""
    if baseline:
        return "check", (f"**{cal.nm_per_px:.4g} nm/px**: a different "
                         f"magnification from the last scale accepted "
                         f"({baseline:.4g} nm/px). Check the bar.")
    return "check", (f"**{cal.nm_per_px:.4g} nm/px**, read from the scale bar. "
                     f"Check the measured span.")


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
                check_image, value, unit, the row holding them, value on the
                working tab, unit on the working tab, tab)
    """
    level, message = review()
    view = scale_check_view()
    # One set of updates per tab: Gradio takes the value out of an update as it
    # applies it, so an update shared by two fields left the second showing 0.
    value, unit = printed_value_fields()
    work_value, work_unit = printed_value_fields()
    return (scale_span_payload(),
            gr.update(value=view, visible=view is not None),   # on the Scale tab
            gr.update(visible=level == "check"),
            message,
            view,                                              # on the working tab
            value, unit, gr.update(visible=value.get("visible", False)),
            work_value, work_unit,
            gr.Tabs(selected=2) if level == "stop" else gr.update())


def printed_value_fields():
    """
    The value read off the bar, ready to correct, for a bar reading; hidden for
    any other scale. OCR misreads the number far more often than it mismeasures
    the bar, so correcting the number is the quick fix.
    """
    cal = getattr(state, "scale_calibration", None)
    if cal is None or cal.method != "box_ocr" or not cal.pixel_length or not cal.scale_nm:
        return gr.update(visible=False), gr.update(visible=False)
    value, unit = sc.split_length(cal.scale_nm)
    return gr.update(value=value, visible=True), gr.update(value=unit, visible=True)


def accept_from_work_tab(value=None, unit=None):
    """
    Confirm the scale from the working tab, without going to look for it.

    The whole cost of checking should be one glance and one click; making the
    analyst switch tabs to accept what they have already looked at is what makes
    supervision expensive enough to skip.
    """
    confirm_scale(value, unit)
    return check_outputs() + (frame_header(), scale_card())
