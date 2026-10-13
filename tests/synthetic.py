"""
Synthetic SEM micrograph generation for tests.

These builders produce images whose ground truth is known exactly — scale bar
length in pixels, the nm value printed next to it, and the particle mask — so
measurements can be checked against a number rather than against a previous run.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# A real font is needed for OCR to have any chance; PIL's bitmap default font is
# too small and too thin to be read reliably.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_micrograph(
    width=1024,
    image_height=704,
    databar_height=96,
    databar_gray=0,
    bar_gray=255,
    bar_length_px=200,
    scale_text="500 nm",
    scale_nm=500.0,
    bar_thickness=8,
    bar_left=60,
    n_particles=40,
    particle_seed=0,
    text_size=40,
):
    """
    Build a synthetic SEM-style micrograph with a databar and scale bar.

    The layout mirrors a real instrument export: a noisy grayscale micrograph on
    top, and below it a solid databar carrying the scale bar and its label.

    Args:
        databar_gray: Databar background level. 0 gives the black databar used by
            FEI/Zeiss; a high value gives a light databar.
        bar_gray: Scale bar level. Should contrast with ``databar_gray``.
        bar_length_px: Exact scale bar length in pixels (ground truth).
        scale_nm: Physical length the bar represents (ground truth).

    Returns:
        tuple: (rgb_image, truth) where truth is a dict with keys
            ``nm_per_px``, ``bar_length_px``, ``scale_nm``, ``particle_mask``,
            ``n_particles_drawn``, ``image_height``, ``databar_top``.
    """
    rng = np.random.default_rng(particle_seed)
    total_height = image_height + databar_height

    # --- micrograph area: dark, mildly textured background ---
    micrograph = rng.normal(38, 6, size=(image_height, width))

    yy, xx = np.mgrid[0:image_height, 0:width]
    truth_mask = np.zeros((image_height, width), dtype=bool)
    for _ in range(n_particles):
        cx = rng.uniform(70, width - 70)
        cy = rng.uniform(70, image_height - 70)
        r = rng.uniform(14, 34)
        aspect = rng.uniform(0.7, 1.4)
        theta = rng.uniform(0, np.pi)
        dx, dy = xx - cx, yy - cy
        xr = dx * np.cos(theta) + dy * np.sin(theta)
        yr = -dx * np.sin(theta) + dy * np.cos(theta)
        blob = (xr / r) ** 2 + (yr / (r * aspect)) ** 2 <= 1.0
        truth_mask |= blob
        micrograph[blob] = rng.normal(185, 10, size=int(blob.sum()))

    micrograph = np.clip(micrograph, 0, 255).astype(np.uint8)

    # --- assemble full frame with databar ---
    full = np.full((total_height, width), databar_gray, dtype=np.uint8)
    full[:image_height] = micrograph
    pil = Image.fromarray(full).convert("RGB")
    draw = ImageDraw.Draw(pil)

    fill = (bar_gray, bar_gray, bar_gray)
    bar_top = image_height + databar_height // 2
    # PIL's rectangle includes both corners, so the far edge is length - 1: the
    # bar covers exactly bar_length_px pixels, which is what "truth" claims.
    draw.rectangle(
        [bar_left, bar_top, bar_left + bar_length_px - 1, bar_top + bar_thickness - 1],
        fill=fill,
    )

    # Label sits above the bar so a search box covering the databar catches both.
    font = _load_font(text_size)
    draw.text((bar_left, image_height + 8), scale_text, fill=fill, font=font)

    truth = {
        "nm_per_px": scale_nm / bar_length_px,
        "bar_length_px": bar_length_px,
        "scale_nm": scale_nm,
        "particle_mask": truth_mask,
        "n_particles_drawn": n_particles,
        "image_height": image_height,
        "databar_top": image_height,
        "databar_height": databar_height,
    }
    return np.array(pil), truth


def make_zeiss_frame(width=1024, height=768, strip_top=690, strip_bottom=749,
                     tick_centres=(40, 116), seed=3):
    """
    A Zeiss-style frame: a white data strip framed by black rules, inset a few
    rows above the bottom edge with micrograph below it, and a "2 µm" bar drawn
    as |—| whose length is measured between the ticks' centres.

    Returns:
        tuple: (rgb_image, truth) with ``databar_top``, ``bar_length_px`` (tick
        centre to centre), ``scale_nm`` and ``nm_per_px``.
    """
    rng = np.random.default_rng(seed)
    frame = np.clip(rng.normal(120, 18, size=(height, width)), 0, 255).astype(np.uint8)
    frame[strip_top:strip_bottom + 1] = 255
    frame[strip_top:strip_top + 2] = 0                     # frame rules
    frame[strip_bottom - 1:strip_bottom + 1] = 0
    pil = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = _load_font(22)
    black = (0, 0, 0)
    left, right = tick_centres
    bar_row = strip_top + 40
    draw.text((left, strip_top + 8), "2 µm", fill=black, font=font)
    draw.rectangle([left - 1, bar_row - 1, right + 1, bar_row], fill=black)     # 2 px rule
    for x in (left, right):
        draw.rectangle([x - 1, bar_row - 9, x, bar_row + 8], fill=black)         # 2 px ticks
    draw.text((260, strip_top + 8), "Mag =  10.00 K X", fill=black, font=font)
    draw.text((520, strip_top + 8), "WD =  9.0 mm", fill=black, font=font)
    draw.text((420, strip_top + 32), "Date :20 Aug 2024", fill=black, font=font)
    length = right - left
    return np.array(pil), {"databar_top": strip_top, "bar_length_px": length,
                           "scale_nm": 2000.0, "nm_per_px": 2000.0 / length}


def make_fei_frame(width=1536, image_height=1024, databar_height=70, bar=(900, 1450),
                   seed=4):
    """
    An FEI-style frame: a dark grey databar below the image, with the "5 µm"
    label printed in a gap in the middle of the bar, which has end ticks.

    Returns:
        tuple: (rgb_image, truth) with ``bar_length_px`` (tick centre to centre)
        and ``scale_nm``.
    """
    rng = np.random.default_rng(seed)
    height = image_height + databar_height
    frame = np.full((height, width), 45, dtype=np.uint8)
    frame[:image_height] = np.clip(rng.normal(90, 25, size=(image_height, width)), 0, 255)
    frame[image_height + 44] = 140                                   # divider line
    pil = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = _load_font(24)
    white = (235, 235, 235)
    left, right = bar
    row = image_height + 20
    label = "5 µm"
    text_w = draw.textlength(label, font=font)
    middle = (left + right) / 2
    gap0, gap1 = middle - text_w / 2 - 6, middle + text_w / 2 + 6
    draw.rectangle([left, row, gap0, row + 1], fill=white)
    draw.rectangle([gap1, row, right, row + 1], fill=white)
    for x in (left, right):
        draw.rectangle([x - 1, row - 8, x, row + 9], fill=white)
    draw.text((middle - text_w / 2, row - 14), label, fill=white, font=font)
    draw.text((40, image_height + 6), "WD  curr  det  HV", fill=white, font=font)
    draw.text((middle - 30, image_height + 48), "Teneo", fill=white, font=font)
    return np.array(pil), {"bar_length_px": right - left, "scale_nm": 5000.0,
                           "databar_top": image_height}


def make_jeol_frame(width=1280, image_height=960, strip_height=64, bar=(1000, 1159),
                    seed=5):
    """
    A JEOL SEM-style frame: a black strip below the image with white text, and
    the scale bar in the strip's top rows, above its "1µm" label.
    """
    rng = np.random.default_rng(seed)
    height = image_height + strip_height
    frame = np.zeros((height, width), dtype=np.uint8)
    frame[:image_height] = np.clip(rng.normal(150, 20, size=(image_height, width)), 0, 255)
    pil = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = _load_font(28)
    white = (255, 255, 255)
    left, right = bar
    draw.rectangle([left, image_height, right, image_height + 5], fill=white)
    draw.text((left + 40, image_height + 22), "1µm", fill=white, font=font)
    draw.text((20, image_height + 22), "CSU   SEI   15.0kV   X15,000   WD 15.0mm",
              fill=white, font=font)
    return np.array(pil), {"bar_length_px": right - left + 1, "scale_nm": 1000.0,
                           "databar_top": image_height}


def databar_region(truth, width_frac=0.55, x_center=0.30):
    """
    Search-region kwargs covering exactly the databar of a ``make_micrograph``
    image, for passing to ``ScaleDetector.detect_scale_bar``.
    """
    total_height = truth["image_height"] + truth["databar_height"]
    return {
        "region_x": x_center,
        "region_y": (truth["databar_top"] + truth["databar_height"] / 2) / total_height,
        "region_width": width_frac,
        "region_height": truth["databar_height"] / total_height,
    }


def make_disk_mask(shape=(400, 400), centers_radii=((100, 100, 30), (300, 250, 45))):
    """
    Binary mask of non-touching disks with exactly known areas, for checking the
    analyzer's area and equivalent-diameter arithmetic.

    Returns:
        tuple: (mask, radii) with mask as uint8 and radii in pixels.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    radii = []
    for cx, cy, r in centers_radii:
        mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 1
        radii.append(r)
    return mask, radii


def make_burnin_micrograph(
    width=1024,
    height=1024,
    background_gray=200,
    noise=9.0,
    bar_gray=0,
    bar_length_px=128,
    bar_thickness=12,
    bar_left=33,
    bar_bottom_margin=28,
    scale_text="1 µm",
    scale_nm=1000.0,
    text_size=44,
    seed=0,
):
    """
    A TEM-style frame with the scale bar burned into the image itself.

    Modelled on the JEOL exports in use: a light, noisy field with **no databar**
    and a solid dark bar plus its label sitting in the bottom-left of the
    micrograph. This is the layout that exposed two real defects — the scale bar
    detector preferring a ragged bright run over the solid dark bar, and the
    databar detector claiming a bar where there was none.

    Returns:
        tuple: (rgb_image, truth) with ``nm_per_px``, ``bar_length_px``,
        ``scale_nm``, ``bar_box`` as (x0, y0, w, h).
    """
    rng = np.random.default_rng(seed)
    field = np.clip(rng.normal(background_gray, noise, size=(height, width)), 0, 255)
    pil = Image.fromarray(field.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(pil)

    fill = (bar_gray, bar_gray, bar_gray)
    bar_top = height - bar_bottom_margin - bar_thickness
    # PIL's rectangle includes both corners, so the far edge is length - 1: the
    # bar covers exactly bar_length_px pixels, which is what "truth" claims.
    draw.rectangle(
        [bar_left, bar_top, bar_left + bar_length_px - 1, bar_top + bar_thickness - 1],
        fill=fill,
    )
    font = _load_font(text_size)
    draw.text((bar_left, bar_top - text_size - 6), scale_text, fill=fill, font=font)

    truth = {
        "nm_per_px": scale_nm / bar_length_px,
        "bar_length_px": bar_length_px,
        "scale_nm": scale_nm,
        "bar_box": (bar_left, bar_top, bar_length_px, bar_thickness),
        "height": height,
        "width": width,
    }
    return np.array(pil), truth


def make_grid_bar_frame(width=1024, height=1024, background_gray=205, noise=9.0,
                        seed=1):
    """
    A TEM frame whose bottom corner is blocked by the specimen grid bar.

    The dark wedge is uniform enough, and contrasts strongly enough with the
    film, to look like a databar to any test based on row statistics alone — but
    it is micrograph, and cropping it throws away data. Its top edge is diagonal,
    which is what actually distinguishes it from a drawn bar.
    """
    rng = np.random.default_rng(seed)
    field = np.clip(rng.normal(background_gray, noise, size=(height, width)), 0, 255)

    yy, xx = np.mgrid[0:height, 0:width]
    # A diagonal band across the bottom-left, as a grid bar appears. Sized to
    # cover roughly a fifth of the frame, matching what the real TEM frames show.
    wedge = yy > (height * 0.55 + 0.45 * xx)
    field[wedge] = rng.normal(4, 2.5, size=int(wedge.sum())).clip(0, 255)

    return np.stack([field.astype(np.uint8)] * 3, axis=-1)


def make_vignetted_frame(width=1024, height=768, background_gray=110, noise=8.0,
                         radius_frac=0.48, particle_gray=225, n_particles=12,
                         seed=2):
    """
    An SEM-style frame shot through a circular aperture.

    The corners fall to black, which is what a whole-frame mask latches onto:
    on the real set those wedges *were* the reported particles. Bright specks are
    scattered inside the illuminated disc so there is something real to find.

    Returns:
        tuple: (rgb_image, truth) with ``particle_mask`` and ``corner_mask``.
    """
    rng = np.random.default_rng(seed)
    field = np.clip(rng.normal(background_gray, noise, size=(height, width)), 0, 255)

    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height / 2, width / 2
    radius = radius_frac * max(height, width)
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
    field[~inside] = rng.normal(2, 1.5, size=int((~inside).sum())).clip(0, 255)

    particles = np.zeros((height, width), dtype=bool)
    for _ in range(n_particles):
        r = rng.uniform(6, 14)
        px = rng.uniform(cx - radius * 0.6, cx + radius * 0.6)
        py = rng.uniform(cy - radius * 0.5, cy + radius * 0.5)
        blob = ((xx - px) ** 2 + (yy - py) ** 2) <= r ** 2
        blob &= inside
        particles |= blob
    field[particles] = rng.normal(particle_gray, 6, size=int(particles.sum())).clip(0, 255)

    image = np.stack([field.astype(np.uint8)] * 3, axis=-1)
    return image, {"particle_mask": particles, "corner_mask": ~inside,
                   "illuminated": inside}
