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
