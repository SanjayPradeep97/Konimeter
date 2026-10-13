"""
Visualization Functions for Gradio App

Provides image overlay and plotting functions optimized for Gradio display.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

# Axis numbers as people write them — 200, not 2×10².
_PLAIN = FuncFormatter(lambda value, _position: f"{value:g}")


def create_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Tint the masked pixels of an image, leaving the rest untouched.

    Args:
        image (np.ndarray): RGB image.
        mask (np.ndarray): Boolean mask of pixels to tint.
        color (tuple): RGB tint.
        alpha (float): Tint strength, 0-1.

    Returns:
        np.ndarray: A new RGB image; the input is not modified.
    """
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    mask = mask.astype(bool)
    overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * np.array(color)).astype(
        overlay.dtype)
    return overlay


# The working image is shown in a box about 760px tall. Gradio shrinks a larger
# frame to fit but never enlarges a smaller one, so a 512px frame sat small and
# soft in the middle of a black box. Such frames are enlarged here instead, by a
# whole number so a click maps back to exactly one pixel (handle_image_click
# divides by display_factor). Gradio's click coordinates assume no enlargement,
# which is why this cannot be left to CSS.
DISPLAY_MIN_PX = 900


def display_factor(shape):
    """How many times a frame of this shape is enlarged for the working view."""
    longest = max(shape[:2])
    return max(1, -(-DISPLAY_MIN_PX // longest)) if longest else 1


def for_display(image):
    """The working view of ``image``: enlarged by display_factor, pixels kept sharp."""
    if image is None:
        return None
    factor = display_factor(image.shape)
    if factor == 1:
        return image
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def _drawing_sizes(shape):
    """
    Line width and label size that read the same on screen whatever the frame
    size. A 2048px frame is shown at about a third of its size, so fixed sizes
    left its particle numbers a few screen pixels tall.
    """
    size = max(shape[:2]) / 1000.0
    font_scale = max(0.6, 0.75 * size)
    return {
        "line": max(2, round(2 * size)),
        "font_scale": font_scale,
        "text": max(2, round(2 * font_scale)),
        "marker": max(20, round(20 * size)),
    }


def create_particle_visualization(image, labeled_mask, regions, show_labels=True,
                                 pending_deletes=None, pending_add_masks=None, pending_merge=None):
    """
    Create visualization with particle contours and numbered labels.

    Args:
        image (np.ndarray): RGB image
        labeled_mask (np.ndarray): Labeled segmentation mask
        regions (list): RegionProperties objects
        show_labels (bool): Whether to show particle numbers
        pending_deletes (list): List of particle labels queued for deletion (yellow outline)
        pending_add_masks (list): List of masks queued for addition (green outline)
        pending_merge (list): List of particle labels selected for merging (blue outline)

    Returns:
        np.ndarray: Annotated image
    """
    # Create a copy to draw on
    vis_image = image.copy()
    sizes = _drawing_sizes(image.shape)

    pending_deletes = pending_deletes or []
    pending_add_masks = pending_add_masks or []
    pending_merge = pending_merge or []

    # Draw contours and labels for each particle
    for idx, region in enumerate(regions, start=1):
        # Get binary mask for this particle
        mask_i = (labeled_mask == region.label).astype(np.uint8)

        # RETR_CCOMP, not RETR_EXTERNAL: a particle can have a hole in it — a
        # nanotube that loops back on itself encloses background, and that
        # background is not part of the particle. RETR_EXTERNAL returns only
        # outer boundaries, so such a particle was drawn as a solid blob however
        # carefully its hole had been cut out. The measurement was right and the
        # picture disagreed with it, which is the wrong way round for a view
        # whose whole job is letting the mask be judged.
        contours, _ = cv2.findContours(mask_i, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Choose color based on status
        if region.label in pending_deletes:
            color = (255, 255, 0)  # Yellow for pending deletion
            thickness = 2 * sizes["line"]
        elif region.label in pending_merge:
            color = (0, 0, 255)  # Blue for pending merge
            thickness = 2 * sizes["line"]
        else:
            color = (255, 0, 0)  # Red for normal
            thickness = sizes["line"]

        # Draw contours
        cv2.drawContours(vis_image, contours, -1, color, thickness)

        # Add numbered label at centroid
        if show_labels:
            y, x = int(region.centroid[0]), int(region.centroid[1])

            # Draw text background
            text = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = sizes["font_scale"]
            text_thickness = sizes["text"]

            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, text_thickness)

            # Draw black rectangle background
            cv2.rectangle(vis_image,
                         (x - text_w//2 - 3, y - text_h//2 - 3),
                         (x + text_w//2 + 3, y + text_h//2 + 3),
                         (0, 0, 0), -1)

            # Draw white text
            cv2.putText(vis_image, text,
                       (x - text_w//2, y + text_h//2),
                       font, font_scale, (255, 255, 255), text_thickness)

    # Draw pending additions in green
    for add_mask in pending_add_masks:
        if add_mask is not None and add_mask.any():
            mask_uint8 = add_mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2 * sizes["line"])

    return for_display(vis_image)


def create_point_refine_visualization(image, refined_mask, point_coords, point_labels):
    """
    Create visualization for point refinement mode showing the refined mask and point markers.

    Args:
        image (np.ndarray): RGB image
        refined_mask (np.ndarray): Boolean mask of the refined particle
        point_coords (list): List of (x, y) click coordinates
        point_labels (list): List of point labels (1=positive, 0=negative)

    Returns:
        np.ndarray: Annotated image with mask and point markers
    """
    vis_image = image.copy()
    sizes = _drawing_sizes(image.shape)

    # Draw the refined mask contour in white
    if refined_mask is not None and refined_mask.any():
        mask_uint8 = refined_mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (255, 255, 255), sizes["line"] + 1)

    # Draw point markers
    for (x, y), label in zip(point_coords, point_labels):
        x_int, y_int = int(x), int(y)
        if label == 1:
            # Positive point: Green + marker
            cv2.drawMarker(vis_image, (x_int, y_int), (0, 255, 0),
                          markerType=cv2.MARKER_CROSS, markerSize=sizes["marker"],
                          thickness=sizes["line"] + 1)
        else:
            # Negative point: Red × marker
            cv2.drawMarker(vis_image, (x_int, y_int), (255, 0, 0),
                          markerType=cv2.MARKER_TILTED_CROSS, markerSize=sizes["marker"],
                          thickness=sizes["line"] + 1)

    return for_display(vis_image)


def visualize_three_masks(image, masks, labels):
    """
    Create a panel visualization showing the mask candidates side by side.

    Args:
        image (np.ndarray): RGB image
        masks: Sequence of boolean masks (H, W)
        labels: One caption per mask — either a ready-made string describing the
            candidate, or a bare confidence score.

    Returns:
        np.ndarray: Combined panel image
    """
    num_masks = len(masks)

    # Create subplots
    fig, axes = plt.subplots(1, num_masks, figsize=(15, 5))

    if num_masks == 1:
        axes = [axes]

    for idx, (mask, label) in enumerate(zip(masks, labels)):
        ax = axes[idx]

        # Create overlay
        overlay = create_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5)

        # Labels may be pre-formatted strings or bare confidence scores.
        if isinstance(label, str):
            title = label if label.lower().startswith("option") else f"Mask {idx + 1}\n{label}"
        else:
            title = f"Mask {idx + 1}\nScore: {label:.3f}"

        ax.imshow(overlay)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    # Convert plot to image
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    # Convert RGBA to RGB
    img_array = img_array[:, :, :3]
    plt.close(fig)

    return img_array


def _column_names(measurements):
    """
    Area and diameter headers in the unit the numbers are actually in.

    Hard-coded nm used to label pixel counts as nanometres whenever an image
    had no scale, which is the one case where the difference matters most.
    """
    unit = 'nm' if measurements.get('unit') == 'nm' else 'px'
    return f'Area ({unit}²)', f'Diameter ({unit})'


def create_results_dataframe(measurements):
    """
    Convert measurements dict to a formatted DataFrame for display.

    Args:
        measurements (dict): Measurements from ParticleAnalyzer

    Returns:
        pd.DataFrame: Formatted results table
    """
    import pandas as pd

    area, diameter = _column_names(measurements)
    ratios = measurements.get('aspect_ratios')
    columns = ['Particle ID', area, diameter] + (['Aspect ratio'] if ratios is not None else [])
    if measurements['num_particles'] == 0:
        return pd.DataFrame(columns=columns)

    # Create DataFrame
    data = {
        'Particle ID': list(range(1, measurements['num_particles'] + 1)),
        area: [f"{a:.1f}" for a in measurements['areas']],
        diameter: [f"{d:.1f}" for d in measurements['diameters']],
    }
    if ratios is not None:
        data['Aspect ratio'] = [f"{r:.2f}" for r in ratios]

    df = pd.DataFrame(data)
    return df


def create_summary_statistics_table(measurements):
    """
    Create a summary statistics table.

    Args:
        measurements (dict): Measurements from ParticleAnalyzer

    Returns:
        pd.DataFrame: Summary statistics
    """
    import pandas as pd

    if measurements['num_particles'] == 0:
        return pd.DataFrame()

    areas = np.array(measurements['areas'])
    diameters = np.array(measurements['diameters'])
    area, diameter = _column_names(measurements)

    stats_data = {
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        area: [
            f"{np.mean(areas):.1f}",
            f"{np.median(areas):.1f}",
            f"{np.std(areas):.1f}",
            f"{np.min(areas):.1f}",
            f"{np.max(areas):.1f}"
        ],
        diameter: [
            f"{np.mean(diameters):.1f}",
            f"{np.median(diameters):.1f}",
            f"{np.std(diameters):.1f}",
            f"{np.min(diameters):.1f}",
            f"{np.max(diameters):.1f}"
        ]
    }
    ratios = measurements.get('aspect_ratios')
    if ratios:
        ratios = np.asarray(ratios, dtype=float)
        stats_data['Aspect ratio'] = [f"{f(ratios):.2f}" for f in
                                      (np.mean, np.median, np.std, np.min, np.max)]

    return pd.DataFrame(stats_data)


def _histogram(ax, values, label, colour, log_scale):
    """
    One size distribution on one axis.

    On a log axis the bins are spaced evenly in log, so each bin covers the same
    ratio of sizes, and the box gives the geometric mean and geometric standard
    deviation, the usual summary of a particle size distribution.
    """
    values = np.asarray(values, dtype=float)
    bins = min(30, len(values))
    positive = values[values > 0]
    if log_scale and len(positive):
        low, high = positive.min(), positive.max()
        if high > low:
            bins = np.geomspace(low, high, bins + 1)
        ax.hist(positive, bins=bins, color=colour, edgecolor='black', alpha=0.7)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(_PLAIN)
        ax.xaxis.set_minor_formatter(NullFormatter())
        logs = np.log(positive)
        stats = (f'Geometric mean: {np.exp(logs.mean()):.3g}\n'
                 f'GSD: {np.exp(logs.std()):.2f}\n'
                 f'Median: {np.median(positive):.3g}')
    else:
        ax.hist(values, bins=bins, color=colour, edgecolor='black', alpha=0.7)
        stats = (f'Mean: {np.mean(values):.3g}\nMedian: {np.median(values):.3g}\n'
                 f'Std: {np.std(values):.3g}')
    ax.set_xlabel(label + (' (log scale)' if log_scale else ''), fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.98, 0.98, stats, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)


# Length over width at which a particle counts as a fibre in occupational fibre
# counting (NIOSH 7400, WHO), marked on the aspect-ratio plot.
FIBRE_ASPECT_RATIO = 3.0


def _aspect_histogram(ax, ratios, log_scale):
    """
    Particle shape: how elongated each one is, with the fibre ratio marked.

    Size alone cannot tell a fibre from a compact particle of the same area;
    this is the plot that can.
    """
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios[np.isfinite(ratios) & (ratios >= 1)]
    high = max(float(ratios.max()), FIBRE_ASPECT_RATIO * 1.5)
    count = min(30, max(len(ratios), 1))
    if log_scale:
        bins = np.geomspace(1, high, count + 1)
        ax.set_xscale('log')
        ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500) if t <= high]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(_PLAIN)
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        bins = np.linspace(1, high, count + 1)
    ax.hist(ratios, bins=bins, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.axvline(FIBRE_ASPECT_RATIO, color='black', linestyle='--', linewidth=1.2)
    ax.text(FIBRE_ASPECT_RATIO, 0.97, ' 3:1', transform=ax.get_xaxis_transform(),
            va='top', ha='left', fontsize=10)
    fibres = float((ratios >= FIBRE_ASPECT_RATIO).mean())
    stats = (f'Median: {np.median(ratios):.2f}\n'
             f'At or above 3:1: {100 * fibres:.0f}%')
    ax.set_xlabel('Aspect ratio, length / width' + (' (log scale)' if log_scale else ''),
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.98, 0.98, stats, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)


def _nothing_to_plot(ax, text):
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12,
            transform=ax.transAxes)
    ax.axis('off')


def create_histogram_plots(measurements, log_scale=False):
    """
    The size and shape distributions of a set of particles, side by side.

    Size is the equivalent diameter; area would only repeat it on another
    scale. Shape is the aspect ratio, which says what size cannot: how many of
    the particles are fibres.

    Args:
        measurements (dict): 'diameters', 'aspect_ratios' and 'unit'. Either
            list may be empty.
        log_scale (bool): Log axes. Particle sizes usually span orders of
            magnitude, and on a linear axis one large particle crowds every
            other into the first bar.

    Returns:
        np.ndarray: The figure, as an RGB image.
    """
    diameters = np.asarray(measurements.get('diameters') or [], dtype=float)
    ratios = measurements.get('aspect_ratios') or []
    unit = measurements.get('unit', 'nm')

    fig = Figure(figsize=(14, 5))
    FigureCanvasAgg(fig)
    size_ax, shape_ax = fig.subplots(1, 2)

    if len(diameters):
        _histogram(size_ax, diameters, f'Equivalent diameter ({unit})', 'coral',
                   log_scale)
        size_ax.set_title(f'Size (n={len(diameters)})', fontsize=14, fontweight='bold')
    else:
        _nothing_to_plot(size_ax, 'No sizes to plot')

    if len(ratios):
        _aspect_histogram(shape_ax, ratios, log_scale)
        shape_ax.set_title(f'Shape (n={len(ratios)})', fontsize=14, fontweight='bold')
    else:
        _nothing_to_plot(shape_ax, 'No aspect ratios recorded')

    fig.tight_layout()
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


# The scale check is drawn at a fixed width so a bar that is 80px in a 2048px
# frame is still legible. Upscaled only — a bar already wider than this is shown
# at its own size rather than shrunk.
SCALE_CHECK_WIDTH = 760
SCALE_CHECK_MAX_ZOOM = 10


def render_scale_check(image, calibration):
    """
    Crop the measured scale bar and draw the span that was measured across it.

    The reported nm/px is the printed length divided by a pixel count, and the
    pixel count is the one part of it that cannot be checked by reading. A
    measurement that stopped at a tick, caught the bar's anti-aliased edge or ran
    along a neighbouring rule looks perfectly reasonable as a number and is
    unmistakable as a picture.

    Args:
        image (np.ndarray): The full frame the calibration was measured in.
        calibration: A ``ScaleCalibration``, or None.

    Returns:
        np.ndarray: RGB crop with the span drawn on it, or None when there is
        nothing to show — no calibration, or one from metadata, which carries a
        pixel size and never measured a bar.
    """
    from sem_particle_analysis import scale_calibration as sc

    if image is None or calibration is None:
        return None
    span = sc.measured_span(calibration)
    if span is None:
        return None

    (ax, ay), (bx, by) = span
    length = max(1.0, ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)

    # Enough room around the span to include whatever is printed with the bar:
    # the label sits within about a bar's length above or below it.
    pad_x = max(16.0, 0.15 * length)
    pad_y = max(30.0, 0.45 * length)
    height, width = image.shape[:2]
    x0 = int(max(0, min(ax, bx) - pad_x))
    x1 = int(min(width, max(ax, bx) + pad_x))
    y0 = int(max(0, min(ay, by) - pad_y))
    y1 = int(min(height, max(ay, by) + pad_y))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    crop = image[y0:y1, x0:x1]
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    crop = np.ascontiguousarray(crop[..., :3].astype(np.uint8))

    thickness = _bar_half_thickness(crop, ((ax + bx) / 2) - x0, ((ay + by) / 2) - y0,
                                    limit=int(pad_y))

    # Nearest-neighbour: this is a view for judging exactly where the ends fell,
    # and interpolation would invent the very edge being judged.
    zoom = min(SCALE_CHECK_MAX_ZOOM, max(1.0, SCALE_CHECK_WIDTH / (x1 - x0)))
    if zoom > 1:
        crop = cv2.resize(crop, None, fx=zoom, fy=zoom,
                          interpolation=cv2.INTER_NEAREST)

    view = _draw_span(crop, (ax - x0) * zoom, (ay - y0) * zoom,
                      (bx - x0) * zoom, (by - y0) * zoom,
                      thickness * zoom, pad_y * zoom)
    return _caption(view, _span_label(calibration, length))


def _bar_half_thickness(crop, x, y, limit):
    """
    How far the bar reaches above and below the row that was measured.

    The dimension line is drawn clear of the bar rather than along it — a red
    line inside a black bar is barely visible, which is the opposite of the
    point. Bars run from two pixels thick to a couple of dozen depending on the
    frame, so the clearance is measured rather than guessed.

    Args:
        crop: The region around the bar, RGB or greyscale.
        x, y: A point on the bar, in crop coordinates.
        limit: How far to walk before giving up.

    Returns:
        int: Half-thickness in crop pixels, at least 1.
    """
    gray = crop if crop.ndim == 2 else crop[..., :3].mean(axis=2)
    height, width = gray.shape[:2]
    x = int(round(min(max(x, 0), width - 1)))
    y = int(round(min(max(y, 0), height - 1)))
    reference = float(gray[y, x])

    # Wide enough to keep following a bar through its own noise, tight enough to
    # stop at the background it sits on — printed bars are near-saturated one way
    # or the other, so the two are far apart.
    tolerance = 60.0
    reach = 1
    for step in (1, -1):
        run, probe = 0, y + step
        while (0 <= probe < height and run < limit
               and abs(float(gray[probe, x]) - reference) <= tolerance):
            run += 1
            probe += step
        reach = max(reach, run)
    return reach


def _span_label(calibration, pixel_span):
    """What the measurement says, in the order it was derived."""
    pixels = calibration.pixel_length or pixel_span
    parts = [f"{pixels:.0f} px"]
    if calibration.scale_nm:
        from sem_particle_analysis.scale_calibration import format_length

        parts.append(f"= {format_length(calibration.scale_nm)}")
    parts.append(f"({calibration.nm_per_px:.4g} nm/px)")
    return "  ".join(parts)


# Drawn twice, dark underneath, so the marks read on a bright databar and on a
# dark micrograph alike.
_SPAN_COLOR = (255, 64, 96)
_SPAN_SHADOW = (0, 0, 0)


def _draw_span(view, ax, ay, bx, by, half_thickness, room_below):
    """
    Mark the measured segment with end ticks and a dimension line.

    Args:
        half_thickness: Half the bar's height, so the line clears it.
        room_below: How much image there is under the bar to draw into.
    """
    offset = int(round(min(max(half_thickness + 6, 10), max(10, room_below - 8))))
    ticks = int(round(max(8, half_thickness + 6)))
    ends = [(int(round(ax)), int(round(ay))), (int(round(bx)), int(round(by)))]

    for color, weight in ((_SPAN_SHADOW, 5), (_SPAN_COLOR, 2)):
        # Below the bar, so the bar itself stays visible above it.
        cv2.line(view, (ends[0][0], ends[0][1] + offset),
                 (ends[1][0], ends[1][1] + offset), color, weight, cv2.LINE_AA)
        # The ticks cross the bar at each measured end, which is the thing being
        # checked: they should land on the ends of the printed bar, not inside
        # it and not past it.
        for x, y in ends:
            cv2.line(view, (x, y - ticks), (x, y + offset + 6), color, weight,
                     cv2.LINE_AA)
    return view


def _caption(view, text):
    """Write the measurement across the bottom of the view."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 8
    band = th + baseline + 2 * pad
    height, width = view.shape[:2]
    if width < tw + 2 * pad:
        view = cv2.copyMakeBorder(view, 0, 0, 0, tw + 2 * pad - width,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
        width = view.shape[1]
    view = cv2.copyMakeBorder(view, 0, band, 0, 0, cv2.BORDER_CONSTANT,
                              value=(16, 16, 16))
    cv2.putText(view, text, (pad, height + pad + th), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return view
