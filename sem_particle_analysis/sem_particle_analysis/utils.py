"""Helpers shared by the library: loading images, finding them on disk, and
summarising a set of measurements."""

import os
import cv2
import numpy as np

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


def load_image(file_path):
    """
    Load an image from file and convert to RGB.

    Args:
        file_path (str): Path to image file

    Returns:
        np.ndarray: RGB image array

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    bgr = cv2.imread(file_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {file_path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_tiff_metadata(file_path):
    """
    Extract metadata from a TIFF file for scale detection.

    Extracts key metadata fields that may contain pixel size or resolution
    information from SEM/TEM microscope images.

    Args:
        file_path (str): Path to TIFF file

    Returns:
        dict: Metadata dictionary containing:
            - 'image_description': Raw ImageDescription tag (str or None)
            - 'x_resolution': X resolution value (float or None)
            - 'y_resolution': Y resolution value (float or None)
            - 'resolution_unit': Unit code (1=none, 2=inch, 3=cm) or None
            - 'software': Software tag if present
            - 'raw_tags': Dict of all available TIFF tags
            - 'is_tiff': Boolean indicating successful TIFF read

    Raises:
        ValueError: If file is not a valid TIFF or tifffile not available
    """
    if not TIFFFILE_AVAILABLE:
        raise ValueError("tifffile library not available. Install with: pip install tifffile")

    # Check file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ('.tif', '.tiff'):
        raise ValueError(f"Not a TIFF file: {file_path}")

    metadata = {
        'image_description': None,
        'x_resolution': None,
        'y_resolution': None,
        'resolution_unit': None,
        'software': None,
        'raw_tags': {},
        'is_tiff': False
    }

    try:
        with tifffile.TiffFile(file_path) as tif:
            metadata['is_tiff'] = True

            # Get the first page's tags
            if tif.pages:
                page = tif.pages[0]
                tags = page.tags

                # Extract common tags
                # Tag 270: ImageDescription
                if 'ImageDescription' in tags:
                    desc = tags['ImageDescription'].value
                    if isinstance(desc, bytes):
                        desc = desc.decode('utf-8', errors='ignore')
                    metadata['image_description'] = desc

                # Tag 282: XResolution
                if 'XResolution' in tags:
                    val = tags['XResolution'].value
                    # Handle rational (tuple) or float
                    if isinstance(val, tuple):
                        metadata['x_resolution'] = val[0] / val[1] if val[1] != 0 else None
                    else:
                        metadata['x_resolution'] = float(val)

                # Tag 283: YResolution
                if 'YResolution' in tags:
                    val = tags['YResolution'].value
                    if isinstance(val, tuple):
                        metadata['y_resolution'] = val[0] / val[1] if val[1] != 0 else None
                    else:
                        metadata['y_resolution'] = float(val)

                # Tag 296: ResolutionUnit (1=none, 2=inch, 3=centimeter)
                if 'ResolutionUnit' in tags:
                    metadata['resolution_unit'] = tags['ResolutionUnit'].value

                # Tag 305: Software
                if 'Software' in tags:
                    software = tags['Software'].value
                    if isinstance(software, bytes):
                        software = software.decode('utf-8', errors='ignore')
                    metadata['software'] = software

                # Extract image dimensions
                if 'ImageWidth' in tags:
                    metadata['image_width'] = tags['ImageWidth'].value
                if 'ImageLength' in tags:
                    metadata['image_height'] = tags['ImageLength'].value

                # Store all tags for debugging/advanced parsing
                # Index by BOTH string name and numeric code for flexible lookup
                for tag_name, tag in tags.items():
                    try:
                        val = tag.value
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='ignore')
                        # Store by string name (existing behavior)
                        metadata['raw_tags'][tag_name] = val
                        # Also store by numeric code for manufacturer parsers
                        if hasattr(tag, 'code'):
                            metadata['raw_tags'][tag.code] = val
                    except Exception:
                        # Skip tags that can't be easily converted
                        pass

    except Exception as e:
        raise ValueError(f"Failed to read TIFF metadata: {e}")

    return metadata


def create_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.4):
    """
    Create an image with a colored mask overlay.

    Args:
        image (np.ndarray): RGB image
        mask (np.ndarray): Boolean mask
        color (tuple): RGB color for overlay (0-255)
        alpha (float): Transparency (0-1)

    Returns:
        np.ndarray: Image with overlay
    """
    overlay = image.copy()
    overlay[mask] = (
        overlay[mask] * (1 - alpha) +
        np.array(color) * alpha
    ).astype(np.uint8)

    return overlay


def find_images_in_folder(folder_path, extensions=None):
    """
    Find all image files in a folder.

    Args:
        folder_path (str): Path to folder
        extensions (list, optional): List of extensions to search for.
            Default: ['png', 'jpg', 'jpeg', 'tif', 'tiff']

    Returns:
        list: Sorted list of image file paths
    """
    import glob
    import os

    if extensions is None:
        extensions = ['png', 'jpg', 'jpeg', 'tif', 'tiff']

    image_paths = []
    for ext in extensions:
        pattern = os.path.join(folder_path, f'*.{ext}')
        image_paths.extend(glob.glob(pattern))

    return sorted(image_paths)


def print_summary(measurements, title="Analysis Summary"):
    """
    Print a formatted summary of particle measurements.

    Args:
        measurements (dict): Measurements dictionary
        title (str): Summary title
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)

    print(f"Number of particles: {measurements['num_particles']}")

    if measurements['num_particles'] == 0:
        print("No particles detected")
        return

    unit = measurements['unit']

    print(f"\nArea Statistics ({unit}²):" if unit == 'nm' else f"\nArea Statistics ({unit}):")
    areas = np.array(measurements['areas'])
    print(f"  Mean:   {np.mean(areas):.2f}")
    print(f"  Median: {np.median(areas):.2f}")
    print(f"  Std:    {np.std(areas):.2f}")
    print(f"  Min:    {np.min(areas):.2f}")
    print(f"  Max:    {np.max(areas):.2f}")

    print(f"\nDiameter Statistics ({unit}):")
    diameters = np.array(measurements['diameters'])
    print(f"  Mean:   {np.mean(diameters):.2f}")
    print(f"  Median: {np.median(diameters):.2f}")
    print(f"  Std:    {np.std(diameters):.2f}")
    print(f"  Min:    {np.min(diameters):.2f}")
    print(f"  Max:    {np.max(diameters):.2f}")

    print("="*60 + "\n")
