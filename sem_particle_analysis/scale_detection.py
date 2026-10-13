"""
Scale Bar Detection and Calibration

Automatically detects scale bars in SEM/TEM images and calculates
pixel-to-nanometer conversion factors.

Supports two detection methods:
1. TIFF Metadata Extraction (primary) - Extracts pixel size from microscope metadata
2. OCR-based Scale Bar Detection (fallback) - Reads scale bar text using OCR
"""

import math
import os
import re
import unicodedata
import logging
import cv2
import numpy as np

from .utils import extract_tiff_metadata

logger = logging.getLogger(__name__)

# Metadata units, in nanometres. "µ" is looked up after NFKC normalisation, which
# turns the micro sign (U+00B5) into the Greek mu (U+03BC); instruments write
# either, and a regex matching case-insensitively treats them as the same letter.
_METADATA_UNIT_TO_NM = {"m": 1e9, "mm": 1e6, "um": 1e3, "μm": 1e3, "nm": 1.0}


def _length_to_nm(value, unit):
    """A metadata length in nanometres; raises ValueError for an unknown unit."""
    key = unicodedata.normalize("NFKC", unit or "").strip().lower()
    if key not in _METADATA_UNIT_TO_NM:
        raise ValueError(f"unknown unit {unit!r}")
    return float(value) * _METADATA_UNIT_TO_NM[key]


class OCRUnavailableError(ValueError):
    """
    Raised when a scale bar must be read but EasyOCR is not installed.

    Subclasses ValueError so that callers already handling a failed OCR attempt
    treat a missing library as just another way for OCR not to produce an answer.
    """


class ScaleDetector:
    """
    Detects and measures scale bars in microscopy images.

    Supports two detection methods:
    1. TIFF metadata extraction (primary) - reads pixel size from image metadata
    2. OCR-based scale bar detection (fallback) - reads scale bar text

    Attributes:
        reader (easyocr.Reader): OCR reader instance
        last_detection (dict): Results from most recent detection
    """

    # Unit conversion factors to nanometers
    UNIT_TO_NM = {
        'nm': 1.0,
        'um': 1000.0,
        'µm': 1000.0,
        'μm': 1000.0,  # Different mu character
        # Common OCR misreads of "µm". The µ glyph is small and frequently comes
        # back as H, p, j, l or i; "rn" is the classic misread of "m".
        'uum': 1000.0,
        'hm': 1000.0,
        'pum': 1000.0,
        'jum': 1000.0,
        'lum': 1000.0,
        'ium': 1000.0,
        'urn': 1000.0,
        'prn': 1000.0,
        'hrn': 1000.0,
        'mm': 1_000_000.0,
        'cm': 10_000_000.0,
        'm': 1_000_000_000.0,
        'a': 0.1,  # Angstroms
        'å': 0.1,  # Angstrom symbol
        'angstrom': 0.1,
        'angstroms': 0.1,
        'pm': 0.001,  # Picometers
    }

    # Valid pixel size range for SEM/TEM images (nm/pixel)
    # 0.01 nm = highest resolution TEM, 200,000 nm = lowest mag SEM
    VALID_PIXEL_SIZE_RANGE = (0.01, 200_000)

    # Maximum plausible scale bar thickness, as a fraction of image height and as
    # an absolute floor for small images. Instruments draw the bar as a thin rule
    # (well under 1% of frame height); anything thicker is a filled databar or
    # some other block, not a scale bar.
    MAX_BAR_THICKNESS_FRAC = 0.02
    MIN_BAR_THICKNESS_PX = 6

    # A scale bar is drawn solid, so its pixels fill its bounding box. Thresholded
    # background noise, opened with a wide kernel, also yields long thin
    # components — but ragged ones. Measured on the NIOSH TEM set, the true bar
    # fills 1.00 of its box while the spurious runs that were beating it on width
    # fill 0.31–0.67.
    MIN_BAR_FILL = 0.8

    # The losing polarity has to have found something this close in size to the
    # winner before the reading counts as genuinely ambiguous. Both polarities
    # nearly always find *something*, so a lower bar here floods every frame with
    # warnings and hides the readings that really are doubtful.
    AMBIGUOUS_POLARITY_AREA_RATIO = 0.6

    # Plausible physical length for a printed scale bar, in nm: from atomic
    # resolution (1 Å) up to 1 mm. A parsed value outside this came from a
    # misread unit, and acting on it would rescale every measurement.
    VALID_SCALE_BAR_NM = (0.1, 1_000_000.0)

    # Instruments label scale bars with a 1/2/5 mantissa (100 nm, 200 nm, 500 nm,
    # 1 µm, ...). A parsed value off this grid — "7 µm", "1100 nm" — is nearly
    # always a misread digit, and a misread digit is a multiplicative error in
    # every measurement, so such results are flagged for manual confirmation.
    # AMT TEM cameras also label bars 400 nm and 600 nm.
    STANDARD_MANTISSAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)

    # Databar geometry. Instruments put the acquisition information in a solid
    # strip below the micrograph; these bound what counts as one.
    DATABAR_MAX_FRACTION = 0.15      # tallest acceptable bar, as a share of height
    DATABAR_SCAN_FRACTION = 0.25     # scan past that, so "no edge" is detectable
    DATABAR_MEDIAN_TOLERANCE = 8.0   # grey levels the background may drift by
    DATABAR_MIN_HEIGHT_PX = 10
    DATABAR_MIN_SEPARATION = 12.0    # a bar must differ from the frame above it
    # A databar contains rules — the scale bar itself, panel dividers — wide
    # enough to move a whole row's median. FEI draws a scale bar spanning nearly
    # half the frame width, so the row carrying it looks nothing like the
    # background. Such a row must not be mistaken for the top edge, so allow a
    # run of this many disagreeing rows before concluding the bar has ended.
    DATABAR_MAX_RULE_THICKNESS_PX = 6
    # A databar is drawn, so it starts with a straight full-width edge: its top
    # row is flat background. A dark region belonging to the micrograph — a TEM
    # grid bar cutting across the corner, a vignette — has no such row, because
    # wherever it is cut the row crosses both dark and bright areas. Measured on
    # the NIOSH set, real FEI databars give a top-row standard deviation of 5.9
    # and grid bars 87–111, so this sits well clear of both.
    DATABAR_EDGE_ROW_STD = 25.0
    # ...and the brightness steps at that edge, from one row to the next. Plain
    # background that darkens or lightens gradually towards the bottom of a frame
    # passes every other test here, and on four NIOSH TEM frames was cropped off
    # as a "databar", 5-9% of the micrograph with it. Their edges stepped by 0-2
    # grey levels; a drawn bar steps by tens.
    DATABAR_MIN_EDGE_STEP = 12.0

    # Registry for custom tag parsers (extensibility for new microscope formats)
    # Format: {tag_code: [(parser_func, manufacturer_name), ...]}

    # Manufacturer detection patterns for ImageDescription parsing
    MANUFACTURER_PATTERNS = {
        'fei': [
            r'\[User\]',
            r'PixelWidth\s*=',
            r'PixelHeight\s*=',
            r'Thermo\s*Fisher',
            r'FEI\s*Company',
            r'\[Beam\]',
            r'\[Scan\]',
        ],
        'zeiss': [
            r'SmartSEM',
            r'AP_PIXEL_SIZE',
            r'AP_WD',
            r'Zeiss',
            r'\bLEO\b',
            r'SUPRA',
            r'SIGMA',
            r'GeminiSEM',
        ],
        'jeol': [
            r'JEOL',
            r'JEM-\d+',
            r'JSM-\d+',
            r'JIB-\d+',
        ],
        'hitachi': [
            r'Hitachi',
            r'SU\d{4}',
            r'S-\d{4}',
            r'TM\d{4}',
            r'Regulus',
            r'FlexSEM',
        ],
        'tescan': [
            r'TESCAN',
            r'MIRA',
            r'VEGA',
            r'CLARA',
        ],
    }

    def __init__(self, use_gpu=False):
        """
        Initialize the scale bar detector.

        The OCR reader is not built here. EasyOCR takes several seconds to load
        its models and drags in a large dependency tree, and neither is needed to
        read a pixel size out of TIFF metadata — which is the tier that succeeds
        on most instrument exports. Constructing a detector is therefore free,
        and OCR is paid for only by the images that actually need it.

        Args:
            use_gpu (bool): Whether to use GPU for OCR (default: False)
        """
        self.use_gpu = use_gpu
        self._reader = None
        self.last_detection = None

    @property
    def reader(self):
        """
        The EasyOCR reader, built on first use.

        Raises:
            ScaleDetectionError: If EasyOCR is not installed. Raised here rather
                than at import time so that metadata-based scale detection, the
                analyzer and the segmenter all remain usable without it.
        """
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:  # pragma: no cover - depends on install
                raise OCRUnavailableError(
                    "Reading a printed scale bar needs EasyOCR, which is not "
                    "installed. Install it with 'pip install easyocr', or use a "
                    "scale from file metadata or entered by hand."
                ) from exc
            print("Initializing OCR reader...")
            self._reader = easyocr.Reader(['en'], gpu=self.use_gpu)
            print("OCR reader initialized")
        return self._reader

    def detect_scale_from_metadata(self, file_path):
        """
        Extract scale information from TIFF metadata.

        Supports multiple SEM microscope manufacturers including FEI/Thermo Fisher,
        Zeiss, JEOL, Hitachi, and TESCAN. Also handles generic TIFF resolution tags.

        Args:
            file_path (str): Path to TIFF image file

        Returns:
            dict: Detection results containing:
                - 'pixel_size_nm': Pixel size in nanometers
                - 'conversion': Conversion factor (nm/pixel), same as pixel_size_nm
                - 'scale_nm': Reference scale (same as pixel_size_nm for single pixel)
                - 'pixel_length': Always 1 (representing one pixel)
                - 'manufacturer': Detected manufacturer name
                - 'confidence': 'high', 'medium', or 'low'
                - 'metadata_source': Description of where scale was found
                - 'raw_metadata': Original metadata dict for debugging

        Raises:
            ValueError: If no scale information found in metadata
        """
        # A JEOL SEM writes the scale to a .txt beside the image rather than into
        # it, for any image format, so that is looked for first.
        from_sidecar = self._scale_from_sidecar(file_path)
        if from_sidecar is not None:
            self.last_detection = from_sidecar
            return from_sidecar

        # Checked by name first, so a PNG or JPEG gets the reason rather than a
        # TIFF reader's error with the file's full path in it.
        if os.path.splitext(str(file_path))[1].lower() not in (".tif", ".tiff"):
            raise ValueError("not a TIFF, and only TIFF files record a pixel size")
        metadata = extract_tiff_metadata(file_path)

        if not metadata['is_tiff']:
            raise ValueError("not a TIFF, and only TIFF files record a pixel size")

        image_desc = metadata.get('image_description', '') or ''

        # Detect manufacturer
        manufacturer = self._detect_manufacturer(image_desc, metadata.get('software', ''), metadata['raw_tags'])
        logger.debug(f"Detected manufacturer: {manufacturer}")

        pixel_size_nm = None
        metadata_source = None
        confidence = 'low'

        # Try manufacturer-specific parsing
        if manufacturer == 'fei':
            pixel_size_nm = self._parse_fei_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'FEI ImageDescription (PixelWidth)'
                confidence = 'high'

        elif manufacturer == 'zeiss':
            pixel_size_nm = self._parse_zeiss_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'Zeiss SmartSEM (AP_PIXEL_SIZE)'
                confidence = 'high'

        elif manufacturer == 'jeol':
            pixel_size_nm = self._parse_jeol_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'JEOL metadata'
                confidence = 'medium'

        elif manufacturer == 'hitachi':
            pixel_size_nm = self._parse_hitachi_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'Hitachi metadata'
                confidence = 'medium'

        elif manufacturer == 'tescan':
            pixel_size_nm = self._parse_tescan_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'TESCAN metadata'
                confidence = 'medium'

        # An AMT camera (on many TEMs) writes its calibration as text.
        if pixel_size_nm is None and image_desc:
            pixel_size_nm = self._parse_amt_description(image_desc)
            if pixel_size_nm:
                metadata_source = 'AMT camera calibration (XpixCal)'
                confidence = 'high'
                manufacturer = 'amt'

        # Try generic pixel size patterns in ImageDescription
        if pixel_size_nm is None and image_desc:
            pixel_size_nm = self._parse_generic_pixel_size(image_desc)
            if pixel_size_nm:
                metadata_source = 'ImageDescription pixel size pattern'
                confidence = 'medium'

        # Try tag 34118 — could be Zeiss CZ_SEM or Hitachi
        if pixel_size_nm is None:
            tag_34118 = metadata['raw_tags'].get(34118) or metadata['raw_tags'].get('CZ_SEM')
            if tag_34118 is not None and isinstance(tag_34118, dict):
                # Check if it's Zeiss format (has 'ap_pixel_size' key)
                if 'ap_pixel_size' in tag_34118 or 'dp_sem' in tag_34118:
                    pixel_size_nm = self._parse_zeiss_cz_sem_tag(metadata['raw_tags'])
                    if pixel_size_nm:
                        metadata_source = 'Zeiss CZ_SEM tag (34118)'
                        confidence = 'high'
                        manufacturer = 'zeiss'
                # Check if it's Hitachi format (has '' key with tuple)
                elif '' in tag_34118:
                    pixel_size_nm = self._parse_hitachi_metadata(None, metadata['raw_tags'])
                    if pixel_size_nm:
                        metadata_source = 'Hitachi tag (34118)'
                        confidence = 'high'
                        manufacturer = 'hitachi'

        # Try tag 34682 — FEI HELIOS
        if pixel_size_nm is None:
            tag_34682 = metadata['raw_tags'].get(34682) or metadata['raw_tags'].get('FEI_HELIOS')
            if tag_34682 is not None:
                pixel_size_nm = self._parse_fei_helios_tag(metadata['raw_tags'])
                if pixel_size_nm:
                    metadata_source = 'FEI HELIOS tag (34682)'
                    confidence = 'high'
                    manufacturer = 'fei'

        # The TIFF's own XResolution/YResolution tags are deliberately not used:
        # programs fill them with a print resolution. On the 17 files in the test
        # set that had nothing else, every one was wrong, by a factor of up to
        # 130,000.
        if pixel_size_nm is None:
            raise ValueError(
                f"No scale information found in TIFF metadata. "
                f"Detected manufacturer: {manufacturer}"
            )

        # Validate pixel size is in plausible range for SEM/TEM
        pixel_size_nm, validity, msg = self._validate_pixel_size(pixel_size_nm, metadata_source or 'metadata')
        if validity == 'invalid':
            logger.warning(f"Pixel size validation failed: {msg}")
            raise ValueError(f"Extracted pixel size is out of range: {msg}")

        result = {
            'pixel_size_nm': pixel_size_nm,
            'conversion': pixel_size_nm,
            'scale_nm': pixel_size_nm,
            'pixel_length': 1,
            'manufacturer': manufacturer,
            'confidence': confidence,
            'metadata_source': metadata_source,
            'raw_metadata': metadata
        }

        self.last_detection = result
        return result

    def _detect_manufacturer(self, image_desc, software, raw_tags=None):
        """
        Detect microscope manufacturer from metadata strings and private tags.

        Args:
            image_desc (str): ImageDescription tag content
            software (str): Software tag content
            raw_tags (dict, optional): All TIFF tags for private tag detection

        Returns:
            str: Manufacturer name ('fei', 'zeiss', 'jeol', 'hitachi', 'tescan', or 'unknown')
        """
        combined = f"{image_desc} {software or ''}"

        for manufacturer, patterns in self.MANUFACTURER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return manufacturer

        # Also check for known private tags
        if raw_tags:
            # FEI HELIOS tag (34682)
            if raw_tags.get(34682) or raw_tags.get('FEI_HELIOS'):
                return 'fei'

            # Tag 34118 could be Zeiss CZ_SEM or Hitachi
            tag_34118 = raw_tags.get(34118) or raw_tags.get('CZ_SEM')
            if isinstance(tag_34118, dict):
                if 'ap_pixel_size' in tag_34118 or 'dp_sem' in tag_34118:
                    return 'zeiss'
                elif '' in tag_34118:
                    return 'hitachi'

        return 'unknown'

    def _parse_fei_helios_tag(self, raw_tags):
        """
        Parse FEI/Thermo Fisher HELIOS private tag (34682).

        FEI Teneo/Helios instruments store metadata in TIFF tag 34682 as a
        nested dict with sections like Scan, EBeam, Image. For example:
            {Scan: {PixelWidth: 1.34896e-06, PixelHeight: 1.34896e-06, ...},
             EBeam: {HFW: 0.002072, ...},
             Image: {ResolutionX: 1536, ResolutionY: 1024, ...}}

        Scan.PixelWidth is in meters.

        Args:
            raw_tags (dict): All TIFF tags (indexed by name or code)

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        tag_value = raw_tags.get(34682) or raw_tags.get('FEI_HELIOS')
        if not isinstance(tag_value, dict):
            return None

        # Primary: Scan.PixelWidth (in meters)
        scan = tag_value.get('Scan', {})
        if isinstance(scan, dict):
            for key in ('PixelWidth', 'PixelHeight'):
                pixel_val = scan.get(key)
                if pixel_val is not None:
                    try:
                        value_m = float(pixel_val)
                        if value_m > 0:
                            return value_m * 1e9  # meters to nm
                    except (ValueError, TypeError):
                        continue

        # Also check EScan section (some FEI instruments use this)
        escan = tag_value.get('EScan', {})
        if isinstance(escan, dict):
            for key in ('PixelWidth', 'PixelHeight'):
                pixel_val = escan.get(key)
                if pixel_val is not None:
                    try:
                        value_m = float(pixel_val)
                        if value_m > 0:
                            return value_m * 1e9
                    except (ValueError, TypeError):
                        continue

        # Fallback: compute from EBeam.HFW / Image.ResolutionX
        ebeam = tag_value.get('EBeam', {})
        image_info = tag_value.get('Image', {})
        if isinstance(ebeam, dict) and isinstance(image_info, dict):
            hfw = ebeam.get('HFW')
            res_x = image_info.get('ResolutionX')
            if hfw is not None and res_x is not None:
                try:
                    hfw_m = float(hfw)
                    width_pixels = int(res_x)
                    if hfw_m > 0 and width_pixels > 0:
                        return (hfw_m * 1e9) / width_pixels
                except (ValueError, TypeError):
                    pass

        return None

    def _parse_fei_metadata(self, image_desc, raw_tags):
        """
        Parse FEI/Thermo Fisher SEM metadata.

        Tries FEI_HELIOS private tag (34682) first, then falls back to
        ImageDescription text parsing (older FEI format).

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        # First try FEI_HELIOS private tag (tag 34682)
        result = self._parse_fei_helios_tag(raw_tags)
        if result is not None:
            return result

        # Fall back to ImageDescription text parsing
        if not image_desc:
            return None

        # Look for PixelWidth or PixelHeight
        patterns = [
            r'PixelWidth\s*=\s*([0-9.eE+-]+)',
            r'PixelHeight\s*=\s*([0-9.eE+-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc)
            if match:
                try:
                    # Value is in meters, convert to nanometers
                    value_m = float(match.group(1))
                    return value_m * 1e9
                except ValueError:
                    continue

        return None

    def _parse_zeiss_cz_sem_tag(self, raw_tags):
        """
        Parse Zeiss CZ_SEM private tag (34118).

        Zeiss instruments store metadata in TIFF tag 34118 as a dict with
        tuple values. For example:
            'ap_pixel_size': ('Pixel Size', 26.37, 'nm')
            'ap_image_pixel_size': ('Image Pixel Size', 26.37, 'nm')
            'ap_width': ('Width', 27.0, 'um')
            'dp_image_store': ('Store resolution', '1024 * 768')

        Args:
            raw_tags (dict): All TIFF tags (indexed by name or code)

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        tag_value = raw_tags.get(34118) or raw_tags.get('CZ_SEM')
        if not isinstance(tag_value, dict):
            return None

        # Primary: the stored image's pixel size. ap_pixel_size refers to the
        # instrument's reference scan, and an image stored at 2048 x 1536 has
        # half of it: taking ap_pixel_size made such frames twice too large.
        for key in ('ap_image_pixel_size', 'ap_pixel_size'):
            entry = tag_value.get(key)
            if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                try:
                    value = float(entry[1])
                    unit = str(entry[2]).lower().strip()
                    conversion = self.UNIT_TO_NM.get(unit, None)
                    if conversion is not None and value > 0:
                        return value * conversion
                except (ValueError, TypeError):
                    continue

        # Fallback: compute from ap_width and image store resolution
        ap_width = tag_value.get('ap_width')
        dp_image_store = tag_value.get('dp_image_store')
        if ap_width and dp_image_store:
            try:
                width_val = float(ap_width[1])
                width_unit = str(ap_width[2]).lower().strip()
                store_str = str(dp_image_store[1])
                image_width = int(store_str.split('*')[0].strip())
                width_nm = width_val * self.UNIT_TO_NM.get(width_unit, 1.0)
                if image_width > 0 and width_nm > 0:
                    return width_nm / image_width
            except (ValueError, TypeError, IndexError):
                pass

        return None

    def _parse_zeiss_metadata(self, image_desc, raw_tags):
        """
        Parse Zeiss SEM metadata.

        Tries CZ_SEM private tag (34118) first, then falls back to
        ImageDescription text parsing (SmartSEM format).

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        # First try CZ_SEM private tag (tag 34118)
        result = self._parse_zeiss_cz_sem_tag(raw_tags)
        if result is not None:
            return result

        # Fall back to ImageDescription text parsing
        if not image_desc:
            return None

        # "Pixel Size = 2.791 nm" states its unit; a bare AP_PIXEL_SIZE value is
        # in metres. A bare "Pixel Size" is left to the generic parser rather
        # than guessed at.
        patterns = [
            (r'Image\s*Pixel\s*Size\s*=\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)\b', None),
            (r'Pixel\s*Size\s*=\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)\b', None),
            (r'AP_PIXEL_SIZE\s*=\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)\b', None),
            (r'AP_PIXEL_SIZE\s*=\s*([0-9.eE+-]+)', "m"),
        ]

        for pattern, implied_unit in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    unit = implied_unit or match.group(2)
                    return _length_to_nm(match.group(1), unit)
                except ValueError:
                    continue

        return None

    def _parse_jeol_metadata(self, image_desc, raw_tags):
        """
        Parse JEOL SEM/TEM metadata.

        JEOL instruments may store pixel size in various formats.

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        if not image_desc:
            return None

        # Look for common JEOL patterns
        patterns = [
            r'PixelSize\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)',
            r'Pixel\s*Size\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)',
            r'Resolution\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)/pixel',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    # Convert to nanometers
                    return _length_to_nm(value, unit)
                except (ValueError, IndexError):
                    continue

        return None

    def _parse_hitachi_metadata(self, image_desc, raw_tags):
        """
        Parse Hitachi SEM metadata.

        Hitachi instruments often store pixel size in TIFF tag 34118 as a dict
        with a tuple containing calibration data. The pixel size (in meters)
        is typically at index 3 of the tuple.

        Format of tag 34118:
            {'': (0, 0, 0, pixel_size_m, magnification, ...)}

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        # First try tag 34118 (Hitachi SEM metadata)
        if 34118 in raw_tags:
            tag_value = raw_tags[34118]
            if isinstance(tag_value, dict) and '' in tag_value:
                data_tuple = tag_value['']
                if isinstance(data_tuple, (tuple, list)) and len(data_tuple) >= 4:
                    try:
                        # Pixel size is at index 3, in meters
                        pixel_size_m = float(data_tuple[3])
                        if pixel_size_m > 0:
                            # Convert meters to nanometers
                            return pixel_size_m * 1e9
                    except (ValueError, TypeError, IndexError):
                        pass

        # Fallback to ImageDescription patterns
        if image_desc:
            patterns = [
                r'PixelSize\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)',
                r'Pixel\s*Size\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)',
            ]

            for pattern in patterns:
                match = re.search(pattern, image_desc, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        unit = match.group(2).lower()

                        return _length_to_nm(value, unit)
                    except (ValueError, IndexError):
                        continue

        return None

    def _parse_tescan_metadata(self, image_desc, raw_tags):
        """
        Parse TESCAN SEM metadata.

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        if not image_desc:
            return None

        # Look for TESCAN patterns
        patterns = [
            r'PixelSize\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)',
            r'Pixel\s*Width\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|μm|mm|m)',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    return _length_to_nm(value, unit)
                except (ValueError, IndexError):
                    continue

        return None

    def _scale_from_sidecar(self, file_path):
        """
        The pixel size from a JEOL SEM's .txt beside the image, as a result dict
        like detect_scale_from_metadata's, or None.
        """
        from .sidecar import parse_jeol, read_sidecar

        info = parse_jeol(read_sidecar(file_path))
        if not info or not info.get("nm_per_px"):
            return None
        nm_per_px = info["nm_per_px"]
        # The bar was drawn for the resolution the sidecar records; an image
        # saved at another size has proportionally larger or smaller pixels.
        width = self._image_width(file_path)
        if info.get("width") and width and width != info["width"]:
            nm_per_px *= info["width"] / width
        nm_per_px, validity, msg = self._validate_pixel_size(nm_per_px, "JEOL sidecar")
        if validity == 'invalid':
            return None
        return {
            'pixel_size_nm': nm_per_px,
            'conversion': nm_per_px,
            'scale_nm': nm_per_px,
            'pixel_length': 1,
            'manufacturer': 'jeol',
            'confidence': 'high',
            'metadata_source': 'JEOL .txt beside the image (SM_MICRON_BAR)',
            'raw_metadata': info,
        }

    @staticmethod
    def _image_width(file_path):
        try:
            from PIL import Image
            with Image.open(file_path) as handle:
                return handle.size[0]
        except Exception:
            return None

    @staticmethod
    def _parse_amt_description(image_desc):
        """
        Pixel size from an AMT camera's ImageDescription, or None.

        AMT writes calibration lines such as ``XpixCal=620.662000`` (pixels per
        unit) and ``Unit=um``.
        """
        cal = re.search(r'XpixCal\s*=\s*([0-9.eE+-]+)', image_desc)
        unit = re.search(r'\bUnit\s*=\s*(nm|um|µm|μm|mm)\b', image_desc)
        if not cal or not unit:
            return None
        try:
            per_unit = float(cal.group(1))
            if per_unit <= 0:
                return None
            return _length_to_nm(1.0, unit.group(1)) / per_unit
        except ValueError:
            return None

    def _parse_generic_pixel_size(self, image_desc):
        """
        Try to find pixel size from generic patterns in ImageDescription.

        Looks for patterns like:
        - "pixel size = 10 nm"
        - "pixelsize: 5.5 um"
        - "10 nm/pixel"

        Args:
            image_desc (str): ImageDescription tag content

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        if not image_desc:
            return None

        patterns = [
            # "pixel size = 10 nm" or "pixelsize: 5.5 um"
            r'pixel\s*size\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|mm|m|pm)',
            # "10 nm/pixel" or "5.5 um / pixel"
            r'([0-9.eE+-]+)\s*(nm|um|µm|mm|m|pm)\s*/\s*pixel',
            # "resolution: 10 nm"
            r'resolution\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|mm|m|pm)',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    # Convert to nanometers using our conversion table
                    conversion = self.UNIT_TO_NM.get(unit, 1.0)
                    return value * conversion
                except (ValueError, IndexError):
                    continue

        return None

    def _validate_pixel_size(self, pixel_size_nm, source="unknown"):
        """
        Validate that a pixel size is in the physically plausible range for SEM/TEM.

        Args:
            pixel_size_nm (float): Pixel size in nanometers
            source (str): Description of where the value came from (for logging)

        Returns:
            tuple: (validated_value_or_None, 'valid'|'invalid', message)
        """
        if pixel_size_nm is None:
            return None, 'invalid', f'{source}: no value'
        if pixel_size_nm <= 0:
            return None, 'invalid', f'{source}: non-positive value {pixel_size_nm}'
        if pixel_size_nm < self.VALID_PIXEL_SIZE_RANGE[0]:
            return None, 'invalid', f'{source}: {pixel_size_nm:.6f} nm/px too small (sub-atomic)'
        if pixel_size_nm > self.VALID_PIXEL_SIZE_RANGE[1]:
            return None, 'invalid', f'{source}: {pixel_size_nm:.1f} nm/px too large (not SEM/TEM scale)'
        return pixel_size_nm, 'valid', f'{source}: {pixel_size_nm:.4f} nm/px within range'

    def detect_scale_bar(self, image, region_x=0.75, region_y=0.95,
                        region_width=0.5, region_height=0.08,
                        polarity='auto', threshold=200):
        """
        Detect and measure the scale bar in an image using adaptive polarity
        and morphological line detection.

        Supports scale bars of any color (bright on dark, dark on bright) at
        any position in the image. The search region is defined by its center
        (region_x, region_y) and size (region_width, region_height), all as
        fractions of image dimensions.

        Args:
            image (np.ndarray): RGB image array
            region_x (float): Horizontal center of search region (0-1, left to right)
            region_y (float): Vertical center of search region (0-1, top to bottom)
            region_width (float): Width of search region as fraction of image (0-1)
            region_height (float): Height of search region as fraction of image (0-1)
            polarity (str): Scale bar polarity:
                - 'auto': Try both bright and dark, pick the one with best line
                - 'bright': Bright bar on dark background (white scale bars)
                - 'dark': Dark bar on bright background (black scale bars)
            threshold (int): Binary threshold sensitivity (0-255)

        Returns:
            dict: Detection results containing:
                - 'pixel_length': Length of scale bar in pixels
                - 'scale_nm': Physical length in nanometers
                - 'conversion': Conversion factor (nm/pixel)
                - 'ocr_text': Raw OCR text
                - 'region': Region coordinates (x0, y0, width, height)
                - 'line_coords': Scale bar line coordinates (leftmost, rightmost, row)
                - 'binary_image': Thresholded image used for detection
                - 'polarity_used': Which polarity succeeded ('bright' or 'dark')

        Raises:
            ValueError: If scale bar cannot be detected or OCR fails
        """
        H, W = image.shape[:2]

        # Calculate search region from center-based coordinates
        box_w = max(1, int(W * region_width))
        box_h = max(1, int(H * region_height))
        cx = int(W * region_x)
        cy = int(H * region_y)
        x0 = max(0, min(cx - box_w // 2, W - box_w))
        y0 = max(0, min(cy - box_h // 2, H - box_h))

        # Extract region
        crop = image[y0:y0 + box_h, x0:x0 + box_w]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        # Determine which polarities to try
        if polarity == 'auto':
            polarities_to_try = ['bright', 'dark']
        else:
            polarities_to_try = [polarity]

        best_line = None
        best_binary = None
        best_polarity = None

        # A scale bar is a thin rule, not a filled block. Cap candidate thickness
        # relative to the full image so the limit doesn't depend on how tightly
        # the user drew the search box.
        max_thickness = max(self.MIN_BAR_THICKNESS_PX,
                            int(round(H * self.MAX_BAR_THICKNESS_FRAC)))

        binaries = {}
        for pol in polarities_to_try:
            if pol == 'bright':
                _, binary255 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            else:
                _, binary255 = cv2.threshold(gray, 255 - threshold, 255, cv2.THRESH_BINARY_INV)
            binaries[pol] = binary255
        colour = self._colour_mask(crop)
        if colour is not None and polarity == 'auto':
            binaries['colour'] = colour.astype(np.uint8) * 255

        # The label first, then the bar beside it. Searching for the bar on its
        # own measured whatever line was largest in the region: a databar's
        # divider, one half of an FEI bar split by its label, a tick's outer edge.
        tokens = self._read_tokens(crop)
        labels = self._label_candidates(tokens)
        ocr_text = " ".join(t["text"] for t in tokens)
        found = None
        for label in labels:
            # The bar is drawn in its label's colour, whatever that is: a fixed
            # threshold missed the mid-grey bar FEI prints on small frames.
            options = self._binaries_like_label(gray, binaries, label["box"])
            found = self._bar_beside_label(options, label, max_thickness)
            if found is not None:
                break
        if labels and found is None:
            # A label with no bar beside it here: the bar lies outside this
            # region. Falling back to the largest line in the region measured
            # something else; better that the next region is tried.
            raise ValueError(
                f"Read the label '{labels[0]['line_text']}' but found no bar beside "
                "it in this region.")

        candidates = {}
        salvaged = False
        if found is not None:
            best_polarity, best_binary, leftmost, rightmost, top_row, pixel_length, label_box = found
            best_polarity = best_polarity.split("-")[0]       # "dark-ink" is dark
            # Running into the region's edge means the bar may continue beyond it.
            spanned_crop = leftmost <= 0 or rightmost >= box_w - 1
            scale_nm = label["nm"]
            ocr_text = label["line_text"] or ocr_text
        else:
            label_box = None
            for pol, binary255 in binaries.items():
                line = self._find_scale_line_morphological(binary255, max_thickness=max_thickness)
                if line is not None:
                    leftmost, rightmost, row, spanned, area = line
                    candidates[pol] = {"length": rightmost - leftmost + 1, "area": area}
                    # A bar with clear background on both sides always beats one
                    # that ran to the crop edge, regardless of size. Otherwise the
                    # larger solid rule wins: comparing the two polarities on width
                    # alone let a wide, ragged run through background beat the real
                    # bar, which is what put the wrong polarity on most TEM frames.
                    better = (
                        best_line is None
                        or (best_line[3] and not spanned)
                        or (best_line[3] == spanned and area > best_line[4])
                    )
                    if better:
                        best_line = line
                        best_binary = binary255
                        best_polarity = pol

            if best_line is None:
                raise ValueError(
                    "No scale bar line found in region. "
                    "Try adjusting the region position, polarity, or threshold."
                )

            leftmost, rightmost, top_row, spanned_crop, _bar_area = best_line
            # Both columns are the bar's own pixels, so it covers one more than
            # their difference; dropping the one overstated nm/px by 1/length.
            pixel_length = rightmost - leftmost + 1

            # Extract scale value and units
            try:
                scale_nm = self._parse_scale_text(ocr_text)
            except ValueError:
                scale_nm = self._salvage_micron_reading(ocr_text)
                if scale_nm is None:
                    raise
                salvaged = True

        if pixel_length < 5:
            raise ValueError(
                f"Detected line too short ({pixel_length}px). "
                f"Adjust region to cover the scale bar."
            )

        # The OCR path previously trusted whatever the text parsed to. A single
        # misread character can move the value by orders of magnitude, so check
        # both the label and the resulting pixel size before returning them.
        low, high = self.VALID_SCALE_BAR_NM
        if not low <= scale_nm <= high:
            raise ValueError(
                f"Implausible scale bar length {scale_nm:g} nm from OCR text "
                f"'{ocr_text}'. Expected {low:g}-{high:g} nm; the unit was "
                f"probably misread."
            )

        conversion = scale_nm / pixel_length
        _value, status, message = self._validate_pixel_size(
            conversion, source="scale bar OCR"
        )
        if status != 'valid':
            raise ValueError(f"{message} (from OCR text '{ocr_text}')")

        # Flag results that should not be trusted without a look at the overlay,
        # rather than silently returning a number.
        warning = None
        if salvaged:
            # A guessed unit is the one thing that can still be a factor-of-1000
            # error, so it is only accepted when the result is a standard bar
            # value, and it is always flagged for confirmation.
            if not self._is_standard_scale_value(scale_nm):
                raise ValueError(
                    f"Could not read the scale unit from '{ocr_text}', and the "
                    f"micron reading it implies ({scale_nm:g} nm) is not a standard "
                    f"bar value. Set the scale manually."
                )
            warning = (f"Unit not readable in “{ocr_text.strip()}”, so µm was "
                       f"assumed. Check the bar's value.")
        elif not self._is_standard_scale_value(scale_nm):
            warning = (f"Unusual bar value read from “{ocr_text.strip()}”, "
                       f"possibly a misread digit. Check the bar's value.")
        elif spanned_crop:
            warning = ("The bar reaches the edge of the area searched, so it may "
                       "be longer. Check both ends.")
        elif len(candidates) > 1:
            # Genuine ambiguity means the polarity that lost also found something
            # bar-like of comparable size. Both polarities almost always find
            # *something*, so warning on any disagreement in length cried wolf on
            # most frames — and, worse, made a correct-but-flagged reading lose to
            # an unflagged wrong one in detect_scale_bar_anywhere, which prefers
            # results carrying no warning.
            winner = candidates[best_polarity]
            rival = max((c for pol, c in candidates.items() if pol != best_polarity),
                        key=lambda c: c["area"])
            comparable = rival["area"] >= self.AMBIGUOUS_POLARITY_AREA_RATIO * winner["area"]
            differs = abs(rival["length"] - winner["length"]) > 0.1 * winner["length"]
            if comparable and differs:
                warning = ("Two different bars could have been measured. Check "
                           "that the marked span covers the scale bar.")

        # What to leave out of analysis when the bar is printed on the picture:
        # the bar and the text on the label's line, not the whole search region.
        mark = None
        if label_box is not None:
            lx0, ly0, lx1, ly1 = label_box
            bx0, bx1 = min(lx0, leftmost), max(lx1, rightmost)
            by0, by1 = min(ly0, top_row - max_thickness), max(ly1, top_row + max_thickness)
            mark = (x0 + bx0, y0 + by0, bx1 - bx0 + 1, by1 - by0 + 1)

        # Store results
        self.last_detection = {
            'region': (x0, y0, box_w, box_h),
            'mark': mark,
            'pixel_length': pixel_length,
            'scale_nm': scale_nm,
            'conversion': conversion,
            'ocr_text': ocr_text,
            'binary_image': best_binary,
            'line_coords': (leftmost, rightmost, top_row),
            'threshold': threshold,
            'polarity_used': best_polarity,
            'polarity_candidates': candidates,
            'warning': warning,
        }

        if warning:
            print(f"{warning}")

        return self.last_detection

    # Scale bar labels are often only 10-20px tall, which EasyOCR reads poorly —
    # it tends to drop the leading digit entirely, turning "2 µm" into "µm".
    # Enlarging the crop first recovers those digits.
    OCR_TARGET_HEIGHT = 200

    # Characters that can appear in a scale label. Restricting the alphabet helps
    # EasyOCR commit to a faint digit instead of discarding it.
    OCR_ALLOWLIST = "0123456789.nmuµ "

    @classmethod
    def _is_standard_scale_value(cls, scale_nm):
        """True if ``scale_nm`` has a mantissa instruments actually print."""
        if scale_nm <= 0:
            return False
        exponent = math.floor(math.log10(scale_nm))
        mantissa = scale_nm / (10 ** exponent)
        return any(abs(mantissa - m) < 0.01 for m in cls.STANDARD_MANTISSAS)

    # Unit tokens EasyOCR produces when it mangles "µm": the µ becomes u/H/X/J/L
    # and the m becomes rr/II/ll/ff/IT. A genuine "nm" is two plain letters and
    # comes back correctly, so an unreadable unit is overwhelmingly a micron.
    _GARBAGE_UNIT = re.compile(r'^[^\dn\s]{1,6}[\]\)\}]?$', re.I)

    @classmethod
    def _salvage_micron_reading(cls, ocr_text):
        """
        Recover the value from a label whose unit OCR mangled beyond matching.

        Returns nanometres, or None when the text gives no usable number or the
        unit might genuinely have been "nm" — guessing wrong here is a factor of
        1000, so anything ambiguous is refused rather than assumed.
        """
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\S*)', ocr_text.strip())
        if not match:
            return None
        value, unit = float(match.group(1)), match.group(2).strip()
        if not unit or not cls._GARBAGE_UNIT.match(unit):
            return None
        return value * 1000.0  # µm

    # Coloured labels and bars (the red ones some TEM exports burn in) are
    # mid-grey once converted to greyscale, and read badly; picked out by colour
    # instead. A pixel counts as coloured when its channels differ by this much.
    COLOUR_CHROMA = 80
    COLOUR_MIN_FRACTION = 0.002

    @classmethod
    def _colour_mask(cls, crop):
        """Pixels of strong colour in an RGB crop, or None if there are few."""
        if crop.ndim != 3 or crop.shape[2] < 3:
            return None
        rgb = crop[..., :3].astype(np.int16)
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        mask = chroma > cls.COLOUR_CHROMA
        return mask if mask.mean() >= cls.COLOUR_MIN_FRACTION else None

    def _read_tokens(self, crop):
        """
        OCR a crop and return the words with their boxes, in crop pixels.

        Effort escalates, since a label like "1 µm" is mostly thin strokes and is
        often lost at default sensitivity: the crop is enlarged first, and the
        first reading that contains a scale label wins, otherwise the one with
        the most text.

        Returns:
            list[dict]: ``text``, ``conf`` and ``box`` (x0, y0, x1, y1).
        """
        sources = []
        colour = self._colour_mask(crop)
        if colour is not None:
            ink = np.where(colour, 0, 255).astype(np.uint8)
            sources.append(np.stack([ink] * 3, axis=-1))
        sources.append(crop)

        images = []
        for source in sources:
            height = source.shape[0]
            if 0 < height < self.OCR_TARGET_HEIGHT:
                factor = min(4.0, self.OCR_TARGET_HEIGHT / height)
                images.append((cv2.resize(source, None, fx=factor, fy=factor,
                                          interpolation=cv2.INTER_CUBIC), factor))
            images.append((source, 1.0))

        attempts = [
            {},
            {"text_threshold": 0.4, "low_text": 0.3},
            {"text_threshold": 0.3, "low_text": 0.2, "allowlist": self.OCR_ALLOWLIST},
        ]
        reader = self.reader
        best, best_length = [], -1
        for image, factor in images:
            for options in attempts:
                try:
                    found = reader.readtext(image, **options)
                except Exception:
                    continue
                tokens = []
                for box, text, conf in found:
                    xs = [point[0] for point in box]
                    ys = [point[1] for point in box]
                    tokens.append({"text": text, "conf": float(conf),
                                   "box": (min(xs) / factor, min(ys) / factor,
                                           max(xs) / factor, max(ys) / factor)})
                if self._label_candidates(tokens):
                    return tokens
                length = sum(len(t["text"]) for t in tokens)
                if length > best_length:
                    best, best_length = tokens, length
        return best

    # Letters OCR returns for digits inside a number on a scale label.
    _DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0", "U": "0",
                                  "I": "1", "l": "1", "|": "1", "S": "5", "Z": "2",
                                  "B": "8", ",": "."})
    _LABEL = re.compile(
        r"^[^\w.]*([0-9OoQDUIl|SZB.,]*[0-9OoQDUIl|SZB])\s*"
        r"(nm|uum|pum|jum|lum|ium|prn|hrn|urn|um|µm|μm|hm|hmn|mm)[^\w]*$", re.IGNORECASE)

    def _label_value(self, text):
        """
        The length a scale label states, in nm, or None.

        The whole text must be a number and a unit, so "15.0 kV" or "WD 9.0 mm"
        never qualify; common OCR slips are repaired: letters for digits
        ("Ium" for 1 µm), a dropped decimal point ("0 5", "05", "U2" for 0.5
        and 0.2), and the usual misreadings of µ.
        """
        text = unicodedata.normalize("NFKC", text or "").strip()
        text = re.sub(r"^(\W*)0\s+(\d)", r"\g<1>0.\2", text)        # "0 5 µm"
        match = self._LABEL.match(text)
        if not match:
            return None
        number, unit = match.group(1).translate(self._DIGIT_FIXES), match.group(2)
        # "Mm" is µm with its µ read as M: bars are labelled "mm", never "Mm".
        unit = "µm" if unit == "Mm" else unit.lower()
        if not any(ch.isdigit() for ch in match.group(1)) and len(number) > 1:
            return None
        if number.startswith("."):
            number = "0" + number
        if len(number) > 1 and number[0] == "0" and number[1] != ".":
            number = "0." + number[1:]                                 # "05", "02"
        try:
            value = float(number)
        except ValueError:
            return None
        if unit == "hmn":
            unit = "µm"
        nm = value * self.UNIT_TO_NM.get(unit, self.UNIT_TO_NM.get("µm"))
        low, high = self.VALID_SCALE_BAR_NM
        return nm if value > 0 and low <= nm <= high else None

    _INSTRUMENT_BEFORE = re.compile(r"(WD|HV|HFW|spot|dwell|curr|det|mag|=|:)\s*$", re.IGNORECASE)
    _INSTRUMENT_AFTER = re.compile(r"^\s*(kV|pA|nA|[xX]\b|KX)", re.IGNORECASE)

    def _label_candidates(self, tokens):
        """
        The scale labels among the words read, best first.

        Words are grouped into lines, and runs of up to three neighbouring words
        tried, so "5" and "µm" read as two words still make one label.

        Returns:
            list[dict]: ``nm``, ``box`` (crop pixels), ``line_text``, ``standard``.
        """
        lines = []
        for token in sorted(tokens or (), key=lambda t: (t["box"][1] + t["box"][3]) / 2):
            centre = (token["box"][1] + token["box"][3]) / 2
            height = max(1.0, token["box"][3] - token["box"][1])
            for line in lines:
                if abs(line["centre"] - centre) <= 0.6 * max(line["height"], height):
                    line["tokens"].append(token)
                    break
            else:
                lines.append({"centre": centre, "height": height, "tokens": [token]})

        found = []
        for line in lines:
            words = sorted(line["tokens"], key=lambda t: t["box"][0])
            line_text = " ".join(w["text"] for w in words)
            for i in range(len(words)):
                for j in range(i + 1, min(i + 3, len(words)) + 1):
                    run = words[i:j]
                    nm = self._label_value(" ".join(w["text"] for w in run))
                    if nm is None:
                        continue
                    before = " ".join(w["text"] for w in words[max(0, i - 2):i])
                    after = " ".join(w["text"] for w in words[j:j + 1])
                    if self._INSTRUMENT_BEFORE.search(before) or self._INSTRUMENT_AFTER.search(after):
                        continue
                    found.append({
                        "nm": nm, "line_text": line_text, "words": j - i,
                        "standard": self._is_standard_scale_value(nm),
                        "box": (min(w["box"][0] for w in run), min(w["box"][1] for w in run),
                                max(w["box"][2] for w in run), max(w["box"][3] for w in run)),
                    })
        found.sort(key=lambda c: (not c["standard"], c["words"]))
        return found

    def _bar_beside_label(self, binaries, label, max_thickness):
        """
        The scale bar that belongs to ``label``: the bar line nearest it.

        Pieces on one row either side of the label are one bar (FEI prints the
        label in a gap in the middle of it). A bar with end ticks is measured
        between the ticks' centres, the convention both FEI and Zeiss draw to.

        Returns:
            tuple ``(polarity, binary, left, right, row, pixel_length, label_box)``
            in crop pixels, or None when no bar sits beside the label.
        """
        # Solid bars first. Only if none sits beside the label are gaps bridged,
        # for a dotted bar: bridging also turns noise and the foot of the label's
        # own letters into short lines.
        for bridge in (False, True):
            found = self._nearest_bar(binaries, label, max_thickness, bridge)
            if found is not None:
                break
        if found is None:
            return None

        polarity, binary, group = found
        left, right = group["x0"], group["x1"]
        row = int(round((group["y0"] + group["y1"]) / 2))
        ticks = self._tick_centres(binary, left, right, group["y0"], group["y1"])
        if ticks is not None:
            left_centre, right_centre = ticks
            pixel_length = right_centre - left_centre
            left = int(round(left_centre))
            right = left + int(round(pixel_length)) - 1
        else:
            pixel_length = right - left + 1
        return polarity, binary, left, right, row, pixel_length, label["box"]

    def _nearest_bar(self, binaries, label, max_thickness, bridge):
        """The bar-like line nearest a label, as (polarity, binary, group), or None."""
        lx0, ly0, lx1, ly1 = label["box"]
        label_h = max(4.0, ly1 - ly0)
        label_w = max(4.0, lx1 - lx0)
        label_cx = (lx0 + lx1) / 2
        label_cy = (ly0 + ly1) / 2
        best = None
        for polarity, binary in binaries.items():
            height, width = binary.shape
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(label_h)), 1))
            source = binary
            if bridge:
                source = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
            opened = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
            count, _labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
            pieces = []
            for k in range(1, count):
                x, y, w, h, area = (int(v) for v in stats[k][:5])
                if w <= 3 * h or h > max_thickness or area < self.MIN_BAR_FILL * w * h:
                    continue
                if x == 0 and x + w >= width:
                    continue                      # a divider across the whole region
                pieces.append({"x0": x, "x1": x + w - 1, "y0": y, "y1": y + h - 1})
            pieces.sort(key=lambda s: s["x0"])
            groups = []
            for piece in pieces:
                row = (piece["y0"] + piece["y1"]) / 2
                for group in groups:
                    if abs((group["y0"] + group["y1"]) / 2 - row) > 2:
                        continue
                    gap = piece["x0"] - group["x1"]
                    split_by_label = (group["x1"] < label_cx < piece["x0"]
                                      and abs(row - label_cy) <= label_h
                                      and gap <= label_w + 4 * label_h)
                    if 0 < gap and (gap <= 3 or split_by_label):
                        group.update(x1=piece["x1"], y0=min(group["y0"], piece["y0"]),
                                     y1=max(group["y1"], piece["y1"]))
                        break
                else:
                    groups.append(dict(piece))
            for group in groups:
                length = group["x1"] - group["x0"] + 1
                if length < 5:
                    continue
                # A line across the whole region is a divider, or a bar too long
                # for this region: ranked last, so a wider region gets its turn.
                spanning = group["x0"] == 0 and group["x1"] >= width - 1
                # A line found only by bridging gaps must have had gaps: a dotted
                # bar, not solid shapes run together.
                if bridge:
                    row_pixels = binary[group["y0"]:group["y1"] + 1, group["x0"]:group["x1"] + 1] > 0
                    if row_pixels.any(axis=0).mean() > 0.9:
                        continue
                row = (group["y0"] + group["y1"]) / 2
                # The foot of the label's own letters, or an underline: a thin
                # line within the label's box. (OCR draws the box generously, so
                # a thick bar just below the text can fall inside it too.)
                thin = group["y1"] - group["y0"] + 1 <= max(3, 0.1 * label_h)
                if (thin and ly0 <= row <= ly1
                        and group["x0"] >= lx0 - 2 and group["x1"] <= lx1 + 2):
                    continue
                vertical = 0.0 if ly0 <= row <= ly1 else min(abs(row - ly0), abs(row - ly1))
                horizontal = max(0.0, group["x0"] - lx1, lx0 - group["x1"])
                if vertical > 3 * label_h or horizontal > 2 * label_w:
                    continue
                # A bar stands out from the rows just above and below it. A strip
                # of a light databar's own background between two lines of text
                # passes every other test, and does not.
                if not self._stands_out(binary, group):
                    continue
                # Nearest first; at much the same distance, the solid bar beats
                # an outline drawn round it, then the longer line wins.
                thickness = group["y1"] - group["y0"] + 1
                # A one-pixel line, or one on the region's top or bottom row, is
                # nearly always a border or an anti-aliased edge: last resort only.
                weak = thickness < 2 or group["y0"] == 0 or group["y1"] >= height - 1
                score = (weak, round((vertical + 2 * horizontal) / max(2.0, label_h / 3)),
                         spanning, -thickness, -length)
                if best is None or score < best[0]:
                    best = (score, polarity, binary, group)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _binaries_like_label(self, gray, binaries, box):
        """
        The thresholded images to look for the bar in: those of the label's own
        colour. A bar is drawn like its label; lines of the other polarity were
        winning, a dark FEI databar panel over its white bar, the white outline
        round a JEOL TEM bar over the black bar inside it.
        """
        ink = self._ink_like_label(gray, box)
        if ink is None:
            return dict(binaries)
        mask, brighter = ink
        same = "bright" if brighter else "dark"
        chosen = {f"{same}-ink": mask}
        if same in binaries:
            chosen[same] = binaries[same]
        if "colour" in binaries:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            if binaries["colour"][max(0, y0):y1, max(0, x0):x1].any():
                chosen["colour"] = binaries["colour"]
        return chosen

    @staticmethod
    def _ink_like_label(gray, box, pad=None):
        """
        Everything drawn in the label's colour, as (binary image, brighter), or
        None when the label hardly differs from its background.

        The label's background is the median of a band around it; its ink is the
        far tail of the pixels inside it on the side away from the background;
        the threshold sits halfway between.
        """
        height, width = gray.shape
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, max(x0 + 1, x1)), min(height, max(y0 + 1, y1))
        inside = gray[y0:y1, x0:x1].astype(np.float64)
        if inside.size < 9:
            return None
        pad = pad or max(3, (y1 - y0) // 2)
        around = gray[max(0, y0 - pad):min(height, y1 + pad),
                      max(0, x0 - pad):min(width, x1 + pad)].astype(np.float64)
        background = float(np.median(around))
        brighter = inside.mean() > background
        ink = np.percentile(inside, 95 if brighter else 5)
        if abs(ink - background) < 25:
            return None
        threshold = (ink + background) / 2
        mask = gray > threshold if brighter else gray < threshold
        return mask.astype(np.uint8) * 255, bool(brighter)

    @staticmethod
    def _stands_out(binary, group, reach=3, most=0.5):
        """
        Whether, on each side of a line, one of the next few rows is mostly
        background. Any of them will do: a bar outlined in white over a dark
        particle has background only in the outline's row.
        """
        height = binary.shape[0]
        x0, x1 = group["x0"], group["x1"] + 1
        for direction, edge in ((-1, group["y0"]), (1, group["y1"])):
            rows = [edge + direction * k for k in range(1, reach + 1)]
            rows = [r for r in rows if 0 <= r < height]
            if rows and all((binary[r, x0:x1] > 0).mean() >= most for r in rows):
                return False
        return True

    @staticmethod
    def _tick_centres(binary, left, right, top, bottom, reach=8):
        """
        Centres of vertical end ticks at both ends of a bar, or None.

        A tick is a column near the end that runs well above or below the bar's
        own thickness. Measured from the outside of the ticks, a Zeiss 2 µm bar
        came out 78 px against the 75.8 its metadata implies; centre to centre it
        is 76.
        """
        height = binary.shape[0]
        thickness = bottom - top + 1
        need = thickness + max(4, thickness)

        def extent(col):
            column = binary[:, col] > 0
            if not column[top:bottom + 1].any():
                return 0
            up = top
            while up > 0 and column[up - 1]:
                up -= 1
            down = bottom
            while down < height - 1 and column[down + 1]:
                down += 1
            return down - up + 1

        def centre(cols):
            hits = [c for c in cols if extent(c) >= need]
            return float(np.mean(hits)) if hits else None

        span = right - left + 1
        if span < 3 * reach:
            return None
        left_centre = centre(range(left, left + reach))
        right_centre = centre(range(right - reach + 1, right + 1))
        if left_centre is None or right_centre is None:
            return None
        return left_centre, right_centre

    def _find_scale_line_morphological(self, binary255, max_thickness=None):
        """
        Find a horizontal scale bar line using morphological operations.

        Uses a wide horizontal kernel to isolate long horizontal features,
        then finds the longest connected horizontal segment.

        Args:
            binary255 (np.ndarray): Binary image (0 or 255), uint8
            max_thickness (int, optional): Reject candidates taller than this many
                pixels. A real scale bar is a thin line; without this guard an
                inverted threshold over a solid databar yields one region-spanning
                blob that looks "wide and thin" whenever the crop is short, and it
                wins on width every time. If None, no thickness limit is applied.

        Returns:
            tuple: ``(leftmost, rightmost, row, spanned_crop, area)`` of the best
            line, or None. ``area`` is its pixel count, used to compare the two
            polarities against each other.
        """
        if not binary255.any():
            return None

        h, w = binary255.shape

        # Create a horizontal morphological kernel — width scaled to region
        # Minimum kernel width: 10% of crop width, ensures we only find long bars
        kernel_w = max(15, w // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))

        # Opening: removes small noise, keeps long horizontal structures
        opened = cv2.morphologyEx(binary255, cv2.MORPH_OPEN, kernel)

        if not opened.any():
            return None

        # Find connected components in the morphologically opened image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)

        if num_labels <= 1:
            return None

        # Find the widest component (skip background label 0).
        # Components spanning the crop edge-to-edge are tracked separately: they
        # are almost always an artifact of the search box clipping a band of the
        # micrograph or databar, not a scale bar. A real bar has background on at
        # least one side. They are only used if nothing better exists, so that a
        # box drawn exactly around the bar still works.
        best_label = best_spanning_label = None
        best_area = best_spanning_area = 0
        for label_id in range(1, num_labels):
            comp_w = stats[label_id, cv2.CC_STAT_WIDTH]
            comp_h = stats[label_id, cv2.CC_STAT_HEIGHT]
            # Scale bar lines are wide and thin (aspect ratio > 3:1)
            if comp_w <= comp_h * 3:
                continue
            # ...and thin in absolute terms. The aspect test alone is not enough:
            # a solid databar filling a short crop still satisfies w > 3h.
            if max_thickness is not None and comp_h > max_thickness:
                continue

            # ...and solid. Without this, a ragged run through thresholded
            # background beats the real bar whenever it happens to be wider.
            comp_area = int(stats[label_id, cv2.CC_STAT_AREA])
            if comp_area < self.MIN_BAR_FILL * comp_w * comp_h:
                continue

            # Rank by area rather than width. Width alone also picks the bar's
            # own anti-aliased edge — a one- or two-pixel-tall sliver a few
            # pixels wider than the bar it borders, which is solid enough to pass
            # every other test and measures slightly long.
            comp_x = stats[label_id, cv2.CC_STAT_LEFT]
            spans_crop = comp_x == 0 and (comp_x + comp_w) >= w
            if spans_crop:
                if comp_area > best_spanning_area:
                    best_spanning_area = comp_area
                    best_spanning_label = label_id
            elif comp_area > best_area:
                best_area = comp_area
                best_label = label_id

        spanned = False
        if best_label is None:
            if best_spanning_label is None:
                return None
            best_label = best_spanning_label
            best_area = best_spanning_area
            spanned = True

        # Get the row and column extents of the best component
        component_mask = (labels == best_label)
        rows, cols = np.where(component_mask)
        # Use the median row (most representative row of the bar)
        bar_row = int(np.median(rows))
        leftmost = int(cols.min())
        rightmost = int(cols.max())

        return (leftmost, rightmost, bar_row, spanned, int(best_area))

    def _parse_scale_text(self, ocr_text):
        """
        Extract scale value and unit from OCR text.

        Supports units: nm, um/µm, mm, cm, m, pm, angstroms (A/Å)
        Rejects matches that are clearly SEM instrument parameters
        (working distance, voltage, beam current, etc.).

        Args:
            ocr_text (str): Raw OCR text

        Returns:
            float: Scale length in nanometers

        Raises:
            ValueError: If scale information cannot be extracted
        """
        # Match patterns like "100 nm", "1 μm", "500nm", "10 A", etc.
        # Extended pattern to include more units
        # Bare "m" and "pm" are deliberately absent. A micrograph scale bar is
        # never labelled in metres, and picometre bars do not occur on these
        # instruments — but "µm" is very often misread as "m" or "pm", which
        # previously yielded scales off by a factor of a million or more.
        # The unit must end the word: "20 Aug" in a date is not 20 ångström.
        pattern = (r'(\d+(?:\.\d+)?)\s*'
                   r'(nm|uum|pum|jum|lum|ium|prn|hrn|urn|um|µm|μm|hm|mm|cm|'
                   r'angstroms|angstrom|å|a)(?![a-z])')

        # Find ALL matches, not just the first — then filter out instrument params
        candidates = []
        for match in re.finditer(pattern, ocr_text, flags=re.I):
            value = float(match.group(1))
            unit = match.group(2).lower()
            start_pos = match.start()

            # Check if this match is preceded by or part of an instrument parameter
            # Look at the text context around the match
            context_before = ocr_text[max(0, start_pos - 15):start_pos].strip()
            context_after = ocr_text[match.end():match.end() + 10].strip()
            full_context = context_before + " " + match.group(0) + " " + context_after

            is_instrument = False

            # Reject if preceded by known parameter labels
            if re.search(r'(WD|HV|spot|dwell|curr|det|mag)\s*$', context_before, re.I):
                is_instrument = True
            # Reject voltage values (kV)
            if re.search(r'kV', context_after, re.I):
                is_instrument = True
            # Reject if it looks like a working distance (X.X mm with decimal)
            if unit == 'mm' and '.' in match.group(1):
                is_instrument = True
            # Reject if unit is 'pm' and could be AM/PM time
            if unit == 'pm' and re.search(r'\d{1,2}[:.]\d{2}', context_before):
                is_instrument = True
            # Reject 'm' if it's part of "mm" already matched or time
            if unit == 'm' and re.search(r'(mm|PM|AM)', full_context, re.I):
                is_instrument = True

            if not is_instrument:
                conversion = self.UNIT_TO_NM.get(unit, 1.0)
                candidates.append({
                    'value': value,
                    'unit': unit,
                    'nm': value * conversion,
                    'text': match.group(0),
                    'position': start_pos,
                })

        if not candidates:
            raise ValueError(
                f"Could not extract scale information from OCR text: '{ocr_text}'"
            )

        # If multiple candidates, prefer the last one (scale bars are usually
        # at the end/right of the databar) and prefer round numbers
        def score_candidate(c):
            """Higher = better candidate for a scale bar label."""
            score = 0
            # Round numbers are more likely scale labels
            if c['value'] == int(c['value']):
                score += 10
            # Common scale bar values
            if c['value'] in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
                score += 20
            # Later position = more likely the scale label (right side of bar)
            score += c['position'] / 100.0
            return score

        best = max(candidates, key=score_candidate)
        return best['nm']

    def detect_databar(self, image, metadata=None):
        """
        Detect whether the image has a databar/info bar at the bottom.

        Many SEM images (especially FEI, Hitachi, JEOL) have an information
        bar at the bottom showing acquisition parameters. This method detects
        it using metadata (when available) or image analysis.

        Args:
            image (np.ndarray): RGB or grayscale image array
            metadata (dict, optional): Raw metadata dict from extract_tiff_metadata()

        Returns:
            dict: {
                'has_databar': bool,
                'databar_height': int (pixels),
                'databar_fraction': float (0-1)
            }
        """
        H, W = image.shape[:2]
        result = {'has_databar': False, 'databar_height': 0, 'databar_fraction': 0.0}

        # Method 1: FEI metadata tells us the scan resolution vs image size
        if metadata:
            raw = metadata.get('raw_tags', {})

            # FEI HELIOS tag: Image.ResolutionY < actual image height = databar
            tag_34682 = raw.get(34682) or raw.get('FEI_HELIOS')
            if isinstance(tag_34682, dict):
                image_info = tag_34682.get('Image', {})
                res_y = image_info.get('ResolutionY')
                if res_y is not None:
                    try:
                        scan_height = int(res_y)
                        if 0 < scan_height < H:
                            result['has_databar'] = True
                            result['databar_height'] = H - scan_height
                            result['databar_fraction'] = result['databar_height'] / H
                            return result
                    except (ValueError, TypeError):
                        pass

        # Method 2: find the solid strip at the bottom by locating its top edge.
        #
        # A databar is a block of one background colour carrying text and a rule.
        # Its rows are therefore not uniform — a row crossing the label has a
        # high standard deviation — but its row *median* is the background level
        # throughout, because the glyphs cover only a small part of the width.
        # Matching on the median tolerates the text while still changing sharply
        # at the boundary with the micrograph.
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        row_medians = np.median(gray, axis=1).astype(float)
        spreads = np.median(np.abs(gray - row_medians[:, None]), axis=1)

        # Usually the strip runs to the bottom edge. Zeiss places it a few rows
        # higher, with a sliver of micrograph below; there the search starts at
        # the strip's lowest row, and the crop takes the sliver with it.
        starts = [0]
        inset_limit = int(H * self.DATABAR_INSET_FRACTION)
        for offset in range(1, inset_limit):
            # A flat row whose neighbour below is textured micrograph, or flat
            # at a clearly different level (black, at a low-magnification edge).
            below_differs = (spreads[H - offset] > 2
                             or abs(row_medians[H - 1 - offset] - row_medians[H - offset])
                             >= self.DATABAR_MIN_SEPARATION)
            if spreads[H - 1 - offset] <= 2 and below_differs:
                starts.append(offset)
                break
        for offset in starts:
            height = self._strip_height(gray, row_medians, offset)
            if height:
                result['has_databar'] = True
                result['databar_height'] = height
                result['databar_fraction'] = height / H
                return result
        return result

    # How far above the bottom edge a databar strip may begin (Zeiss: 3-18 rows
    # of a 768-row frame).
    DATABAR_INSET_FRACTION = 0.04

    def _strip_height(self, gray, row_medians, offset):
        """
        Rows from the top of a databar strip whose lowest row is ``offset`` rows
        above the bottom edge, down to the bottom edge; 0 if there is no strip.
        """
        H = gray.shape[0]
        bottom = H - 1 - offset
        # The strip's own level, from its lowest rows rather than one row: Zeiss
        # frames the strip with a black rule, and starting from the rule's level
        # the strip was never found at all.
        base = float(np.median(row_medians[max(0, bottom - 11):bottom + 1]))

        # Walk up while the background level holds. Scanning further than the
        # largest acceptable bar is deliberate: it is what makes "no boundary
        # found" distinguishable from "a very tall bar".
        #
        # A single disagreeing row does not end the bar — it is usually a rule
        # drawn across it. Only a run of them means the micrograph has started.
        # `height` is where the walk has reached; `confirmed` is the last row
        # that actually matched, so trailing rule rows are not counted in.
        scan_limit = min(H - offset, int(H * self.DATABAR_SCAN_FRACTION))
        height = confirmed = 0
        misses = 0
        while height < scan_limit:
            if abs(row_medians[bottom - height] - base) <= self.DATABAR_MEDIAN_TOLERANCE:
                misses = 0
                height += 1
                confirmed = height
            else:
                misses += 1
                if misses > self.DATABAR_MAX_RULE_THICKNESS_PX:
                    break
                height += 1
        height = confirmed
        # Take in a frame rule along the strip's top edge: a thin run of uniform
        # rows unlike the strip, which would otherwise be left on the
        # micrograph. A long run is not a rule but uniform micrograph (black, at
        # a low-magnification edge), and stays.
        def flat(row):
            values = gray[row].astype(float)
            return np.median(np.abs(values - np.median(values))) <= 2

        run, level = 0, None
        while run < self.DATABAR_MAX_RULE_THICKNESS_PX:
            row = bottom - height - run
            if row < 0 or not flat(row) or abs(row_medians[row] - base) <= self.DATABAR_MEDIAN_TOLERANCE:
                break
            if level is not None and abs(row_medians[row] - level) > self.DATABAR_MEDIAN_TOLERANCE:
                break
            level = row_medians[row] if level is None else level
            run += 1
        above_run = bottom - height - run
        if run and (above_run < 0 or not flat(above_run)
                    or abs(row_medians[above_run] - level) > self.DATABAR_MEDIAN_TOLERANCE):
            height += run
        height += offset
        top = H - height

        # Too short to be a databar, or no top edge inside the scan window. The
        # latter means the whole bottom quarter of the frame looks like its last
        # row — a featureless image, not an image with a bar stuck on it. The
        # previous implementation scanned largest-first and returned the first
        # window that passed, so it reported the largest uniform-looking window
        # rather than the actual edge: it overestimated the height on real
        # databars, and claimed one on images that had none. Both cost real
        # micrograph area, since the Scale tab trims whatever is reported.
        if not (self.DATABAR_MIN_HEIGHT_PX <= height - offset
                and height <= int(H * self.DATABAR_MAX_FRACTION)):
            return 0

        above = gray[:top]
        if above.size == 0:
            return 0

        # A databar is visibly a different block from the image above it.
        separation = abs(float(np.median(gray[top:bottom + 1])) - float(np.median(above)))
        if separation < self.DATABAR_MIN_SEPARATION:
            return 0

        # ...and it begins with a drawn edge. Checking the top two rows rather
        # than only the first tolerates a divider rule along the very top. The
        # spread is the median deviation, not the standard deviation: JEOL puts
        # the scale bar in the strip's top rows, and a bar across a seventh of
        # the width made those rows' standard deviation look like a grid bar's.
        edge_rows = gray[top:top + 2].astype(float)
        spread = np.median(np.abs(edge_rows - np.median(edge_rows, axis=1, keepdims=True)), axis=1)
        if spread.min() > self.DATABAR_EDGE_ROW_STD:
            return 0

        # A drawn edge is straight and runs the full width: just inside it the
        # strip's level holds nearly everywhere, and just outside it nearly
        # nowhere. A grid bar's edge is diagonal, so neither row is consistent.
        inside = gray[min(bottom, top + 2)].astype(float)
        outside = gray[top - 1].astype(float)
        if ((np.abs(inside - base) <= 3 * self.DATABAR_MEDIAN_TOLERANCE).mean() < 0.8
                or (np.abs(outside - base) > self.DATABAR_MEDIAN_TOLERANCE).mean() < 0.5):
            return 0

        # A drawn edge is a step, not a slope: the rows just above it against
        # the strip's own level (not the rows just below, which may be a frame
        # rule taken in above).
        above_edge = np.median(row_medians[max(0, top - 3):top])
        if abs(above_edge - base) < self.DATABAR_MIN_EDGE_STEP:
            return 0
        return height

    # Where scale bars are found, as (region_x, region_y, region_width,
    # region_height). SEM instruments put the bar in a databar at bottom-right;
    # JEOL TEM burns it into the bottom-left of the frame itself.
    SEARCH_REGIONS = (
        ("bottom-left", 0.14, 0.955, 0.28, 0.085),
        ("bottom-right", 0.80, 0.955, 0.34, 0.085),
        ("bottom-left-tall", 0.16, 0.93, 0.32, 0.14),
        ("bottom-right-tall", 0.78, 0.93, 0.40, 0.14),
        ("bottom-strip", 0.5, 0.96, 1.0, 0.08),
    )

    def detect_scale_bar_anywhere(self, image, regions=None, **kwargs):
        """
        Look for a scale bar in each of the usual places and return the best hit.

        Callers otherwise have to know in advance which corner a given microscope
        writes to, which differs by vendor and even by export setting. A result
        with no warning is preferred over a flagged one; a flagged result is
        preferred over nothing.

        Args:
            image (np.ndarray): RGB image.
            regions: Iterable of (name, x, y, width, height); defaults to
                ``SEARCH_REGIONS``.
            **kwargs: Passed through to ``detect_scale_bar``.

        Returns:
            dict: As ``detect_scale_bar``, plus ``region_name``.

        Raises:
            ValueError: If no region yields a usable scale bar.
        """
        flagged = None
        errors = []
        for name, x, y, width, height in (regions or self.SEARCH_REGIONS):
            try:
                result = self.detect_scale_bar(
                    image, region_x=x, region_y=y,
                    region_width=width, region_height=height, **kwargs
                )
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                continue
            result['region_name'] = name
            if not result.get('warning'):
                return result
            if flagged is None:
                flagged = result

        if flagged is not None:
            return flagged
        raise ValueError(
            "No scale bar found in any of the usual positions. Tried — "
            + "; ".join(errors)
        )

