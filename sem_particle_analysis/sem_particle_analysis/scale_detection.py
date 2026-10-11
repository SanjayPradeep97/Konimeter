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
import logging
import cv2
import numpy as np

from .utils import extract_tiff_metadata

logger = logging.getLogger(__name__)


class OCRUnavailableError(ValueError):
    """
    Raised when a scale bar must be read but EasyOCR is not installed.

    Subclasses ValueError so that callers already handling a failed OCR attempt —
    notably ``detect_scale(method='auto')``, which falls back to metadata — treat
    a missing library as just another way for OCR not to produce an answer.
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
    STANDARD_MANTISSAS = (1.0, 1.5, 2.0, 2.5, 3.0, 5.0)

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

    # Registry for custom tag parsers (extensibility for new microscope formats)
    # Format: {tag_code: [(parser_func, manufacturer_name), ...]}
    _TAG_PARSERS = {}

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

    # Confidence scoring for method comparison
    CONFIDENCE_SCORES = {'high': 3, 'medium': 2, 'low': 1}

    @classmethod
    def register_tag_parser(cls, tag_code, manufacturer, parser_func):
        """
        Register a custom parser for a specific TIFF tag code.

        This allows adding support for new microscope formats without
        modifying the core ScaleDetector code. The parser function receives
        the tag value and should return pixel size in nanometers, or None.

        Args:
            tag_code (int): TIFF tag code to handle
            manufacturer (str): Manufacturer name for identification
            parser_func (callable): Function(tag_value) -> float (nm) or None

        Example:
            ScaleDetector.register_tag_parser(
                tag_code=50000,
                manufacturer='oxford',
                parser_func=lambda v: float(v.get('PixelSize', 0)) * 1e9
            )
        """
        if tag_code not in cls._TAG_PARSERS:
            cls._TAG_PARSERS[tag_code] = []
        cls._TAG_PARSERS[tag_code].append((parser_func, manufacturer))

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
        self.last_metadata_result = None
        self.last_ocr_result = None

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

    # Where the OCR fallback used to look, and still does when a caller names a
    # region explicitly.
    DEFAULT_OCR_REGION = dict(region_x=0.75, region_y=0.95,
                              region_width=0.5, region_height=0.08)

    def detect_scale(self, image, file_path=None, method='auto',
                     region_x=None, region_y=None,
                     region_width=None, region_height=None,
                     polarity='auto', threshold=200):
        """
        Detect scale using the best available method.

        This is the main entry point for scale detection. In 'auto' mode, it
        tries TIFF metadata first, then OCR. When both succeed, it cross-checks
        the results and selects the most reliable one based on confidence scoring.

        Args:
            image (np.ndarray): RGB image array (required for OCR fallback)
            file_path (str, optional): Path to image file for metadata extraction
            method (str): Detection method - 'auto', 'metadata', or 'ocr'
                - 'auto': Try both, compare confidence, cross-check (default)
                - 'metadata': Only use metadata extraction (raises if fails)
                - 'ocr': Only use OCR detection
            region_x (float, optional): Horizontal center of OCR search region (0-1)
            region_y (float, optional): Vertical center of OCR search region (0-1)
            region_width (float, optional): Width fraction for OCR search region (0-1)
            region_height (float, optional): Height fraction for OCR search region (0-1)
                Leave all four unset — the usual case — to search every position a
                scale bar is normally printed in. Naming any of them restricts the
                search to that one region instead.
            polarity (str): Scale bar polarity - 'auto', 'bright', or 'dark'
            threshold (int): Binary threshold for OCR (0-255)

        Returns:
            dict: Detection results containing:
                - 'conversion': Conversion factor (nm/pixel)
                - 'method': Detection method used ('metadata' or 'ocr')
                - 'scale_nm': Physical scale in nanometers
                - 'pixel_length': Pixel length (1 for metadata, detected for OCR)
                - 'cross_check': Cross-check info if both methods succeeded
                - Additional method-specific fields

        Raises:
            ValueError: If scale cannot be detected by specified method
        """
        metadata_result = None
        ocr_result = None

        # Reset cross-check state
        self.last_metadata_result = None
        self.last_ocr_result = None

        # --- Try metadata extraction ---
        if method in ('auto', 'metadata') and file_path:
            if self._is_tiff(file_path):
                try:
                    logger.info(f"Attempting TIFF metadata extraction from: {file_path}")
                    metadata_result = self.detect_scale_from_metadata(file_path)
                    metadata_result['method'] = 'metadata'
                    self.last_metadata_result = metadata_result
                    logger.info(f"Metadata extraction successful: {metadata_result['conversion']:.4f} nm/pixel")
                except ValueError as e:
                    logger.warning(f"Metadata extraction failed: {e}")
                    if method == 'metadata':
                        raise

        # For 'metadata' only mode, return what we have
        if method == 'metadata':
            if metadata_result:
                self.last_detection = metadata_result
                return metadata_result
            raise ValueError("Metadata extraction failed and no fallback allowed")

        # --- Try OCR ---
        if method in ('auto', 'ocr'):
            try:
                logger.info("Attempting OCR-based scale bar detection")
                region = {k: v for k, v in (
                    ('region_x', region_x), ('region_y', region_y),
                    ('region_width', region_width), ('region_height', region_height),
                ) if v is not None}

                if region:
                    # An explicitly named region is used as given.
                    ocr_result = self.detect_scale_bar(
                        image, **{**self.DEFAULT_OCR_REGION, **region},
                        polarity=polarity, threshold=threshold,
                    )
                else:
                    # Nothing named, so look in all the usual places rather than
                    # only the bottom-right. That default suits an SEM databar and
                    # misses a TEM bar burned into the bottom-left entirely, which
                    # left every TEM frame in a batch run measured in pixels while
                    # the web app — which already swept — read them fine.
                    ocr_result = self.detect_scale_bar_anywhere(
                        image, polarity=polarity, threshold=threshold)
                ocr_result['method'] = 'ocr'
                ocr_result['confidence'] = 'medium'  # OCR is always medium confidence
                self.last_ocr_result = ocr_result
                logger.info(f"OCR detection successful: {ocr_result['conversion']:.4f} nm/pixel")
            except ValueError as e:
                logger.warning(f"OCR detection failed: {e}")
                if method == 'ocr':
                    raise

        # --- Select best result and attach cross-check info ---
        chosen = self._select_best_result(metadata_result, ocr_result)

        if chosen is None:
            raise ValueError("Could not detect scale using any available method")

        self.last_detection = chosen
        return chosen

    def _select_best_result(self, metadata_result, ocr_result):
        """
        Select the best detection result and attach cross-check info.

        When both methods succeed, compares confidence scores and attaches
        cross-check information showing whether the two methods agree.

        Args:
            metadata_result (dict or None): Result from metadata extraction
            ocr_result (dict or None): Result from OCR detection

        Returns:
            dict or None: The best result with cross-check info attached
        """
        if metadata_result and ocr_result:
            # Cross-check: compare the two conversions
            meta_conv = metadata_result['conversion']
            ocr_conv = ocr_result['conversion']
            ratio = meta_conv / ocr_conv if ocr_conv > 0 else float('inf')
            agrees = 0.9 <= ratio <= 1.1  # Within 10%

            cross_check = {
                'metadata_conversion': meta_conv,
                'ocr_conversion': ocr_conv,
                'agrees': agrees,
                'ratio': ratio,
            }

            # Select based on confidence
            meta_score = self.CONFIDENCE_SCORES.get(
                metadata_result.get('confidence', 'low'), 0)
            ocr_score = self.CONFIDENCE_SCORES.get(
                ocr_result.get('confidence', 'low'), 0)

            if meta_score >= ocr_score:
                chosen = metadata_result
            else:
                chosen = ocr_result

            chosen['cross_check'] = cross_check
            return chosen

        # Only one result available
        return metadata_result or ocr_result

    def _is_tiff(self, file_path):
        """Check if file is a TIFF based on extension."""
        if not file_path:
            return False
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ('.tif', '.tiff')

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
        # Extract metadata
        metadata = extract_tiff_metadata(file_path)

        if not metadata['is_tiff']:
            raise ValueError("Not a valid TIFF file")

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

        # Try generic resolution tags as fallback
        if pixel_size_nm is None:
            pixel_size_nm = self._parse_generic_resolution(metadata)
            if pixel_size_nm:
                metadata_source = 'TIFF XResolution/YResolution tags'
                confidence = 'medium'
                if manufacturer == 'unknown':
                    confidence = 'low'

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

        # Try registered custom tag parsers
        if pixel_size_nm is None and self._TAG_PARSERS:
            for tag_code, parsers in self._TAG_PARSERS.items():
                tag_val = metadata['raw_tags'].get(tag_code)
                if tag_val is not None:
                    for parser_func, mfr_name in parsers:
                        try:
                            parsed = parser_func(tag_val)
                            if parsed and parsed > 0:
                                pixel_size_nm = parsed
                                manufacturer = mfr_name
                                metadata_source = f'Registered parser ({mfr_name}, tag {tag_code})'
                                confidence = 'medium'
                                break
                        except Exception as e:
                            logger.debug(f"Custom parser for {mfr_name} failed: {e}")
                            continue
                if pixel_size_nm:
                    break

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

        # Primary: ap_pixel_size tuple -> ('Pixel Size', 26.37, 'nm')
        for key in ('ap_pixel_size', 'ap_image_pixel_size'):
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

        # Look for AP_PIXEL_SIZE
        patterns = [
            r'AP_PIXEL_SIZE\s*=\s*([0-9.eE+-]+)',
            r'Pixel\s*Size\s*=\s*([0-9.eE+-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    # Value is in meters, convert to nanometers
                    value_m = float(match.group(1))
                    return value_m * 1e9
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
            r'PixelSize\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
            r'Pixel\s*Size\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
            r'Resolution\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)/pixel',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    # Convert to nanometers
                    if unit in ('m',):
                        return value * 1e9
                    elif unit in ('um', 'µm'):
                        return value * 1000
                    else:  # nm
                        return value
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
                r'PixelSize\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
                r'Pixel\s*Size\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
            ]

            for pattern in patterns:
                match = re.search(pattern, image_desc, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        unit = match.group(2).lower()

                        if unit in ('m',):
                            return value * 1e9
                        elif unit in ('um', 'µm'):
                            return value * 1000
                        else:
                            return value
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
            r'PixelSize\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
            r'Pixel\s*Width\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    if unit in ('m',):
                        return value * 1e9
                    elif unit in ('um', 'µm'):
                        return value * 1000
                    else:
                        return value
                except (ValueError, IndexError):
                    continue

        return None

    # Standard display/scanner DPI values that are NOT SEM calibration
    STANDARD_DPI_VALUES = {72, 96, 150, 192, 200, 240, 300, 360, 600}

    def _parse_generic_resolution(self, metadata):
        """
        Parse standard TIFF XResolution/YResolution tags.

        These tags specify pixels per unit (where unit is defined by ResolutionUnit).
        ResolutionUnit: 1=none, 2=inch, 3=centimeter

        Args:
            metadata (dict): Metadata dict from extract_tiff_metadata()

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        x_res = metadata.get('x_resolution')
        y_res = metadata.get('y_resolution')
        res_unit = metadata.get('resolution_unit')

        # Need at least one resolution value
        resolution = x_res or y_res
        if resolution is None or resolution <= 0:
            return None

        # Reject standard display/scanner DPI values that are meaningless for SEM
        if res_unit in (2, 3) and resolution in self.STANDARD_DPI_VALUES:
            logger.info(f"Ignoring standard DPI value {resolution} (not SEM calibration)")
            return None

        # ResolutionUnit: 2=inch, 3=centimeter
        # pixels_per_unit is the resolution value
        # pixel_size = 1 / pixels_per_unit (in that unit)

        if res_unit == 2:  # Inch
            # resolution = pixels per inch
            # pixel_size_inch = 1 / resolution
            # pixel_size_nm = pixel_size_inch * 25.4e6 (25.4 mm/inch, 1e6 nm/mm)
            pixel_size_nm = 25.4e6 / resolution
            return pixel_size_nm

        elif res_unit == 3:  # Centimeter
            # resolution = pixels per cm
            # pixel_size_cm = 1 / resolution
            # pixel_size_nm = pixel_size_cm * 1e7 (1e7 nm/cm)
            pixel_size_nm = 1e7 / resolution
            return pixel_size_nm

        # Unit 1 (none) or unknown - can't determine scale
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

        candidates = {}
        for pol in polarities_to_try:
            if pol == 'bright':
                _, binary255 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            else:
                _, binary255 = cv2.threshold(gray, 255 - threshold, 255, cv2.THRESH_BINARY_INV)

            line = self._find_scale_line_morphological(binary255, max_thickness=max_thickness)
            if line is not None:
                leftmost, rightmost, row, spanned, area = line
                candidates[pol] = {"length": rightmost - leftmost, "area": area}
                # A bar with clear background on both sides always beats one that
                # ran to the crop edge, regardless of size. Otherwise the larger
                # solid rule wins: comparing the two polarities on width alone let
                # a wide, ragged run through background beat the real bar, which
                # is what put the wrong polarity on most of the TEM frames.
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
        pixel_length = rightmost - leftmost

        if pixel_length < 5:
            raise ValueError(
                f"Detected line too short ({pixel_length}px). "
                f"Adjust region to cover the scale bar."
            )

        ocr_text = self._read_text(crop)

        # Extract scale value and units
        salvaged = False
        try:
            scale_nm = self._parse_scale_text(ocr_text)
        except ValueError:
            scale_nm = self._salvage_micron_reading(ocr_text)
            if scale_nm is None:
                raise
            salvaged = True

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
            warning = (
                f"Unit unreadable in '{ocr_text}' — assumed µm, giving "
                f"{scale_nm / 1000:g} µm / {pixel_length}px. Confirm against the "
                f"image before trusting these measurements."
            )
        elif not self._is_standard_scale_value(scale_nm):
            warning = (
                f"Scale bar read as {scale_nm:g} nm from OCR text '{ocr_text}', "
                f"which is not a standard bar value — most likely a misread digit "
                f"(1 and 7 are commonly confused). Confirm against the image."
            )
        elif spanned_crop:
            warning = (
                f"Measured bar ({pixel_length}px) runs the full width of the search "
                f"region, so its true ends may be outside the box. Widen the region "
                f"so there is background on both sides of the bar, then re-detect."
            )
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
                warning = (
                    f"Ambiguous scale bar: bright polarity measured "
                    f"{candidates['bright']['length']}px, dark polarity "
                    f"{candidates['dark']['length']}px, and both look equally like a "
                    f"bar. Used '{best_polarity}' ({winner['length']}px). Verify the "
                    f"overlay, or set polarity explicitly."
                )

        # Store results
        self.last_detection = {
            'region': (x0, y0, box_w, box_h),
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

    def _read_text(self, crop):
        """
        OCR a crop, escalating effort until a digit comes back.

        A label like "1 µm" is mostly thin strokes, and at default sensitivity
        EasyOCR routinely returns just "µm" — the value silently lost. Each
        attempt below is progressively more aggressive; the first reading that
        contains a digit wins, since the number is the part that matters.

        Returns:
            str: Recognised text, space-joined.
        """
        images = [crop]
        height = crop.shape[0]
        if 0 < height < self.OCR_TARGET_HEIGHT:
            factor = min(4.0, self.OCR_TARGET_HEIGHT / height)
            enlarged = cv2.resize(crop, None, fx=factor, fy=factor,
                                  interpolation=cv2.INTER_CUBIC)
            # Keep the original as a fallback so nothing that used to be
            # readable stops being read.
            images = [enlarged, crop]

        attempts = [
            {},
            {"text_threshold": 0.4, "low_text": 0.3},
            {"text_threshold": 0.3, "low_text": 0.2, "allowlist": self.OCR_ALLOWLIST},
        ]

        # Resolved once, outside the loop: a missing EasyOCR must surface as its
        # own error rather than being swallowed by the per-attempt except and
        # reported as "no text found".
        reader = self.reader

        best = ""
        for image in images:
            for options in attempts:
                try:
                    found = reader.readtext(image, **options)
                except Exception:
                    continue
                text = " ".join(txt for _, txt, _ in found)
                if any(ch.isdigit() for ch in text):
                    return text
                if len(text) > len(best):
                    best = text
        return best

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

    # Known SEM instrument parameter patterns that should NOT be treated as scale
    # These are values like "8.8 mm" (working distance), "10.00 kV", etc.
    _INSTRUMENT_PARAM_PATTERNS = [
        r'WD\s*$',           # Working distance label right before the value
        r'^\s*\d+\.\d+\s*kV',  # Accelerating voltage
        r'^\s*\d+\s*pA',     # Beam current
        r'^\s*\d+\s*nA',     # Beam current
        r'^\s*\d+\s*ps',     # Dwell time
        r'^\s*\d+\s*us',     # Dwell time
        r'PM\b',             # AM/PM time marker
        r'AM\b',             # AM/PM time marker
    ]

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
        pattern = (r'(\d+(?:\.\d+)?)\s*'
                   r'(nm|uum|pum|jum|lum|ium|prn|hrn|urn|um|µm|μm|hm|mm|cm|'
                   r'angstroms|angstrom|å|a)')

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
        base = row_medians[H - 1]

        # Walk up from the last row while the background level holds. Scanning
        # further than the largest acceptable bar is deliberate: it is what makes
        # "no boundary found" distinguishable from "a very tall bar".
        #
        # A single disagreeing row does not end the bar — it is usually a rule
        # drawn across it. Only a run of them means the micrograph has started.
        # `height` is where the walk has reached; `confirmed` is the last row
        # that actually matched, so trailing rule rows are not counted in.
        scan_limit = min(H, int(H * self.DATABAR_SCAN_FRACTION))
        height = confirmed = 0
        misses = 0
        while height < scan_limit:
            if abs(row_medians[H - 1 - height] - base) <= self.DATABAR_MEDIAN_TOLERANCE:
                misses = 0
                height += 1
                confirmed = height
            else:
                misses += 1
                if misses > self.DATABAR_MAX_RULE_THICKNESS_PX:
                    break
                height += 1
        height = confirmed

        # Too short to be a databar, or no top edge inside the scan window. The
        # latter means the whole bottom quarter of the frame looks like its last
        # row — a featureless image, not an image with a bar stuck on it. The
        # previous implementation scanned largest-first and returned the first
        # window that passed, so it reported the largest uniform-looking window
        # rather than the actual edge: it overestimated the height on real
        # databars, and claimed one on images that had none. Both cost real
        # micrograph area, since cli.crop_databar trims whatever is reported.
        if not (self.DATABAR_MIN_HEIGHT_PX <= height <= int(H * self.DATABAR_MAX_FRACTION)):
            return result

        above = gray[:H - height]
        if above.size == 0:
            return result

        # A databar is visibly a different block from the image above it.
        separation = abs(float(np.median(gray[H - height:])) - float(np.median(above)))
        if separation < self.DATABAR_MIN_SEPARATION:
            return result

        # ...and it begins with a drawn edge. Checking the top two rows rather
        # than only the first tolerates a divider rule along the very top.
        edge_rows = gray[H - height:H - height + 2].astype(float)
        if edge_rows.std(axis=1).min() > self.DATABAR_EDGE_ROW_STD:
            return result

        result['has_databar'] = True
        result['databar_height'] = height
        result['databar_fraction'] = height / H
        return result

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

    def crop_scale_bar(self, image, crop_percent=7.0):
        """
        Crop the bottom portion of an image to remove the scale bar.

        Args:
            image (np.ndarray): Input image
            crop_percent (float): Percentage to crop from bottom

        Returns:
            np.ndarray: Cropped image
        """
        height = image.shape[0]
        crop_height = int(height * (1 - crop_percent / 100))
        return image[:crop_height, :].copy()

    def get_conversion_factor(self):
        """
        Get the conversion factor from the last detection.

        Returns:
            float: Conversion factor (nm/pixel)

        Raises:
            RuntimeError: If no detection has been performed
        """
        if self.last_detection is None:
            raise RuntimeError("No scale bar has been detected yet")
        return self.last_detection['conversion']

    def set_manual_scale(self, scale_nm, pixel_length):
        """
        Manually set the scale calibration.

        Args:
            scale_nm (float): Physical length in nanometers
            pixel_length (int): Length in pixels

        Returns:
            float: Calculated conversion factor (nm/pixel)

        Raises:
            ValueError: If values are not positive
        """
        if scale_nm <= 0 or pixel_length <= 0:
            raise ValueError("Scale and pixel length must be positive values")

        conversion = scale_nm / pixel_length

        self.last_detection = {
            'pixel_length': pixel_length,
            'scale_nm': scale_nm,
            'conversion': conversion,
            'ocr_text': 'Manual entry',
            'manual': True
        }

        return conversion
