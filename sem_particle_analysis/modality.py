"""
Which kind of instrument produced a frame, and what that implies.

SEM and TEM micrographs need opposite handling in three places, and getting any
of them backwards is silent rather than loud:

* **Databar.** SEM exports usually carry one below the image and it must be
  cropped off; TEM frames have none and cropping discards micrograph.
* **Polarity.** Particles collected on an SEM filter substrate scatter more than
  the membrane and read *brighter*; in TEM, electron-dense material reads
  *darker* than the support film. A detector that guesses this from absolute
  contrast picks the wrong side whenever something else in the frame — a dark
  aperture vignette, a grid bar — out-contrasts the particles.
* **Analysable area.** What counts as "not sample" differs by instrument.

The kind is read from the metadata the instrument wrote, and falls back to frame
geometry only when the file says nothing. It is always reported with its
evidence so a wrong guess is visible and correctable rather than silent.
"""

from dataclasses import dataclass
from typing import Optional

SEM = "SEM"
TEM = "TEM"
UNKNOWN = "unknown"

# What each kind implies for particle polarity, as the ``dark_features`` argument
# ParticleSegmenter.rank_candidates takes. None means "decide from contrast",
# which is only right when we genuinely do not know.
_DARK_PARTICLES = {SEM: False, TEM: True, UNKNOWN: None}


@dataclass(frozen=True)
class Modality:
    """
    A frame's instrument kind, with the evidence for it.

    Attributes:
        kind: ``SEM``, ``TEM`` or ``UNKNOWN``.
        instrument: Manufacturer key when known — 'fei', 'jeol', 'zeiss', ...
        detail: Human-readable justification, shown in the UI and recorded in
            the provenance file.
        from_metadata: True when the instrument said so itself, rather than the
            kind being inferred from how the frame looks.
    """

    kind: str = UNKNOWN
    instrument: Optional[str] = None
    detail: str = ""
    from_metadata: bool = False

    @property
    def dark_particles(self):
        """``dark_features`` for the segmenter, or None to decide by contrast."""
        return _DARK_PARTICLES.get(self.kind)

    @property
    def expects_databar(self):
        """Whether a strip below the micrograph is expected for this kind."""
        return self.kind == SEM

    @property
    def label(self):
        maker = f" ({self.instrument.upper()})" if self.instrument else ""
        return f"{self.kind}{maker}"

    def to_dict(self):
        return {
            "kind": self.kind,
            "instrument": self.instrument,
            "detail": self.detail,
            "from_metadata": self.from_metadata,
            "dark_particles": self.dark_particles,
        }


# Manufacturers that only ever make one of the two, so the maker settles the kind.
_MAKER_KIND = {
    "zeiss": SEM,
    "hitachi": SEM,
    "tescan": SEM,
}


def _text_of(value):
    """Flatten a TIFF tag value to searchable text, NULs stripped."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).replace("\x00", "")


def from_metadata(metadata):
    """
    Read the instrument kind out of raw TIFF metadata.

    Args:
        metadata (dict): As returned by ``utils.extract_tiff_metadata``.

    Returns:
        Modality: ``UNKNOWN`` when the file carries nothing conclusive.
    """
    if not metadata:
        return Modality(detail="no metadata to read")

    tags = metadata.get("raw_tags") or {}

    # FEI/Thermo write a structured dict that names the instrument type outright.
    fei = tags.get(34682) or tags.get("FEI_HELIOS")
    if isinstance(fei, dict):
        system = fei.get("System") or {}
        declared = str(system.get("Type", "")).strip().upper()
        model = system.get("SystemType") or ""
        if declared in (SEM, TEM):
            return Modality(
                kind=declared, instrument="fei", from_metadata=True,
                detail=f"FEI tag 34682 System.Type={declared}"
                       + (f" on a {model}" if model else ""),
            )
        return Modality(kind=SEM, instrument="fei", from_metadata=True,
                        detail="FEI tag 34682 present; FEI TIFFs of this shape are SEM")

    # Zeiss / Hitachi share tag 34118 but only build scanning instruments.
    if isinstance(tags.get(34118) or tags.get("CZ_SEM"), dict):
        maker = "zeiss" if ("ap_pixel_size" in (tags.get(34118) or tags.get("CZ_SEM"))
                            or "dp_sem" in (tags.get(34118) or tags.get("CZ_SEM"))) else "hitachi"
        return Modality(kind=SEM, instrument=maker, from_metadata=True,
                        detail=f"{maker} tag 34118")

    # JEOL write a free-text block naming the imaging mode.
    for code, value in tags.items():
        if not isinstance(code, int) or code < 65000:
            continue
        text = _text_of(value)
        if "Mode" in text and "TEM" in text:
            return Modality(kind=TEM, instrument="jeol", from_metadata=True,
                            detail=f"JEOL tag {code} reports TEM mode")
        if "Mode" in text and "SEM" in text:
            return Modality(kind=SEM, instrument="jeol", from_metadata=True,
                            detail=f"JEOL tag {code} reports SEM mode")

    # Anything the description or software tag names outright.
    haystack = " ".join(filter(None, (
        _text_of(metadata.get("image_description") or ""),
        _text_of(metadata.get("software") or ""),
    )))
    for maker, kind in _MAKER_KIND.items():
        if maker in haystack.lower():
            return Modality(kind=kind, instrument=maker, from_metadata=True,
                            detail=f"{maker} named in the file description")

    return Modality(detail="metadata names no instrument")


def from_appearance(image, databar_height=0):
    """
    Guess the kind from how the frame looks, for files whose metadata is silent.

    Weak evidence deliberately reported as such: an appended databar means a
    scanning instrument wrote it, and a bare square frame is characteristic of a
    TEM camera readout. Anything else stays unknown rather than being guessed.

    Args:
        image (np.ndarray): The full frame, databar included.
        databar_height (int): Rows of databar found, 0 if none.

    Returns:
        Modality: Never ``from_metadata``.
    """
    if databar_height > 0:
        return Modality(kind=SEM, detail=f"{databar_height}px databar below the image")

    height, width = image.shape[:2]
    if height == width:
        return Modality(kind=TEM,
                        detail=f"square {width}px frame with no databar")

    return Modality(detail="no databar and a non-square frame")


def detect(image, metadata=None, databar_height=0):
    """
    Best available reading of a frame's instrument kind.

    Metadata wins; appearance is consulted only when the file says nothing.

    Returns:
        Modality
    """
    stated = from_metadata(metadata)
    if stated.kind != UNKNOWN:
        return stated

    guessed = from_appearance(image, databar_height=databar_height)
    if guessed.kind != UNKNOWN:
        return Modality(kind=guessed.kind, instrument=None,
                        detail=f"{stated.detail}; inferred from {guessed.detail}")
    return Modality(detail=f"{stated.detail}; {guessed.detail}")


def resolve(name):
    """
    Turn a user-supplied choice into a Modality, for overriding detection.

    Args:
        name: 'SEM', 'TEM', 'auto'/None, or an existing Modality.

    Returns:
        Modality or None when the caller means "detect it".
    """
    if isinstance(name, Modality):
        return name
    if name is None:
        return None
    text = str(name).strip().upper()
    if text in ("", "AUTO"):
        return None
    if text in (SEM, TEM):
        return Modality(kind=text, detail="set by hand", from_metadata=False)
    if text == UNKNOWN.upper():
        return Modality(detail="set by hand")
    raise ValueError(f"Unknown modality {name!r}. Use 'SEM', 'TEM' or 'auto'.")
