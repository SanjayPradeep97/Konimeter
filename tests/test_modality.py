"""
Reading a frame's instrument kind, and what it implies.

The kind decides which way round the particles are and whether a databar is
expected. Both are silent when wrong: the wrong polarity measures the background,
and the wrong databar assumption either leaves one in frame or crops micrograph
away.
"""

import pytest

from sem_particle_analysis import modality as mod


def tags(raw=None):
    """Metadata shaped like utils.extract_tiff_metadata. Tag codes are ints, so
    they are passed as a dict rather than as keyword arguments."""
    return {"raw_tags": dict(raw or {}), "image_description": None, "software": None}


class TestFromMetadata:
    def test_fei_states_its_own_type(self):
        found = mod.from_metadata(tags({34682: {"System": {"Type": "SEM",
                                                           "SystemType": "Teneo"}}}))
        assert found.kind == mod.SEM
        assert found.instrument == "fei"
        assert found.from_metadata is True
        assert "Teneo" in found.detail

    def test_an_fei_tag_without_a_type_still_means_scanning(self):
        found = mod.from_metadata(tags({34682: {"Scan": {"PixelWidth": 1e-9}}}))
        assert found.kind == mod.SEM
        assert found.instrument == "fei"

    def test_jeol_reports_tem_mode(self):
        text = "Gun LaB6 HT 120 ( Mode TEM uP SA Zoom Image ( Magn 11000x"
        found = mod.from_metadata(tags({65027: text}))
        assert found.kind == mod.TEM
        assert found.instrument == "jeol"
        assert found.from_metadata is True

    def test_jeol_reports_sem_mode(self):
        found = mod.from_metadata(tags({65027: "Mode SEM something"}))
        assert found.kind == mod.SEM

    def test_jeol_text_survives_nul_padding(self):
        # The real tag stores the text with NULs between characters.
        padded = "\x00".join("Mode TEM uP")
        assert mod.from_metadata(tags({65027: padded})).kind == mod.TEM

    def test_zeiss_tag_means_sem(self):
        found = mod.from_metadata(tags({34118: {"ap_pixel_size": ("Pixel Size", 26.4, "nm")}}))
        assert found.kind == mod.SEM
        assert found.instrument == "zeiss"

    def test_hitachi_tag_means_sem(self):
        found = mod.from_metadata(tags({34118: {"": ("something",)}}))
        assert found.kind == mod.SEM
        assert found.instrument == "hitachi"

    def test_a_maker_named_in_the_description(self):
        metadata = {"raw_tags": {}, "image_description": "Carl Zeiss SmartSEM",
                    "software": None}
        assert mod.from_metadata(metadata).instrument == "zeiss"

    def test_nothing_conclusive_is_unknown(self):
        assert mod.from_metadata(tags()).kind == mod.UNKNOWN
        assert mod.from_metadata(None).kind == mod.UNKNOWN


class TestFilesThatNameTheirInstrument:
    def test_jeol_tem_center_tags_mean_tem(self):
        # JEOL's TEM software writes numeric private tags and no text; on the
        # test set they are on every JEOL TEM TIFF and on no SEM file.
        found = mod.from_metadata(tags({65006: "-0.0", 65007: "-0.0", 65009: "1.0",
                                        65010: "1.0", 65027: b"\x00"}))
        assert found.kind == mod.TEM and found.instrument == "jeol"

    def test_an_amt_camera_means_tem(self):
        metadata = {"raw_tags": {}, "software": None,
                    "image_description": "NIOSH\r40000\rXpixCal=620.66\rUnit=um"}
        found = mod.from_metadata(metadata)
        assert found.kind == mod.TEM and found.instrument == "amt"

    def test_a_jeol_sem_says_so_in_its_txt(self):
        text = "$CM_INSTRUMENT JSM 6500F\n$$SM_COLUMN_MODE SEM\n"
        assert mod.from_sidecar(text).kind == mod.SEM

    def test_a_jsm_without_a_column_mode_is_still_an_sem(self):
        assert mod.from_sidecar("$CM_INSTRUMENT JSM 7600F\n").kind == mod.SEM


class TestDetect:
    def test_the_file_is_read(self):
        found = mod.detect(metadata=tags({34682: {"System": {"Type": "SEM"}}}))
        assert found.kind == mod.SEM
        assert found.from_metadata is True

    def test_the_txt_beside_the_image_is_read(self):
        found = mod.detect(metadata=tags(), sidecar="$CM_INSTRUMENT JSM 6500F\n")
        assert found.kind == mod.SEM

    def test_a_silent_file_is_unknown_not_guessed(self):
        # Frames that say nothing are left to the microscope chosen on Setup;
        # guessing from a square frame or a printed bar called old SEM JPEGs TEM.
        found = mod.detect(metadata=tags())
        assert found.kind == mod.UNKNOWN
        assert found.detail


class TestImplications:
    def test_sem_particles_are_bright_and_tem_particles_are_dark(self):
        # Material on an SEM filter substrate scatters more than the membrane;
        # electron-dense material in TEM blocks the beam.
        assert mod.Modality(kind=mod.SEM).dark_particles is False
        assert mod.Modality(kind=mod.TEM).dark_particles is True

    def test_an_unknown_kind_leaves_polarity_to_contrast(self):
        assert mod.Modality().dark_particles is None

    def test_label_names_the_instrument_when_known(self):
        assert mod.Modality(kind=mod.TEM, instrument="jeol").label == "TEM (JEOL)"
        assert mod.Modality(kind=mod.SEM).label == "SEM"

    def test_to_dict_carries_the_evidence(self):
        data = mod.Modality(kind=mod.TEM, instrument="jeol", detail="tag 65027",
                            from_metadata=True).to_dict()
        assert data["kind"] == "TEM"
        assert data["dark_particles"] is True
        assert data["detail"] == "tag 65027"
        assert data["from_metadata"] is True


class TestResolve:
    @pytest.mark.parametrize("value", [None, "auto", "AUTO", ""])
    def test_auto_means_detect_it(self, value):
        assert mod.resolve(value) is None

    @pytest.mark.parametrize("value,kind", [("SEM", mod.SEM), ("tem", mod.TEM)])
    def test_an_explicit_choice_is_honoured(self, value, kind):
        found = mod.resolve(value)
        assert found.kind == kind
        assert found.from_metadata is False

    def test_an_existing_modality_passes_through(self):
        original = mod.Modality(kind=mod.SEM, instrument="fei")
        assert mod.resolve(original) is original

    def test_an_unknown_name_is_an_error(self):
        with pytest.raises(ValueError):
            mod.resolve("cryo-FIB")
