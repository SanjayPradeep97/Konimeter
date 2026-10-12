"""
Loading micrographs: every format an instrument or a user is likely to hand over
comes back as 8-bit RGB with its contrast intact.

A 12-bit image in a 16-bit TIFF used to load nearly black, so the scale bar,
the databar and the particles were all lost; a float TIFF did not load at all.
"""

import numpy as np
import pytest
from PIL import Image

from sem_particle_analysis.utils import load_image

tifffile = pytest.importorskip("tifffile")


def test_a_12_bit_image_in_a_16_bit_tiff_keeps_its_contrast(tmp_path):
    data = np.linspace(0, 4094, 64 * 64).reshape(64, 64).astype(np.uint16)
    tifffile.imwrite(tmp_path / "frame.tif", data)
    image = load_image(str(tmp_path / "frame.tif"))
    assert image.shape == (64, 64, 3) and image.dtype == np.uint8
    assert image.min() == 0 and image.max() == 255


def test_a_float_tiff_loads(tmp_path):
    tifffile.imwrite(tmp_path / "frame.tif",
                     np.linspace(-1, 3, 64 * 64, dtype=np.float32).reshape(64, 64))
    image = load_image(str(tmp_path / "frame.tif"))
    assert image.dtype == np.uint8 and image.max() == 255


def test_8_bit_values_are_left_alone(tmp_path):
    Image.fromarray(np.full((8, 8), 77, np.uint8)).save(tmp_path / "gray.png")
    assert (load_image(str(tmp_path / "gray.png")) == 77).all()


def test_colour_order_is_rgb_and_alpha_is_dropped(tmp_path):
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 10, 20, 30, 128
    Image.fromarray(rgba, "RGBA").save(tmp_path / "colour.png")
    assert load_image(str(tmp_path / "colour.png"))[0, 0].tolist() == [10, 20, 30]


def test_a_folder_name_with_accents_and_mu_opens(tmp_path):
    folder = tmp_path / "Échantillon µ"
    folder.mkdir()
    Image.fromarray(np.full((8, 8), 5, np.uint8)).save(folder / "frame.png")
    assert load_image(str(folder / "frame.png")).shape == (8, 8, 3)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "missing.tif"))
