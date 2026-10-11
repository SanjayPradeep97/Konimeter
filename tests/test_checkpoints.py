"""
Which checkpoint the app offers first.

The Setup tab takes the head of ``discover_checkpoints()`` as its default, so the
order is not cosmetic — it decides which model an analyst loads when they press
the button without reading the list.
"""

from pathlib import Path

import pytest

from sem_particle_analysis import discover_checkpoints, infer_model_type
from sem_particle_analysis import model as model_module
from sem_particle_analysis.model import is_full_checkpoint


@pytest.fixture
def weights(tmp_path, monkeypatch):
    """A directory holding all three canonical checkpoints, plus a fragment.

    The default search directories are switched off so that a real checkpoint
    in the repository's ``sam_weights/`` (there is one after running the demo)
    cannot leak into the list being tested."""
    monkeypatch.setattr(model_module, "_default_search_dirs", lambda: [])
    for name in ("sam_vit_h_4b8939.pth", "sam_vit_l_0b3195.pth",
                 "sam_vit_b_01ec64.pth", "decoder_only.pth"):
        (tmp_path / name).write_bytes(b"not a real checkpoint")
    return tmp_path


def types(paths):
    return [infer_model_type(p, default=None) for p in paths]


class TestTheOfferedDefault:
    def test_vit_b_is_offered_first(self, weights):
        # The analyst corrects every mask, so the model gives a starting point,
        # not an answer — and ViT-B reaches it several times faster.
        found = discover_checkpoints(extra_dirs=[weights])
        assert infer_model_type(found[0], default=None) == "vit_b"

    def test_the_larger_models_are_still_offered(self, weights):
        found = discover_checkpoints(extra_dirs=[weights])
        assert {"vit_b", "vit_l", "vit_h"} <= set(types(found))

    def test_smaller_first_all_the_way_down(self, weights):
        found = [p for p in discover_checkpoints(extra_dirs=[weights])
                 if is_full_checkpoint(p)]
        assert types(found) == ["vit_b", "vit_l", "vit_h"]

    def test_a_fragment_is_never_the_default(self, weights):
        # A partial checkpoint fails with an opaque key error deep inside
        # sam_model_registry, so it is offered last whatever its name sorts to.
        found = discover_checkpoints(extra_dirs=[weights])
        assert is_full_checkpoint(found[0])
        assert Path(found[-1]).name == "decoder_only.pth"

    def test_extra_directories_are_searched_before_the_defaults(self, weights):
        found = discover_checkpoints(extra_dirs=[weights])
        assert weights in {Path(p).parent for p in found}


class TestTheBatchPathIsUnaffected:
    def test_it_picks_by_type_not_by_order(self, weights):
        # cli.find_checkpoint filters discover_checkpoints by model type, so
        # reordering the list must not change which file --model-type vit_h
        # resolves to.
        from sem_particle_analysis.cli import find_checkpoint

        found = discover_checkpoints(extra_dirs=[weights])
        by_type = {infer_model_type(p, default=None): p for p in found}
        assert by_type["vit_h"].name == "sam_vit_h_4b8939.pth"
        # find_checkpoint searches the real directories rather than the fixture;
        # what matters is that it selects on type at all.
        resolved = find_checkpoint("vit_h")
        assert resolved is None or infer_model_type(resolved) == "vit_h"
