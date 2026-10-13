"""
The analysis app's interface builds, and its handlers fit their outputs.

Building a Blocks resolves every component a handler is wired to, so a name
used in the wiring but never created fails here instead of at launch — where
it stops the app from starting at all, before a single page is served.
"""

import pytest

pytest.importorskip("gradio")


@pytest.fixture(scope="module")
def app():
    from sem_analysis_app.ui import create_interface

    return create_interface()


def wired(app, name):
    """Every dependency whose handler is the function called ``name``."""
    return [f for f in app.fns.values()
            if getattr(getattr(f, "fn", None), "__name__", "") == name]


def test_the_interface_builds(app):
    assert app is not None


def test_export_fills_both_of_its_outputs(app):
    # Export hands over a file and says what it did. It returns two values, so
    # it has to be wired to two components; one of them once did not exist.
    from sem_analysis_app.callbacks import export_results
    from sem_analysis_app.state import state

    saved, state.results_manager = state.results_manager, None
    try:
        returned = export_results()
    finally:
        state.results_manager = saved

    handlers = wired(app, "export_results")
    assert handlers, "Export is not wired to anything"
    for handler in handlers:
        assert len(handler.outputs) == len(returned)


@pytest.mark.parametrize("name, returned", [
    ("browse_for_folder", 5),
    ("restore_setup", 7),
    ("detect_scale_again", 5),
    ("check_outputs", 11),
    ("read_box_scale", 2),
    ("apply_two_points", 2),
    ("confirm_scale", 2),
])
def test_scale_and_setup_handlers_fill_their_outputs(app, name, returned):
    handlers = wired(app, name)
    assert handlers, f"{name} is not wired to anything"
    for handler in handlers:
        assert len(handler.outputs) == returned


def test_check_outputs_returns_what_it_is_wired_to(app):
    from sem_analysis_app.callbacks import check_outputs

    assert len(check_outputs()) == 11


def test_there_is_no_typed_folder_path(app):
    # The folder is chosen in the system's dialog; nothing on the page takes a
    # typed path to it any more.
    labels = [getattr(block, "label", None) for block in app.blocks.values()]
    assert "Image folder" not in labels
