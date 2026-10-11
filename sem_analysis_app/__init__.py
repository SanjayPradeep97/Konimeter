"""
Gradio web application for SEM/TEM particle analysis.

Layout:
    state.py       the shared AppState instance
    callbacks/     event handlers, one module per tab
    ui.py          interface definition and event wiring
    visualization.py  overlay and figure rendering

Run it with ``konimeter``, or ``python -m sem_analysis_app``.
"""

__all__ = ["create_interface"]


def __getattr__(name):
    """
    Build the interface only when it is asked for.

    Importing it eagerly pulls in Gradio, which meant the application state — a
    plain object with no UI dependencies of its own — could not be imported, or
    tested, without the whole web stack present.
    """
    if name == "create_interface":
        from .ui import create_interface

        return create_interface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
