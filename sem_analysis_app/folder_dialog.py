"""
The operating system's own folder picker, opened from the app.

A browser cannot hand a web page the path of a folder, only copies of the files
in it, and results have to be saved beside the originals. So the picker is
opened by the app itself, on the computer it runs on, in a separate process:
macOS and Tk both insist that windows belong to a program's main thread, and
Gradio runs callbacks in worker threads.
"""

import os
import subprocess
import sys

PROMPT = "Choose a folder of images"

# Run in a child Python on Windows and Linux. It prints the chosen path, or
# nothing when the dialog is cancelled.
_TK_SCRIPT = r"""
import sys
import tkinter
from tkinter import filedialog

root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.update()
path = filedialog.askdirectory(parent=root, title=sys.argv[1],
                               initialdir=sys.argv[2] or None, mustexist=True)
root.destroy()
sys.stdout.buffer.write((path or "").encode("utf-8"))
"""


class FolderDialogError(RuntimeError):
    """The folder picker could not be opened."""


def _applescript_text(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _choose_macos(start):
    where = (f" default location (POSIX file {_applescript_text(start)})"
             if start else "")
    script = ("activate\n"
              f"POSIX path of (choose folder with prompt {_applescript_text(PROMPT)}{where})")
    result = subprocess.run(["osascript", "-e", script], capture_output=True,
                            text=True, encoding="utf-8")
    if result.returncode != 0:
        if "-128" in result.stderr:      # the analyst pressed Cancel
            return None
        raise FolderDialogError(result.stderr.strip() or "osascript failed")
    return result.stdout.strip() or None


def _choose_tk(start):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run([sys.executable, "-c", _TK_SCRIPT, PROMPT, start or ""],
                            capture_output=True, creationflags=flags)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise FolderDialogError(detail[-1] if detail else "the folder dialog failed")
    return result.stdout.decode("utf-8").strip() or None


def choose_folder(start=None):
    """
    Ask for a folder with the system's dialog.

    Args:
        start: Folder to open the dialog in, if it exists.

    Returns:
        str or None: The chosen folder, or None if the dialog was cancelled.

    Raises:
        FolderDialogError: If no dialog could be shown.
    """
    start = start if start and os.path.isdir(start) else os.path.expanduser("~")
    try:
        path = _choose_macos(start) if sys.platform == "darwin" else _choose_tk(start)
    except OSError as exc:
        raise FolderDialogError(str(exc)) from exc
    return os.path.normpath(path) if path else None
