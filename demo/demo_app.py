"""demo_app.py -- segment a particle, and see it classified.

The two halves of the paper's pipeline joined end to end, on 60 held-out test
images that ship with the repository (20 Fiber, 20 Cluster, 20 Matrix; see
demo_images.csv, select_demo_images.py and ATTRIBUTION.md):

    open a micrograph  ->  SAM (ViT-B) makes a first guess at the particle
    click to correct   ->  SAM re-runs on the clicks (positive: particle,
                           negative: background)
    every mask         ->  DINOv2 hypercolumn features are pooled INSIDE the
                           mask and the paper's five-fold MLP ensemble
                           predicts Fiber / Cluster / Matrix / MatrixSurface

The first guess needs no click: SAM is prompted, out of sight, at the darkest
large object away from the frame edge (prompting.auto_point), and the largest
dark one of its three masks is shown.  A click on the particle replaces that
starting point; a click on the background trims the mask instead; after that,
clicks add up.  The classifier never sees the image as a whole: only the
pixels inside the mask.

Start it with run_demo.bat / run_demo.sh, or

    python demo_app.py

The SAM ViT-B checkpoint (358 MB) is downloaded from the page the first time;
DINOv2's weights (330 MB) are fetched by timm on first use.
"""
from __future__ import annotations
import csv, html, os, sys, threading, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "classification"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

# status lines use a few unicode glyphs; a cp1252 console would choke
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import gradio as gr
from PIL import Image, ImageDraw

import download_sam_weights as weights
from classify import CLASSES
from prompting import MODE_RULE, MODE_SCORE, auto_point, choose_mask

COLOURS = {"Fiber": (200, 60, 60), "Cluster": (40, 150, 60),
           "Matrix": (50, 90, 210), "MatrixSurface": (160, 80, 200)}
DISPLAY_W = 1024                  # images are 1350 x 1040; shown at this width
POSITIVE, NEGATIVE = "Particle (+)", "Background (−)"


# ----------------------------------------------------------------- data
def demo_images():
    """stem -> (image path, expert label)."""
    images = {}
    for row in csv.DictReader(open(HERE / "demo_images.csv")):
        stem = Path(row["filename"]).stem
        path = HERE / "images" / row["category"] / f"{stem}.png"
        if path.exists():
            images[stem] = (path, row["category"])
    return images


IMAGES = demo_images()
# A fixed shuffled order, so neighbouring entries do not share a class.
CHOICES = sorted(IMAGES)
np.random.default_rng(0).shuffle(CHOICES)


# ----------------------------------------------------------------- models
def sam_checkpoint():
    """SAM_CHECKPOINT if it names a file, else sam_weights/sam_vit_b_01ec64.pth."""
    named = os.environ.get("SAM_CHECKPOINT")
    if named and Path(named).is_file():
        return Path(named)
    return weights.checkpoint_path("vit_b")


class Models:
    """SAM and the classifier, loaded once, on first need."""
    sam = None
    clf = None
    image = None                      # the stem SAM's predictor is set to
    lock = threading.Lock()

    @classmethod
    def ready(cls):
        return cls.sam is not None and cls.clf is not None

    @classmethod
    def load(cls, progress=None):
        with cls.lock:
            if cls.sam is None:
                if progress:
                    progress(0.2, desc="Loading SAM ViT-B")
                from sam_model import SAMModel
                cls.sam = SAMModel(sam_checkpoint())
            if cls.clf is None:
                if progress:
                    progress(0.6, desc="Loading DINOv2 and the classifier "
                                       "(330 MB is downloaded the first time)")
                from classify import CNTClassifier
                cls.clf = CNTClassifier()

    @classmethod
    def show(cls, stem, rgb):
        """Encode an image with SAM unless it is the one already encoded."""
        if cls.image != stem:
            cls.sam.set_image(rgb)
            cls.image = stem


# ----------------------------------------------------------------- session
class Session:
    """One browser tab's image, prompts and mask."""
    def __init__(self):
        self.stem = None; self.gray = None; self.rgb = None
        self.points = []; self.labels = []
        self.seeded = False           # points[0] is the automatic prompt, not shown
        self.history = []             # earlier (points, labels, seeded)
        self.mask = None; self.pred = None; self.probs = None
        self.mode = MODE_RULE
        self.timing = ""

    @property
    def first_guess(self):
        """Nothing clicked yet: only the automatic prompt."""
        return self.seeded and len(self.points) == 1

    @property
    def clicks(self):
        """The points the analyst placed, with their labels."""
        return list(zip(self.points, self.labels))[1 if self.seeded else 0:]

    def remember(self):
        self.history.append((list(self.points), list(self.labels), self.seeded))


def segment(s):
    """Run SAM on the session's points and classify the mask it chooses."""
    Models.show(s.stem, s.rgb)
    t0 = time.time()
    masks, scores, _ = Models.sam.predict(np.array(s.points, float),
                                          np.array(s.labels, int))
    s.mask = choose_mask(masks, scores, s.gray, s.mode)
    classify(s, f"SAM {1000 * (time.time() - t0):.0f} ms")


def classify(s, sam_time=""):
    if s.mask is None or not s.mask.any():
        s.pred, s.probs = None, None
        return
    t0 = time.time()
    s.pred, s.probs = Models.clf.predict(s.gray, s.mask)
    clf_time = f"classifier {1000 * (time.time() - t0):.0f} ms"
    s.timing = ", ".join(t for t in (sam_time, clf_time) if t)


def first_guess(s):
    s.points, s.labels = [list(auto_point(s.gray))], [1]
    s.seeded, s.history = True, []
    segment(s)


# ----------------------------------------------------------------- drawing
def render(s):
    if s.gray is None:
        return None
    h, w = s.gray.shape
    dh = int(round(h * DISPLAY_W / w))
    out = np.asarray(Image.fromarray(s.rgb).resize((DISPLAY_W, dh), Image.BILINEAR),
                     dtype=np.float32)
    if s.mask is not None and s.mask.any():
        m = np.asarray(Image.fromarray(s.mask.astype(np.uint8) * 255)
                       .resize((DISPLAY_W, dh), Image.NEAREST)) > 127
        colour = np.array(COLOURS.get(s.pred, (255, 200, 0)), np.float32)
        out[m] = 0.6 * out[m] + 0.4 * colour
        from skimage.segmentation import find_boundaries
        out[find_boundaries(m, mode="outer")] = colour
    im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    scale = DISPLAY_W / w
    for (x, y), label in s.clicks:
        cx, cy = x * scale, y * scale
        fill = (60, 210, 60) if label == 1 else (230, 50, 50)
        draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=fill, outline="white", width=2)
    return im


def report(s):
    """The prediction, as HTML."""
    if s.gray is None:
        return '<div class="muted">Choose a micrograph.</div>'
    if s.pred is None:
        return ('<div class="muted">No mask. Click on a particle, or press '
                '<b>Reset</b>.</div>')
    _, truth = IMAGES[s.stem]
    agrees = s.pred == truth
    rows = "".join(
        f'<div class="bar-row"><span class="bar-name">{c}</span>'
        f'<span class="bar"><span style="width:{100 * p:.1f}%;background:'
        f'rgb{COLOURS[c]}"></span></span><span class="bar-pct">{100 * p:.1f}%</span></div>'
        for c, p in zip(CLASSES, s.probs))
    area = int(s.mask.sum())
    clicks = len(s.clicks)
    how = ("SAM's first guess, before any click" if not clicks
           else f"SAM, from {clicks} click{'' if clicks == 1 else 's'}")
    note = ""
    inside, outside = float(s.gray[s.mask].mean()), float(s.gray[~s.mask].mean())
    if inside >= outside:
        note = ("This mask is brighter than its surroundings, so it is probably "
                "support film. Click on the dark particle.")
    elif area > 0.5 * s.mask.size:
        note = "The mask covers most of the frame. Click on the film to trim it."
    colour = "rgb{}".format(COLOURS[s.pred])
    return (
        f'<div class="pred-label" style="color:{colour}">{s.pred}</div>'
        f'<div class="pred-truth">Expert label: <b>{truth}</b> · '
        f'<span class="{"ok" if agrees else "bad"}">'
        f'{"agrees" if agrees else "differs"}</span></div>'
        f'<div class="bars">{rows}</div>'
        f'<div class="muted">{html.escape(how)} · {100 * area / s.mask.size:.1f}% of the '
        f'frame{" · " + s.timing if s.timing else ""}</div>'
        + (f'<div class="note">{html.escape(note)}</div>' if note else ""))


def view(s):
    return render(s), report(s), s


# ----------------------------------------------------------------- events
def models_status():
    """What the setup panel says, and whether it is shown."""
    if sam_checkpoint().is_file():
        return gr.update(visible=False), ""
    return gr.update(visible=True), ""


def download_sam(progress=gr.Progress()):
    def report_bytes(done, total):
        if total:
            progress(done / total, desc=f"Downloading SAM ViT-B: {done / 2**20:.0f} "
                                        f"of {total / 2**20:.0f} MB")
    try:
        weights.download("vit_b", progress=report_bytes)
    except Exception as problem:
        return gr.update(visible=True), (f"The download failed ({problem}). Check the "
                                         "internet connection and try again.")
    return gr.update(visible=False), "SAM ViT-B downloaded."


def need_models(progress):
    """Load the models if they can be; a message saying why not if they cannot."""
    if Models.ready():
        return None
    if not sam_checkpoint().is_file():
        return "Download the SAM model first, with the button above."
    Models.load(progress)
    return None


def open_image(stem, s, progress=gr.Progress()):
    s = Session()
    if not stem:
        return view(s)
    problem = need_models(progress)
    if problem:
        return None, f'<div class="note">{problem}</div>', s
    s.stem = stem
    path, _ = IMAGES[stem]
    s.gray = np.asarray(Image.open(path).convert("L"))
    s.rgb = np.stack([s.gray] * 3, -1)
    progress(0.8, desc="Segmenting")
    first_guess(s)
    return view(s)


def next_image(stem):
    if stem not in CHOICES:
        return CHOICES[0]
    return CHOICES[(CHOICES.index(stem) + 1) % len(CHOICES)]


def click(kind, s, evt: gr.SelectData):
    if s is None or s.gray is None:
        return view(s or Session())
    x, y = (evt.index or (None, None))[:2]
    if x is None or y is None:
        return view(s)
    scale = s.gray.shape[1] / DISPLAY_W
    point, label = [x * scale, y * scale], 1 if kind == POSITIVE else 0
    s.remember()
    if s.first_guess and label == 1:
        # A click on the particle replaces the automatic prompt: it says where
        # the particle is. A background click trims the first guess instead.
        s.points, s.labels, s.seeded = [point], [label], False
    else:
        s.points.append(point); s.labels.append(label)
    segment(s)
    return view(s)


def undo(s):
    if s is None or s.gray is None or not s.history:
        return view(s or Session())
    s.points, s.labels, s.seeded = s.history.pop()
    segment(s)
    return view(s)


def reset(s):
    if s is None or s.gray is None:
        return view(s or Session())
    first_guess(s)
    return view(s)


def set_mode(mode, s):
    s.mode = mode
    if s.gray is not None:
        segment(s)
    return view(s)


# ----------------------------------------------------------------- page
CSS = """
.pred-label {font-size: 30px; font-weight: 700; line-height: 1.2;}
.pred-truth {font-size: 14px; margin: 2px 0 12px;}
.pred-truth .ok {color: #15803d; font-weight: 600;}
.pred-truth .bad {color: #b91c1c; font-weight: 600;}
.dark .pred-truth .ok {color: #4ade80;}
.dark .pred-truth .bad {color: #f87171;}
.bars {margin-bottom: 10px;}
.bar-row {display: flex; align-items: center; gap: 8px; font-size: 13px; margin: 3px 0;}
.bar-name {width: 104px;}
.bar {flex: 1; height: 10px; border-radius: 5px; overflow: hidden;
      background: var(--background-fill-secondary);}
.bar span {display: block; height: 100%;}
.bar-pct {width: 52px; text-align: right; font-variant-numeric: tabular-nums;}
.muted {font-size: 13px; color: var(--body-text-color-subdued);}
.note {font-size: 13px; margin-top: 8px; padding: 6px 10px; border-radius: 6px;
       background: rgba(217, 119, 6, .12);}
.hint p {font-size: 13px; color: var(--body-text-color-subdued); margin: 0;}
.field-label p {font-size: 14px; margin: 0;}
.picker {align-items: center;}
"""


def build_app():
    with gr.Blocks(title="CNT segment and classify") as app:
        gr.Markdown(
            "## Segment a carbon-nanotube particle, and see it classified\n"
            "SAM outlines the particle; DINOv2 features pooled **inside the outline** "
            "go to the paper's five-fold MLP ensemble (92.7 %, 166/179, on the held-out "
            f"test set). {len(IMAGES)} held-out test micrographs, none seen in training.")
        state = gr.State(Session())

        with gr.Column(visible=False) as setup:
            gr.Markdown("**The SAM ViT-B model (358 MB) is needed once.** It is saved in "
                        "`sam_weights/` and found there from then on.")
            download_btn = gr.Button("Download SAM ViT-B", variant="primary")
        setup_status = gr.Markdown("")

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                image = gr.Image(type="pil", show_label=False, interactive=False,
                                 buttons=[])
            with gr.Column(scale=2):
                gr.Markdown("**Micrograph**", elem_classes=["field-label"])
                with gr.Row(equal_height=True, elem_classes=["picker"]):
                    choice = gr.Dropdown(CHOICES, value=None, label="Micrograph",
                                         show_label=False, container=False, scale=4)
                    next_btn = gr.Button("Next", scale=1, min_width=80)
                result = gr.HTML('<div class="muted">Choose a micrograph.</div>')
                gr.Markdown("If the outline is wrong, click on the image. **Particle** "
                            "marks the particle, **Background** what to leave out.",
                            elem_classes=["hint"])
                kind = gr.Radio([POSITIVE, NEGATIVE], value=POSITIVE, show_label=False)
                with gr.Row():
                    undo_btn = gr.Button("Undo")
                    reset_btn = gr.Button("Reset")
                with gr.Accordion("Which of SAM's three masks is kept", open=False):
                    mode = gr.Radio([MODE_RULE, MODE_SCORE], value=MODE_RULE,
                                    show_label=False)

        outputs = [image, result, state]
        app.load(models_status, outputs=[setup, setup_status])
        download_btn.click(download_sam, outputs=[setup, setup_status])
        choice.change(open_image, [choice, state], outputs)
        next_btn.click(next_image, [choice], [choice])
        image.select(click, [kind, state], outputs)
        undo_btn.click(undo, [state], outputs)
        reset_btn.click(reset, [state], outputs)
        mode.change(set_mode, [mode, state], outputs)

    # SAM's predictor holds one image at a time, so requests run one by one.
    for event in app.fns.values():
        event.concurrency_id = "demo"
        event.concurrency_limit = 1
    return app


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Segment a CNT particle with SAM and classify it.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)
    if not IMAGES:
        raise SystemExit(f"no demo images under {HERE / 'images'}; is the checkout complete?")
    if sam_checkpoint().is_file():
        print("loading SAM ViT-B, DINOv2 and the classifier heads ...")
        Models.load()
    else:
        print("SAM ViT-B not downloaded yet: the page offers to download it.")
    build_app().launch(server_name=args.host, server_port=args.port, css=CSS,
                       theme=gr.themes.Soft(), show_error=True,
                       inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
