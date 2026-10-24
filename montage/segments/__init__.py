"""Segment registry.

Each segment is one function `render(ctx) -> image` registered with
@segment("<name in timeline.json>"). It returns a float image in 0..1, either
HxW (black & white; the cold tint is added in post) or HxWx3 (colour).
Film grain / vignette / dust are applied afterwards by the pipeline.
"""
import importlib
import pkgutil
import zlib
from dataclasses import dataclass

import numpy as np

from ..fx import Canvas

REGISTRY = {}
THUMBS = {}          # optional: segment name -> fn(tl, seg) -> frames for the contact sheet


def segment(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


@dataclass
class Ctx:
    tl: object          # Timeline
    seg: object         # Segment
    frame: int          # frame index inside the segment
    W: int
    H: int
    ss: int             # supersampling for vector drawing

    @property
    def fps(self):
        return self.tl.fps

    @property
    def t(self):                      # seconds since segment start
        return self.frame / self.fps

    @property
    def u(self):                      # 0..1 progress through the segment
        return self.frame / max(1, self.seg.n_frames - 1)

    @property
    def duration(self):
        return self.seg.n_frames / self.fps

    @property
    def shot(self):
        return self.seg.shot_at(self.frame)[0]

    @property
    def shot_t(self):
        return self.seg.shot_at(self.frame)[1] / self.fps

    @property
    def shot_u(self):
        _, f, n = self.seg.shot_at(self.frame)
        return f / max(1, n - 1)

    @property
    def scale(self):                  # 1.0 at 1080p; use for line widths
        return self.H / 1080.0

    @property
    def params(self):
        return self.seg.params

    @property
    def palette(self):
        return self.seg.palette

    def seed(self, *extra):
        """Deterministic seed list for this segment (+ anything extra)."""
        return [self.tl.seed, self.seg.index,
                *[zlib.crc32(e.encode()) if isinstance(e, str) else int(e) for e in extra]]

    def rng(self, *extra):
        """Per-frame RNG (different every frame, identical on every run)."""
        return np.random.default_rng(self.seed(self.frame, *extra))

    def canvas(self):
        return Canvas(self.W, self.H, self.ss)

    def other(self, name, frame):
        """Context for rendering a frame of a different segment (for ghosts / match cuts)."""
        return Ctx(self.tl, self.tl.segment(name), frame, self.W, self.H, self.ss)


def thumbnails(name):
    """Decorator: choose which frames of a segment go on its contact sheet."""
    def deco(fn):
        THUMBS[name] = fn
        return fn
    return deco


def placeholder(ctx):
    """Used for segments not written yet: dark frame + name, so a full render still runs."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("L", (ctx.W, ctx.H), 12)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", int(48 * ctx.scale))
    except OSError:
        font = ImageFont.load_default()
    d.text((ctx.W / 2, ctx.H / 2), f"{ctx.seg.slug}  {ctx.seg.title}\n(not yet rendered)",
           fill=90, font=font, anchor="mm", align="center")
    return np.asarray(img, np.float32) / 255.0


def get(name):
    return REGISTRY.get(name, placeholder)


def implemented(name):
    return name in REGISTRY


for _m in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_m.name}")
