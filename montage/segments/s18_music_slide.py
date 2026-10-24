"""18 音樂的下滑 — the picture almost stops; the sound leads. The last image of
the mirrors hangs on screen, sinking a fraction and dimming while the music
slides down and fades out.
"""
import numpy as np

from . import get, segment
from ..draw import to_rgb
from ..fx import blur, ease, resize


_HELD = {}


def held_frame(ctx):
    """The last frame of 17, rendered once per process and size."""
    key = (ctx.W, ctx.H)
    if key not in _HELD:
        src = ctx.tl.segment("mirrors")
        _HELD[key] = to_rgb(get("mirrors")(ctx.other("mirrors", src.n_frames - 1)))
    return _HELD[key]


@segment("music_slide")
def render(ctx):
    img = held_frame(ctx)
    u = ease(ctx.u)
    W, H = ctx.W, ctx.H
    # sink a little and dim almost to black
    dy = int(round(u * 0.03 * H))
    out = np.zeros_like(img)
    out[dy:] = img[:H - dy]
    out = out * (1 - 0.85 * u)
    g = out.mean(-1, keepdims=True)
    out = out * (1 - u) + g * u                          # drain the last colour
    return np.clip(out + 0.05 * blur(out, 20 * ctx.scale), 0, 1)
