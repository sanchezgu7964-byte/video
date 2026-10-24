"""29 「這一切都給了你」 — everything was given to you. One frame from every
earlier segment, in order, flashing past at two or three frames each.
"""
from . import get, segment
from ..draw import to_rgb


@segment("all_given")
def render(ctx):
    earlier = [s for s in ctx.tl.segments if s.index < ctx.seg.index]
    src = earlier[ctx.shot % len(earlier)]
    a, n = src.shot_frames[len(src.shot_frames) // 2]          # the middle of its middle shot
    img = get(src.name)(ctx.other(src.name, a + n // 2))
    return to_rgb(img, ctx.tl.look.get("bw_tint", [1, 1, 1]) if img.ndim == 2 else (1, 1, 1))
