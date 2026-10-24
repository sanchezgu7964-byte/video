"""05 黑屏與疊影 — hard cut to black. Fragments of earlier shots flash up,
half-transparent and stacked on top of each other, gone after a few frames.
"""
from functools import lru_cache

import numpy as np

from . import get, segment, thumbnails
from ..draw import to_rgb
from ..fx import resize


@lru_cache(maxsize=4)
def flash_plan(seed, seg_index, n_frames, fps, flashes, flash_len, sources):
    """List of flashes: (start_frame, length, [(source, frame, opacity, zoom, dx, dy), ...])."""
    rng = np.random.default_rng([seed, seg_index, 505])
    L = max(1, int(round(flash_len * fps)))
    lo, hi = int(0.6 * fps), n_frames - int(0.45 * fps)
    starts = []
    while len(starts) < flashes:
        s = int(rng.integers(lo, hi))
        if all(abs(s - o) > L + 4 for o in starts):
            starts.append(s)
    plan = []
    for s in sorted(starts):
        layers = []
        for src in rng.choice(sources, size=int(rng.integers(2, 4)), replace=False):
            layers.append((str(src), float(rng.uniform(0.1, 0.95)), float(rng.uniform(0.6, 1.0)),
                           float(rng.uniform(1.0, 1.35)), float(rng.uniform(-0.08, 0.08)),
                           float(rng.uniform(-0.06, 0.06))))
        plan.append((s, L + int(rng.integers(0, 2)), layers))
    return plan


def zoomed(img, z, dx, dy):
    H, W = img.shape[:2]
    if z <= 1.001 and dx == 0 and dy == 0:
        return img
    cw, ch = W / z, H / z
    x0 = int(np.clip((W - cw) / 2 + dx * W, 0, W - cw))
    y0 = int(np.clip((H - ch) / 2 + dy * H, 0, H - ch))
    crop = img[y0:y0 + int(ch), x0:x0 + int(cw)]
    return np.stack([resize(crop[..., c], W, H) for c in range(3)], -1)


@segment("blackout")
def render(ctx):
    p = ctx.params
    tint = ctx.tl.look.get("bw_tint", [1, 1, 1])
    plan = flash_plan(ctx.tl.seed, ctx.seg.index, ctx.seg.n_frames, ctx.fps, int(p.get("flashes", 6)),
                      float(p.get("flash_length", 0.12)),
                      tuple(p.get("sources", ["viaduct", "rail_yard", "smoke", "last_judgment"])))
    img = np.zeros((ctx.H, ctx.W, 3), np.float32)
    op = float(p.get("opacity", 0.35))
    for start, L, layers in plan:
        k = ctx.frame - start
        if not 0 <= k < L:
            continue
        env = (1 - k / L) ** 1.4
        for src, pos, o, z, dx, dy in layers:
            seg = ctx.tl.segment(src)
            f = int(pos * (seg.n_frames - 1))
            layer = to_rgb(get(src)(ctx.other(src, f)), tint)
            img = img + zoomed(layer, z, dx, dy) * (op * o * env)
    return np.clip(img, 0, 1)


@thumbnails("blackout")
def blackout_thumbs(tl, seg):
    """The evenly spaced default would land between flashes: show each flash's first frame."""
    p = seg.params
    plan = flash_plan(tl.seed, seg.index, seg.n_frames, tl.fps, int(p.get("flashes", 6)),
                      float(p.get("flash_length", 0.12)),
                      tuple(p.get("sources", ["viaduct", "rail_yard", "smoke", "last_judgment"])))
    return [0] + [start for start, _, _ in plan] + [seg.n_frames - 1]
