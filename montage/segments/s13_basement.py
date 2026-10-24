"""13 地下室 — sudden stillness. Almost total darkness; a small patch of light
from a barred vent creeps across the stone, revealing it only where it falls.
"""
from functools import lru_cache

import numpy as np

from . import segment
from ..fx import Canvas, blur, resize
from ..geom import perlin


@lru_cache(maxsize=2)
def stone(seed, W, H):
    """Rough stone wall: irregular courses of blocks + noise, computed once per size."""
    P = perlin(seed + 131)
    w, h = W // 2, H // 2
    y, x = np.mgrid[0:h, 0:w].astype(np.float32) / h
    rough = 0.6 + 0.25 * P.fbm2(x * 9, y * 9, 4) + 0.15 * P.noise2(x * 60, y * 60)
    course = 0.11
    row = np.floor(y / course)
    off = (row % 2) * 0.09 + P.noise2(row * 3.1, 0.5) * 0.04
    bx = (x + off) / 0.19
    jx = np.abs(bx - np.round(bx)) * 0.19
    jy = np.abs(y / course - np.round(y / course)) * course
    joints = np.exp(-np.minimum(jx, jy) / 0.004)
    g = rough * (1 - 0.6 * joints)
    return resize(g.astype(np.float32), W, H)


@segment("basement")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    p = ctx.params
    tex = stone(ctx.tl.seed, W, H)
    speed = float(p.get("speed", 0.03))
    # the patch drifts slowly down and to the right, like the sun moving outside
    cx = W * 0.36 + speed * ctx.t * H * 1.4
    cy = H * 0.42 + speed * ctx.t * H * 0.5
    pw, ph = H * 0.2, H * 0.13
    sk = H * 0.06
    patch = Canvas(W, H, ctx.ss)
    patch.polygon([(cx - pw / 2 + sk, cy - ph / 2), (cx + pw / 2 + sk, cy - ph / 2),
                   (cx + pw / 2 - sk, cy + ph / 2), (cx - pw / 2 - sk, cy + ph / 2)], 1.0)
    bars = Canvas(W, H, ctx.ss)
    for k in range(1, 5):
        bx = cx - pw / 2 + k * pw / 5
        bars.line([(bx + sk, cy - ph / 2 - 5), (bx - sk, cy + ph / 2 + 5)], 1.0, H * 0.012)
    light = blur(np.clip(patch.array() - bars.array(), 0, 1), 3 * sc)
    light = light + 0.25 * blur(light, 50 * sc)
    glow = float(p.get("glow", 0.12))
    warm = np.array([1.0, 0.86, 0.66], np.float32)
    img = (tex * light * glow * 2.2)[..., None] * warm
    # the rest of the room exists only as a trace
    img = img + (tex * 0.006)[..., None] * np.array([0.8, 0.85, 1.0])
    # a few motes hanging in the beam
    rng = np.random.default_rng(ctx.seed("motes"))
    motes = Canvas(W, H, ctx.ss)
    for _ in range(40):
        x0, y0 = rng.uniform(-0.15, 0.15) * H, rng.uniform(-0.1, 0.1) * H
        ph_ = rng.uniform(0, 6.28)
        x = cx + x0 + np.sin(ctx.t * 0.4 + ph_) * 8 * sc
        y = cy + y0 + ctx.t * 3 * sc
        motes.dot(x, y, rng.uniform(0.5, 1.2) * sc, rng.uniform(0.3, 0.8))
    img = img + (motes.array() * blur(light, 20 * sc) * glow * 3)[..., None] * warm
    return np.clip(img, 0, 1)
