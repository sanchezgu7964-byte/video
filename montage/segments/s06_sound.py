"""06 聲音出現 — still dark. A faint point of light at the centre and slow
ripples spreading out from it: a sound arriving from far below.

ring_times() is shared with the soundtrack, so each ripple has its own low ping.
"""
import numpy as np

from . import segment
from ..fx import blur
from ..geom import perlin


def ring_times(params, duration):
    n = int(params.get("rings", 5))
    return [0.25 + i * (duration - 0.75) / max(1, n - 1) for i in range(n)]


@segment("sound")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    p = ctx.params
    b = float(p.get("brightness", 0.18))
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = (x - W / 2) / H, (y - H / 2) / H
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    P = perlin(ctx.tl.seed + 61)
    wob = 0.006 * P.noise2(np.cos(ang) * 2 + ctx.t * 0.3, np.sin(ang) * 2)
    img = np.zeros((H, W), np.float32)
    t = ctx.t
    for i, t0 in enumerate(ring_times(p, ctx.duration)):
        age = t - t0
        if age <= 0:
            continue
        rad = 0.14 * age ** 0.8
        width = 0.004 + 0.01 * age
        amp = b * np.exp(-age / 1.3) * (1 - np.exp(-age / 0.05))
        img += amp * np.exp(-((r + wob - rad) / width) ** 2)
    # the source: a tiny breathing point
    pulse = 0.6 + 0.4 * np.sin(t * 2 * np.pi / 0.9) ** 2
    fade_in = min(1.0, t / 0.6)
    img += fade_in * b * 1.6 * pulse * np.exp(-r / 0.006) + fade_in * b * 0.35 * np.exp(-r / 0.05)
    return np.clip(img + 0.4 * blur(img, 12 * sc), 0, 1)
