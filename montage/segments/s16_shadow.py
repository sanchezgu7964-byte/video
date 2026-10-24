"""16 遺忘的影子 — a figure stands against a wall lit from the side; its shadow
peels away from the body, drifts off and fades: the forgotten shadow.
"""
import numpy as np

from . import segment
from ..fx import Canvas, blur, ease, smoothstep


def figure(cv, x, base, h, value=1.0, skew=0.0, sx=1.0):
    """Standing figure in screen space (feet at (x, base)); skew shears it sideways (for shadows)."""
    head = [(0.11 * np.cos(a), 1.62 + 0.13 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 20)]
    body = [(-0.2, 1.45), (0.2, 1.45), (0.24, 0.9), (0.15, 0.9), (0.13, 0.0), (0.03, 0.0), (0.0, 0.8),
            (-0.03, 0.0), (-0.13, 0.0), (-0.15, 0.9), (-0.24, 0.9)]
    k = h / 1.75
    for poly in (head, body):
        cv.polygon([(x + (px * sx + skew * pz) * k, base - pz * k) for px, pz in poly], value)


@segment("shadow")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    p = ctx.params
    u = ctx.u
    base = H * 0.86
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    # wall lit from the left, a strip of floor
    wall = 0.06 + 0.16 * np.exp(-((x / W - 0.3) / 0.45) ** 2) * (0.7 + 0.3 * (1 - y / H))
    floor = (y > base).astype(np.float32)
    img = wall * (1 - floor) + floor * (0.05 + 0.04 * np.exp(-(x / W - 0.3) ** 2 / 0.2))
    fx = W * 0.36
    d = smoothstep(float(p.get("detach_start", 0.25)), 0.75, u)
    fade = 1 - smoothstep(0.55, float(p.get("fade_end", 0.9)), u)
    sh = Canvas(W, H, ctx.ss)
    # the shadow falls up the wall to the right; once free it drifts away and lifts
    figure(sh, fx + W * 0.12 + d * W * 0.32, base - d * H * 0.06, H * 0.62 * (1 + 0.1 * d), 1.0,
           skew=0.35 + 0.2 * d, sx=1.0)
    shadow = blur(sh.array(), (2 + 10 * d) * sc) * 0.75 * fade
    img = img * (1 - shadow)
    fig = Canvas(W, H, ctx.ss)
    figure(fig, fx, base, H * 0.62)
    m = fig.array()
    img = img * (1 - m) + 0.012 * m
    col = np.stack([img * 1.04, img, img * 0.95], -1)
    return np.clip(col, 0, 1)
