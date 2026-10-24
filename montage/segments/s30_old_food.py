"""30 英雄的古老食糧 — the ancient food of heroes. A dead stop; colour gone.
Three heavy, slow shots, each an abstraction:
  falsehood   — a mask that cracks and parts
  failure     — upright lines toppling, one after another, to the ground
  humiliation — a figure pressed down by a descending weight
"""
import numpy as np

from . import segment
from ..fx import Canvas, blur, ease, smoothstep
from .s16_shadow import figure


def room(W, H):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    return 0.03 + 0.1 * np.exp(-((x / W - 0.5) ** 2 / 0.12 + (y / H - 0.35) ** 2 / 0.2))


def shot_mask(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    u = ease(ctx.shot_u)
    img = room(W, H)
    cx, cy, rx, ry = W / 2, H * 0.48, H * 0.24, H * 0.33
    rng = np.random.default_rng(ctx.seed("crack"))
    ys = np.linspace(cy - ry * 1.1, cy + ry * 1.1, 18)
    jag = cx + rng.normal(0, H * 0.012, len(ys))
    gap = u * H * 0.05
    for side in (-1, 1):
        cv = Canvas(W, H, ctx.ss)
        th = np.linspace(-np.pi / 2, np.pi / 2, 60) if side > 0 else np.linspace(np.pi / 2, 3 * np.pi / 2, 60)
        arc = [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in th]
        arc = [p for p in arc if (p[0] - cx) * side >= -2]
        edge = list(zip(jag, ys))
        poly = arc + (edge[::-1] if side > 0 else edge)
        ang = side * u * 0.05
        c, s = np.cos(ang), np.sin(ang)
        poly = [(cx + c * (x - cx) - s * (y - cy) + side * gap, cy + s * (x - cx) + c * (y - cy) + u * H * 0.01)
                for x, y in poly]
        cv.polygon(poly, 1.0)
        m = cv.array()
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        shade = 0.35 + 0.45 * np.clip(1 - np.hypot((x - cx + rx * 0.4) / rx, (y - cy + ry * 0.3) / ry) * 0.7, 0, 1)
        img = img * (1 - m) + shade * m
    holes = Canvas(W, H, ctx.ss)
    for ex in (-1, 1):
        x0 = cx + ex * rx * 0.42 + ex * gap
        holes.d.ellipse([(x0 - rx * 0.17) * holes.ss, (cy - ry * 0.2) * holes.ss, (x0 + rx * 0.17) * holes.ss,
                         (cy - ry * 0.05) * holes.ss], fill=255)
    holes.line([(cx - rx * 0.3 - gap, cy + ry * 0.45), (cx + rx * 0.3 + gap, cy + ry * 0.45)], 1.0, 4 * sc)
    hm = holes.array()
    return img * (1 - hm) + 0.01 * hm


def shot_fall(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    img = room(W, H)
    base = H * 0.82
    cv = Canvas(W, H, ctx.ss)
    for k in range(9):
        x0 = W * (0.12 + 0.095 * k)
        f = ease(smoothstep(0.05 + 0.08 * k, 0.35 + 0.08 * k, ctx.shot_u))
        ang = -f * (np.pi / 2 - 0.04 * (k % 3))                 # topples to the left, to the floor
        L = H * 0.6
        cv.line([(x0, base), (x0 + np.sin(ang) * L, base - np.cos(ang) * L)], 1.0, 7 * sc)
    m = cv.array()
    floor = np.zeros((H, W), np.float32)
    floor[int(base):] = 1
    img = img * (1 - floor) + 0.05 * floor
    return img * (1 - m) + 0.55 * m


def shot_pressed(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    u = ease(ctx.shot_u)
    img = room(W, H)
    base = H * 0.84
    slab_y = H * (0.02 + 0.42 * u)
    h = H * 0.62 * (1 - 0.5 * u)                              # the figure is pressed lower
    fig = Canvas(W, H, ctx.ss)
    figure(fig, W / 2, base, h, 1.0, skew=0.28 * u, sx=1 + 0.35 * u)
    slab = Canvas(W, H, ctx.ss)
    slab.polygon([(0, 0), (W, 0), (W, slab_y), (0, slab_y)], 1.0)
    fm, sm = fig.array(), slab.array()
    img = img * (1 - 0.6 * blur(sm, 40 * sc) * (1 - sm))         # the slab's shadow
    img = img * (1 - fm) + 0.015 * fm
    return img * (1 - sm) + 0.16 * sm


@segment("old_food")
def render(ctx):
    img = (shot_mask, shot_fall, shot_pressed)[ctx.shot % 3](ctx)
    return np.clip(img, 0, 1)
