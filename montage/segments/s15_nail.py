"""15 指甲 — macro: the pale half-moon at the root of a fingernail, its curve
extending very slowly out of the dark. Quiet, faintly unsettling.
Shallow depth of field: only a band around the lunula is sharp.
"""
import numpy as np

from . import segment
from ..fx import Canvas, blur, ease
from ..geom import perlin


@segment("nail")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    u = ctx.u
    g = float(ctx.params.get("growth", 0.35))
    cx, cy, R = W * 0.5, H * 1.02, H * 0.62
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    # nail plate: a broad rounded shape rising from the bottom, faint ridges along it
    plate = Canvas(W, H, ctx.ss)
    t = np.linspace(np.pi, 2 * np.pi, 80)
    plate.polygon([(cx + 1.05 * R * np.cos(a) * 1.15, cy + 1.35 * R * np.sin(a)) for a in t], 1.0)
    pm = plate.array()
    P = perlin(ctx.tl.seed + 151)
    ridges = 0.85 + 0.15 * P.noise2(x / (W * 0.012), y / (H * 0.6))
    sheen = np.exp(-(((x - cx) / W + 0.12 - 0.04 * u) / 0.07) ** 2) * 0.25
    plate_col = pm * (0.09 + 0.05 * ridges + sheen * (y / H))
    # the lunula: a crescent whose arc lengthens through the shot
    half = np.radians(14 + 70 * ease(u) * g / 0.35)
    a = np.linspace(-np.pi / 2 - half, -np.pi / 2 + half, 120)
    moon = Canvas(W, H, ctx.ss)
    outer = [(cx + R * np.cos(v), cy + R * 0.82 * np.sin(v)) for v in a]
    inner = [(cx + R * 0.93 * np.cos(v), cy + 0.06 * H + R * 0.78 * np.sin(v)) for v in a[::-1]]
    moon.polygon(outer + inner, 0.5)
    moon.line(outer, 1.0, 2.0 * sc)
    mm = moon.array()
    sharp = plate_col + mm * 0.45
    # depth of field: focus band follows the top of the arc
    focus_y = cy - R * 0.82
    w = np.clip(np.abs(y - focus_y) / (H * 0.18), 0, 1)
    soft = blur(sharp, 22 * sc)
    img = sharp * (1 - w) + soft * w
    img = img + 0.25 * blur(mm, 18 * sc) * 0.4
    col = np.stack([img * 1.06, img * 0.98, img * 0.92], -1)      # skin-warm, barely saturated
    return np.clip(col, 0, 1)
