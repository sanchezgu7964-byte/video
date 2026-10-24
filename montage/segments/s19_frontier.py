"""19 邊境、馬匹、拂曉 — frontier, horses, dawn. The second great warm image of
the film, rhyming with the sunrise of 04: an open plain at first light, a thin
line across it that is a border, and far off a few horses running.
"""
import numpy as np

from . import segment
from ..draw import pal
from ..fx import Canvas, blur, ease, resize
from ..geom import perlin
from .s22_charge import leg


def full_horse(cv, x, ground, L, phase, value=1.0):
    """Whole horse in profile, running right. L = leg length in pixels."""
    hy = ground - L * 1.05
    s = cv.ss
    cv.d.ellipse([(x - L * 0.25) * s, (hy - L * 0.62) * s, (x + L * 1.55) * s, (hy + L * 0.18) * s], fill=int(value * 255))
    # neck and head, bobbing with the stride
    bob = np.sin(2 * np.pi * phase) * L * 0.06
    neck = [(x + L * 1.25, hy - L * 0.45), (x + L * 1.55, hy - L * 0.25), (x + L * 1.95, hy - L * 1.0 + bob),
            (x + L * 1.75, hy - L * 1.15 + bob)]
    head = [(x + L * 1.72, hy - L * 1.15 + bob), (x + L * 2.0, hy - L * 1.05 + bob), (x + L * 2.35, hy - L * 0.72 + bob),
            (x + L * 2.25, hy - L * 0.62 + bob), (x + L * 1.95, hy - L * 0.8 + bob)]
    cv.polygon(neck, value)
    cv.polygon(head, value)
    cv.line([(x - L * 0.2, hy - L * 0.4), (x - L * 0.55, hy - L * 0.1 + bob), (x - L * 0.6, hy + L * 0.25)], value, L * 0.1)
    for off, dx in ((0.0, 0.05), (0.1, 0.2), (0.52, 1.15), (0.62, 1.3)):
        leg(cv, x + dx * L, hy, L, phase + off)


@segment("frontier")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    u = ctx.u
    top, low, sun_c, land = (pal(ctx, k, d) for k, d in
                             (("sky_top", "#3b4a7a"), ("sky_low", "#ffb25c"), ("sun", "#fff0c8"), ("land", "#3a2416")))
    hz = H * 0.62
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.clip((hz - y) / hz, 0, 1)[..., None]
    sky = low * (1 - a) ** 1.6 + top * (1 - (1 - a) ** 1.6)
    sx, sy = W * 0.68, hz - H * (0.02 + 0.07 * ease(u))
    d = np.sqrt((x - sx) ** 2 + (y - sy) ** 2) / H
    sky = sky * (0.75 + 0.25 * u) + sun_c * (0.9 * np.exp(-d / 0.04) + 0.35 * np.exp(-d / 0.25))[..., None]
    sky = np.where((d < 0.035)[..., None], sun_c * 1.1, sky)
    # the plain: dark warm earth, grass texture, light raking across it
    P = perlin(ctx.tl.seed + 191)
    gy = np.clip((y - hz) / (H - hz), 1e-3, 1)
    tex = 0.75 + 0.25 * P.fbm2(x / (W * 0.08) / (0.2 + gy), y / (H * 0.02) / (0.2 + gy), 3)
    rake = 0.35 + 0.65 * np.exp(-np.abs(x - sx) / (W * 0.35)) * np.exp(-(gy) / 0.25)
    ground = land * (tex * (1.0 + 1.6 * rake))[..., None] + sun_c * (0.22 * np.exp(-gy / 0.06))[..., None]
    img = np.where((y < hz)[..., None], sky, ground)
    # the border: one thin line across the land, a few posts along it
    cv = Canvas(W, H, ctx.ss)
    ax, ay, bx, by = -0.05 * W, H * 0.9, 1.05 * W, hz + H * 0.03
    cv.line([(ax, ay), (bx, by)], 1.0, 1.4 * sc)
    for k in range(1, 30):
        f = (k / 30) ** 0.7
        px, py = ax + (bx - ax) * f, ay + (by - ay) * f
        hgt = H * 0.05 * (1 - f) + 2 * sc
        cv.line([(px, py), (px, py - hgt)], 1.0, max(1.0, 3 * (1 - f)) * sc)
    line = cv.array()
    img = img * (1 - 0.6 * line[..., None]) + sun_c * 0.75 * line[..., None]          # the border catches the light
    # distant horses running along the horizon
    hc = Canvas(W, H, ctx.ss)
    rng = np.random.default_rng(ctx.seed("horses"))
    for k in range(int(ctx.params.get("horses", 7))):
        L = H * rng.uniform(0.018, 0.03)
        x0 = rng.uniform(-0.1, 0.6) * W + ctx.t * W * rng.uniform(0.035, 0.05)
        full_horse(hc, x0, hz + H * rng.uniform(0.002, 0.02), L, ctx.t * 2.4 + rng.uniform(0, 1))
    hm = hc.array()
    img = img * (1 - 0.92 * hm[..., None]) + 0.02 * hm[..., None]
    return np.clip(img + 0.08 * blur(img, 30 * sc), 0, 1)
