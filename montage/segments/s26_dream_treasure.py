"""26 夢如埋藏的寶藏 — dreams like buried treasure. The underground framing of
13 (stone, darkness, a single source of light) returns, but now the light is
warm gold: a cavity deep in the earth glowing, gold dust rising through cracks.
"""
import numpy as np

from . import segment
from ..fx import Canvas, blur, ease
from .s13_basement import stone


@segment("dream_treasure")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    tex = stone(ctx.tl.seed, W, H)
    u = ease(ctx.u)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    gold = np.array([1.0, 0.78, 0.35], np.float32)
    cx, cy = W * 0.5, H * 0.68
    # the cavity: a rough opening whose glow swells slowly
    cav = Canvas(W, H, ctx.ss)
    rng = np.random.default_rng(ctx.seed("cavity"))
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    rad = H * 0.13 * (1 + 0.18 * rng.standard_normal(40))
    cav.polygon([(cx + r * 1.6 * np.cos(a), cy + r * 0.8 * np.sin(a)) for a, r in zip(ang, rad)], 1.0)
    cm = cav.array()
    g = float(ctx.params.get("glow", 0.6)) * (0.55 + 0.45 * u)
    d = np.sqrt(((x - cx) / 1.6) ** 2 + (y - cy) ** 2) / H
    light = g * (0.9 * np.exp(-d / 0.12) + 0.35 * np.exp(-d / 0.35))
    # a few thin, angular cracks leaking light upward
    cr = Canvas(W, H, ctx.ss)
    for k in range(4):
        px, py = cx + rng.uniform(-0.12, 0.12) * W, cy - H * 0.09
        pts = [(px, py)]
        for _ in range(5):
            px += rng.uniform(-0.03, 0.03) * W
            py -= rng.uniform(0.05, 0.1) * H
            pts.append((px, py))
        cr.line(pts, 1.0, 0.9 * sc)
    crack = cr.array()
    img = (tex * (0.02 + light * 0.9))[..., None] * gold
    img = img + gold * (crack * g * 0.5)[..., None] + gold * (blur(crack, 10 * sc) * g * 0.35)[..., None]
    # the hoard: a heap of coins in the hollow, each catching the light
    coins = Canvas(W, H, ctx.ss)
    shine = Canvas(W, H, ctx.ss)
    for _ in range(260):
        r = rng.uniform(0.006, 0.014) * H
        ang = rng.uniform(0, 2 * np.pi)
        rr = np.sqrt(rng.uniform(0, 1))
        px, py = cx + np.cos(ang) * rr * H * 0.19, cy + np.sin(ang) * rr * H * 0.09
        coins.d.ellipse([(px - r) * coins.ss, (py - r * 0.45) * coins.ss, (px + r) * coins.ss, (py + r * 0.45) * coins.ss],
                        fill=int(255 * rng.uniform(0.5, 1.0)))
        if rng.random() < 0.3:
            shine.dot(px - r * 0.3, py - r * 0.15, r * 0.25, 1.0)
    co, shn = coins.array() * cm, shine.array() * cm
    img = img * (1 - cm[..., None]) + gold * 0.25 * g * cm[..., None]
    img = img + gold * (co * (0.55 + 0.6 * g))[..., None] + np.array([1.0, 0.95, 0.8]) * (shn * g)[..., None]
    # gold dust rising
    dust = Canvas(W, H, ctx.ss)
    for _ in range(220):
        x0 = cx + rng.normal(0, 0.12) * W
        sp = rng.uniform(0.03, 0.09) * H
        ph = rng.uniform(0, 1)
        yy = cy - ((ctx.t * sp + ph * H * 0.6) % (H * 0.6))
        dust.dot(x0 + np.sin(ctx.t + ph * 6) * 8 * sc, yy, rng.uniform(0.6, 2.0) * sc, rng.uniform(0.3, 1.0))
    dm = dust.array()
    img = img + gold * (dm * (0.3 + 0.7 * g))[..., None]
    return np.clip(img + 0.3 * blur(img, 40 * sc), 0, 1)
