"""21 代數和火焰 — algebra and fire, six alternating flashes: glowing formula
lines and plotted curves, then procedural flame; each shot keeps a ghost of the other.
"""
import numpy as np
from PIL import Image, ImageDraw

from . import segment
from PIL import ImageFont
from ..fx import Canvas, blur, resize
from ..geom import perlin

FORMULAS = [["eⁱᵖ + 1 = 0".replace("ᵖ", "π"), "x² + y² = r²", "∑ 1/n² = π²/6"],
            ["f(x) = ∫ g(t) dt", "√2 ∉ ℚ", "a·b = |a||b| cos θ"],
            ["φ = (1 + √5) / 2", "lim (1 + 1/n)ⁿ = e", "z ↦ z² + c"]]


def formula_layer(ctx, k):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(64 * sc))
    lines = FORMULAS[k % len(FORMULAS)]
    drift = ctx.shot_u * 40 * sc
    for i, s in enumerate(lines):
        d.text((W * (0.12 + 0.22 * i), H * (0.22 + 0.24 * i) - drift), s, fill=255, font=f)
    lay = np.asarray(img, np.float32) / 255.0
    cv = Canvas(W, H, ctx.ss)
    x = np.linspace(0, W, 200)
    ph = ctx.t * 2 + k
    cv.line(list(zip(x, H * 0.6 + H * 0.12 * np.sin(x / W * 9 + ph))), 0.8, 1.6 * sc)
    cv.line(list(zip(x, H * 0.8 - H * 0.5 * ((x / W - 0.5) * 1.6) ** 2)), 0.6, 1.2 * sc)
    th = np.linspace(0, 6 * np.pi, 300)
    cv.line(list(zip(W * 0.8 + th * H * 0.012 * np.cos(th + ph), H * 0.35 + th * H * 0.012 * np.sin(th + ph))), 0.5, 1.0 * sc)
    for gx in np.linspace(0, W, 13):
        cv.line([(gx, 0), (gx, H)], 0.08, 1.0)
    lay = np.maximum(lay, cv.array())
    return lay + 0.8 * blur(lay, 10 * sc) + 0.4 * blur(lay, 40 * sc)


def fire_layer(ctx, k):
    W, H = ctx.W, ctx.H
    gw, gh = W // 4, H // 4
    y, x = np.mgrid[0:gh, 0:gw].astype(np.float32) / gh
    P = perlin(ctx.tl.seed + 211 + k)
    t = ctx.t
    warp = P.fbm2(x * 3, y * 3 + t * 2.5, 3) * 0.25
    n = 0.5 + 0.5 * P.fbm2(x * 4 + warp, y * 3 + t * 3.2, 5)
    rise = np.clip(y * 1.25 - 0.2, 0, 1)                 # hotter at the bottom
    f = np.clip(n * 1.6 * rise - 0.35 + 0.4 * rise, 0, 1)
    return resize(f, W, H)


def fire_colour(f):
    stops = np.array([0.0, 0.3, 0.55, 0.8, 1.0])
    cols = np.array([[0, 0, 0], [0.45, 0.05, 0.02], [0.95, 0.35, 0.05], [1.0, 0.75, 0.3], [1.0, 0.97, 0.85]])
    return np.stack([np.interp(f, stops, cols[:, c]) for c in range(3)], -1)


@segment("algebra_fire")
def render(ctx):
    k = ctx.shot
    form = formula_layer(ctx, k // 2)
    fire = fire_layer(ctx, k)
    algebra_col = np.array([0.65, 0.85, 1.0], np.float32)
    if k % 2 == 0:
        img = form[..., None] * algebra_col + 0.25 * fire_colour(fire * 0.8)
    else:
        img = fire_colour(fire) + 0.22 * form[..., None] * algebra_col
    return np.clip(img, 0, 1)
