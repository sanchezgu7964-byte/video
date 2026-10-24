"""20 青銅砝碼與薩迦 — two shots: a heavy bronze weight standing in the dark,
then a long saga scroll covered in rune-like marks.
"""
import numpy as np

from . import segment
from ..draw import CanvasRGB, pal
from ..fx import Canvas, blur, ease
from ..geom import perlin


def shot_weight(ctx):
    """A squat bronze cylinder weight with a ring handle, lit from a high window."""
    W, H, sc = ctx.W, ctx.H, ctx.scale
    bronze, bg = pal(ctx, "bronze", "#8c5a2b"), pal(ctx, "bg", "#120c08")
    z = 1 + 0.06 * ease(ctx.shot_u)
    cx, base = W * 0.5, H * 0.78
    R, hgt, e = H * 0.26 * z, H * 0.26 * z, 0.32                  # radius, height, ellipse squash (view from above)
    top = base - hgt
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    side = Canvas(W, H, ctx.ss)
    side.polygon([(cx - R, top), (cx + R, top), (cx + R, base), (cx - R, base)], 1.0)
    side.d.ellipse([(cx - R) * side.ss, (base - R * e) * side.ss, (cx + R) * side.ss, (base + R * e) * side.ss], fill=255)
    lid = Canvas(W, H, ctx.ss)
    lid.d.ellipse([(cx - R) * lid.ss, (top - R * e) * lid.ss, (cx + R) * lid.ss, (top + R * e) * lid.ss], fill=255)
    ring = Canvas(W, H, ctx.ss)
    rr = R * 0.32
    ring.d.ellipse([(cx - rr) * ring.ss, (top - R * e * 0.2 - rr * 1.25) * ring.ss, (cx + rr) * ring.ss,
                    (top - R * e * 0.2 + rr * 0.15) * ring.ss], outline=255, width=int(R * 0.09 * ring.ss))
    ms, ml, mr = side.array(), lid.array(), ring.array()
    r = np.clip((x - cx) / R, -1, 1)
    n = np.sqrt(np.clip(1 - r * r, 0, 1))
    shade_side = 0.1 + 0.6 * np.clip(-0.55 * r + 0.55 * n, 0, 1) ** 1.6 + 0.9 * np.exp(-((r + 0.45) / 0.06) ** 2)
    P = perlin(ctx.tl.seed + 201)
    patina = 0.82 + 0.18 * P.fbm2(x / (W * 0.02), y / (H * 0.02), 3)
    green = np.clip(P.noise2(x / (W * 0.05), y / (H * 0.05)), 0, 1)[..., None] * np.array([-0.06, 0.06, 0.03])
    side_c = bronze * (shade_side * patina)[..., None] + green
    lid_c = bronze * (0.95 + 0.4 * np.exp(-(((x - cx) / R + 0.3) ** 2 + ((y - top) / (R * e) + 0.2) ** 2) / 0.15))[..., None] * patina[..., None]
    ring_c = bronze * (0.5 + 0.9 * np.clip(-(x - cx) / rr, 0, 1))[..., None]
    floor = (y > base).astype(np.float32)[..., None]
    pool = np.exp(-((x - cx) / (W * 0.3)) ** 2 - ((y - base) / (H * 0.15)) ** 2)[..., None]
    img = bg * (0.6 + 0.8 * pool) * (1 - floor) + (bg * 1.3 + bronze * 0.1 * pool) * floor
    img = img * (1 - 0.6 * blur(ms, 30 * sc)[..., None] * floor)
    img = img * (1 - ms[..., None]) + side_c * ms[..., None]
    img = img * (1 - ml[..., None]) + lid_c * ml[..., None]
    img = img * (1 - mr[..., None]) + ring_c * mr[..., None]
    return img


RUNE_STROKES = [[(0, 0), (0, 1)], [(0, 0.2), (0.5, 0)], [(0, 0.5), (0.5, 0.3)], [(0, 0.3), (0.5, 0.6)],
                [(0.5, 0), (0.5, 1)], [(0, 0), (0.5, 0.5)], [(0.5, 0.5), (0, 1)], [(0, 0.6), (0.5, 0.8)]]


def shot_saga(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    parch, ink, bg = pal(ctx, "parchment", "#c9b28a"), pal(ctx, "ink", "#2b1d12"), pal(ctx, "bg", "#120c08")
    cv = CanvasRGB(W, H, ctx.ss, bg)
    pan = -ctx.shot_u * W * 0.08
    x0, x1, y0, y1 = -W * 0.1 + pan, W * 1.25 + pan, H * 0.2, H * 0.8
    cv.rect(x0, y0, x1, y1, parch)
    rng = np.random.default_rng(ctx.seed("runes"))
    lh = H * 0.075
    for row in range(7):
        yy = y0 + H * 0.06 + row * lh
        x = x0 + W * 0.08
        while x < x1 - W * 0.08:
            gw = lh * 0.45
            for _ in range(int(rng.integers(2, 4))):
                (ax, ay), (bx, by) = RUNE_STROKES[rng.integers(len(RUNE_STROKES))]
                cv.line([(x + ax * gw, yy + ay * lh * 0.7), (x + bx * gw, yy + by * lh * 0.7)], ink * rng.uniform(0.8, 1.3),
                        max(1.0, 2.2 * sc))
            x += gw * rng.uniform(1.2, 1.6) + (lh * 0.6 if rng.random() < 0.18 else 0)
    # rolled ends
    for ex in (x0, x1):
        cv.rect(ex - W * 0.025, y0 - H * 0.03, ex + W * 0.025, y1 + H * 0.03, parch * 0.7)
    img = cv.array()
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    P = perlin(ctx.tl.seed + 202)
    stain = 0.85 + 0.15 * P.fbm2(x / (W * 0.06), y / (H * 0.06), 4)
    light = 0.55 + 0.6 * np.exp(-((x - W * 0.45) / (W * 0.5)) ** 2 - ((y - H * 0.45) / (H * 0.6)) ** 2)
    return img * (stain * light)[..., None]


@segment("bronze_saga")
def render(ctx):
    img = shot_weight(ctx) if ctx.shot == 0 else shot_saga(ctx)
    return np.clip(img, 0, 1)
