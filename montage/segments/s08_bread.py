"""08 麵包 — five short warm shots of bread built from simple geometry:
round loaf, baguette, a slice, a tray of rolls, and finally the loaf broken open.

Volume comes from "inflating" each flat mask: blur it into a dome, take the
gradient as a surface normal, light it from the upper left.
"""
import numpy as np

from . import segment
from ..draw import over, pal
from ..fx import Canvas, blur, ease
from ..geom import perlin


def ellipse_pts(cx, cy, rx, ry, rot=0.0, n=120, a0=0.0, a1=2 * np.pi):
    t = np.linspace(a0, a1, n)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = np.cos(rot), np.sin(rot)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def mask_of(ctx, polys):
    cv = Canvas(ctx.W, ctx.H, ctx.ss)
    for p in polys:
        cv.polygon([tuple(q) for q in p], 1.0)
    return cv.array()


def inflate(mask, radius, light=(-0.6, -0.8)):
    """Dome shading for a flat mask -> (lit 0..1-ish, dome height)."""
    h = blur(mask, radius, 2)
    h = blur(h * mask, radius * 0.5, 2)
    gy, gx = np.gradient(h)
    k = radius * 1.2
    lit = 0.55 + k * (gx * -light[0] + gy * -light[1])
    return np.clip(lit, 0.1, 1.4), h


def crust(ctx, mask, radius, base, texture_scale=40.0):
    lit, h = inflate(mask, radius)
    H, W = mask.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float32) / H
    P = perlin(ctx.tl.seed + 81 + ctx.shot)
    speck = 0.85 + 0.15 * P.fbm2(x * texture_scale, y * texture_scale, 2)
    browned = np.clip(h * 1.6, 0, 1)                    # tops brown more than the edges
    col = base * (0.75 + 0.35 * browned[..., None]) * (lit * speck)[..., None]
    return col, lit


def backdrop(ctx):
    H, W = ctx.H, ctx.W
    bg, light = pal(ctx, "bg", "#1a0f08"), pal(ctx, "light", "#ffcf80")
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((x - W * 0.45) ** 2 + (y - H * 0.4) ** 2) / H
    return bg * (0.6 + 0.8 * np.exp(-d / 0.45))[..., None] + light * (0.06 * np.exp(-d / 0.3))[..., None]


def shot_boule(ctx):
    W, H = ctx.W, ctx.H
    img = backdrop(ctx)
    cx, cy, r = W / 2, H * 0.52, H * 0.33
    m = mask_of(ctx, [ellipse_pts(cx, cy, r, r * 0.94)])
    shadow = blur(mask_of(ctx, [ellipse_pts(cx + r * 0.08, cy + r * 0.12, r, r * 0.94)]), 30 * ctx.scale)
    img = img * (1 - 0.75 * shadow[..., None])
    col, _ = crust(ctx, m, r * 0.5, pal(ctx, "crust", "#b5651d"))
    img = over(img, m, col)
    # square score: four ears of paler crumb
    cv = Canvas(W, H, ctx.ss)
    rot = 0.05 + 0.15 * ctx.shot_u
    for k in range(4):
        a = rot + k * np.pi / 2
        c, s = np.cos(a), np.sin(a)
        p0 = (cx + c * r * 0.42 - s * r * 0.5, cy + s * r * 0.42 + c * r * 0.5)
        p1 = (cx + c * r * 0.42 + s * r * 0.5, cy + s * r * 0.42 - c * r * 0.5)
        cv.line([p0, p1], 1.0, r * 0.06)
    score = cv.array() * m
    img = over(img, score * 0.85, pal(ctx, "crumb", "#f3d9a4") * 0.95)
    return img


def shot_baguette(ctx):
    W, H = ctx.W, ctx.H
    img = backdrop(ctx)
    rot = np.radians(-14 + 4 * ctx.shot_u)
    cx, cy = W * (0.5 - 0.03 * ctx.shot_u), H * 0.52
    body = ellipse_pts(cx, cy, W * 0.42, H * 0.12, rot)
    m = mask_of(ctx, [body])
    img = img * (1 - 0.7 * blur(mask_of(ctx, [body + [0, H * 0.06]]), 25 * ctx.scale)[..., None])
    col, _ = crust(ctx, m, H * 0.09, pal(ctx, "crust", "#b5651d") * 1.1, 70)
    img = over(img, m, col)
    cv = Canvas(W, H, ctx.ss)
    c, s = np.cos(rot), np.sin(rot)
    for k in range(5):
        t = -0.62 + k * 0.31
        px, py = cx + c * W * 0.42 * t, cy + s * W * 0.42 * t
        d = np.array([np.cos(rot + 0.5), np.sin(rot + 0.5)]) * H * 0.075
        cv.line([(px - d[0] * 1.6, py - d[1] * 0.9), (px + d[0] * 1.6, py + d[1] * 0.9)], 1.0, H * 0.022)
    img = over(img, cv.array() * m * 0.9, pal(ctx, "crumb", "#f3d9a4"))
    return img


def shot_slice(ctx):
    W, H = ctx.W, ctx.H
    img = backdrop(ctx)
    z = 1 + 0.06 * ctx.shot_u
    cx, cy, w, h = W / 2, H * 0.55, H * 0.42 * z, H * 0.36 * z
    top = ellipse_pts(cx, cy - h * 0.15, w, h * 0.75, 0, 60, np.pi, 2 * np.pi)
    shape = np.vstack([top, [[cx + w, cy + h * 0.65], [cx - w, cy + h * 0.65]]])
    m = mask_of(ctx, [shape])
    inner = mask_of(ctx, [np.array([cx, cy]) + (shape - [cx, cy]) * [0.9, 0.86] + [0, h * 0.06]])
    col, lit = crust(ctx, m, H * 0.05, pal(ctx, "crust", "#b5651d") * 0.9)
    img = over(img, m, col)
    # crumb with holes
    rng = np.random.default_rng([ctx.tl.seed, 803])
    cv = Canvas(W, H, ctx.ss)
    for _ in range(260):
        hx, hy = rng.uniform(cx - w, cx + w), rng.uniform(cy - h * 0.9, cy + h * 0.65)
        rx, ry = rng.uniform(2, 14) * ctx.scale * z, rng.uniform(2, 10) * ctx.scale * z
        cv.d.ellipse([(hx - rx) * cv.ss, (hy - ry) * cv.ss, (hx + rx) * cv.ss, (hy + ry) * cv.ss], fill=255)
    holes = cv.array() * inner
    crumb = pal(ctx, "crumb", "#f3d9a4") * (0.85 + 0.2 * lit[..., None])
    img = over(img, inner, crumb)
    img = img * (1 - 0.55 * holes[..., None])
    return img


def shot_rolls(ctx):
    W, H = ctx.W, ctx.H
    img = backdrop(ctx) * 0.8
    r = H * 0.13
    polys, cuts = [], Canvas(W, H, ctx.ss)
    pan = -0.05 * W * ctx.shot_u
    for i in range(4):
        for j in range(2):
            cx, cy = W * (0.2 + i * 0.2) + pan + (j * W * 0.1), H * (0.32 + j * 0.36)
            polys.append(ellipse_pts(cx, cy, r, r * 0.9, 0.3 * i))
            cuts.line([(cx - r * 0.5, cy + r * 0.2), (cx + r * 0.5, cy - r * 0.2)], 1.0, r * 0.08)
    m = mask_of(ctx, polys)
    sh = blur(mask_of(ctx, [p + [r * 0.15, r * 0.2] for p in polys]), 18 * ctx.scale)
    img = img * (1 - 0.7 * sh[..., None])
    col, _ = crust(ctx, m, r * 0.5, pal(ctx, "crust", "#b5651d") * 1.05)
    img = over(img, m, col)
    img = over(img, cuts.array() * m * 0.8, pal(ctx, "crumb", "#f3d9a4"))
    return img


def shot_break(ctx):
    """The loaf is torn in two: halves pull apart, crumb faces exposed, crumbs fall."""
    W, H = ctx.W, ctx.H
    img = backdrop(ctx)
    u = ease(min(1.0, ctx.shot_u * 1.25))
    cx, cy, rx, ry = W / 2, H * 0.5, H * 0.5, H * 0.3
    rng = np.random.default_rng([ctx.tl.seed, 804])
    ys = np.linspace(cy - ry * 1.05, cy + ry * 1.05, 26)
    jag = cx + rng.normal(0, H * 0.018, len(ys)) + np.sin(np.linspace(0, 3, len(ys))) * H * 0.02
    gap = u * W * 0.13
    crust_c, crumb_c = pal(ctx, "crust", "#b5651d"), pal(ctx, "crumb", "#f3d9a4")
    for side in (-1, 1):
        shift = np.array([side * gap / 2, -u * H * 0.02])
        ang = side * u * 0.12
        arc = ellipse_pts(cx, cy, rx, ry, 0, 80, np.pi / 2, 3 * np.pi / 2) if side < 0 else \
            ellipse_pts(cx, cy, rx, ry, 0, 80, -np.pi / 2, np.pi / 2)
        # keep the arc on this side of the tear, then close along the jagged edge
        arc = arc[(arc[:, 0] - cx) * side >= 0]
        edge = np.stack([jag, ys], 1)              # top -> bottom
        if side > 0:                                 # right arc ends at the bottom: go back up
            edge = edge[::-1]
        poly = np.vstack([arc, edge])
        c, s = np.cos(ang), np.sin(ang)
        pivot = np.array([cx, cy])
        poly = (poly - pivot) @ np.array([[c, s], [-s, c]]) + pivot + shift
        m = mask_of(ctx, [poly])
        col, lit = crust(ctx, m, H * 0.12, crust_c)
        img = img * (1 - 0.6 * blur(m, 30 * ctx.scale)[..., None] * 0.6)
        img = over(img, m, col)
        # exposed crumb: a band along the tear, widening as it opens
        e2 = (np.stack([jag, ys], 1) - pivot) @ np.array([[c, s], [-s, c]]) + pivot + shift
        band = np.vstack([e2, (e2 + [side * (H * 0.03 + gap * 0.25), 0])[::-1]])
        cm = mask_of(ctx, [band]) * m * min(1.0, u * 3)
        P = perlin(ctx.tl.seed + 85)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / H
        pores = 0.8 + 0.25 * P.noise2(xx * 90, yy * 90)
        img = over(img, cm, crumb_c * pores[..., None])
    # falling crumbs
    cv = Canvas(W, H, ctx.ss)
    for k in range(40):
        t0 = rng.uniform(0.1, 0.6)
        tt = ctx.shot_u - t0
        if tt <= 0:
            continue
        x0 = cx + rng.normal(0, H * 0.02)
        y0 = cy + rng.uniform(-ry, ry) * 0.8
        vx = rng.normal(0, 0.15) * H
        cv.dot(x0 + vx * tt, y0 + 2.2 * H * tt * tt, rng.uniform(1.5, 4.5) * ctx.scale, 1.0)
    img = over(img, cv.array(), crumb_c * 0.9)
    return img


SHOTS = [shot_boule, shot_baguette, shot_slice, shot_rolls, shot_break]


@segment("bread")
def render(ctx):
    return np.clip(SHOTS[ctx.shot % len(SHOTS)](ctx), 0, 1)
