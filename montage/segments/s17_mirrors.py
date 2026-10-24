"""17 忙於反映的鏡子 — mirrors busy reflecting. Each mirror holds an endless
corridor of frames inside frames (a recursive, continuously zooming nest) with a
small figure repeated on every level. The glass wobbles; from shot to shot the
mirrors multiply (1, 1, 2, 4, 6, 9, 16) and the zoom runs faster.
"""
import numpy as np

from . import segment
from ..draw import CanvasRGB, pal
from ..fx import blur


def rect_pts(cx, cy, hw, hh, rot):
    c, s = np.cos(rot), np.sin(rot)
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def figure_pts(cx, base_y, h, rot, ocx, ocy):
    """A plain standing figure (head + tapered body) as two polygons, rotated about (ocx, ocy)."""
    c, s = np.cos(rot), np.sin(rot)
    R = lambda x, y: (ocx + c * (x - ocx) - s * (y - ocy), ocy + s * (x - ocx) + c * (y - ocy))
    w = h * 0.22
    body = [R(cx - w * 0.5, base_y), R(cx - w * 0.35, base_y - h * 0.72), R(cx + w * 0.35, base_y - h * 0.72),
            R(cx + w * 0.5, base_y)]
    t = np.linspace(0, 2 * np.pi, 16)
    head = [R(cx + w * 0.32 * np.cos(a), base_y - h * 0.86 + w * 0.36 * np.sin(a)) for a in t]
    return body, head


def mirror(ctx, w, h, t, phase, speed, wob, depth):
    """Render one mirror's recursive interior into its own (clipped) w x h image."""
    glass, frame, light = pal(ctx, "glass", "#5d6b74"), pal(ctx, "frame", "#2a2c2e"), pal(ctx, "light", "#dfe8ee")
    cv = CanvasRGB(w, h, ctx.ss, glass * 0.5)
    cx, cy = w / 2, h / 2
    s = 0.8
    z = (t * speed + phase) % 1.0
    vx, vy = 0.06 * w * np.sin(t * 0.7 + phase * 5), 0.05 * h * np.cos(t * 0.9 + phase * 3)
    figs = []
    for L in range(depth):
        k = s ** (L - z + 1)                     # level 0 sits just inside the fixed outer frame
        hw, hh = w * 0.5 * k, h * 0.5 * k
        ccx, ccy = cx + vx * (1 - k), cy + vy * (1 - k)
        rot = wob * 0.035 * (L - z + 1) * np.sin(t * 1.6 + phase * 7 + L * 0.45)
        fade = 0.86 ** (L - z + 1)
        b = 0.07
        cv.polygon(rect_pts(ccx, ccy, hw, hh, rot), frame * (0.6 + 0.5 * fade))
        cv.polygon(rect_pts(ccx, ccy, hw * (1 - b), hh * (1 - b * w / h), rot), light * 0.3 * fade)
        cv.polygon(rect_pts(ccx, ccy, hw * (1 - b * 1.4), hh * (1 - b * 1.4 * w / h), rot), glass * 0.62 * fade)
        figs.append((ccx - hw * 0.55, ccy + hh * (1 - b * 1.6), hh * 0.8, rot, ccx, ccy, fade))
    # the reflected figure, innermost first, so the line of reflections stays visible
    for fx, fy, fh, rot, ox, oy, fade in reversed(figs):
        body, head = figure_pts(fx, fy, fh, rot, ox, oy)
        dark = np.array([0.03, 0.033, 0.036]) + glass * 0.12 * (1 - fade)
        cv.polygon(body, dark)
        cv.polygon(head, dark)
    return cv.array()


@segment("mirrors")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    p = ctx.params
    counts = p.get("mirrors_per_shot", [1, 1, 2, 4, 6, 9, 16])
    n = int(counts[min(ctx.shot, len(counts) - 1)])
    cols = int(np.ceil(np.sqrt(n * 16 / 9 * 0.62))) if n > 1 else 1
    cols = {1: 1, 2: 2, 4: 2, 6: 3, 9: 3, 16: 4}.get(n, cols)
    rows = int(np.ceil(n / cols))
    wall = np.array([0.025, 0.027, 0.03])
    cv = CanvasRGB(W, H, ctx.ss, wall)
    gap = 0.035 * H
    cw = (W - gap * (cols + 1)) / cols
    ch = (H - gap * (rows + 1)) / rows
    # mirrors are portrait-ish when alone, then fill whatever cell they get
    if n == 1:
        cw, ch = H * 0.62, H * 0.86
    speed = 0.25 + 0.22 * ctx.shot
    wob = float(p.get("wobble", 1.0)) * (0.6 + 0.35 * ctx.shot)
    rng = np.random.default_rng(ctx.seed("phases", ctx.shot))
    phases = rng.uniform(0, 1, n)
    k = 0
    boxes = []
    for r in range(rows):
        for c in range(cols):
            if k >= n:
                break
            if n == 1:
                x0, y0 = (W - cw) / 2, (H - ch) / 2
            else:
                x0, y0 = gap + c * (cw + gap), gap + r * (ch + gap)
            cv.rect(x0, y0, x0 + cw, y0 + ch, pal(ctx, "frame", "#2a2c2e") * 1.3)    # fixed outer frame
            boxes.append((x0, y0))
            k += 1
    img = cv.array()
    border = max(2, int(0.035 * min(cw, ch)))
    for k, (x0, y0) in enumerate(boxes):
        xi, yi = int(round(x0)) + border, int(round(y0)) + border
        w, h = int(cw) - 2 * border, int(ch) - 2 * border
        inner = mirror(ctx, w, h, ctx.t, phases[k], speed, wob, int(p.get("depth", 14)))
        img[yi:yi + h, xi:xi + w] = inner[:H - yi, :W - xi]
    # a slow sheen sliding across the glass
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    pos = (ctx.shot_u * 1.6 - 0.3) * (W + H)
    sheen = np.exp(-(((x + 0.6 * y) - pos) / (0.08 * W)) ** 2) * 0.08
    img = img + sheen[..., None] * (img.mean(-1, keepdims=True) > 0.05)
    return np.clip(img + 0.1 * blur(img, 18 * sc), 0, 1)
