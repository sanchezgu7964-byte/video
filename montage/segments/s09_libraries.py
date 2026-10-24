"""09 東方和西方的圖書館 — ten accelerating flashes of libraries.

West: upright spines with gilt bands. East: flat stacks of thread-bound books
and the round ends of scrolls in pigeonholes. Three layouts alternate:
a shelf seen straight on, a wall-grid of cells, and an endless shelved
corridor in one-point perspective.
"""
import numpy as np

from . import segment
from ..draw import CanvasRGB, pal
from ..fx import blur

SEQUENCE = [("spines", "west"), ("cells", "east"), ("corridor", "west"), ("spines", "east"),
            ("cells", "west"), ("corridor", "east"), ("spines", "west"), ("cells", "east"),
            ("corridor", "west"), ("corridor", "east")]


def book_colors(ctx, side, rng, n):
    if side == "west":
        base = [pal(ctx, "west", "#3a4a3a"), np.array([0.42, 0.12, 0.1]), np.array([0.3, 0.2, 0.12]),
                np.array([0.12, 0.14, 0.25]), np.array([0.5, 0.36, 0.2])]
    else:
        base = [pal(ctx, "east", "#8c2f1e"), pal(ctx, "paper", "#d8cfb8") * 0.9, np.array([0.55, 0.45, 0.25]),
                np.array([0.2, 0.25, 0.35]), np.array([0.65, 0.58, 0.45])]
    idx = rng.integers(0, len(base), n)
    return [base[i] * rng.uniform(0.7, 1.15) for i in idx]


def fill_cell(cv, ctx, side, x0, y0, x1, y1, rng, scale=1.0):
    """Fill a shelf compartment with books (west) or stacks + scrolls (east)."""
    gold = pal(ctx, "paper", "#d8cfb8") * 0.85
    if side == "west":
        x = x0
        cols = book_colors(ctx, side, rng, 400)
        k = 0
        while x < x1:
            w = rng.uniform(0.010, 0.032) * ctx.H * scale
            h = (y1 - y0) * rng.uniform(0.62, 0.97)
            col = cols[k % len(cols)]
            k += 1
            cv.rect(x, y1 - h, min(x + w, x1), y1, col)
            for b in (0.12, 0.2, 0.82):
                yy = y1 - h + h * b
                cv.line([(x + w * 0.12, yy), (min(x + w * 0.88, x1), yy)], gold * rng.uniform(0.5, 1), 1.0 * scale)
            x += w + rng.uniform(0, 0.002) * ctx.H
    else:
        cols = book_colors(ctx, side, rng, 400)
        x = x0
        k = 0
        while x < x1 - 4:
            w = min((x1 - x0) * rng.uniform(0.25, 0.45), x1 - x)
            if rng.random() < 0.4:                       # scrolls: a pile of circles
                r = (y1 - y0) * rng.uniform(0.09, 0.14)
                for row in range(int((y1 - y0) / (2 * r))):
                    for c in range(int(w / (2 * r))):
                        cx, cy = x + r + c * 2 * r, y1 - r - row * 2 * r * 0.95
                        cv.ellipse(cx, cy, r * 0.95, r * 0.95, cols[k % len(cols)])
                        cv.ellipse(cx, cy, r * 0.35, r * 0.35, cols[k % len(cols)] * 0.4)
                        k += 1
            else:                                        # stack of flat thread-bound books
                y = y1
                while y > y0 + (y1 - y0) * rng.uniform(0.1, 0.4):
                    t = rng.uniform(0.04, 0.09) * (y1 - y0)
                    inset = rng.uniform(0, 0.08) * w
                    cv.rect(x + inset, y - t, x + w - inset * 0.5, y, cols[k % len(cols)])
                    cv.line([(x + inset + w * 0.1, y - t * 0.5), (x + w * 0.95 - inset, y - t * 0.5)],
                            cols[k % len(cols)] * 0.55, 0.8 * scale)
                    k += 1
                    y -= t + 1
            x += w + rng.uniform(0.004, 0.015) * ctx.H


def shot_spines(ctx, side, rng):
    W, H = ctx.W, ctx.H
    wood = np.array([0.16, 0.1, 0.06]) if side == "west" else np.array([0.25, 0.08, 0.05])
    cv = CanvasRGB(W, H, ctx.ss, pal(ctx, "dark", "#0d0b09"))
    pan = -ctx.shot_u * 0.08 * W
    for k, (y0, y1) in enumerate(((0.06, 0.47), (0.53, 0.94))):
        cv.rect(0, y1 * H, W, (y1 + 0.04) * H, wood)
        fill_cell(cv, ctx, side, pan - 0.05 * W, y0 * H, W * 1.12 + pan, y1 * H, rng, 1.6)
    img = cv.array()
    y = np.linspace(0, 1, H)[:, None, None]
    return img * (0.55 + 0.6 * np.exp(-((y - 0.45) / 0.5) ** 2))


def shot_cells(ctx, side, rng):
    W, H = ctx.W, ctx.H
    wood = np.array([0.2, 0.13, 0.08]) if side == "west" else np.array([0.32, 0.1, 0.06])
    cv = CanvasRGB(W, H, ctx.ss, wood * 0.6)
    z = 1.0 + 0.1 * ctx.shot_u
    nx, ny = 9, 5
    cw, ch = W * z / nx, H * z / ny
    ox, oy = W / 2 - nx * cw / 2, H / 2 - ny * ch / 2
    for i in range(nx):
        for j in range(ny):
            x0, y0 = ox + i * cw + cw * 0.06, oy + j * ch + ch * 0.08
            x1, y1 = ox + (i + 1) * cw - cw * 0.06, oy + (j + 1) * ch - ch * 0.04
            cv.rect(x0, y0, x1, y1, pal(ctx, "dark", "#0d0b09"))
            fill_cell(cv, ctx, side, x0, y0 + ch * 0.05, x1, y1, rng, 0.8)
    return cv.array()


def shot_corridor(ctx, side, rng):
    """Two walls of shelves receding to one vanishing point; darkness at the end."""
    W, H = ctx.W, ctx.H
    cv = CanvasRGB(W, H, ctx.ss, (0, 0, 0))
    vx, vy = W / 2, H * 0.48
    f = H * 0.9
    move = ctx.shot_u * 1.4                                  # walking forward (world units)
    half_w, h_floor, h_ceil = 1.2, -1.1, 1.6
    proj = lambda x, y, z: (vx + f * x / z, vy - f * y / z)
    wood = np.array([0.22, 0.14, 0.08]) if side == "west" else np.array([0.34, 0.12, 0.07])
    cols = book_colors(ctx, side, rng, 2000)
    zs = np.arange(0.5, 60, 0.11) - (move % 0.11)
    shelves = np.linspace(h_floor + 0.1, h_ceil - 0.1, 7)
    k = 0
    for z0 in zs[::-1]:                                          # far to near
        z1 = z0 + 0.1
        fog = np.exp(-z0 / 9.0)
        for sx in (-1, 1):
            for a, b in zip(shelves[:-1], shelves[1:]):
                hb = a + (b - a) * rng.uniform(0.6, 0.92)
                if side == "east" and (k % 3 == 0):
                    hb = a + (b - a) * 0.35
                p = [proj(sx * half_w, a, z0), proj(sx * half_w, a, z1),
                     proj(sx * half_w, hb, z1), proj(sx * half_w, hb, z0)]
                cv.polygon(p, cols[k % len(cols)] * fog)
                k += 1
    # shelf boards, floor and ceiling lines on top
    for sx in (-1, 1):
        for a in shelves:
            cv.line([proj(sx * half_w, a, 0.4), proj(sx * half_w, a, 60)], wood * 1.4, 2.0)
    cv.line([proj(-half_w, h_floor, 0.4), proj(-half_w, h_floor, 60)], wood, 3)
    img = cv.array()
    # the end of the corridor glows faintly, like another gallery beyond
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((x - vx) ** 2 + (y - vy) ** 2) / H
    lamp = pal(ctx, "paper", "#d8cfb8") * 0.35
    return img + lamp * np.exp(-d / 0.05)[..., None]


@segment("libraries")
def render(ctx):
    kind, side = SEQUENCE[ctx.shot % len(SEQUENCE)]
    rng = np.random.default_rng(ctx.seed("shot", ctx.shot))       # same books on every frame of a shot
    img = {"spines": shot_spines, "cells": shot_cells, "corridor": shot_corridor}[kind](ctx, side, rng)
    return np.clip(img + 0.08 * blur(img, 20 * ctx.scale), 0, 1)
