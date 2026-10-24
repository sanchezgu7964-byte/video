"""07 星星 — three shots, three skies:
  1. a dense field with a Milky Way band, slow push in
  2. circular star trails around an off-frame pole (long exposure building up)
  3. a sparse sky where a constellation draws itself
"""
from functools import lru_cache

import numpy as np

from . import segment
from ..draw import draw_stars, make_stars
from ..fx import Canvas, blur
from ..geom import perlin


@lru_cache(maxsize=2)
def field(seed, aspect):
    rng = np.random.default_rng([seed, 701])
    P = perlin(seed + 71)

    def band(x, y):                      # diagonal band of higher density, ragged by noise
        d = (y - 0.35 - (x / aspect) * 0.4)
        return 0.15 + 0.85 * np.exp(-(d / 0.14) ** 2) * (0.6 + 0.4 * P.noise2(x * 4, y * 4))

    pts, mag = make_stars(rng, 4200, aspect, band)
    return pts, mag * 0.85


def shot_field(ctx):
    W, H = ctx.W, ctx.H
    aspect = W / H
    pts, mag = field(ctx.tl.seed, round(aspect, 4))
    z = 1.0 + 0.05 * ctx.shot_u
    cx, cy = aspect / 2, 0.5
    xform = lambda p: (np.stack([(p[:, 0] - cx) * z + cx, (p[:, 1] - cy) * z + cy], 1)) * H
    img = draw_stars(W, H, ctx.ss, pts, mag, ctx.t, 0.2, ctx.tl.seed + 1, 0.8, xform)
    # diffuse glow of the band
    x, y = np.meshgrid(np.arange(W) / H, np.arange(H) / H)
    d = (y - 0.35 - (x / aspect) * 0.4)
    neb = 0.07 * np.exp(-(d / 0.11) ** 2) * (0.5 + 0.5 * perlin(ctx.tl.seed + 72).fbm2(x * 3, y * 3, 3))
    return img + np.clip(neb, 0, None)


def shot_trails(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    rng = np.random.default_rng([ctx.tl.seed, 702])
    n = 520
    pole = np.array([W * 0.86, -H * 0.18])
    r = np.sqrt(rng.uniform((0.15 * H) ** 2, (1.35 * W) ** 2, n))
    a0 = rng.uniform(0, 2 * np.pi, n)
    mag = rng.uniform(0, 1, n) ** 4
    sweep = np.radians(2 + 26 * ctx.shot_u)                 # the exposure grows
    cv = Canvas(W, H, ctx.ss)
    for ri, ai, m in zip(r, a0, mag):
        th = np.linspace(ai, ai + sweep, max(3, int(ri * sweep / 6)))
        x, y = pole[0] + ri * np.cos(th), pole[1] + ri * np.sin(th)
        if np.all((x < -10) | (x > W + 10) | (y < -10) | (y > H + 10)):
            continue
        cv.line(list(zip(x, y)), 0.18 + 0.8 * m, (0.6 + 1.4 * m) * sc)
    img = cv.array()
    return img + 0.35 * blur(img, 6 * sc)


@lru_cache(maxsize=2)
def constellation(seed, aspect):
    rng = np.random.default_rng([seed, 703])
    pts, mag = make_stars(rng, 380, aspect)
    # nine bright stars on a loose ring, plus the order in which they get joined
    ang = np.sort(rng.uniform(0, 2 * np.pi, 9))
    rad = rng.uniform(0.18, 0.34, 9)
    big = np.stack([aspect / 2 + rad * np.cos(ang) * 1.3, 0.5 + rad * np.sin(ang)], 1)
    edges = [(i, (i + 1) % 9) for i in range(9)] + [(0, 4), (2, 6)]
    return pts, mag * 0.6, big, edges


def shot_constellation(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    pts, mag, big, edges = constellation(ctx.tl.seed, round(W / H, 4))
    img = draw_stars(W, H, ctx.ss, pts, mag, ctx.t, 0.3, ctx.tl.seed + 3)
    cv = Canvas(W, H, ctx.ss)
    prog = ctx.shot_u * 1.25 * len(edges)
    for k, (i, j) in enumerate(edges):
        f = np.clip(prog - k, 0, 1)
        if f <= 0:
            continue
        a, b = big[i] * H, big[j] * H
        cv.line([tuple(a), tuple(a + (b - a) * f)], 0.32, 0.9 * sc)
    for x, y in big * H:
        cv.dot(x, y, 3.2 * sc, 1.0)
    lines = cv.array()
    return img + lines + 0.8 * blur(lines, 10 * sc)


SHOTS = [shot_field, shot_trails, shot_constellation]


@segment("stars")
def render(ctx):
    return np.clip(SHOTS[ctx.shot % len(SHOTS)](ctx), 0, 1)
