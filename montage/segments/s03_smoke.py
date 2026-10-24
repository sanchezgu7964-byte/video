"""03 黑煙升上夜空 — black smoke rising into a night sky, slowly covering the stars.

Smoke = particles advected by curl noise (derived from 3D Perlin noise), splatted
into a density field, blurred, and textured with more noise for wisps.
"""
from functools import lru_cache

import numpy as np

from . import segment
from ..draw import draw_stars, make_stars
from ..fx import Canvas, blur, resize
from ..geom import perlin

PREROLL = 1.8          # seconds simulated before the shot starts, so smoke already exists


@lru_cache(maxsize=2)
def simulate(seed, n, n_frames, fps, rise, aspect):
    """Positions (n_frames, n, 2) in normalised coords (x 0..aspect, y 0 = bottom) and ages."""
    rng = np.random.default_rng([seed, 303])
    P = perlin(seed + 31)
    sources = np.array([[aspect * 0.58, 0.16], [aspect * 0.34, 0.12], [aspect * 0.8, 0.1]])
    weight = np.array([0.55, 0.3, 0.15])
    src = rng.choice(3, n, p=weight)
    dur = n_frames / fps
    # emission accelerates through the shot: more and more smoke
    birth = -PREROLL + (dur + PREROLL) * rng.uniform(0, 1, n) ** 0.85
    pos = sources[src] + rng.normal(0, 0.008, (n, 2))
    jitter = rng.uniform(0.7, 1.3, n)
    dt = 1.0 / fps
    steps = int(round((dur + PREROLL) * fps))
    out_pos = np.zeros((n_frames, n, 2), np.float32)
    out_age = np.zeros((n_frames, n), np.float32)
    t = -PREROLL
    e = 0.01
    for k in range(steps):
        alive = birth <= t
        x, y = pos[alive, 0], pos[alive, 1]
        f = 2.2
        nz = lambda a, b: P.noise3(a * f, b * f, t * 0.18)
        # curl of a scalar noise field: divergence-free swirling
        cx = (nz(x, y + e) - nz(x, y - e)) / (2 * e)
        cy = -(nz(x + e, y) - nz(x - e, y)) / (2 * e)
        age = t - birth[alive]
        spread = np.clip(age / 2.5, 0.15, 1.0)
        vx = 0.07 * cx * spread + 0.015
        vy = 0.06 * cy * spread + rise * jitter[alive] * (1.4 - 0.4 * spread)
        pos[alive, 0] += vx * dt
        pos[alive, 1] += vy * dt
        # unborn particles ride along with their source
        t += dt
        fi = k - int(round(PREROLL * fps))
        if 0 <= fi < n_frames:
            out_pos[fi] = pos
            out_age[fi] = np.where(birth <= t, t - birth, -1)
    return out_pos, out_age


@lru_cache(maxsize=2)
def stars(seed, aspect, n):
    return make_stars(np.random.default_rng([seed, 304]), n, aspect)


@lru_cache(maxsize=2)
def skyline(seed, aspect):
    """Factory silhouette along the bottom: blocks, sheds with saw-tooth roofs, chimneys."""
    rng = np.random.default_rng([seed, 305])
    polys = []
    x = -0.05
    while x < aspect + 0.05:
        w = rng.uniform(0.05, 0.2)
        h = rng.uniform(0.03, 0.09)
        if rng.random() < 0.35:                       # saw-tooth shed roof
            n = int(w / 0.025) + 1
            pts = [(x, 0)]
            for i in range(n):
                xa = x + i * w / n
                pts += [(xa, h), (xa + w / n * 0.8, h + 0.02)]
            pts += [(x + w, h), (x + w, 0)]
            polys.append(pts)
        else:
            polys.append([(x, 0), (x, h), (x + w, h), (x + w, 0)])
        x += w + rng.uniform(0, 0.02)
    for cx, h in ((aspect * 0.58, 0.16), (aspect * 0.34, 0.12), (aspect * 0.8, 0.1)):
        polys.append([(cx - 0.012, 0), (cx - 0.009, h), (cx + 0.009, h), (cx + 0.012, 0)])
    return polys


@segment("smoke")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    p = ctx.params
    aspect = W / H
    y = (np.arange(H, dtype=np.float32) + 0.5) / H
    # sky: deep at the top, a faint city glow near the horizon
    sky = (0.05 + 0.3 * np.exp(-(1 - y) / 0.22))[:, None] * np.ones((1, W), np.float32)
    # a hazy moon for the smoke to drift across
    mx, my = aspect * 0.66, 0.3
    xx = (np.arange(W, dtype=np.float32) + 0.5) / H
    d = np.sqrt((xx[None, :] - mx) ** 2 + (y[:, None] - my) ** 2)
    maria = 0.88 + 0.12 * perlin(ctx.tl.seed + 33).noise2(xx[None, :] * 60, y[:, None] * 60)
    sky = sky + 0.95 * maria * np.clip((0.042 - d) / 0.003, 0, 1) + 0.32 * np.exp(-d / 0.07) + 0.12 * np.exp(-d / 0.3)

    pts, mag = stars(ctx.tl.seed, aspect, int(p.get("stars", 900)))
    pts = pts * np.array([1, 0.85])
    sky = sky + draw_stars(W, H, ctx.ss, pts, mag, ctx.t, 0.3, ctx.tl.seed)

    pos, age = simulate(ctx.tl.seed, int(p.get("particles", 6000)), ctx.seg.n_frames, ctx.fps,
                        float(p.get("rise_speed", 0.06)), round(aspect, 4))
    P_, A_ = pos[ctx.frame], age[ctx.frame]
    live = A_ >= 0
    gw, gh = W // 4, H // 4
    dens = []
    for lo, hi in ((0, 1.2), (1.2, 99)):              # young/small and old/large puffs
        m = live & (A_ >= lo) & (A_ < hi)
        g = np.zeros((gh, gw), np.float32)
        xi = np.clip((P_[m, 0] / aspect * gw).astype(int), 0, gw - 1)
        yi = np.clip(((1 - P_[m, 1]) * gh).astype(int), 0, gh - 1)
        inb = (P_[m, 1] < 1.15) & (P_[m, 0] > -0.1) & (P_[m, 0] < aspect + 0.1)
        np.add.at(g, (yi[inb], xi[inb]), 1.0)
        dens.append(g)
    D = blur(dens[0], 5 * sc, 1) * 1.3 + blur(dens[1], 12 * sc, 1)
    gx, gy = np.meshgrid(np.arange(gw) / gh, np.arange(gh) / gh)
    Pn = perlin(ctx.tl.seed + 32)
    wisp = 0.55 + 0.45 * Pn.noise3(gx * 7, gy * 7 + ctx.t * 0.25, ctx.t * 0.3)
    wisp *= 0.75 + 0.25 * Pn.noise3(gx * 19, gy * 19 + ctx.t * 0.4, 5 + ctx.t * 0.5)
    D = resize(D * wisp, W, H)
    alpha = 1 - np.exp(-D * float(p.get("density", 1.6)))
    # smoke: almost black, faintly lit from the furnaces below and the sky glow above
    # smoke: nearly black; thin edges pick up a little of the sky behind them
    lit = 0.012 + 0.05 * np.exp(-(1 - y) / 0.1)[:, None] + 0.25 * sky * (1 - alpha) ** 2
    img = sky * (1 - alpha) + lit * alpha

    cv = Canvas(W, H, ctx.ss)
    for poly in skyline(ctx.tl.seed, round(aspect, 4)):
        cv.polygon([(x * H, (1 - yy) * H) for x, yy in poly], 1.0)
    sil = cv.array()
    img = img * (1 - sil) + 0.008 * sil
    return np.clip(img, 0, 1)
