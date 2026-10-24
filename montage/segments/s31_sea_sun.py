"""31 海洋與太陽 — the sea and the sun. The widest image of the film: a slow,
heaving ocean under a high sun, a path of glitter on the water. Full colour, but still.
"""
import numpy as np

from . import segment
from ..draw import pal
from ..fx import blur, resize
from ..geom import Camera, perlin


@segment("sea_sun")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    t = ctx.t
    top, low, sea, sun_c = (pal(ctx, k, d) for k, d in
                            (("sky_top", "#2f6fb0"), ("sky_low", "#cfe3ef"), ("sea", "#0f3f63"), ("sun", "#fffbe8")))
    cam = Camera((0.0, -t * 0.6, 9.0), (0.0, 1000.0 - t * 0.6, 9.0 + 1000 * np.tan(np.radians(3))), 70.0, W, H)
    sun_dir = np.array([0.0, np.cos(np.radians(15)), np.sin(np.radians(15))])
    sx, sy, _ = cam.project(cam.pos + sun_dir * 1e5)
    hx, hy, _ = cam.project(np.array([0.0, 1e7, 9.0]))
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.clip((hy - y) / hy, 0, 1)[..., None]
    sky = low * (1 - a) ** 0.7 + top * (1 - (1 - a) ** 0.7)
    d = np.hypot(x - sx, y - sy) / H
    sky = sky + sun_c * (0.6 * np.exp(-d / 0.03) + 0.35 * np.exp(-d / 0.18) + 0.15 * np.exp(-d / 0.6))[..., None]
    sky = np.where((d < 0.028)[..., None], sun_c * 1.15, sky)
    # sea surface (half resolution): two swell layers + chop, normals from finite differences
    gw, gh = W // 2, H // 2
    X, Y, dist, hit = cam.ground(gw, gh)
    P = perlin(ctx.tl.seed + 311)
    hfun = lambda xx, yy: (0.6 * P.noise2(xx / 22 + t * 0.12, yy / 40 + t * 0.05)
                           + 0.25 * P.fbm2(xx / 5 - t * 0.2, yy / 7 + t * 0.15, 3))
    e = 0.3
    h0 = hfun(X, Y)
    dhx = (hfun(X + e, Y) - h0) / e
    dhy = (hfun(X, Y + e) - h0) / e
    n = np.stack([-dhx, -dhy, np.ones_like(dhx) * 1.6], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    v = np.stack([X - cam.pos[0], Y - cam.pos[1], -np.full_like(X, cam.pos[2])], -1)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    r = v - 2 * (v * n).sum(-1, keepdims=True) * n
    spec = np.clip(r @ sun_dir, 0, 1) ** 260 * 3.0 + np.clip(r @ sun_dir, 0, 1) ** 25 * 0.25
    fres = np.clip(1 - np.abs((v * n).sum(-1)), 0, 1) ** 4
    sky_ref = low * 0.85
    water = sea[None, None] * (0.75 + 0.5 * h0[..., None]) * (1 - fres[..., None]) + sky_ref * fres[..., None]
    water = water + sun_c * spec[..., None]
    haze = (1 - np.exp(-dist / 1800.0))[..., None]
    water = water * (1 - haze) + low * 0.9 * haze
    water = np.where(hit[..., None], water, low)
    water = np.stack([resize(water[..., c], W, H) for c in range(3)], -1)
    img = np.where((y > hy)[..., None], water, sky)
    return np.clip(img + 0.06 * blur(img, 30 * sc), 0, 1)
