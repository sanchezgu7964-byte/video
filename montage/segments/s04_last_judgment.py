"""04 最後審判 — a bare plain, a railway running to the horizon, the sun rising
exactly at the vanishing point. The frame starts black & white and floods with
colour: the first colour in the film.
"""
from functools import lru_cache

import numpy as np

from . import segment
from ..draw import desat, luma, pal
from ..fx import Canvas, blur, ease, resize, smoothstep
from ..geom import Camera, perlin

EYE = 1.6


def camera(W, H, u):
    tilt = np.radians(5.0)
    pos = np.array([0.55, -0.6 * u, EYE])
    return Camera(pos, pos + np.array([-0.55 * 0.0, 1000.0, 1000.0 * np.tan(tilt)]), 62.0, W, H)


def sun_elevation(u):
    return np.radians(-3.5 + 11.0 * ease(min(1.0, u * 1.05)))


@lru_cache(maxsize=2)
def poles():
    return [(3.4, 12.0 + 42.0 * k) for k in range(45)]


@segment("last_judgment")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    u = ctx.u
    cam = camera(W, H, u)
    tint = ctx.tl.look.get("bw_tint", [1, 1, 1])
    c_top, c_mid, c_low = pal(ctx, "sky_top", "#141a4a"), pal(ctx, "sky_mid", "#b8325c"), pal(ctx, "sky_low", "#ff8a3a")
    c_sun, c_ground = pal(ctx, "sun", "#fff2c8"), pal(ctx, "ground", "#2b1a12")

    hx, hy, _ = cam.project(np.array([cam.pos[0], 1e7, EYE]))      # horizon row
    e = sun_elevation(u)
    sx, sy, _ = cam.project(np.array([0.0, 1e6, EYE + 1e6 * np.tan(e)]))
    vx, _, _ = cam.project(np.array([0.0, 1e7, 0.0]))
    k = 0.12 + 0.88 * smoothstep(np.radians(-3.5), np.radians(5.0), e)   # daylight amount

    # ---- sky ---------------------------------------------------------------
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]
    a = np.clip((hy - y) / max(hy, 1), 0, 1)                           # 0 at horizon, 1 at top
    stops = np.array([0.0, 0.18 + 0.1 * k, 1.0])
    cols = np.stack([c_low, c_mid, c_top])
    sky = np.stack([np.interp(a, stops, cols[:, c]) for c in range(3)], -1)
    sky = sky * (0.25 + 0.75 * k)
    d = np.sqrt((x - sx) ** 2 + (y - sy) ** 2) / H
    glow = 0.9 * np.exp(-d / 0.05) + 0.45 * np.exp(-d / 0.22) + 0.15 * np.exp(-d / 0.7)
    sky = sky + c_sun * (glow * (0.35 + 0.65 * k))[..., None]
    disc = np.clip((0.045 - d) * H / 2.0, 0, 1)
    sky = sky * (1 - disc[..., None]) + 1.25 * c_sun * disc[..., None]

    # ---- ground (half resolution ray cast) ---------------------------------
    gw, gh = W // 2, H // 2
    X, Y, dist, hit = cam.ground(gw, gh)
    P = perlin(ctx.tl.seed + 41)
    tex = 0.6 + 0.4 * P.fbm2(X / 6, Y / 6, 3)
    cracks = np.exp(-np.abs(P.noise2(X / 1.7, Y / 1.7)) * 14)
    tex = tex * (1 - 0.6 * cracks)
    gx = (np.arange(gw) + 0.5)[None, :] * W / gw
    sheen = np.exp(-np.abs(gx - sx) / (0.18 * W)) * np.exp(-dist / 400.0) * 0 + \
        np.exp(-np.abs(gx - sx) / (0.12 * W)) * (1 - np.exp(-dist / 60.0))
    ground = c_ground[None, None] * (tex * (0.25 + 0.75 * k))[..., None] + \
        (c_sun * 0.45 * k)[None, None] * sheen[..., None]
    fogf = (1 - np.exp(-dist / 260.0))[..., None]
    horizon_col = (c_low * (0.3 + 0.7 * k) + c_sun * 0.25 * k)
    ground = ground * (1 - fogf) + horizon_col * fogf
    ground = np.where(hit[..., None], ground, horizon_col)
    ground = np.stack([resize(ground[..., c], W, H) for c in range(3)], -1)
    below = np.clip(y - hy + 0.5, 0, 1)[..., None] * np.ones((1, W, 1), np.float32)
    img = sky * (1 - below) + ground * below

    # ---- sleepers, rails, telegraph poles -----------------------------------
    slp, rail, pole = Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss)
    for yy in np.arange(1.2, 260, 0.65):
        q = np.array([[-1.3, yy - 0.11, 0.05], [1.3, yy - 0.11, 0.05], [1.3, yy + 0.11, 0.05], [-1.3, yy + 0.11, 0.05]])
        qx, qy, z = cam.project(q)
        if np.all(z > 0.3):
            slp.polygon(list(zip(qx, qy)), 1.0)
    ys = np.concatenate([np.linspace(0.9, 50, 120), np.geomspace(50, 1e5, 120)])
    for rx in (-0.7175, 0.7175):
        L = np.stack([np.full_like(ys, rx - 0.035), ys, np.full_like(ys, 0.17)], 1)
        R = np.stack([np.full_like(ys, rx + 0.035), ys, np.full_like(ys, 0.17)], 1)
        lx, ly, _ = cam.project(L)
        rx_, ry, _ = cam.project(R)
        rail.polygon(list(zip(lx, ly)) + list(zip(rx_[::-1], ry[::-1])), 1.0)
    tops = []
    for px, py in poles():
        b = cam.project(np.array([px, py, 0.0]))
        t = cam.project(np.array([px, py, 7.0]))
        if b[2] < 1:
            continue
        w = max(0.6, 0.18 * cam.focal / b[2])
        pole.line([(b[0], b[1]), (t[0], t[1])], 1.0, w)
        arm = [cam.project(np.array([px + o, py, 6.4])) for o in (-0.9, 0.9)]
        pole.line([(arm[0][0], arm[0][1]), (arm[1][0], arm[1][1])], 1.0, w * 0.7)
        tops.append((px, py))
    for (x0, y0), (x1, y1) in zip(tops[:-1], tops[1:]):
        for wx in (-0.7, 0.7):
            s_ = np.linspace(0, 1, 16)
            Wp = np.stack([x0 + wx + 0 * s_, y0 + (y1 - y0) * s_, 6.5 - 0.9 * 4 * s_ * (1 - s_)], 1)
            qx, qy, z = cam.project(Wp)
            if np.all(z > 1):
                pole.line(list(zip(qx, qy)), 0.8, 0.8 * sc)

    img = img * (1 - 0.85 * slp.array()[..., None])
    rm = rail.array()
    yy_ = np.clip((y - hy) / (H - hy), 0, 1)
    shine = c_sun * (0.25 + 0.95 * k) * (1 - 0.6 * yy_)[..., None] + 0.08
    img = img * (1 - rm[..., None]) + shine * rm[..., None]
    pm = pole.array()
    img = img * (1 - pm[..., None]) + (0.02 + 0.05 * c_ground) * pm[..., None]
    img = img + 0.25 * blur(np.clip(img - 0.85, 0, None).mean(-1), 40 * sc)[..., None] * c_sun

    # ---- from black & white into colour ------------------------------------
    p = ctx.params
    s = smoothstep(float(p.get("color_in_start", 0.15)), float(p.get("color_in_end", 0.7)), u)
    img = desat(np.clip(img, 0, 1.2), 1 - s, tint)
    return np.clip(img, 0, 1)
