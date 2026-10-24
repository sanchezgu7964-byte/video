"""01 高架橋 — a railway viaduct seen from high above, receding into haze.

The drawing function `draw_viaduct` is reused by the ending (segment 14),
which returns to this exact composition.
"""
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from . import segment
from ..fx import Canvas, blur, resize
from ..geom import Camera, WorldTexture, normalize, perlin
from ..timeline import ROOT

LIGHT = normalize([0.42, 0.8, -0.36])       # direction the light travels
TEX_X0, TEX_SIZE, TEX_N = -500.0, 1200.0, 2048


def river_x(y):
    return 28 + 24 * np.sin(y / 70.0) + 9 * np.sin(y / 23.0 + 1.0)


@lru_cache(maxsize=2)
def ground_texture(seed):
    cache = os.path.join(ROOT, "out", "cache", f"viaduct_ground_v4_{seed}_{TEX_N}.npy")
    if os.path.exists(cache):
        return WorldTexture(np.load(cache), TEX_X0, TEX_X0, TEX_SIZE)
    n = TEX_N
    c = TEX_X0 + (np.arange(n) / (n - 1)) * TEX_SIZE
    X, Y = np.meshgrid(c, c)
    P = perlin(seed + 11)
    g = 0.16 + 0.07 * P.fbm2(X / 140, Y / 140, 4) + 0.025 * P.fbm2(X / 9, Y / 9, 3)
    # field parcels: blocky noise quantised into a few tones
    parcels = np.floor((P.noise2(X / 55 + 3.3, Y / 38 - 1.7) * 3 + 3)) / 6.0
    g += 0.035 * (parcels - 0.5)
    # roads / hedgerows as straight faint lines
    img = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed + 12)
    for _ in range(14):
        a = rng.uniform(0, np.pi)
        x0, y0 = rng.uniform(0, n, 2)
        L = n * 1.5
        d.line((x0 - np.cos(a) * L, y0 - np.sin(a) * L, x0 + np.cos(a) * L, y0 + np.sin(a) * L),
               fill=int(rng.uniform(80, 200)), width=int(rng.integers(2, 5)))
    roads = np.asarray(img, np.float32) / 255.0
    g += 0.05 * roads
    # river: darker water with a pale bank
    dist = np.abs(X - river_x(Y))
    water = np.clip((9.5 - dist) / 2.0, 0, 1)
    bank = np.clip((12.5 - dist) / 2.5, 0, 1) - water
    ripple = 0.5 + 0.5 * P.noise2(X / 3.0, Y / 14.0)
    g = g * (1 - water) + water * (0.06 + 0.07 * ripple) + 0.07 * bank
    g = np.clip(g, 0.02, 1).astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, g)
    return WorldTexture(g, TEX_X0, TEX_X0, TEX_SIZE)


@lru_cache(maxsize=4)
def bridge(span=18.0, height=34.0, width=9.0, pier=3.6, x0=-420.0, x1=700.0, crown=5.0, arc_n=18):
    """Static 3D geometry, one entry per span."""
    yn, yf = -width / 2, width / 2
    r = (span - pier) / 2
    s = height - crown - r
    th = np.linspace(0, np.pi, arc_n)
    spans = []
    xs = np.arange(x0, x1, span)
    for xl in xs:
        xr = xl + span
        cx = (xl + xr) / 2
        arc = np.stack([cx + r * np.cos(th), s + r * np.sin(th)], 1)        # right -> left

        def wall(y):
            pts = [(xl, 0), (xl, height), (xr, height), (xr, 0), (cx + r, 0), (cx + r, s)]
            pts += [tuple(p) for p in arc[1:-1]]
            pts += [(cx - r, s), (cx - r, 0)]
            return np.array([(x, y, z) for x, z in pts])

        intrados = []
        for k in range(arc_n - 1):
            a, b = arc[k], arc[k + 1]
            q = np.array([(a[0], yn, a[1]), (b[0], yn, b[1]), (b[0], yf, b[1]), (a[0], yf, a[1])])
            mid = (a + b) / 2
            nrm = normalize([cx - mid[0], 0, s - mid[1]])
            intrados.append((q, nrm))
        sides = [
            (np.array([(cx + r, yn, 0), (cx + r, yn, s), (cx + r, yf, s), (cx + r, yf, 0)]), np.array([-1.0, 0, 0])),
            (np.array([(cx - r, yn, 0), (cx - r, yn, s), (cx - r, yf, s), (cx - r, yf, 0)]), np.array([1.0, 0, 0])),
        ]
        deck = np.array([(xl, yn, height), (xr, yn, height), (xr, yf, height), (xl, yf, height)])
        near_arc = np.array([(x, yn, z) for x, z in arc])
        spans.append(dict(cx=cx, far=wall(yf), near=wall(yn), intrados=intrados, sides=sides,
                          deck=deck, near_arc=near_arc,
                          cornice=np.array([(xl, yn, height - 1.6), (xr, yn, height - 1.6)]),
                          impost=np.array([(xl, yn, s), (cx - r, yn, s)]),
                          impost2=np.array([(cx + r, yn, s), (xr, yn, s)])))
    return dict(spans=spans, x0=x0, x1=xs[-1] + span, yn=yn, yf=yf, height=height)


def shade(n, albedo, amb=0.32):
    return albedo * (amb + (1 - amb) * max(0.0, float(np.dot(n, -LIGHT))))


def viaduct_camera(u, drift, W, H):
    k = drift * (0.85 * u + 0.15 * u * u)
    pos = np.array([-235.0, -105.0, 160.0]) + k * np.array([30.0, 4.0, -8.0])
    tgt = np.array([110.0, 22.0, 0.0]) + k * np.array([30.0, 0.0, 0.0])
    return Camera(pos, tgt, 46.0, W, H)


def draw_viaduct(ctx, cam, p):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    span, height = float(p.get("span", 18.0)), float(p.get("height", 34.0))
    fog_d, fog_c = float(p.get("fog_distance", 260.0)), float(p.get("fog_level", 0.2))
    geo = bridge(span, height)

    def fog(c, dist):
        f = 1 - np.exp(-np.maximum(dist - 200.0, 0) / fog_d)
        return c * (1 - f) + fog_c * f

    # ---- ground (half resolution ray cast) --------------------------------
    gw, gh = W // 2, H // 2
    X, Y, dist, hit = cam.ground(gw, gh)
    ground = ground_texture(ctx.tl.seed).sample(X, Y)

    # shadow of the walls + deck, projected along the light onto z = 0
    sh = Canvas(gw, gh, 2)
    t_of = lambda P: P[:, 2:3] / -LIGHT[2]
    for sp in geo["spans"]:
        for poly in (sp["near"], sp["far"], sp["deck"]):
            G = poly + t_of(poly) * LIGHT
            sx, sy, z = cam.project(G)
            if np.all(z > 1):
                sh.polygon(list(zip(sx * gw / W, sy * gh / H)), 1.0)
    shadow = sh.array()
    ground = ground * (1 - float(p.get("shadow", 0.6)) * shadow)
    ground = np.where(hit, fog(ground, dist), fog_c)
    img = resize(ground, W, H)

    # ---- bridge (painter's algorithm, far spans first) --------------------
    col, alpha, hi = Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss)
    rng = np.random.default_rng(ctx.seed("spans"))
    tone = rng.uniform(-0.025, 0.025, len(geo["spans"]))

    def proj(P):
        sx, sy, z = cam.project(P)
        return list(zip(sx, sy)), z

    order = sorted(range(len(geo["spans"])),
                   key=lambda i: -np.linalg.norm(np.array([geo["spans"][i]["cx"], 0, height / 2]) - cam.pos))
    for i in order:
        sp = geo["spans"][i]
        d = np.linalg.norm(np.array([sp["cx"], 0, height / 2]) - cam.pos)
        pts, z = proj(sp["near"])
        if np.any(z < 1):
            continue
        xs = np.array([q[0] for q in pts])
        if xs.max() < -W * 0.2 or xs.min() > W * 1.2:
            continue
        a = 0.5 + tone[i]

        def fill(P, c):
            pp, _ = proj(P)
            col.polygon(pp, fog(c, d))
            alpha.polygon(pp, 1.0)

        fill(sp["far"], shade(np.array([0, 1.0, 0]), a))
        for q, n in sp["sides"]:
            fill(q, shade(n, a * 0.9))
        for q, n in sp["intrados"]:
            fill(q, shade(n, a * 0.7))
        fill(sp["near"], shade(np.array([0, -1.0, 0]), a))
        fill(sp["deck"], shade(np.array([0, 0, 1.0]), 0.5 + tone[i]))
        # drawn details on the face
        col.line(proj(sp["near_arc"])[0], fog(0.34, d), 1.1 * sc)
        col.line(proj(sp["cornice"])[0], fog(0.30, d), 1.0 * sc)
        col.line(proj(sp["impost"])[0], fog(0.26, d), 0.8 * sc)
        col.line(proj(sp["impost2"])[0], fog(0.26, d), 0.8 * sc)

    # ---- highlights along the deck: parapets and rails --------------------
    xs = np.linspace(geo["x0"], geo["x1"], 400)
    for y, z, v, w in ((geo["yn"], height, 0.95, 1.4), (geo["yf"], height, 0.5, 1.0),
                       (-0.75, height + 0.2, 0.85, 1.0), (0.75, height + 0.2, 0.85, 1.0)):
        P = np.stack([xs, np.full_like(xs, y), np.full_like(xs, z)], 1)
        sx, sy, zc = cam.project(P)
        ok = zc > 5
        dd = np.linalg.norm(P - cam.pos, axis=1)
        # draw in chunks so the fog can dim the far end
        idx = np.where(ok)[0]
        for k in range(0, len(idx) - 1, 20):
            j = idx[k:k + 21]
            hi.line(list(zip(sx[j], sy[j])), fog(v, dd[j].mean()), w * sc)

    a_ = alpha.array()
    c_ = col.array()
    h_ = hi.array()
    img = img * (1 - a_) + c_
    img = np.maximum(img, h_)
    img = img + 0.12 * blur(img, 18 * sc) + 0.35 * blur(h_, 6 * sc)
    return np.clip(img, 0, 1)


@segment("viaduct")
def render(ctx):
    cam = viaduct_camera(ctx.u, float(ctx.params.get("drift", 1.0)), ctx.W, ctx.H)
    return draw_viaduct(ctx, cam, ctx.params)
