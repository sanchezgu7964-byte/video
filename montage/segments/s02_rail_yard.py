"""02 鐵的迷宮 — a railway yard from above at night.

Parallel tracks, a ladder fanning out into sidings, crossovers between the
main tracks. A train of plain rectangular cars rolls through and takes the
diverging route at a turnout. Rails catch the yard lamps: light pools plus a
specular glint that slides along the steel as the camera drifts.
"""
import os
from functools import lru_cache

import numpy as np

from . import segment
from ..fx import Canvas, blur, resize
from ..geom import Camera, WorldTexture, perlin
from ..timeline import ROOT

SP = 4.6            # track spacing (m)
GAUGE = 1.435
X_MIN, X_MAX = -420.0, 420.0
LAMP_H = 22.0


def _resample(P, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    return np.stack([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])], 1), s


def _straight(x0, x1, y):
    x = np.arange(x0, x1, 0.5)
    return np.stack([x, np.full_like(x, y)], 1)


def _scurve(x0, L, ya, yb):
    u = np.linspace(0, 1, int(L / 0.5) + 1)
    return np.stack([x0 + L * u, ya + (yb - ya) * u * u * (3 - 2 * u)], 1)


def _bezier(a, c, b, n=40):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 2 * np.asarray(a) + 2 * (1 - u) * u * np.asarray(c) + u * u * np.asarray(b)


@lru_cache(maxsize=2)
def layout(seed, M, S, train_track, switch_x, switch_len):
    rng = np.random.default_rng([seed, 202])
    tracks = []
    for i in range(M):
        tracks.append(_straight(X_MIN, X_MAX, i * SP))
    # ladder: a diagonal leaves the top main track and feeds the sidings
    th = np.radians(11.0)
    L0, yb = -95.0, (M - 1) * SP
    xb = lambda y: L0 + (y - yb) / np.tan(th)
    c = 9.0
    diag_end_y = (M + S - 1) * SP
    diag = np.concatenate([_bezier((L0 - 2 * c, yb), (L0, yb), (L0 + c, yb + c * np.tan(th))),
                           np.stack([np.linspace(L0 + c, xb(diag_end_y), 200),
                                     np.linspace(yb + c * np.tan(th), diag_end_y, 200)], 1)])
    tracks.append(diag)
    for j in range(M, M + S):
        y = j * SP
        x = xb(y)
        a = (x - c, y - c * np.tan(th))
        tracks.append(np.concatenate([_bezier(a, (x, y), (x + c, y)), _straight(x + c, X_MAX, y)]))
    # crossovers between neighbouring main tracks
    xovers = [(train_track, switch_x)]
    for i in range(M - 1):
        for _ in range(3):
            x = rng.uniform(-220, 260)
            if all(abs(x - xo) > 70 or abs(i - io) > 1 for io, xo in xovers):
                xovers.append((i, x))
    for i, x in xovers:
        flip = (i, x) != (train_track, switch_x) and rng.random() < 0.5
        ya, yb2 = (i + 1) * SP, i * SP
        tracks.append(_scurve(x, switch_len, *((ya, yb2) if flip else (yb2, ya))))
    # the train's route: along train_track, through the turnout, onward on the next track
    y0, y1 = train_track * SP, (train_track + 1) * SP
    route = np.concatenate([_straight(X_MIN, switch_x, y0), _scurve(switch_x, switch_len, y0, y1),
                            _straight(switch_x + switch_len + 0.5, X_MAX, y1)])
    route, rs = _resample(route, 0.25)

    # rails, sleepers and ballast beds as flat arrays
    rails, sleepers = [], []
    for P in tracks:
        P, _ = _resample(P, 0.5)
        tg = np.gradient(P, axis=0)
        tg /= np.linalg.norm(tg, axis=1, keepdims=True)
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        rails += [P + nr * GAUGE / 2, P - nr * GAUGE / 2]
        k = np.arange(0, len(P), 0.65 / 0.5).astype(int)
        sleepers.append(np.stack([P[k] + nr[k] * 1.3, P[k] - nr[k] * 1.3], 1))
    lamps = []
    for k, x in enumerate(np.arange(-200, 300, 46)):
        lamps.append((x + rng.uniform(-8, 8), (3.5 if k % 2 else 10.5) * SP + rng.uniform(-1, 1)))
    signals = [(rng.uniform(-150, 200), (rng.integers(0, M + S) + 0.5) * SP) for _ in range(9)]
    return dict(tracks=tracks, rails=rails, sleepers=np.concatenate(sleepers), route=route, route_s=rs,
                lamps=lamps, signals=signals, M=M, S=S)


@lru_cache(maxsize=2)
def ground_tex(seed):
    cache = os.path.join(ROOT, "out", "cache", f"yard_ground_v1_{seed}.npy")
    n, x0, size = 2048, -500.0, 1000.0
    if not os.path.exists(cache):
        c = x0 + np.arange(n) / (n - 1) * size
        X, Y = np.meshgrid(c, c)
        P = perlin(seed + 21)
        g = 0.5 + 0.35 * P.fbm2(X / 30, Y / 30, 3) + 0.25 * P.noise2(X / 1.3, Y / 1.3)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, np.clip(g, 0, 1).astype(np.float32))
    return WorldTexture(np.load(cache), x0, x0, size)


def yard_camera(t, u, p, W, H, M, S):
    follow = float(p.get("camera_follow", 4.5))
    tx = -12.0 + follow * t
    ty = (M + S) * SP * 0.36
    yaw = np.radians(-19.0 - 3.0 * u)
    elev, dist = np.radians(64.0), 104.0 - 6.0 * u
    back = np.array([np.sin(yaw), -np.cos(yaw)])          # horizontal offset away from the view
    pos = np.array([tx + back[0] * dist * np.cos(elev), ty + back[1] * dist * np.cos(elev),
                    dist * np.sin(elev)])
    return Camera(pos, (tx, ty, 0.0), 47.0, W, H)


def _runs(mask):
    """Start/stop indices of contiguous True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.where(d == 1)[0], np.where(d == -1)[0])


def train_cars(lay, t, p):
    n = int(p.get("train_cars", 11))
    v = float(p.get("train_speed", 9.0))
    sx = float(p.get("switch_x", 0.0))
    s_switch = sx - X_MIN
    head = s_switch - 28.0 + v * t
    route, rs = lay["route"], lay["route_s"]
    at = lambda s: np.stack([np.interp(s, rs, route[:, 0]), np.interp(s, rs, route[:, 1])], -1)
    cars, s = [], head
    for c in range(n):
        L = 19.0 if c == 0 else 15.5
        sc = s - L / 2
        f, b = at(sc + 0.36 * L), at(sc - 0.36 * L)
        d = (f - b) / np.linalg.norm(f - b)
        cars.append((at(sc), d, L, 3.1, c))
        s -= L + 1.1
    return cars


@segment("rail_yard")
def render(ctx):
    p = ctx.params
    W, H, sc = ctx.W, ctx.H, ctx.scale
    M, S = int(p.get("main_tracks", 8)), int(p.get("sidings", 8))
    lay = layout(ctx.tl.seed, M, S, int(p.get("train_track", 3)), float(p.get("switch_x", 0.0)),
                 float(p.get("switch_length", 34.0)))
    cam = yard_camera(ctx.t, ctx.u, p, W, H, M, S)

    # visible ground window (+ margin) for culling
    gx, gy, _, _ = cam.ground(9, 9)
    x_lo, x_hi, y_lo, y_hi = gx.min() - 20, gx.max() + 20, gy.min() - 20, gy.max() + 20
    inside = lambda P: (P[..., 0] > x_lo) & (P[..., 0] < x_hi) & (P[..., 1] > y_lo) & (P[..., 1] < y_hi)
    px_per_m = cam.focal / np.linalg.norm(cam.pos - np.array([*cam.pos[:2], 0]))

    def proj2(P):
        sx, sy, _ = cam.project(np.concatenate([P, np.zeros(P.shape[:-1] + (1,))], -1))
        return np.stack([sx, sy], -1)

    # ---- light field on the ground (half resolution) ----------------------
    gw, gh = W // 2, H // 2
    X, Y, _, _ = cam.ground(gw, gh)
    pool_k = float(p.get("lamp_pool", 0.8))
    light = 0.2 + 0.08 * (X - X.min()) / (np.ptp(X) + 1e-6)
    glint = np.zeros_like(X)
    cg = cam.pos[:2]
    hc = cam.pos[2]
    shimmer = 0.75 + 0.25 * perlin(ctx.tl.seed + 5).noise2(X / 3.0 + ctx.t * 0.7, Y / 40.0)
    for lx, ly in lay["lamps"]:
        light += pool_k * np.exp(-((X - lx) ** 2 + (Y - ly) ** 2) / (2 * 15.0 ** 2))
        r = np.array([lx, ly]) + (cg - np.array([lx, ly])) * LAMP_H / (LAMP_H + hc)
        glint += np.exp(-((X - r[0]) / 3.2) ** 2 - ((Y - r[1]) / 18.0) ** 2)
    glint *= shimmer
    tex = ground_tex(ctx.tl.seed).sample(X, Y)
    light_f = resize(light, W, H)
    glint_f = resize(glint, W, H)
    tex_f = resize(tex, W, H)

    # ---- vector layers ------------------------------------------------------
    bed, sl, rl = Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss)
    for P in lay["tracks"]:
        m = inside(P)
        for a, b in _runs(m):
            if b - a > 1:
                q = proj2(P[a:b:4] if b - a > 8 else P[a:b])
                bed.line(q.tolist(), 1.0, 3.4 * px_per_m)
    S_ = lay["sleepers"]
    S_ = S_[inside(S_[:, 0]) | inside(S_[:, 1])]
    sl.lines(proj2(S_), 1.0, max(1.0, 0.24 * px_per_m))
    for P in lay["rails"]:
        m = inside(P)
        for a, b in _runs(m):
            if b - a > 1:
                rl.line(proj2(P[a:b]).tolist(), 1.0, max(1.0, 0.11 * px_per_m) * 1.15)

    # ---- the train ---------------------------------------------------------
    tc, ta = Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss)
    for ctr, d, L, w, idx in train_cars(lay, ctx.t, p):
        if not inside(ctr[None])[0]:
            continue
        nrm = np.array([-d[1], d[0]])
        corner = lambda a, b: ctr + d * a * L / 2 + nrm * b * w / 2
        body = proj2(np.array([corner(1, 1), corner(1, -1), corner(-1, -1), corner(-1, 1)]))
        tc.polygon(body.tolist(), 0.26)
        ta.polygon(body.tolist(), 1.0)
        tc.line(np.vstack([body, body[:1]]).tolist(), 0.55, 1.1 * sc)
        for b in (-0.45, 0.45):                         # roof ribs
            q = proj2(np.array([ctr + d * (L / 2 - 1.2) + nrm * b, ctr - d * (L / 2 - 1.2) + nrm * b]))
            tc.line(q.tolist(), 0.36, 0.9 * sc)
        if idx == 0:                                    # locomotive cab and fan grilles
            cab = proj2(np.array([corner(1, 0.9), corner(1, -0.9), corner(0.72, -0.9), corner(0.72, 0.9)]))
            tc.polygon(cab.tolist(), 0.07)
            for k in (-0.35, 0.0, 0.35):
                g = ctr - d * k * L / 2
                tc.dot(*proj2(g[None])[0], 0.55 * px_per_m, 0.42)
        else:
            for k in np.linspace(-0.3, 0.3, 3):         # hatches
                g = ctr + d * k * L
                hq = proj2(np.array([g + d * 1.2 + nrm * 0.9, g + d * 1.2 - nrm * 0.9,
                                     g - d * 1.2 - nrm * 0.9, g - d * 1.2 + nrm * 0.9]))
                tc.line(np.vstack([hq, hq[:1]]).tolist(), 0.3, 0.8 * sc)

    # ---- lamps (posts + heads) and signal dots -----------------------------
    lamp = Canvas(W, H, ctx.ss)
    for lx, ly in lay["lamps"]:
        base = cam.project(np.array([lx, ly, 0.0]))
        head = cam.project(np.array([lx, ly, LAMP_H]))
        lamp.line([(base[0], base[1]), (head[0], head[1])], 0.3, 1.6 * sc)
        lamp.dot(head[0], head[1], 4.5 * sc, 1.0)
    for k, (sx_, sy_) in enumerate(lay["signals"]):
        q = proj2(np.array([[sx_, sy_]]))[0]
        on = 0.55 + 0.45 * (np.sin(ctx.t * (1.3 + 0.4 * k) + k) > -0.6)
        lamp.dot(q[0], q[1], 1.6 * sc, 0.9 * on)

    rails = rl.array() * (light_f * 0.95 + float(p.get("glint", 1.3)) * glint_f)
    img = (tex_f * 0.05 * light_f
           + bed.array() * (0.02 + 0.05 * tex_f) * light_f
           + sl.array() * 0.085 * light_f
           + 0.8 * rails)
    a = ta.array()
    img = img * (1 - a) + tc.array() * (0.55 + 0.6 * light_f)
    lam = lamp.array()
    img = np.maximum(img, lam)
    img = img + 0.4 * blur(rails * (1 - a), 7 * sc) + 1.6 * blur(lam, 16 * sc) + 0.8 * blur(lam, 70 * sc)
    return np.clip(img, 0, 1)
