"""22 血液裡的衝鋒 — the charge in the blood. The fastest passage of the film:
14 shrinking shots of nothing but hooves and dust, smeared by motion blur and
tinted black → blood red → hot orange. Horses are black silhouettes, backlit
by a glowing haze of dust.

Each shot uses one of four compositions:
  legs     — galloping legs from the side, bellies cut by the frame, ground racing past
  strike   — one huge hoof slamming down, dust bursting out
  cloud    — a churning dust cloud with legs flickering through
  streaks  — abstract speed lines with legs flashing across
"""
import numpy as np

from . import segment
from ..draw import pal
from ..fx import Canvas, blur, hblur, resize, rotate
from ..geom import perlin

KINDS = ["legs", "strike", "legs", "cloud", "streaks", "legs", "strike", "cloud",
         "legs", "streaks", "strike", "legs", "cloud", "streaks", "legs", "strike"]


def leg(cv, hx, hy, L, phase, sc=1.0):
    """One galloping leg as a silhouette: forearm, cannon, fetlock, pastern, hoof.
    Returns the hoof's ground point."""
    a1 = 0.75 * np.sin(2 * np.pi * phase)
    a2 = a1 + 1.15 * max(0.0, np.sin(2 * np.pi * phase + 1.0))      # knee folds back on the swing
    a3 = a2 - 0.55
    d = lambda a: np.array([np.sin(a), np.cos(a)])
    hip = np.array([hx, hy])
    knee = hip + d(a1) * L * 0.46
    fet = knee + d(a2) * L * 0.4
    pas = fet + d(a3) * L * 0.13

    def limb(p0, p1, w0, w1):
        dd = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
        n = np.array([-dd[1], dd[0]])
        cv.polygon([tuple(p0 + n * w0), tuple(p1 + n * w1), tuple(p1 - n * w1), tuple(p0 - n * w0)], 1.0)

    limb(hip, knee, L * 0.13, L * 0.075)
    limb(knee, fet, L * 0.05, L * 0.04)
    limb(fet, pas, L * 0.045, L * 0.04)
    cv.dot(*knee, L * 0.065, 1.0)
    cv.dot(*fet, L * 0.055, 1.0)
    dd = d(a3)
    n = np.array([-dd[1], dd[0]])
    hoof = [pas - n * L * 0.045, pas + n * L * 0.05, pas + dd * L * 0.1 + n * L * 0.075, pas + dd * L * 0.1 - n * L * 0.065]
    cv.polygon([tuple(q) for q in hoof], 1.0)
    return pas + dd * L * 0.1


def horse(cv, x, ground, L, t, f, ph):
    """Belly + four legs; the body is cut off by the top of the frame."""
    hy = ground - L * 1.05
    cv.d.ellipse([(x - L * 0.35) * cv.ss, (hy - L * 0.9) * cv.ss, (x + L * 1.75) * cv.ss, (hy + L * 0.22) * cv.ss], fill=255)
    feet = []
    for off, dx in ((0.0, 0.05), (0.1, 0.2), (0.52, 1.25), (0.62, 1.4)):
        feet.append(leg(cv, x + dx * L, hy, L, t * f + ph + off))
    return feet


def dust(cv, rng, n, origins, shot_t, H, speed=1.0):
    """Particles thrown back and up from the hooves, drawn as short streaks (their motion blur)."""
    for _ in range(n):
        ox, oy = origins[rng.integers(len(origins))]
        birth = rng.uniform(-0.4, max(0.05, shot_t))
        age = shot_t - birth
        if age <= 0:
            continue
        vx = rng.uniform(-1.6, -0.3) * H * speed
        vy = -rng.uniform(0.1, 0.8) * H * speed
        k = 2.5

        def pos(a):
            f = (1 - np.exp(-k * a)) / k
            return ox + vx * f - H * 0.9 * a, oy + vy * f + 0.25 * H * a * a
        x1, y1 = pos(age)
        x0, y0 = pos(max(0.0, age - 0.05))
        cv.line([(x0, y0), (x1, y1)], np.exp(-age / 0.6) * rng.uniform(0.4, 1.0), rng.uniform(1, 4) * H / 1080)


def haze(ctx, ground, t, strength=1.0):
    """Glowing dust-laden air, brightest just above the ground, drifting fast."""
    W, H = ctx.W, ctx.H
    gw, gh = W // 4, H // 4
    y, x = np.mgrid[0:gh, 0:gw].astype(np.float32) / gh
    P = perlin(ctx.tl.seed + 221)
    n = 0.5 + 0.5 * P.fbm2(x * 3 + t * 6, y * 3, 3)
    g = (0.25 + 0.5 * np.exp(-np.abs(y - ground / H) / 0.18)) * (0.6 + 0.6 * n) * strength
    return resize(hblur(g, 8), W, H)


def shot_legs(ctx, rng, t):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    ground = H * rng.uniform(0.74, 0.86)
    bg = haze(ctx, ground, t)
    gr = Canvas(W, H, ctx.ss)
    for _ in range(70):
        x = (rng.uniform(0, 1.4) * W - t * W * 2.5) % (W * 1.4) - W * 0.2
        y = ground + rng.uniform(0, H - ground)
        gr.line([(x, y), (x + rng.uniform(30, 160) * sc, y)], rng.uniform(0.2, 0.6), rng.uniform(1, 3) * sc)
    groundmask = np.zeros((H, W), np.float32)
    groundmask[int(ground):] = 1
    bg = bg * (1 - groundmask * 0.65) + hblur(gr.array(), 40 * sc) * 0.4
    sil = Canvas(W, H, ctx.ss)
    origins = []
    for h in range(int(rng.integers(2, 4))):
        L = H * 0.5 * rng.uniform(0.75, 1.15)
        x = rng.uniform(-0.15, 0.75) * W + np.sin(t * 2 + h) * W * 0.02
        origins += [tuple(p) for p in horse(sil, x, ground, L, t, rng.uniform(2.3, 2.8), rng.uniform(0, 1))
                    if p[1] > ground - L * 0.12]
    origins = origins or [(W * 0.5, ground)]
    dl = Canvas(W, H, ctx.ss)
    dust(dl, rng, 900, origins, ctx.shot_t, H, 1.2)
    s = hblur(sil.array(), 6 * sc)
    d = dl.array()
    return bg * (1 - s) + 0.02 * s + d + 0.5 * blur(d, 12 * sc)


def shot_strike(ctx, rng, t):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    ground = H * 0.84
    u = ctx.shot_u
    impact = 0.45
    L = H * 1.05
    bg = haze(ctx, ground, t, 1.1)
    bg[int(ground):] *= 0.4
    x = W * rng.uniform(0.3, 0.55)
    drop = min(1.0, u / impact)
    hy = ground - L * 0.98 - (1 - drop ** 2) * H * 0.45
    sil = Canvas(W, H, ctx.ss)
    foot = leg(sil, x, hy, L, 0.47 + 0.03 * drop)          # near-vertical, coming straight down
    sil.polygon([(x - L * 0.3, -10), (x + L * 0.3, -10), (x + L * 0.15, hy + L * 0.1), (x - L * 0.15, hy + L * 0.1)], 1.0)
    s = sil.array()
    if drop < 1:
        s = hblur(s.T, 20 * sc * (1 - drop)).T              # vertical smear while falling
    img = bg * (1 - s) + 0.02 * s
    if u > impact:
        a = u - impact
        dl = Canvas(W, H, ctx.ss)
        fx = foot[0]
        for _ in range(1300):
            ang = rng.uniform(np.pi * 1.02, np.pi * 1.98)
            v = rng.uniform(0.4, 2.4) * H
            pt = lambda aa: (fx + np.cos(ang) * v * aa * (1 - 0.5 * aa),
                             ground + np.sin(ang) * v * aa * (1 - 0.5 * aa) * 0.55 + H * 0.5 * aa * aa)
            dl.line([pt(max(0, a - 0.04)), pt(a)], rng.uniform(0.3, 1.0) * np.exp(-a / 0.4), rng.uniform(1, 3) * sc)
        d = dl.array()
        img = img + d + 0.7 * blur(d, 20 * sc) + np.exp(-a / 0.05) * 0.45
    return img


def shot_cloud(ctx, rng, t):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    gw, gh = W // 4, H // 4
    y, x = np.mgrid[0:gh, 0:gw].astype(np.float32) / gh
    P = perlin(ctx.tl.seed + 222 + ctx.shot)
    c = 0.5 + 0.5 * P.fbm2(x * 2.5 + t * 9, y * 2.5 - t * 1.5, 4)
    c = resize(hblur(np.clip(c, 0, 1) ** 1.5, 10), W, H) * 1.1
    sil = Canvas(W, H, ctx.ss)
    L = H * 0.75
    for j in range(3):
        if (int(ctx.shot_t * 24) + j) % 2 == 0:
            leg(sil, W * (0.15 + 0.32 * j), -H * 0.05, L, t * 2.6 + j * 0.31)
    s = hblur(sil.array(), 28 * sc)
    return c * (1 - 0.85 * s)


def shot_streaks(ctx, rng, t):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    cv = Canvas(W, H, ctx.ss)
    for _ in range(160):
        y = rng.uniform(0, H)
        x = (rng.uniform(0, 2) * W - t * W * 4) % (W * 2) - W * 0.5
        cv.line([(x, y), (x + rng.uniform(0.1, 0.6) * W, y)], rng.uniform(0.15, 0.9), rng.uniform(1, 6) * sc)
    bg = hblur(cv.array(), 30 * sc) * 0.9 + 0.25
    sil = Canvas(W, H, ctx.ss)
    for j in range(4):
        if (int(ctx.shot_t * 24) + j) % 3 == 0:
            leg(sil, W * (0.12 + 0.24 * j), H * 0.05, H * 0.75, t * 2.6 + j * 0.27)
    s = hblur(sil.array(), 22 * sc)
    return bg * (1 - s)


SHOTS = {"legs": shot_legs, "strike": shot_strike, "cloud": shot_cloud, "streaks": shot_streaks}


@segment("charge")
def render(ctx):
    rng = np.random.default_rng(ctx.seed("shot", ctx.shot))
    kind = KINDS[ctx.shot % len(KINDS)]
    g = SHOTS[kind](ctx, rng, ctx.t)
    g = rotate(g, float(rng.uniform(-9, 9)))
    g = np.clip(g, 0, 1.5)
    dark, mid, hot = pal(ctx, "dark", "#140304"), pal(ctx, "mid", "#8e1414"), pal(ctx, "hot", "#ffb070")
    a = np.clip(g / 0.55, 0, 1)[..., None]
    b = np.clip((g - 0.55) / 0.45, 0, 1)[..., None]
    img = dark * (1 - a) + mid * a
    img = img * (1 - b) + hot * b
    return np.clip(img, 0, 1)
