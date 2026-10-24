"""24 忍冬花 — honeysuckle, suddenly slow. Extreme close-up of the flower's
curling lip petals and long stamens, very shallow depth of field: one bloom
sharp, others dissolved into warm bokeh.
"""
import numpy as np

from . import segment
from ..draw import pal
from ..fx import Canvas, blur, ease


def bloom(cv_petal, cv_throat, cv_stamen, cx, cy, s, rot):
    """Two-lipped honeysuckle flower: a tube, a strap-like lower lip curling back,
    a four-lobed upper lip, and long arching stamens."""
    c, si = np.cos(rot), np.sin(rot)
    R = lambda x, y: (cx + (c * x - si * y) * s, cy + (si * x + c * y) * s)
    # tube
    cv_throat.polygon([R(-1.6, -0.06), R(0.0, -0.12), R(0.0, 0.12), R(-1.6, 0.06)], 1.0)
    # lower lip: a soft strap curling down and back, rounded at the tip
    t = np.linspace(0, 1, 40)
    spine = [(0.0 + 0.9 * np.sin(a * 2.2), 0.1 + 0.9 * (1 - np.cos(a * 2.2)) * 0.6) for a in t]
    w = 0.15 * np.sqrt(1 - 0.85 * t)
    left = [R(x - wi * 0.3, y - wi) for (x, y), wi in zip(spine, w)]
    right = [R(x + wi * 0.3, y + wi) for (x, y), wi in zip(spine, w)]
    cv_petal.polygon(left + right[::-1], 1.0)
    cv_petal.dot(*R(*spine[-1]), 0.03 * s, 1.0)
    # upper lip: four rounded lobes fanning upward (each an ellipse)
    for k, ang in enumerate(np.linspace(-1.45, -0.35, 4)):
        L = 0.7 + 0.08 * (k % 2)
        th = np.linspace(0, 2 * np.pi, 30)
        ex, ey = L * 0.5 * np.cos(th) + L * 0.5, 0.15 * np.sin(th)
        ca, sa = np.cos(ang), np.sin(ang)
        cv_petal.polygon([R(ca * a_ - sa * b_, sa * a_ + ca * b_) for a_, b_ in zip(ex, ey)], 1.0)
    # stamens: long arcs ending in anthers
    for k in range(5):
        a = -0.6 + k * 0.18
        pts = [R(0.05 + 1.5 * q * np.cos(a), -0.02 + 1.5 * q * np.sin(a) - 0.35 * q * q) for q in np.linspace(0, 1, 20)]
        cv_stamen.line(pts, 1.0, max(1.0, 0.012 * s))
        cv_stamen.dot(*pts[-1], 0.03 * s, 1.0)


@segment("honeysuckle")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    petal, throat, leaf, bg = (pal(ctx, k, d) for k, d in
                               (("petal", "#f6e7c1"), ("throat", "#e8b04a"), ("leaf", "#4f6b3a"), ("bg", "#2a2418")))
    u = ease(ctx.u)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = bg * (0.6 + 0.6 * np.exp(-((x / W - 0.6) ** 2 + (y / H - 0.4) ** 2) / 0.2))[..., None]
    rng = np.random.default_rng(ctx.seed("bokeh"))
    # background blooms and leaves: rendered then heavily blurred (out of focus)
    far_p, far_t, far_s, leaves = (Canvas(W // 2, H // 2, 1) for _ in range(4))
    for _ in range(9):
        bloom(far_p, far_t, far_s, rng.uniform(0, W / 2), rng.uniform(0, H / 2), rng.uniform(60, 120) * sc,
              rng.uniform(-3, 3))
    for _ in range(14):
        cx, cy, r = rng.uniform(0, W / 2), rng.uniform(0, H / 2), rng.uniform(30, 90) * sc
        leaves.d.ellipse([cx - r, cy - r * 0.45, cx + r, cy + r * 0.45], fill=200)
    from ..fx import resize
    blur_far = lambda c: resize(blur(c.array(), 30 * sc, 2), W, H)
    lf, fp, ft = blur_far(leaves), blur_far(far_p), blur_far(far_t)
    img = img * (1 - 0.7 * lf[..., None]) + leaf * 0.5 * lf[..., None]
    img = img * (1 - 0.5 * fp[..., None]) + petal * 0.55 * fp[..., None] + throat * 0.3 * ft[..., None]
    # the bloom in focus, slowly drifting closer
    p, t, s_ = Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss)
    bloom(p, t, s_, W * (0.5 + 0.02 * u), H * (0.62 - 0.02 * u), H * (0.5 + 0.05 * u), -0.2 + 0.05 * u)
    pm, tm, sm = p.array(), t.array(), s_.array()
    shade = 0.75 + 0.35 * (1 - y / H)
    img = img * (1 - tm[..., None]) + throat * (shade * 0.9)[..., None] * tm[..., None]
    img = img * (1 - pm[..., None]) + petal * shade[..., None] * pm[..., None]
    img = img * (1 - sm[..., None]) + np.array([0.95, 0.85, 0.55]) * sm[..., None]
    soft = blur(img, 10 * sc)
    return np.clip(img * 0.65 + soft * 0.5, 0, 1)                 # a soft, glowing lens
