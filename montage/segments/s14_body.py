"""14 身體 — only feet and the ground, in silhouette, walking slowly; the
floor holds a faint reflection.

step_times() is shared with the soundtrack (one footstep sound per footfall).
"""
import numpy as np


def step_times(params, duration):
    """Seconds (from segment start) at which a foot lands."""
    sps = float(params.get("steps_per_second", 0.9))
    period = 1.0 / sps / 2.0                   # two feet per stride cycle
    return [0.35 + k * period for k in range(int((duration - 0.35) / period) + 1)]


from functools import lru_cache

from . import segment
from ..fx import Canvas, blur, ease


def shoe(cv, x, y, s, tilt, value=1.0):
    """Side view of a shoe whose sole touches (x, y) at the heel, toe pointing right."""
    c, si = np.cos(tilt), np.sin(tilt)
    R = lambda px, py: (x + c * px - si * py, y + si * px + c * py)
    pts = [(-0.02, 0), (0.95, 0), (1.0, -0.08), (0.85, -0.2), (0.45, -0.32), (0.2, -0.42), (0.0, -0.4), (-0.06, -0.18)]
    cv.polygon([R(px * s, py * s) for px, py in pts], value)
    return R(0.1 * s, -0.38 * s)                       # ankle


def foot_state(t, lands, side, S, x0, lift):
    """(heel_x, heel_y_offset, tilt) of one foot. Feet alternate: side 0 lands on even steps."""
    mine = [(i, tl) for i, tl in enumerate(lands) if i % 2 == side]
    pos = x0 + (side - 1) * S                          # back foot starts one step behind
    for k, (i, tl) in enumerate(mine):
        if t < tl:
            prev_t = mine[k - 1][1] if k > 0 else tl - 2 * (lands[1] - lands[0])
            lift_t = prev_t + (lands[1] - lands[0])    # leaves the ground when the other foot lands
            if t < lift_t:
                return pos, 0.0, 0.0
            u = (t - lift_t) / (tl - lift_t)
            target = x0 + i * S
            return pos + (target - pos) * ease(u), -lift * np.sin(np.pi * u), -0.35 * np.sin(np.pi * u) + 0.2 * u * (1 - u)
        pos = x0 + i * S
    return pos, 0.0, 0.0


@segment("body")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    p = ctx.params
    ground = H * 0.7
    lands = step_times(p, ctx.duration + 2)
    S = W * 0.2                                         # stride (world, in pixels)
    x0 = W * 0.2
    pace = S / (lands[1] - lands[0])                    # body speed
    cam = pace * ctx.t - W * 0.3                        # the camera follows the walker
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    bg = (0.025 + 0.12 * np.exp(-np.abs(y - ground / H) / 0.12)) * np.ones((1, W), np.float32)
    sil = Canvas(W, H, ctx.ss)
    for side in (0, 1):
        hx, dy, tilt = foot_state(ctx.t, lands, side, S, x0, H * 0.08)
        hx -= cam
        ank = shoe(sil, hx, ground + dy, H * 0.3, tilt)
        hip_x = x0 + pace * ctx.t - cam + S * 0.25
        top = (hip_x + (ank[0] - hip_x) * 0.35, -H * 0.05)
        sil.polygon([(ank[0] - H * 0.05, ank[1] + H * 0.01), (ank[0] + H * 0.045, ank[1] + H * 0.01),
                     (top[0] + H * 0.085, top[1]), (top[0] - H * 0.075, top[1])], 1.0)
    seams = Canvas(W, H, ctx.ss)
    for k in range(-2, 12):
        sx = k * W * 0.23 - (cam % (W * 0.23))
        seams.line([(sx, ground + 2), (sx - W * 0.5, H)], 1.0, 1.2 * sc)
    m = sil.array()
    # the floor: a faint reflection of the light and of the feet
    gi = int(ground)
    refl = np.zeros_like(m)
    span = min(gi, H - gi)
    refl[gi:gi + span] = m[gi - 1:gi - span - 1:-1] if gi - span - 1 >= 0 else m[gi - 1::-1][:span]
    falloff = np.exp(-np.clip(y - ground / H, 0, None) / 0.08)
    floor = (0.02 + float(p.get("sheen", 0.35)) * 0.35 * bg * falloff) * (1 - 0.9 * blur(refl, 3 * sc, 1))
    floor = floor * (1 - 0.5 * seams.array())
    img = np.where(y < ground / H, bg, floor)
    img = img * (1 - m) + 0.006 * m
    return np.clip(img + 0.05 * blur(img, 30 * sc), 0, 1)
