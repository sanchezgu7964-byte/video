"""25 愛與愛的前夜、懷念 — two hands reach toward each other and almost touch
(they never do). Next shot: only one hand remains, beside an empty space
where a faint warmth lingers.
"""
import numpy as np

from . import segment
from ..fx import Canvas, blur, ease


def reaching_hand(cv, tip_x, tip_y, s, side, slope):
    """A reaching hand: tapered forearm from the frame edge, palm, an extended index
    finger ending exactly at (tip_x, tip_y); side=-1 comes from the left.
    slope tilts the whole arm (radians, positive = arm rises toward the tip)."""
    d = -side
    c, si = np.cos(slope), np.sin(slope)
    R = lambda a, b: (tip_x + d * s * (c * a + si * b), tip_y + s * (-si * a + c * b))
    edge = 4.0
    cv.polygon([R(-edge, -0.2), R(-1.25, -0.12), R(-1.25, 0.42), R(-edge, 0.62)], 1.0)           # forearm
    cv.polygon([R(-1.35 + 0.42 * np.cos(a), 0.2 + 0.33 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 28)], 1.0)
    pts = [R(-1.0 + q, 0.04 * (1 - q) ** 2) for q in np.linspace(0, 1, 14)]
    cv.line(pts, 1.0, s * 0.11)                                                                    # index finger
    cv.dot(*R(0, 0), s * 0.05, 1.0)
    for k in range(3):                                                                             # curled fingers
        cv.line([R(-1.02, 0.14 + 0.1 * k), R(-0.8, 0.2 + 0.1 * k), R(-0.86, 0.3 + 0.1 * k)], 1.0, s * 0.1)
    cv.line([R(-1.25, 0.38), R(-0.95, 0.36)], 1.0, s * 0.1)                                        # thumb


@segment("love")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    u = ease(ctx.shot_u)
    cx, cy = W / 2, H * 0.47
    s = H * 0.3
    warm = np.array([1.0, 0.72, 0.45], np.float32)
    cv = Canvas(W, H, ctx.ss)
    if ctx.shot == 0:
        gap = H * (0.09 - 0.07 * u)                        # closes, but never to zero
        reaching_hand(cv, cx - gap / 2, cy + H * 0.01, s, -1, -0.12)
        reaching_hand(cv, cx + gap / 2, cy - H * 0.01, s, 1, 0.1)
        glow_x, glow = cx, 0.55 + 0.45 * u
    else:
        reaching_hand(cv, cx - H * 0.02, cy + H * 0.01, s, -1, -0.12)
        glow_x, glow = cx + H * 0.12, 0.35 * (1 - 0.6 * u)  # where the other hand was
    d = np.sqrt((x - glow_x) ** 2 + (y - cy) ** 2) / H
    bg = np.array([0.08, 0.05, 0.05]) + warm * (glow * (0.5 * np.exp(-d / 0.12) + 0.2 * np.exp(-d / 0.45)))[..., None]
    m = cv.array()
    rim = np.clip(blur(m, 3 * sc, 1) - m, 0, 1) * 2
    img = bg * (1 - m[..., None]) + np.array([0.025, 0.018, 0.016]) * m[..., None] + warm * (rim * glow * 0.6)[..., None]
    return np.clip(img, 0, 1)
