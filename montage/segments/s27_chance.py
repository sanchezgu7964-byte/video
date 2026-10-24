"""27 慷慨的機遇 — generous chance. Two dice tumbling through the air in slow
arcs; at freeze_at the picture stops dead on the last frame.
"""
import numpy as np

from . import segment
from ..draw import CanvasRGB
from ..fx import blur

FACES = [  # (normal, up, right, pips)
    ((0, 0, 1), (0, 1, 0), (1, 0, 0), 1), ((0, 0, -1), (0, 1, 0), (-1, 0, 0), 6),
    ((1, 0, 0), (0, 1, 0), (0, 0, -1), 3), ((-1, 0, 0), (0, 1, 0), (0, 0, 1), 4),
    ((0, 1, 0), (0, 0, -1), (1, 0, 0), 2), ((0, -1, 0), (0, 0, 1), (1, 0, 0), 5)]
PIPS = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)], 4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
        5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)], 6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)]}


def rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def die(cv, cx, cy, size, R, light):
    """Orthographic die: draw the faces turned toward the viewer (+z), back to front."""
    faces = []
    for n, up, rt, pips in FACES:
        n, up, rt = (R @ np.array(v, float) for v in (n, up, rt))
        if n[2] <= 0:
            continue
        faces.append((n[2], n, up, rt, pips))
    for _, n, up, rt, pips in sorted(faces, key=lambda f: f[0]):
        c = n * 0.5
        corners = [c + (a * rt + b * up) * 0.5 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pts = [(cx + p[0] * size, cy - p[1] * size) for p in corners]
        lam = 0.35 + 0.65 * max(0.0, float(n @ light))
        cv.polygon(pts, np.array([0.93, 0.9, 0.84]) * lam)
        for px, py in PIPS[pips]:
            q = c + (px * 0.28 * rt + py * 0.28 * up)
            ex = size * 0.075 * np.sqrt(max(n[2], 0.05))
            cv.ellipse(cx + q[0] * size, cy - q[1] * size, size * 0.075, ex, np.array([0.08, 0.05, 0.05]) * lam + 0.02)


@segment("chance")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    freeze = float(ctx.params.get("freeze_at", 2.3))
    t = min(ctx.t, freeze)
    bg = np.array([0.05, 0.07, 0.06])
    cv = CanvasRGB(W, H, ctx.ss, bg)
    light = np.array([-0.4, 0.6, 0.7])
    light /= np.linalg.norm(light)
    for k, (x0, vx, y0, vy, spin, size) in enumerate(((0.2, 0.2, 0.75, -0.75, (2.1, 3.3, 1.2), 0.17),
                                                      (0.3, 0.17, 0.85, -0.68, (-2.8, 1.7, 2.6), 0.14))):
        x = (x0 + vx * t) * W
        y = (y0 + vy * t + 0.25 * t * t) * H
        R = rot(spin[0] * t + k, spin[1] * t + 2 * k, spin[2] * t)
        die(cv, x, y, size * H, R, light)
    img = cv.array()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    spot = np.exp(-((xx / W - 0.5) ** 2 + (yy / H - 0.45) ** 2) / 0.12)[..., None]
    img = img * (0.55 + 0.6 * spot)
    if ctx.t >= freeze:
        k = (ctx.t - freeze) / max(1e-3, ctx.duration - freeze)
        img = img * (1 + 0.35 * np.exp(-k * 12))              # a small flash as the frame locks
    return np.clip(img + 0.05 * blur(img, 20 * sc), 0, 1)
