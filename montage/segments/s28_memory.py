"""28 令人目眩的回憶 — dizzying memory. A dolly zoom: the camera pulls back while
the lens narrows, so the figure in front keeps exactly the same size while the
colonnade and chequered floor behind it are stretched and pulled.
The colonnade's arches echo the viaduct, the floor the chessboard.

dolly(u) is shared with the soundtrack (pitch and filter follow the pull).
"""
import numpy as np

from . import segment
from ..draw import pal
from ..fx import Canvas, blur, resize, smoothstep
from ..geom import Camera

EYE = 1.55


def dolly(u):
    """0..1 progress of the pull: slow start, rush in the middle, a held last half-second."""
    return float(smoothstep(0.08, 0.9, u))


def camera(ctx, u):
    p = ctx.params
    d0, d1 = float(p.get("distance_from", 4.0)), float(p.get("distance_to", 20.0))
    fov0 = np.radians(float(p.get("fov_from", 82.0)))
    k = dolly(u)
    d = d0 + (d1 - d0) * k
    K = np.tan(fov0 / 2) * d0                           # half-width of view at the subject: constant
    fov = 2 * np.degrees(np.arctan(K / d))
    pos = np.array([0.0, -d, EYE])
    return Camera(pos, (0.0, 0.0, 1.25), fov, ctx.W, ctx.H), fov


def figure_poly():
    """A standing figure in the plane y = 0 (x, z), metres."""
    head = [(0.11 * np.cos(a), 1.62 + 0.13 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 20)]
    body = [(-0.2, 1.45), (0.2, 1.45), (0.24, 0.9), (0.15, 0.9), (0.13, 0.0), (0.03, 0.0), (0.0, 0.8),
            (-0.03, 0.0), (-0.13, 0.0), (-0.15, 0.9), (-0.24, 0.9)]
    return head, body


@segment("memory")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    u = ctx.u
    cam, fov = camera(ctx, u)
    sky, fa, fb = pal(ctx, "sky", "#f0b27a"), pal(ctx, "floor_a", "#d8c3a0"), pal(ctx, "floor_b", "#5a3f33")
    colc, figc = pal(ctx, "column", "#b07d62"), pal(ctx, "figure", "#1a1012")

    # sky: warm, brightest at the vanishing point (the end of the colonnade)
    vx, vy, _ = cam.project(np.array([0.0, 1e6, 0.0]))
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dvp = np.sqrt((x - vx) ** 2 + (y - vy) ** 2) / H
    img = sky * (0.45 + 0.75 * np.exp(-dvp / 0.35))[..., None] + np.array([1.0, 0.95, 0.85]) * (0.6 * np.exp(-dvp / 0.06))[..., None]

    # floor: chessboard tiles, fogging into the sky with distance
    X, Y, dist, hit = cam.ground(W, H)
    raw = ((np.floor(X) + np.floor(Y)) % 2)
    footprint = dist / cam.focal * 1.5 / np.maximum(np.abs(cam.pos[2] / np.maximum(dist, 1e-3)), 0.02)
    chk = (0.5 + (raw - 0.5) * np.clip(1 - footprint / 0.7, 0, 1))[..., None]      # fade to grey when tiles get sub-pixel
    floor = fa * chk + fb * (1 - chk)
    fogd = np.clip(np.hypot(X, Y) / 120.0, 0, 1)[..., None]
    floor = floor * (1 - fogd) + sky * 0.9 * fogd
    floor = np.where(hit[..., None], floor, sky)
    below = (y > vy)[..., None]
    img = np.where(below, floor, img)

    # colonnade: two rows of columns joined by arches, drawn far to near
    lit = Canvas(W, H, ctx.ss)        # coverage of the sunlit half of each column (and arches)
    dark = Canvas(W, H, ctx.ss)       # coverage of the shadowed half
    fogc = Canvas(W, H, ctx.ss)       # fog amount, painted with the same shapes
    ys = np.arange(1.5, 160, 3.0)
    for yc in ys[::-1]:
        for xc in (-3.2, 3.2):
            fog = min(1.0, yc / 140.0)
            for half, cv in ((-1, lit), (1, dark)):
                xa, xb = (xc - 0.28, xc) if half < 0 else (xc, xc + 0.28)
                q = np.array([(xa, yc, 0), (xb, yc, 0), (xb, yc, 5.0), (xa, yc, 5.0)])
                qx, qy, z = cam.project(q)
                if np.any(z < 0.2):
                    continue
                pts = list(zip(qx, qy))
                cv.polygon(pts, 1.0)
                (dark if cv is lit else lit).polygon(pts, 0.0)
                fogc.polygon(pts, fog)
            th = np.linspace(0, np.pi, 14)
            arc = np.stack([np.full(14, xc), yc + 1.5 + 1.5 * np.cos(th), 5.0 + 1.5 * np.sin(th)], 1)
            ax, ay, az = cam.project(arc)
            if np.all(az > 0.2):
                wdt = max(1.0, 0.16 * cam.focal / az.mean())
                lit.line(list(zip(ax, ay)), 1.0, wdt)
                dark.line(list(zip(ax, ay)), 0.0, wdt)
                fogc.line(list(zip(ax, ay)), fog, wdt)
    lm, dm, fg = lit.array()[..., None], dark.array()[..., None], fogc.array()[..., None]
    cover = np.clip(lm + dm, 0, 1)
    colour = (colc * 1.25 * lm + colc * 0.45 * dm) / np.maximum(cover, 1e-3)
    colour = colour * (1 - fg) + sky * fg
    img = img * (1 - cover) + colour * cover

    # the figure: same size on screen throughout (that is the dolly zoom)
    fig = Canvas(W, H, ctx.ss)
    head, body = figure_poly()
    for poly in (head, body):
        P = np.array([(px, 0.0, pz) for px, pz in poly])
        sx, sy, _ = cam.project(P)
        fig.polygon(list(zip(sx, sy)), 1.0)
    fm = fig.array()
    rim = np.clip(blur(fm, 4 * sc, 1) - fm, 0, 1) * 2.2
    img = img * (1 - fm[..., None]) + figc * fm[..., None] + sky * rim[..., None] * 0.6

    # vertigo: radial smear and colour fringing grow with the speed of the pull
    k = dolly(min(1, u + 0.02)) - dolly(max(0, u - 0.02))
    amt = np.clip(k * 9, 0, 1)
    if amt > 0.02:
        acc = img.copy()
        for i, z in enumerate((1.012, 1.025, 1.04)):
            zz = 1 + (z - 1) * amt
            cw, ch = W / zz, H / zz
            box = ((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)
            from PIL import Image
            acc = acc + np.stack([np.asarray(Image.fromarray(img[..., c]).resize((W, H), Image.BILINEAR, box=box))
                                  for c in range(3)], -1)
        img = acc / 4
        shift = int(round(6 * sc * amt))
        if shift:
            img[..., 0] = np.roll(img[..., 0], shift, axis=1)
            img[..., 2] = np.roll(img[..., 2], -shift, axis=1)
    return np.clip(img, 0, 1)
