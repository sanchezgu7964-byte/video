"""12 畫廊、天窗 — accelerating again. Picture frames on gallery walls alternate
with skylights throwing shafts of light into a dark hall. The paintings in the
frames are small echoes of earlier motifs: an arch, rails, a sun, a board.
"""
import numpy as np

from . import segment
from ..draw import CanvasRGB, pal
from ..fx import Canvas, blur


def painting(cv, ctx, x0, y0, x1, y1, motif, rng):
    w, h = x1 - x0, y1 - y0
    ground = np.array(rng.choice([[0.12, 0.1, 0.09], [0.55, 0.48, 0.38], [0.2, 0.22, 0.3], [0.45, 0.2, 0.12]]))
    cv.rect(x0, y0, x1, y1, ground)
    ink = np.clip(1 - ground * 0.9, 0.05, 0.95) * np.array([1, 0.95, 0.85])
    if motif == "arch":
        for k in range(4):
            ax = x0 + w * (0.12 + k * 0.22)
            cv.rect(ax, y0 + h * 0.35, ax + w * 0.16, y1 - h * 0.08, ink)
            cv.ellipse(ax + w * 0.08, y0 + h * 0.35, w * 0.08, h * 0.12, ink)
    elif motif == "rails":
        for k in range(7):
            yy = y0 + h * (0.15 + k * 0.11)
            cv.line([(x0 + w * 0.05, yy), (x1 - w * 0.05, yy - h * 0.1)], ink, max(1.0, h * 0.012))
    elif motif == "sun":
        cv.rect(x0, y0 + h * 0.62, x1, y1, ink * 0.4)
        cv.ellipse(x0 + w * 0.5, y0 + h * 0.6, w * 0.16, w * 0.16, np.array([0.95, 0.55, 0.25]))
        cv.rect(x0, y0 + h * 0.62, x1, y1, ground * 0.6)
    elif motif == "board":
        n = 4
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 0:
                    cv.rect(x0 + w * (0.1 + 0.8 * i / n), y0 + h * (0.1 + 0.8 * j / n),
                            x0 + w * (0.1 + 0.8 * (i + 1) / n), y0 + h * (0.1 + 0.8 * (j + 1) / n), ink)
    elif motif == "figure":
        cv.ellipse(x0 + w * 0.5, y0 + h * 0.3, w * 0.12, h * 0.1, ink)
        cv.polygon([(x0 + w * 0.3, y1), (x0 + w * 0.42, y0 + h * 0.42), (x0 + w * 0.58, y0 + h * 0.42), (x0 + w * 0.7, y1)], ink)


def framed(cv, ctx, x0, y0, x1, y1, rng):
    frame = pal(ctx, "frame", "#3b2e22")
    wall = pal(ctx, "wall", "#e6e1d6")
    b = min(x1 - x0, y1 - y0) * rng.uniform(0.06, 0.1)
    cv.rect(x0, y0, x1, y1, frame)
    cv.polygon([(x0, y0), (x1, y0), (x1 - b, y0 + b), (x0 + b, y0 + b)], frame * 1.5)     # bevel in the light
    cv.polygon([(x0, y0), (x0 + b, y0 + b), (x0 + b, y1 - b), (x0, y1)], frame * 1.2)
    m = b * rng.uniform(0.4, 1.6)
    cv.rect(x0 + b, y0 + b, x1 - b, y1 - b, wall * 0.97)                                  # mount
    motif = rng.choice(["arch", "rails", "sun", "board", "figure"])
    painting(cv, ctx, x0 + b + m, y0 + b + m, x1 - b - m, y1 - b - m, motif, rng)


def wall_light(H, W, cx=0.5, top=True):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 0.55 + 0.55 * np.exp(-((y / H) / 0.6) ** 2) if top else np.ones((H, W))
    return (v * (0.8 + 0.25 * np.exp(-((x / W - cx) / 0.4) ** 2)))[..., None]


def shot_salon(ctx, rng, variant):
    """A wall hung edge to edge with frames, the camera sliding along it."""
    W, H = ctx.W, ctx.H
    cv = CanvasRGB(W, H, ctx.ss, pal(ctx, "wall", "#e6e1d6") * 0.72)
    shadow = Canvas(W, H, ctx.ss)
    pan = (ctx.shot_u - 0.5) * W * (0.12 if variant % 2 else -0.12)
    if variant == 2:                                          # one large frame, close
        boxes = [(W * 0.22, H * 0.1, W * 0.78, H * 0.9)]
    else:
        boxes = []
        x = -W * 0.2
        while x < W * 1.2:
            col_w = rng.uniform(0.12, 0.26) * W
            y = H * rng.uniform(0.04, 0.12)
            while y < H * 0.9:
                h = rng.uniform(0.18, 0.42) * H
                w = col_w * rng.uniform(0.7, 1.0)
                if y + h < H * 0.98:
                    boxes.append((x + (col_w - w) / 2, y, x + (col_w + w) / 2, y + h))
                y += h + rng.uniform(0.03, 0.06) * H
            x += col_w + rng.uniform(0.02, 0.05) * W
    for x0, y0, x1, y1 in boxes:
        o = 0.012 * H
        shadow.polygon([(x0 + pan + o, y0 + o * 1.5), (x1 + pan + o, y0 + o * 1.5),
                        (x1 + pan + o, y1 + o * 1.5), (x0 + pan + o, y1 + o * 1.5)], 1.0)
    img = cv.array()
    img = img * (1 - 0.45 * blur(shadow.array(), 14 * ctx.scale)[..., None])
    cv2 = CanvasRGB(W, H, ctx.ss, (0, 0, 0))
    m = Canvas(W, H, ctx.ss)
    for x0, y0, x1, y1 in boxes:
        framed(cv2, ctx, x0 + pan, y0, x1 + pan, y1, rng)
        m.polygon([(x0 + pan, y0), (x1 + pan, y0), (x1 + pan, y1), (x0 + pan, y1)], 1.0)
    ma = m.array()[..., None]
    img = img * (1 - ma) + cv2.array() * ma
    return img * wall_light(H, W)


def shot_skylight(ctx, rng, variant):
    """A dark hall; skylight panes in the ceiling; slanting shafts and bright patches on the floor."""
    W, H, sc = ctx.W, ctx.H, ctx.scale
    beam_c = pal(ctx, "beam", "#fff6dc")
    dark = pal(ctx, "shadow", "#1c1a18")
    img = np.ones((H, W, 3), np.float32) * dark * 0.55
    panes = Canvas(W, H, ctx.ss)
    beams = Canvas(W, H, ctx.ss)
    floor = Canvas(W, H, ctx.ss)
    n = [3, 4, 5, 6][variant % 4]
    slant = [0.22, -0.3, 0.12, -0.18][variant % 4] * W
    drift = (ctx.shot_u - 0.5) * 0.05 * W
    top, bottom = H * 0.1, H * 0.86
    pw = W * 0.7 / n
    for k in range(n):
        x0 = W * 0.15 + k * pw + pw * 0.22 + drift
        x1 = x0 + pw * 0.5
        panes.polygon([(x0, 0), (x1, 0), (x1, top), (x0, top)], 1.0)
        fx0, fx1 = x0 + slant, x1 + slant
        beams.polygon([(x0, top), (x1, top), (fx1 + pw * 0.06, bottom), (fx0 - pw * 0.06, bottom)], 1.0)
        floor.polygon([(fx0 - pw * 0.06, bottom), (fx1 + pw * 0.06, bottom),
                       (fx1 + pw * 0.14, H * 1.02), (fx0 - pw * 0.14, H * 1.02)], 1.0)
        panes.line([((x0 + x1) / 2, 0), ((x0 + x1) / 2, top)], 0.0, 2 * sc)
    yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32) / H
    from ..geom import perlin
    streak = 0.75 + 0.25 * perlin(ctx.tl.seed + 121).noise2(xx_ * 14 - yy_ * 14 * slant / (bottom - top) * H / W, yy_ * 0.8 + ctx.t * 0.2)
    y = np.linspace(0, 1, H)[:, None]
    b = blur(beams.array(), 8 * sc) * (0.42 - 0.28 * (y - 0.1)) * streak
    img = img + beam_c * b[..., None]
    img = img + beam_c * blur(floor.array(), 5 * sc)[..., None] * 0.8
    img = img * (1 - panes.array()[..., None]) + beam_c * 1.1 * panes.array()[..., None]
    # dust motes drifting in the light
    motes = Canvas(W, H, ctx.ss)
    r2 = np.random.default_rng(ctx.seed("motes", ctx.shot))
    for _ in range(260):
        x, yy = r2.uniform(0, W), r2.uniform(top, bottom)
        x += np.sin(ctx.t * 1.3 + yy) * 6 * sc
        yy += ctx.t * 12 * sc
        motes.dot(x, yy, r2.uniform(0.6, 1.6) * sc, r2.uniform(0.4, 1.0))
    img = img + beam_c * (motes.array() * beams.array())[..., None] * 0.8
    return img + 0.15 * blur(img, 40 * sc)


@segment("gallery")
def render(ctx):
    rng = np.random.default_rng(ctx.seed("shot", ctx.shot))
    v = ctx.shot // 2
    img = shot_salon(ctx, rng, v) if ctx.shot % 2 == 0 else shot_skylight(ctx, rng, v)
    return np.clip(img, 0, 1)
