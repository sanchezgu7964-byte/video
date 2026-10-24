"""10 紙牌 — slower. Three layers of the card:
  1. symbol      — a single card, its suit drawn large, turning slowly
  2. divination  — a spread of ten cards dealt one by one
  3. gambling    — a card table seen from above, a hand on each side

table_scene(ctx, m) is shared with 11 棋盤: m = 0 is the card table, m = 1 the
chessboard, and in between the felt tightens into the board (the match cut).
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from . import segment
from ..draw import CanvasRGB, hand, over, pal, rim
from ..fx import Canvas, blur, ease, hex_rgb, resize, smoothstep
from ..geom import Camera

# --------------------------------------------------------------------------
# symbols (unit coordinates, y down)
# --------------------------------------------------------------------------
def heart_pts(n=80):
    t = np.linspace(0, 2 * np.pi, n)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return np.stack([x / 17, (y + 2) / 17], 1)


def symbol_polys(kind):
    """List of (polygon, 'fg'|'bg') in unit coords."""
    if kind == "diamond":
        return [(np.array([(0, -1), (0.68, 0), (0, 1), (-0.68, 0)]), "fg")]
    if kind == "heart":
        return [(heart_pts(), "fg")]
    if kind == "spade":
        h = heart_pts() * [1, -1] + [0, -0.12]
        stem = np.array([(0, 0.15), (0.32, 0.95), (-0.32, 0.95)])
        return [(h, "fg"), (stem, "fg")]
    if kind == "club":
        out = []
        for cx, cy in ((0, -0.45), (-0.45, 0.12), (0.45, 0.12)):
            t = np.linspace(0, 2 * np.pi, 40)
            out.append((np.stack([cx + 0.42 * np.cos(t), cy + 0.42 * np.sin(t)], 1), "fg"))
        out.append((np.array([(0, -0.1), (0.3, 0.95), (-0.3, 0.95)]), "fg"))
        return out
    if kind == "sun":
        t = np.linspace(0, 2 * np.pi, 48)
        rays = []
        for k in range(12):
            a = k * np.pi / 6
            rays.append((np.array([(0.55 * np.cos(a - 0.12), 0.55 * np.sin(a - 0.12)),
                                   (0.95 * np.cos(a), 0.95 * np.sin(a)),
                                   (0.55 * np.cos(a + 0.12), 0.55 * np.sin(a + 0.12))]), "fg"))
        return rays + [(np.stack([0.5 * np.cos(t), 0.5 * np.sin(t)], 1), "fg")]
    if kind == "moon":
        t = np.linspace(0, 2 * np.pi, 48)
        return [(np.stack([0.7 * np.cos(t), 0.7 * np.sin(t)], 1), "fg"),
                (np.stack([0.28 + 0.62 * np.cos(t), -0.12 + 0.62 * np.sin(t)], 1), "bg")]
    if kind == "tower":
        return [(np.array([(-0.35, 0.95), (-0.3, -0.55), (0.3, -0.55), (0.35, 0.95)]), "fg"),
                (np.array([(-0.38, -0.55), (-0.38, -0.8), (-0.22, -0.8), (-0.22, -0.68), (-0.08, -0.68),
                           (-0.08, -0.8), (0.08, -0.8), (0.08, -0.68), (0.22, -0.68), (0.22, -0.8),
                           (0.38, -0.8), (0.38, -0.55)]), "fg"),
                (np.array([(0.6, -1.0), (0.1, -0.2), (0.35, -0.25), (-0.05, 0.4), (0.5, -0.4), (0.25, -0.35)]), "red")]
    if kind == "star":
        a = np.arange(16) * np.pi / 8
        r = np.where(np.arange(16) % 2 == 0, 0.95, 0.38)
        return [(np.stack([r * np.sin(a), -r * np.cos(a)], 1), "fg")]
    if kind == "wheel":
        t = np.linspace(0, 2 * np.pi, 48)
        out = [(np.stack([0.85 * np.cos(t), 0.85 * np.sin(t)], 1), "fg"),
               (np.stack([0.7 * np.cos(t), 0.7 * np.sin(t)], 1), "bg")]
        for k in range(8):
            a = k * np.pi / 4
            c, s = np.cos(a), np.sin(a)
            out.append((np.array([(-0.05 * s, 0.05 * c), (0.72 * c - 0.05 * s, 0.72 * s + 0.05 * c),
                                  (0.72 * c + 0.05 * s, 0.72 * s - 0.05 * c), (0.05 * s, -0.05 * c)]), "fg"))
        return out
    return []


def card_image(ctx, w, h, face, kind=None, rank=None):
    """RGBA card. face: 'suit' | 'tarot' | 'back'."""
    S = 2
    W_, H_ = int(w * S), int(h * S)
    card, red, ink = pal(ctx, "card", "#efe9dc"), pal(ctx, "red", "#b3202a"), pal(ctx, "ink", "#111111")
    c255 = lambda c: tuple(int(np.clip(v, 0, 1) * 255) for v in c) + (255,)
    img = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(0.07 * W_)
    d.rounded_rectangle([0, 0, W_ - 1, H_ - 1], r, fill=c255(card), outline=c255(card * 0.6), width=max(1, S))
    if face == "back":
        d.rounded_rectangle([int(0.08 * W_), int(0.06 * H_), int(0.92 * W_), int(0.94 * H_)], r // 2, fill=c255(red * 0.8))
        step = W_ / 7
        for k in range(-10, 20):
            x0 = k * step
            d.line([(x0, 0.06 * H_), (x0 + 0.88 * H_, 0.94 * H_)], fill=c255(card * 0.85), width=max(1, S))
            d.line([(x0 + 0.88 * H_, 0.06 * H_), (x0, 0.94 * H_)], fill=c255(card * 0.85), width=max(1, S))
        mask = Image.new("L", (W_, H_), 0)
        ImageDraw.Draw(mask).rounded_rectangle([int(0.08 * W_), int(0.06 * H_), int(0.92 * W_), int(0.94 * H_)], r // 2, fill=255)
        base = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
        ImageDraw.Draw(base).rounded_rectangle([0, 0, W_ - 1, H_ - 1], r, fill=c255(card), outline=c255(card * 0.6), width=max(1, S))
        base.paste(img, (0, 0), mask)
        img = base
    else:
        col = red if kind in ("heart", "diamond", "sun") else ink
        size = 0.36 if face == "suit" else 0.33
        for poly, which in symbol_polys(kind):
            c = {"fg": col, "bg": card, "red": red}[which]
            pts = [(W_ / 2 + x * W_ * size, H_ / 2 + y * W_ * size) for x, y in poly]
            d.polygon(pts, fill=c255(c))
        if face == "tarot":
            d.rectangle([int(0.08 * W_), int(0.06 * H_), int(0.92 * W_), int(0.94 * H_)], outline=c255(ink), width=2 * S)
        if rank:
            for (x, y), rot in (((0.13, 0.1), False), ((0.87, 0.9), True)):
                d.text((x * W_, y * H_), rank, fill=c255(col), anchor="mm",
                       font=_font(int(0.13 * W_)))
                for poly, which in symbol_polys(kind)[:2]:
                    pts = [(x * W_ + px * W_ * 0.05, (y + (0.07 if not rot else -0.07)) * H_ + py * W_ * 0.05)
                           for px, py in poly]
                    d.polygon(pts, fill=c255(col))
    return img.resize((int(w), int(h)), Image.LANCZOS)


@lru_cache(maxsize=8)
def _font(size):
    from ..contact import font
    return font(size)


def felt_background(ctx, spot=(0.5, 0.5), strength=1.0):
    H, W = ctx.H, ctx.W
    felt = pal(ctx, "felt", "#0f3b2a")
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((x - W * spot[0]) ** 2 + (y - H * spot[1]) ** 2) / H
    return felt * (0.25 + 0.95 * strength * np.exp(-d / 0.5))[..., None]


def paste(bg_arr, card, cx, cy, angle, alpha=1.0):
    """Paste a PIL RGBA card onto a float RGB array, rotated about its centre, with a soft shadow."""
    H, W = bg_arr.shape[:2]
    rc = card.rotate(angle, resample=Image.BICUBIC, expand=True)
    a = np.asarray(rc, np.float32) / 255.0
    h, w = a.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return bg_arr
    sub = a[ya - y0:yb - y0, xa - x0:xb - x0]
    m = sub[..., 3:4] * alpha
    # shadow
    sh = np.zeros((H, W), np.float32)
    sx, sy = int(0.012 * H), int(0.018 * H)
    ya2, yb2, xa2, xb2 = min(H, ya + sy), min(H, yb + sy), min(W, xa + sx), min(W, xb + sx)
    sh[ya2:yb2, xa2:xb2] = sub[:yb2 - ya2, :xb2 - xa2, 3] * alpha
    bg_arr = bg_arr * (1 - 0.55 * blur(sh, 10, 2)[..., None])
    bg_arr[ya:yb, xa:xb] = bg_arr[ya:yb, xa:xb] * (1 - m) + sub[..., :3] * m
    return bg_arr


# --------------------------------------------------------------------------
# shot 1: the symbol
# --------------------------------------------------------------------------
def shot_symbol(ctx):
    H, W = ctx.H, ctx.W
    img = felt_background(ctx, (0.5, 0.45), 0.9)
    s = 1.0 + 0.08 * ctx.shot_u
    card = card_image(ctx, H * 0.5 * s, H * 0.7 * s, "suit", "spade", "A")
    img = paste(img, card, W / 2, H * 0.5, -6 + 10 * ctx.shot_u)
    return img


# --------------------------------------------------------------------------
# shot 2: divination — a ten-card spread dealt in order
# --------------------------------------------------------------------------
SPREAD = [  # (x, y, angle, face, kind) in units of card width from frame centre
    (-0.9, 0.0, 0, "tarot", "sun"), (-0.9, 0.0, 90, "back", None),
    (-0.9, -1.25, 0, "tarot", "star"), (-0.9, 1.25, 0, "back", None),
    (-2.1, 0.0, 0, "tarot", "moon"), (0.3, 0.0, 0, "back", None),
    (1.9, 1.5, 0, "back", None), (1.9, 0.5, 0, "tarot", "wheel"),
    (1.9, -0.5, 0, "back", None), (1.9, -1.5, 0, "tarot", "tower"),
]


def shot_spread(ctx):
    H, W = ctx.H, ctx.W
    img = felt_background(ctx, (0.45, 0.5), 0.8)
    cw = H * 0.17
    n = len(SPREAD)
    prog = ctx.shot_u * 1.15 * n
    for k, (x, y, ang, face, kind) in enumerate(SPREAD):
        f = np.clip(prog - k, 0, 1)
        if f <= 0:
            continue
        e = ease(f)
        cx = W / 2 + x * cw * (0.92) + (1 - e) * W * 0.25
        cy = H / 2 + y * cw * 1.02 * 0.92 - (1 - e) * H * 0.3
        card = card_image(ctx, cw * 0.88, cw * 1.22, face, kind)
        img = paste(img, card, cx, cy, ang + (1 - e) * 25, min(1.0, f * 2))
    return img


# --------------------------------------------------------------------------
# shot 3 / chess: the table seen from above, a hand on each side
# --------------------------------------------------------------------------
def table_camera(W, H, t):
    pos = np.array([0.0, -88.0 + 3.0 * np.sin(t * 0.3), 96.0])
    return Camera(pos, (0.0, 4.0, 0.0), 58.0, W, H)


PIECES = [  # (file, rank, type, side) on a 8x8 board, files/ranks 0..7
    (4, 0, "king", 0), (3, 0, "queen", 0), (0, 1, "pawn", 0), (1, 1, "pawn", 0), (5, 1, "pawn", 0),
    (6, 2, "knight", 0), (2, 3, "bishop", 0), (4, 3, "pawn", 0), (7, 0, "rook", 0),
    (4, 7, "king", 1), (2, 7, "rook", 1), (3, 4, "pawn", 1), (5, 6, "pawn", 1), (6, 6, "pawn", 1),
    (1, 5, "knight", 1), (5, 4, "bishop", 1), (7, 6, "pawn", 1), (3, 6, "queen", 1),
]
HEIGHTS = {"pawn": 4.2, "rook": 5.2, "knight": 5.8, "bishop": 6.4, "queen": 7.4, "king": 8.2}


def piece_outline(kind, h):
    """Side profile as polygon in (x, z) units of cm, symmetric about x=0."""
    r = 1.6
    prof = {"pawn": [(r, 0), (r, 0.5), (0.9, 1.0), (0.6, 2.4), (1.0, 2.7), (1.05, 3.3), (0.85, 3.9), (0.4, 4.2), (0, 4.2)],
            "rook": [(r, 0), (r, 0.5), (1.1, 1.0), (1.0, 4.2), (1.35, 4.3), (1.35, 5.2), (0, 5.2)],
            "bishop": [(r, 0), (r, 0.5), (0.9, 1.0), (0.55, 4.0), (1.0, 4.3), (0.95, 5.4), (0.5, 6.0), (0.2, 6.4), (0, 6.4)],
            "queen": [(r, 0), (r, 0.5), (1.0, 1.0), (0.6, 5.2), (1.35, 6.0), (1.1, 7.0), (0.5, 7.1), (0.3, 7.4), (0, 7.4)],
            "king": [(r, 0), (r, 0.5), (1.0, 1.0), (0.65, 5.6), (1.25, 6.2), (0.95, 7.0), (0.25, 7.1), (0.25, 7.6),
                     (0.6, 7.6), (0.6, 7.85), (0.25, 7.85), (0.25, 8.2), (0, 8.2)],
            "knight": [(r, 0), (r, 0.5), (1.0, 1.0), (1.1, 3.0), (1.4, 4.4), (1.1, 5.4), (0.2, 5.8), (-0.6, 5.4),
                       (-1.5, 4.8), (-1.4, 4.2), (-0.4, 4.1), (-1.0, 2.6), (-1.0, 1.0), (-r, 0.5), (-r, 0)]}[kind]
    if kind == "knight":
        return np.array(prof)
    right = np.array(prof)
    left = right[::-1] * [-1, 1]
    return np.vstack([right, left])


def table_scene(ctx, m, t):
    """m: 0 = card table, 1 = chessboard."""
    H, W, sc = ctx.H, ctx.W, ctx.scale
    cam = table_camera(W, H, t)
    merged = {**ctx.tl.segment("cards").palette, **ctx.palette}      # chess reuses the card colours
    pal = lambda _ctx, key, default: hex_rgb(merged.get(key, default))
    felt, lightc, darkc = pal(ctx, "felt", "#0f3b2a"), pal(ctx, "light", "#d9d2c0"), pal(ctx, "dark", "#2a2622")
    gw, gh = W // 2, H // 2
    X, Y, dist, hit = cam.ground(gw, gh)
    # felt superellipse that tightens into the square board
    a = 70 + (32 - 70) * m
    b = 46 + (32 - 46) * m
    p = 2 + 14 * m ** 1.5
    se = np.abs(X / a) ** p + np.abs(Y / b) ** p
    inside = np.clip((1 - se) * 18, 0, 1)
    pool = np.exp(-(X ** 2 + (Y - 2) ** 2) / (2 * 45.0 ** 2))
    wood = np.array([0.09, 0.05, 0.03]) * (0.4 + 0.8 * pool)[..., None]
    surface = felt * (0.3 + 1.1 * pool)[..., None]
    sq = 7.5
    chk = ((np.floor(X / sq + 4) + np.floor(Y / sq + 4)) % 2)[..., None]
    on_board = ((np.abs(X) < 4 * sq) & (np.abs(Y) < 4 * sq))[..., None]
    board = (lightc * chk + darkc * (1 - chk)) * (0.45 + 0.75 * pool)[..., None]
    reveal = smoothstep(0.35, 0.9, m)
    surface = np.where(on_board, surface * (1 - reveal) + board * reveal, surface)
    frame_col = np.array([0.18, 0.1, 0.05]) * (0.5 + 0.8 * pool)[..., None]
    rim_band = (inside > 0) & (se > (0.82 if m < 0.5 else 0.9))
    surface = np.where(rim_band[..., None], frame_col * (0.6 + 0.4 * (1 - m)) + surface * 0.2, surface)
    img = wood * (1 - inside[..., None]) + surface * inside[..., None]
    img = np.stack([resize(img[..., c], W, H) for c in range(3)], -1)

    # cards on the felt, fading out as the board arrives
    card_a = 1 - smoothstep(0.05, 0.45, m)
    if card_a > 0:
        cv = CanvasRGB(W, H, ctx.ss, (0, 0, 0))
        msk = Canvas(W, H, ctx.ss)
        cards = [(-12 + 7.2 * k, 0.0, 0.0, True) for k in range(5)] + \
                [(-40, -4, 0.3, False), (-37, 3, -0.2, False), (40, -3, -0.25, False), (37, 4, 0.15, False)]
        rng = np.random.default_rng([ctx.tl.seed, 1011])
        for x, y, rot, up in cards:
            c, s = np.cos(rot), np.sin(rot)
            corners = [(x + c * dx - s * dy, y + s * dx + c * dy, 0.05) for dx, dy in
                       ((-3.15, -4.4), (3.15, -4.4), (3.15, 4.4), (-3.15, 4.4))]
            qx, qy, _ = cam.project(np.array(corners))
            q = list(zip(qx, qy))
            col = pal(ctx, "card", "#efe9dc") if up else pal(ctx, "red", "#b3202a") * 0.75
            cv.polygon(q, col)
            msk.polygon(q, 1.0)
            if up:
                kind = rng.choice(["diamond", "heart", "spade", "club"])
                sym_col = pal(ctx, "red", "#b3202a") if kind in ("diamond", "heart") else pal(ctx, "ink", "#111111")
                for poly, which in symbol_polys(kind):
                    P3 = np.array([(x + px * 1.6, y + py * 1.6, 0.06) for px, py in poly])
                    sx, sy, _ = cam.project(P3)
                    cv.polygon(list(zip(sx, sy)), sym_col if which == "fg" else pal(ctx, "card", "#efe9dc"))
        ma = msk.array() * card_a
        img = img * (1 - ma[..., None]) + cv.array() * ma[..., None]

    # chess pieces rising out of the board
    piece_a = smoothstep(0.45, 1.0, m)
    if piece_a > 0:
        cv = CanvasRGB(W, H, ctx.ss, (0, 0, 0))
        msk = Canvas(W, H, ctx.ss)
        items = []
        for f, r, kind, side in PIECES:
            x, y = (f - 3.5) * sq, (r - 3.5) * sq
            items.append((np.linalg.norm(np.array([x, y, 0]) - cam.pos), x, y, kind, side))
        for _, x, y, kind, side in sorted(items, reverse=True):
            prof = piece_outline(kind, HEIGHTS[kind]) * [1.25, 1.25 * piece_a]
            if side == 1 and kind == "knight":
                prof = prof * [-1, 1]
            pts3 = np.array([(x + px, y, pz) for px, pz in prof])
            sx, sy, _ = cam.project(pts3)
            base = cam.project(np.array([(x + 2.0 * np.cos(t_), y + 2.0 * np.sin(t_), 0) for t_ in np.linspace(0, 2 * np.pi, 24)]))
            col = np.array([0.84, 0.8, 0.7]) if side == 0 else np.array([0.07, 0.06, 0.06])
            cv.polygon(list(zip(base[0], base[1])), col * 0.8)
            msk.polygon(list(zip(base[0], base[1])), 1.0)
            cv.polygon(list(zip(sx, sy)), col)
            msk.polygon(list(zip(sx, sy)), 1.0)
            cv.line(list(zip(sx, sy)) + [(sx[0], sy[0])], col * 0.45 if side == 0 else np.array([0.3, 0.28, 0.25]), 1.0 * sc)
            # a highlight on the lamp side
            hl = [(sx[i], sy[i]) for i in range(len(sx) // 2)] if kind != "knight" else []
            if hl:
                cv.line(hl, np.clip(col + 0.25, 0, 1), 1.2 * sc)
        pm = msk.array() * piece_a
        img = img * (1 - pm[..., None]) + cv.array() * pm[..., None]

    # the two hands, constant across the match cut
    hm = Canvas(W, H, ctx.ss)
    reach = smoothstep(0.75, 1.0, t / 4.0) if m > 0.99 else 0.0
    hand(hm, W * (0.1 + 0.012 * np.sin(t * 0.8)) + reach * W * 0.06, H * 0.6, H * 0.15, -1, 0.15)
    hand(hm, W * (0.9 - 0.01 * np.sin(t * 0.6 + 1)), H * 0.5, H * 0.14, 1, 0.3)
    hmask = hm.array()
    img = img * (1 - 0.5 * blur(hmask, 40 * sc)[..., None])            # their shadow on the table
    img = over(img, hmask, np.array([0.02, 0.018, 0.016]))
    img = img + rim(hmask, 3 * sc)[..., None] * np.array([0.22, 0.18, 0.12])
    # lamp vignette
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((x - W / 2) ** 2 + (y - H * 0.5) ** 2) / H
    return img * (0.55 + 0.6 * np.exp(-d / 0.55))[..., None]


SHOTS = [shot_symbol, shot_spread, lambda ctx: table_scene(ctx, 0.0, ctx.shot_t)]


@segment("cards")
def render(ctx):
    img = SHOTS[ctx.shot % len(SHOTS)](ctx)
    return np.clip(img, 0, 1)
