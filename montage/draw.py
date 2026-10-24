"""Shared drawing helpers for the later segments: colour compositing, stars, hands."""
import numpy as np

from .fx import Canvas, blur, hex_rgb, smoothstep


def rgb(gray, tint=(1.0, 1.0, 1.0)):
    return gray[..., None] * np.asarray(tint, np.float32)


def to_rgb(img, tint=(1.0, 1.0, 1.0)):
    return rgb(img, tint) if img.ndim == 2 else img


def over(img, mask, color):
    """Composite a flat colour (or HxWx3 image) through a 0..1 mask."""
    m = mask[..., None]
    return img * (1 - m) + np.asarray(color, np.float32) * m


def luma(img):
    return img @ np.array([0.299, 0.587, 0.114], np.float32) if img.ndim == 3 else img


def desat(img, amount, tint=(1.0, 1.0, 1.0)):
    """amount 0 = untouched colour, 1 = (tinted) black & white."""
    g = rgb(luma(img), tint)
    return img * (1 - amount) + g * amount


def pal(ctx, key, default):
    return hex_rgb(ctx.palette.get(key, default))


def grid(W, H):
    """Pixel-centre coordinates normalised to frame height: x in 0..W/H, y in 0..1 (down)."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    return (x + 0.5) / H, (y + 0.5) / H


def vgrad(H, W, stops):
    """Vertical gradient. stops: list of (y 0..1, rgb)."""
    y = (np.arange(H, dtype=np.float32) + 0.5) / H
    ys = [s[0] for s in stops]
    cols = np.array([s[1] for s in stops], np.float32)
    col = np.stack([np.interp(y, ys, cols[:, c]) for c in range(3)], -1)
    return np.broadcast_to(col[:, None, :], (H, W, 3)).copy()


# --------------------------------------------------------------------------
# stars
# --------------------------------------------------------------------------
def make_stars(rng, n, aspect, density=None):
    """Positions (x in 0..aspect, y in 0..1) and magnitudes 0..1 (1 = brightest)."""
    if density is None:
        pts = np.stack([rng.uniform(0, aspect, n), rng.uniform(0, 1, n)], 1)
    else:                                     # rejection-sample against a density function
        pts = []
        while len(pts) < n:
            c = np.stack([rng.uniform(0, aspect, n), rng.uniform(0, 1, n)], 1)
            keep = rng.uniform(0, 1, n) < density(c[:, 0], c[:, 1])
            pts.extend(c[keep].tolist())
        pts = np.array(pts[:n])
    mag = rng.uniform(0, 1, n) ** 6                     # many faint, few bright
    return pts, mag


def draw_stars(W, H, ss, pts, mag, t=0.0, twinkle=0.25, seed=0, size=1.0, xform=None):
    """Return a gray star layer. xform maps normalised pts -> pixel coords."""
    cv = Canvas(W, H, ss)
    sc = H / 1080.0
    if xform is None:
        xy = pts * H
    else:
        xy = xform(pts)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28, len(pts))
    fr = rng.uniform(1.5, 5.0, len(pts))
    tw = 1 - twinkle * (0.5 + 0.5 * np.sin(t * fr + ph))
    for (x, y), m, k in zip(xy, mag, tw):
        if -5 < x < W + 5 and -5 < y < H + 5:
            cv.dot(x, y, (0.55 + 1.6 * m) * sc * size, (0.25 + 0.75 * m) * k)
    s = cv.array()
    return s + 0.6 * blur(s, 6 * sc) + 0.25 * blur(s, 26 * sc)


# --------------------------------------------------------------------------
# hand silhouette (for the card table / chess board)
# --------------------------------------------------------------------------
def hand(cv, x, y, s, side, curl=0.0, value=1.0):
    """Stylised flat hand entering from the left (side=-1) or right (side=+1) edge.

    (x, y) = centre of the palm in pixels, s = palm size in pixels."""
    d = -side                                      # direction the fingers point
    W = cv.W
    # forearm to the frame edge
    edge = 0 if side < 0 else W
    cv.polygon([(x - d * s * 0.3, y - s * 0.42), (edge - d * s * 0.2, y - s * 0.62),
                (edge - d * s * 0.2, y + s * 0.62), (x - d * s * 0.3, y + s * 0.5)], value)
    cv.d.ellipse([(x - s * 0.55) * cv.ss, (y - s * 0.5) * cv.ss, (x + s * 0.55) * cv.ss,
                  (y + s * 0.55) * cv.ss], fill=int(value * 255))
    lens = [0.78, 0.9, 0.84, 0.66]
    for k, L in enumerate(lens):
        fy = y - s * 0.36 + k * s * 0.24
        L *= s * (1 - 0.45 * curl)
        fx0 = x + d * s * 0.35
        fx1 = fx0 + d * L
        w = s * 0.2
        cv.line([(fx0, fy), (fx1, fy + s * 0.04 * (k - 1.5))], value, w)
        cv.dot(fx1, fy + s * 0.04 * (k - 1.5), w / 2, value)
    # thumb, angled towards the viewer
    tx0, ty0 = x + d * s * 0.15, y + s * 0.42
    tx1, ty1 = tx0 + d * s * 0.5, ty0 + s * 0.28
    cv.line([(tx0, ty0), (tx1, ty1)], value, s * 0.22)
    cv.dot(tx1, ty1, s * 0.11, value)


def rim(mask, px):
    """Thin bright outline on the outside of a silhouette mask."""
    return np.clip(blur(mask, px, 2) - mask, 0, 1) * 2.0


__all__ = ["rgb", "to_rgb", "over", "luma", "desat", "pal", "grid", "vgrad",
           "make_stars", "draw_stars", "hand", "rim", "smoothstep"]


class CanvasRGB:
    """Supersampled RGB canvas for flat graphic shots. Colours are 0..1 floats."""

    def __init__(self, W, H, ss=2, bg=(0, 0, 0)):
        from PIL import Image, ImageDraw
        self.W, self.H, self.ss = W, H, ss
        self.img = Image.new("RGB", (W * ss, H * ss), self._c(bg))
        self.d = ImageDraw.Draw(self.img)

    @staticmethod
    def _c(col):
        return tuple(int(np.clip(c, 0, 1) * 255 + 0.5) for c in col)

    def polygon(self, pts, col):
        s = self.ss
        if len(pts) >= 3:
            self.d.polygon([(float(x) * s, float(y) * s) for x, y in pts], fill=self._c(col))

    def rect(self, x0, y0, x1, y1, col):
        self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], col)

    def line(self, pts, col, width=1.0):
        s = self.ss
        self.d.line([(float(x) * s, float(y) * s) for x, y in pts], fill=self._c(col),
                    width=max(1, int(round(width * s))))

    def ellipse(self, cx, cy, rx, ry, col):
        s = self.ss
        self.d.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], fill=self._c(col))

    def array(self):
        from PIL import Image
        img = self.img.resize((self.W, self.H), Image.BOX) if self.ss > 1 else self.img
        return np.asarray(img, np.float32) / 255.0
