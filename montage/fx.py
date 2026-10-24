"""Image utilities and the film look (grain, vignette, flicker, dust)."""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------
# small maths
# --------------------------------------------------------------------------
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def resize(arr, W, H, resample=Image.BILINEAR):
    """Resize a float32 HxW or HxWxC array."""
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:
        return np.array(Image.fromarray(arr).resize((W, H), resample), np.float32)
    return np.stack([resize(arr[..., c], W, H, resample) for c in range(arr.shape[2])], -1)


def box_blur(a, r):
    """Separable box blur with radius r (pixels), edges clamped. a: HxW float."""
    r = int(r)
    if r < 1:
        return a
    out = a
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float64)
        if axis == 0:
            out = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
        else:
            out = (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)
    return out.astype(np.float32)


def blur(a, radius, down=4):
    """Cheap wide gaussian-ish blur: downsample, triple box blur, upsample."""
    H, W = a.shape[:2]
    if a.ndim == 3:
        return np.stack([blur(a[..., c], radius, down) for c in range(a.shape[2])], -1)
    w, h = max(1, W // down), max(1, H // down)
    s = resize(a, w, h, Image.BOX)
    rr = max(1, int(round(radius / down / 1.7)))
    for _ in range(3):
        s = box_blur(s, rr)
    return resize(s, W, H, Image.BILINEAR)


def glow(a, radius, strength, down=4):
    return a + strength * blur(a, radius, down)


# --------------------------------------------------------------------------
# supersampled vector canvas
# --------------------------------------------------------------------------
class Canvas:
    """Draw antialiased shapes by rendering at ss x resolution then box-filtering.

    Coordinates are given in output pixels; values in 0..1.
    """

    def __init__(self, W, H, ss=2):
        self.W, self.H, self.ss = W, H, ss
        self.img = Image.new("L", (W * ss, H * ss), 0)
        self.d = ImageDraw.Draw(self.img)

    def _pts(self, pts):
        s = self.ss
        return [(float(x) * s, float(y) * s) for x, y in pts]

    def polygon(self, pts, value):
        if len(pts) >= 3:
            self.d.polygon(self._pts(pts), fill=int(np.clip(value, 0, 1) * 255 + 0.5))

    def line(self, pts, value, width=1.0):
        if len(pts) >= 2:
            w = max(1, int(round(width * self.ss)))
            self.d.line(self._pts(pts), fill=int(np.clip(value, 0, 1) * 255 + 0.5), width=w,
                        joint="curve" if w > 2 else None)

    def lines(self, segs, value, width=1.0):
        """segs: array (N, 2, 2) of independent segments."""
        w = max(1, int(round(width * self.ss)))
        f = int(np.clip(value, 0, 1) * 255 + 0.5)
        s = self.ss
        d = self.d
        for (x0, y0), (x1, y1) in np.asarray(segs, np.float64) * s:
            d.line((x0, y0, x1, y1), fill=f, width=w)

    def dot(self, x, y, r, value):
        s = self.ss
        d = self.d
        d.ellipse((x * s - r * s, y * s - r * s, x * s + r * s, y * s + r * s),
                  fill=int(np.clip(value, 0, 1) * 255 + 0.5))

    def array(self):
        img = self.img.resize((self.W, self.H), Image.BOX) if self.ss > 1 else self.img
        return np.asarray(img, np.float32) / 255.0


# --------------------------------------------------------------------------
# film look
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def vignette_mask(W, H, strength):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (x / (W - 1) * 2 - 1) * (W / H) / (16 / 9)
    ny = y / (H - 1) * 2 - 1
    r = np.sqrt(nx * nx * 0.85 + ny * ny * 1.1)
    m = 1.0 - strength * smoothstep(0.45, 1.45, r) ** 1.2
    return m.astype(np.float32)


def film_grain(rng, W, H, size):
    gw, gh = max(2, int(W / size)), max(2, int(H / size))
    n = rng.standard_normal((gh, gw)).astype(np.float32)
    return resize(n, W, H, Image.BILINEAR) * 1.25


def dust(rng, W, H, amount, scale):
    """A few specks and the rare hair, as an additive (+/-) layer."""
    if amount <= 0:
        return None
    cv_w = Canvas(W, H, 1)
    cv_b = Canvas(W, H, 1)
    n = rng.poisson(amount * 3)
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(0.6, 2.2) * scale
        (cv_w if rng.random() < 0.6 else cv_b).dot(x, y, r, rng.uniform(0.3, 0.9))
    if rng.random() < amount * 0.12:             # hair / scratch
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        pts = [(x0, y0)]
        ang = rng.uniform(0, np.pi * 2)
        for _ in range(12):
            ang += rng.normal(0, 0.35)
            x0 += np.cos(ang) * 9 * scale
            y0 += np.sin(ang) * 9 * scale
            pts.append((x0, y0))
        cv_b.line(pts, 0.7, 1.2 * scale)
    if rng.random() < amount * 0.2:              # vertical scratch for a few frames
        x = rng.uniform(0.1, 0.9) * W
        cv_w.line([(x, 0), (x + rng.normal(0, 3), H)], 0.25, 1.0 * scale)
    return cv_w.array() - cv_b.array()


def finish(img, look_kind, look, rng, scale):
    """Apply the film treatment. img: HxW (gray) or HxWx3, floats ~0..1. Returns uint8 HxWx3."""
    if img.ndim == 2:
        tint = np.array(look.get("bw_tint", [1, 1, 1]), np.float32)
        img = img[..., None] * tint
    H, W = img.shape[:2]
    lum = img.mean(axis=2)
    flick = 1.0 + look.get("flicker", 0.0) * rng.standard_normal()
    img = img * flick * vignette_mask(W, H, float(look.get("vignette", 0.0)))[..., None]
    g = look.get("grain", 0.0)
    if g > 0:
        grain = film_grain(rng, W, H, look.get("grain_size", 1.6) * scale)
        weight = 0.35 + 2.6 * np.clip(lum, 0, 1) * np.clip(1 - lum, 0, 1)
        img = img + (g * grain * weight)[..., None]
    dd = dust(rng, W, H, look.get("dust", 0.0), scale)
    if dd is not None:
        img = img + dd[..., None]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def hblur(a, r):
    """Horizontal-only box blur (motion smear), radius r in pixels. a: HxW."""
    r = int(r)
    if r < 1:
        return a
    p = np.pad(a, [(0, 0), (r + 1, r)], mode="edge")
    c = np.cumsum(p, axis=1, dtype=np.float64)
    return ((c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)).astype(np.float32)


def rotate(a, deg):
    """Rotate a float HxW image about its centre (edges filled from the border)."""
    if abs(deg) < 1e-3:
        return a
    H, W = a.shape
    im = Image.fromarray(np.asarray(a, np.float32)).rotate(deg, resample=Image.BILINEAR, expand=False)
    # zoom in just enough that no empty corner shows
    r = np.radians(abs(deg))
    k = np.cos(r) + max(W / H, H / W) * np.sin(r)
    cw, ch = W / k, H / k
    box = ((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)
    return np.asarray(im.resize((W, H), Image.BILINEAR, box=box), np.float32)
