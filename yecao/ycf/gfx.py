"""合成工具：浮點 RGB 畫布、cairo 向量圖層、粒子潑灑、模糊與泛光、底片顆粒。"""
import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


class Layer:
    """一張透明的 cairo 圖層；畫完後以 premultiplied RGBA 浮點陣列取出。"""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        self.ctx = cairo.Context(self.surf)
        self.ctx.set_antialias(cairo.ANTIALIAS_GOOD)

    def clear(self):
        self.ctx.save()
        self.ctx.set_operator(cairo.OPERATOR_CLEAR)
        self.ctx.paint()
        self.ctx.restore()

    def rgba(self) -> np.ndarray:
        self.surf.flush()
        a = np.ndarray((self.H, self.W, 4), np.uint8, self.surf.get_data())
        # cairo 在小端機器上是 BGRA
        return a[..., [2, 1, 0, 3]].astype(np.float32) / 255


def over(dst: np.ndarray, layer_rgba: np.ndarray, opacity: float = 1.0) -> None:
    a = layer_rgba[..., 3:4] * opacity
    dst *= 1 - a
    dst += layer_rgba[..., :3] * opacity


def path_polys(ctx, polys, close=True):
    for p in polys:
        if len(p) < 2:
            continue
        ctx.move_to(float(p[0, 0]), float(p[0, 1]))
        for x, y in p[1:]:
            ctx.line_to(float(x), float(y))
        if close:
            ctx.close_path()


def fill_polys(ctx, polys, rgb, alpha=1.0):
    ctx.new_path()
    path_polys(ctx, polys)
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    ctx.set_source_rgba(float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))
    ctx.fill()


def splat(buf: np.ndarray, xy: np.ndarray, rgb: np.ndarray, w: np.ndarray | float = 1.0,
          bilinear=True) -> None:
    """把粒子以加法潑到浮點畫布上。rgb 為 (N,3) 或 (3,)。"""
    H, W = buf.shape[:2]
    if len(xy) == 0:
        return
    x, y = xy[:, 0], xy[:, 1]
    w = np.broadcast_to(np.asarray(w, np.float32), x.shape)
    rgb = np.broadcast_to(np.asarray(rgb, np.float32), (len(x), 3))
    if bilinear:
        x0, y0 = np.floor(x - 0.5), np.floor(y - 0.5)
        fx, fy = x - 0.5 - x0, y - 0.5 - y0
        parts = [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]
    else:
        x0, y0 = np.floor(x), np.floor(y)
        parts = [(0, 0, np.ones_like(x))]
    flat = buf.reshape(-1, 3)
    for dx, dy, ww in parts:
        xi, yi = (x0 + dx).astype(np.int64), (y0 + dy).astype(np.int64)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = yi[ok] * W + xi[ok]
        wt = (w * ww)[ok]
        for c in range(3):
            flat[:, c] += np.bincount(idx, weights=wt * rgb[ok, c], minlength=W * H)


def resize(img: np.ndarray, W: int, H: int) -> np.ndarray:
    chans = [np.asarray(Image.fromarray(img[..., c].astype(np.float32), "F").resize((W, H), Image.BILINEAR))
             for c in range(img.shape[2])]
    return np.stack(chans, -1)


def bloom(img: np.ndarray, sigma: float, down: int = 4) -> np.ndarray:
    H, W = img.shape[:2]
    small = resize(img, W // down, H // down)
    small = gaussian_filter(small, (sigma / down, sigma / down, 0))
    return resize(small, W, H)


def lin_gradient_v(H, W, stops):
    """垂直漸層。stops: [(y_frac, rgb), ...]"""
    ys = np.linspace(0, 1, H, dtype=np.float32)
    out = np.zeros((H, 3), np.float32)
    fr = np.array([s[0] for s in stops])
    for c in range(3):
        out[:, c] = np.interp(ys, fr, [s[1][c] for s in stops])
    return np.broadcast_to(out[:, None, :], (H, W, 3)).copy()


class Finisher:
    """收尾：暗角、底片顆粒、量化。顆粒以影格編號為種子，保證可重現。"""

    def __init__(self, W, H, seed):
        self.W, self.H, self.seed = W, H, seed
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vig = (1 - 0.32 * np.clip(r - 0.35, 0, None) ** 1.6)[..., None].astype(np.float32)
        rng = np.random.default_rng(seed)
        # 預先產生 16 張低解析顆粒，依影格輪替並平移
        self.grain = [rng.standard_normal((H // 2 + 8, W // 2 + 8)).astype(np.float32) for _ in range(16)]

    def __call__(self, img: np.ndarray, frame: int, grain=0.018) -> np.ndarray:
        img = img * self.vig
        g = self.grain[frame % 16]
        ox, oy = (frame * 7) % 8, (frame * 3) % 8
        g = g[oy:oy + self.H // 2, ox:ox + self.W // 2]
        g = np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
        lum = img.mean(-1, keepdims=True)
        img = img + g * grain * (0.35 + np.sqrt(np.clip(lum, 0, 1)))
        # 有序抖動避免暗部色帶
        return np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** p


def ease_in(x, p=2):
    return np.clip(x, 0, 1) ** p


def ramp(t, t0, t1):
    if t1 <= t0:
        return float(t >= t0)
    return float(np.clip((t - t0) / (t1 - t0), 0, 1))


def mix(a, b, k):
    return a + (b - a) * k
