"""刻版：用刀與墨直接造型的木刻。

上一版是「濾鏡」：先畫明暗，再自動鋪刀痕，所以每一塊都落到差不多的密度。這一版反過來，畫面一筆一筆刻出來：

- 塊面：`ink` 留下大塊的黑，`carve` 把整片刻成白。先定黑白，再談細節。
- 刀線：`knife` 是一刀——兩頭尖、中間有力。眉、眼、鼻翼、嘴角、衣褶都是一刀一刀放上去的。
- 刀組：`engrave` 讓一組線順著一條形體的走向（臉頰的弧、肩的弧、布的褶），寬度隨明暗變化：
  在黑底上刻白線，越近光線越粗，到暗處自己收成刀尖。中間調只來自這種刀組，從不鋪平均的排線。
- 背景：`rules` 是肖像背後那種橫刀，長短、粗細、疏密都跟著明暗走。

座標是 1920×1080 的「版面」；取景（推、移）是 cairo 矩陣，版上的刀線跟著放大，永遠銳利。
`Print` 把版印到紙上：近中性的紙、黑墨、極少的漏白。`plain=True` 時完全不加印痕——
造型必須在純黑白下就站得住。
"""
from contextlib import contextmanager

import cairo
import numpy as np

from .noise import noise3

W0, H0 = 1920, 1080


# ── 幾何 ───────────────────────────────────────────────────
def spline(pts, n=None, closed=False, step=6.0):
    """Catmull-Rom：把幾個控制點變成平順的曲線。n 不給時，依長度每 step 取一點。"""
    P = np.asarray(pts, np.float64)
    if closed:
        P = np.concatenate([P[-1:], P, P[:2]])
    else:
        P = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    segs = len(P) - 3
    if n is None:
        L = np.linalg.norm(np.diff(P[1:-1], axis=0), axis=1).sum()
        n = max(4, int(L / step))
    t = np.linspace(0, segs, n if not closed else n + 1)[: (n if closed else None)]
    i = np.minimum(t.astype(int), segs - 1)
    u = (t - i)[:, None]
    p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
    out = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
    return out.astype(np.float32)


def resample(pts, step=5.0):
    P = np.asarray(pts, np.float64)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return P.astype(np.float32)
    s = np.linspace(0, d[-1], max(2, int(d[-1] / step) + 1))
    return np.stack([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])], 1).astype(np.float32)


def normals(P):
    d = np.gradient(P, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return np.stack([-d[:, 1], d[:, 0]], 1)


def offset(P, dist):
    P = np.asarray(P, np.float32)
    return P + normals(P) * np.asarray(dist, np.float32).reshape(-1, 1)


def xform(pts, ox, oy, s=1.0, rot=0.0, flip=False):
    """局部座標 → 版面：縮放、（可左右翻）、旋轉、平移。"""
    P = np.asarray(pts, np.float32).copy()
    if flip:
        P[:, 0] = -P[:, 0]
    c, si = np.cos(rot), np.sin(rot)
    return np.stack([ox + s * (P[:, 0] * c - P[:, 1] * si), oy + s * (P[:, 0] * si + P[:, 1] * c)], 1)


def ellipse(cx, cy, rx, ry, rot=0.0, n=40):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return xform(np.stack([rx * np.cos(th), ry * np.sin(th)], 1), cx, cy, 1, rot)


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def _h(*k):
    """可重現的 0..1 雜湊。"""
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return (h & 0xFFFFFF) / 0xFFFFFF


def rough(poly, amp=1.4, seed=0, closed=True, step=7.0):
    """刀口的毛：沿邊緣以平順的噪聲微微推進推出。"""
    P = np.asarray(poly, np.float32)
    if closed:
        P = np.concatenate([P, P[:1]])
    P = resample(P, step)
    if closed:
        P = P[:-1]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = (noise3(s / 23.0, seed * 1.7, 0.5, 3) + 0.5 * noise3(s / 6.0, seed * 1.7, 3.5, 4)).astype(np.float32)
    return P + normals(P) * (amp * n)[:, None]


# ── 版 ─────────────────────────────────────────────────────
class Block:
    """一塊版。紙＝1，墨＝0。所有尺寸都是版面單位。"""

    def __init__(self, W=W0, H=H0):
        self.W, self.H = W, H
        self.surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
        self.ctx = cairo.Context(self.surf)
        self.ctx.set_antialias(cairo.ANTIALIAS_GOOD)
        self.ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        self.view = (1.0, W0 / 2, H0 / 2)

    def begin(self, view=(1.0, W0 / 2, H0 / 2), black=False):
        """新的一張。view = (放大倍數, 中心 x, 中心 y)，以版面座標計。"""
        self.view = view
        c = self.ctx
        c.identity_matrix()
        c.reset_clip()
        v = 0.0 if black else 1.0
        c.set_source_rgb(v, v, v)
        c.paint()
        z, cx, cy = view
        c.translate(self.W / 2, self.H / 2)
        c.scale(z * self.W / W0, z * self.W / W0)
        c.translate(-cx, -cy)

    # 塊面
    def _path(self, polys, close=True):
        c = self.ctx
        c.new_path()
        for p in polys:
            p = np.asarray(p)
            if len(p) < 2:
                continue
            c.move_to(float(p[0, 0]), float(p[0, 1]))
            for x, y in p[1:]:
                c.line_to(float(x), float(y))
            if close:
                c.close_path()

    def fill(self, polys, v):
        self._path(polys)
        self.ctx.set_source_rgb(v, v, v)
        self.ctx.fill()

    def ink(self, polys):
        self.fill(polys, 0.0)

    def carve(self, polys):
        self.fill(polys, 1.0)

    @contextmanager
    def clip(self, polys, invert=False):
        c = self.ctx
        c.save()
        self._path(polys)
        if invert:
            c.save()
            c.identity_matrix()
            c.rectangle(0, 0, self.W, self.H)
            c.restore()
            c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        c.clip()
        c.set_fill_rule(cairo.FILL_RULE_WINDING)
        try:
            yield
        finally:
            c.restore()

    # 刀
    def knife(self, pts, w, v=1.0, taper=(0.35, 0.5), smooth=True, power=0.8, seed=0, swell=0.12):
        """一刀。pts 是刀走的路線；w 是最寬處（可以是每一點的陣列）；taper 是兩端收尖所佔的比例。
        v=1 刻白（黑底上的白線），v=0 留黑（白底上的黑線）。"""
        P = spline(pts, step=3.0) if smooth and len(pts) > 2 else resample(pts, 3.0)
        if len(P) < 2:
            return
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        L = s[-1]
        if L < 0.5:
            return
        u = s / L
        a, b = taper
        prof = np.ones_like(u)
        if a > 0:
            prof = np.minimum(prof, np.clip(u / a, 0, 1) ** power)
        if b > 0:
            prof = np.minimum(prof, np.clip((1 - u) / b, 0, 1) ** power)
        ww = np.asarray(w, np.float32)
        if ww.ndim:
            ww = np.interp(u, np.linspace(0, 1, len(ww)), ww)
        # 刀走時手的輕重
        ww = ww * (1 + swell * noise3(u * 3.1, seed * 0.37, 7.7, 2))
        half = 0.5 * np.maximum(ww * prof, 0)
        N = normals(P)
        poly = np.concatenate([P + N * half[:, None], (P - N * half[:, None])[::-1]])
        self.fill([poly], v)

    def engrave(self, curves, wfn, v=1.0, wmax=6.0, wmin=0.5, seed=0, taper=0.12, step=4.0, breaks=0.0, swell=0.06):
        """一組順著形體走的刀線。wfn(x, y) → 0..1 給每一點的寬度比例（乘上 wmax）。
        寬度低於 wmin 的地方，刀線斷開——斷處自然收成刀尖。breaks：隨機斷刀的機率。"""
        items = []
        for C in curves:
            P = resample(C, step)
            if len(P) >= 2:
                items.append((P, np.clip(np.asarray(wfn(P[:, 0], P[:, 1]), np.float32), 0, 1.5) * wmax))
        self.strokes(items, v, wmin, seed, taper, breaks, swell)

    def strokes(self, items, v=1.0, wmin=0.5, seed=0, taper=0.12, breaks=0.0, swell=0.06):
        """items: [(點列, 每點寬度)]。寬度不足 wmin 處斷開，每一段兩頭收尖。"""
        for k, (P, w) in enumerate(items):
            P = np.asarray(P, np.float32)
            w = np.asarray(w, np.float32).copy()
            if breaks > 0:
                s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
                cell = (s / 41).astype(int)
                hit = np.array([_h(seed, k, c) < breaks for c in cell])
                w = np.where(hit & ((s / 41) % 1 < 0.22), 0, w)
            on = w > wmin
            idx = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            for a, b in zip(idx[::2], idx[1::2]):
                if b - a < 2:
                    continue
                seg, ws = P[a:b], w[a:b]
                n = len(seg)
                t = np.linspace(0, 1, n)
                tp = max(taper, 1.5 / n)
                prof = np.minimum(np.clip(t / tp, 0, 1), np.clip((1 - t) / tp, 0, 1)) ** 0.7
                self.knife(seg, ws * prof, v, taper=(0, 0), smooth=False, seed=seed * 131 + k + a, swell=swell)

    def rules(self, region, y0, y1, tone, spacing=13.0, v=1.0, seg=(260, 900), gap=(6, 40), angle=0.0,
              wobble=2.5, seed=0, x0=-100, x1=W0 + 100, wmax=None):
        """背景的橫刀。tone(x, y) → 0（黑）..1（白）。黑底上的白刀越近亮處越粗，到最亮處連成一片白。"""
        wmax = spacing * 0.8 if wmax is None else wmax
        curves = []
        y = y0
        k = 0
        while y < y1:
            sp = spacing * (0.8 + 0.4 * _h(seed, k, 1))
            x = x0 - _h(seed, k, 2) * seg[1]
            j = 0
            while x < x1:
                L = seg[0] + (seg[1] - seg[0]) * _h(seed, k, j, 3)
                xs = np.linspace(x, x + L, max(3, int(L / 8)))
                ys = y + wobble * noise3(xs / 180.0, k * 0.71, seed * 0.13, 2)
                P = np.stack([xs, ys], 1)
                if angle:
                    P = xform(P - (W0 / 2, H0 / 2), W0 / 2, H0 / 2, 1, angle)
                curves.append(P)
                x += L + gap[0] + (gap[1] - gap[0]) * _h(seed, k, j, 4)
                j += 1
            y += sp
            k += 1
        with self.clip(region):
            if v >= 0.5:
                self.engrave(curves, lambda x, y: tone(x, y), 1.0, wmax, wmin=0.45, seed=seed, taper=0.08, swell=0.0)
            else:
                self.engrave(curves, lambda x, y: 1 - tone(x, y), 0.0, wmax, wmin=0.45, seed=seed, taper=0.08, swell=0.0)

    def halo(self, poly, w, seed=0, breaks=0.25, closed=True):
        """形體外緣刻一道白，形體才從黑裡分出來。要在填黑之前呼叫；斷斷續續的，不是描邊。"""
        P = np.asarray(poly, np.float32)
        if closed:
            P = np.concatenate([P, P[:1]])
        P = resample(P, 2.0)
        self.strokes([(P, np.full(len(P), 2 * w, np.float32))], 1.0, wmin=0.3, seed=seed, taper=0.03,
                     breaks=breaks, swell=0.25)

    def text(self, face, s, x, y, size, v=0.0, vertical=False, track=1.0, warp=None):
        """字：v=0 墨字，v=1 刻出的白字。warp(P) 可把字形貼到彎曲的布或紙上。"""
        for j, ch in enumerate(s):
            g = face.glyph(ch)
            if not g.contours:
                continue
            if vertical:
                ox, oy = x - size * g.advance / 2, y + size * (j * track + 0.88)
            else:
                ox, oy = x + size * j * track, y
            polys = [q * size + (ox, oy) for q in g.contours]
            if warp is not None:
                polys = [warp(resample(np.concatenate([q, q[:1]]), max(1.0, size / 60))) for q in polys]
            self.fill(polys, v)

    def mask(self):
        """回傳墨量（H, W），1＝墨。"""
        self.surf.flush()
        a = np.ndarray((self.H, self.W, 4), np.uint8, self.surf.get_data())
        return 1.0 - a[..., 1].astype(np.float32) / 255


# ── 印 ─────────────────────────────────────────────────────
class Print:
    """把版印到紙上。印痕是克制的：近中性的紙，黑墨，大片黑裡極少的漏白，墨的輕重只差幾個百分點。"""

    PAPER = np.array([0.925, 0.915, 0.885], np.float32)
    INK = np.array([0.045, 0.043, 0.042], np.float32)

    def __init__(self, W=W0, H=H0, seed=0):
        rng = np.random.default_rng(seed)
        from scipy.ndimage import gaussian_filter, zoom
        lo = rng.random((H // 24 + 1, W // 24 + 1)).astype(np.float32)
        self.press = zoom(gaussian_filter(lo, 1.2), 24, order=1)[:H, :W]          # 壓力不勻
        sp = rng.random((H // 3 + 1, W // 3 + 1)).astype(np.float32)
        sp = zoom(sp, 3, order=0)[:H, :W]
        self.speck = gaussian_filter(sp, 0.8)                                    # 漏白的小斑
        self.fiber = 1 + 0.012 * (gaussian_filter(rng.random((H, W)).astype(np.float32), [0.6, 3.0]) - 0.5) * 6

    def __call__(self, ink, plain=False):
        if plain:
            v = 1 - ink
            return np.repeat(v[..., None], 3, -1)
        k = ink.copy()
        p = (self.press - self.press.mean()) / (self.press.std() + 1e-6)
        k = k * (0.985 + 0.008 * np.clip(p, -1.5, 1.5) / 1.5)
        k = np.where((ink > 0.98) & (self.speck > 0.958 + 0.01 * p), k * 0.7, k)
        img = self.PAPER[None, None] * self.fiber[..., None] * (1 - k[..., None]) + self.INK * k[..., None]
        return img.astype(np.float32)
