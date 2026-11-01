"""最後一幕：在旋轉的分鏡裡，一筆一筆畫出一本《野草》的封面。

深褐的底、兩道細白的地平線、一抹綠地、白的河岸、幾叢金色的野草、直排的書名。
字草的世界在這裡收成一本書：地平線、野草、土色，都是片子裡出現過的東西。

鏡頭：每一個分鏡都在慢慢旋轉、跟著正在畫的那一筆；最後一個拉遠、轉正，停在整個封面上。
（不放出版社的推薦語與標誌；文字只有書名、作者、〈希望〉裡的一句，與 Wild Grass。）
"""
import cairo
import numpy as np

from . import fonts
from .carve import resample, spline
from .gfx import ease, ease_in, ease_out, ramp

PW, PH = 700.0, 1000.0          # 封面（版面像素），置於畫面中央
BG = (0.030, 0.027, 0.025)
PAGE = (0.235, 0.198, 0.158)
WHITE = (0.93, 0.915, 0.88)
GREEN = (0.27, 0.78, 0.31)
GOLD = (0.86, 0.66, 0.33)
SEED = (0.92, 0.56, 0.17)
INKD = (0.20, 0.17, 0.14)
DUR = 16.2

# 分鏡：(開始, 結束, 起始取景, 結束取景)；取景 = (放大, 封面 u, 封面 v, 旋轉弧度)，None 表示跟著筆走
SHOTS = [
    (0.0, 2.6, (3.0, None, 0.676, -0.46), (3.0, None, 0.676, -0.30)),     # 下面那道地平線
    (2.6, 5.0, (2.5, 0.33, 0.72, 0.30), (2.6, 0.34, 0.73, 0.17)),         # 一叢野草，一葉一葉
    (5.0, 7.2, (1.9, None, 0.62, -0.20), (1.9, None, 0.62, -0.09)),       # 綠地被刷開
    (7.2, 9.2, (1.65, 0.36, 0.86, 0.15), (1.7, 0.38, 0.86, 0.05)),        # 白的河岸、Wild Grass
    (9.2, 11.6, (2.3, 0.56, 0.24, -0.11), (2.4, 0.56, 0.24, -0.04)),      # 書名
    (11.6, DUR, (1.4, 0.55, 0.40, 0.08), (1.0, 0.5, 0.5, 0.0)),           # 拉遠、轉正，停住
]


def P(u, v):
    """封面座標 (u, v) ∈ [0,1]² → 版面像素。"""
    return np.stack([960 + (np.asarray(u) - 0.5) * PW, 540 + (np.asarray(v) - 0.5) * PH], -1).astype(np.float32)


LOWER = spline(P(*np.array([(0.06, 0.677), (0.2, 0.668), (0.35, 0.673), (0.5, 0.668), (0.65, 0.673), (0.78, 0.684),
                             (0.88, 0.69), (0.97, 0.694)]).T), step=3)
UPPER = spline(P(*np.array([(0.55, 0.588), (0.66, 0.585), (0.75, 0.579), (0.85, 0.573), (0.93, 0.569),
                            (1.0, 0.567)]).T), step=3)
GREEN_POLY = spline(P(*np.array([(0.28, 0.640), (0.36, 0.622), (0.46, 0.607), (0.56, 0.596), (0.63, 0.589),
                                 (0.75, 0.590), (0.88, 0.592), (1.0, 0.594), (1.0, 0.637), (0.86, 0.634),
                                 (0.7, 0.632), (0.56, 0.633), (0.45, 0.640), (0.36, 0.646)]).T), closed=True, step=3)
RIVER = spline(P(*np.array([(0.0, 0.757), (0.08, 0.758), (0.2, 0.752), (0.3, 0.767), (0.42, 0.773), (0.48, 0.786),
                            (0.37, 0.792), (0.33, 0.803), (0.45, 0.82), (0.56, 0.842), (0.64, 0.872), (0.72, 0.905),
                            (0.78, 0.95), (0.8, 1.0), (0.0, 1.0)]).T), closed=True, step=3)


def _tuft(base, n, L0, L1, lean0, lean1, seed):
    """一叢草：從同一點長出 n 片葉，扇形散開，梢往一側彎。回傳每片葉的中線（版面像素）。"""
    rng = np.random.default_rng(seed)
    bx, by = P(*base)
    out = []
    for k in range(n):
        f = k / max(n - 1, 1)
        L = (L0 + (L1 - L0) * rng.random()) * PH
        lean = lean0 + (lean1 - lean0) * f + rng.normal(0, 0.06)
        curl = rng.uniform(0.15, 0.5) * np.sign(lean - 0.05)
        u = np.linspace(0, 1, 18)
        ang = lean + curl * u ** 1.6
        xs = bx + np.cumsum(np.sin(ang)) * L / 18 + rng.normal(0, 2)
        ys = by - np.cumsum(np.cos(ang)) * L / 18
        out.append((np.stack([xs, ys], 1).astype(np.float32), 0.0065 * PH * rng.uniform(0.8, 1.2)))
    return out


BIG = _tuft((0.325, 0.775), 12, 0.11, 0.2, -0.62, 0.18, 11)
SMALL = _tuft((0.395, 0.805), 5, 0.06, 0.1, -0.4, 0.25, 12)
RIGHT = _tuft((0.745, 0.585), 7, 0.045, 0.08, -0.3, 0.4, 13)


class Cover:
    def __init__(self, W=1920, H=1080, seed=0):
        self.W, self.H = W, H
        self.surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
        self.ctx = cairo.Context(self.surf)
        self.ctx.set_antialias(cairo.ANTIALIAS_GOOD)
        self.title = fonts.face("fonts/serif_black")
        self.small = fonts.face("fonts/serif_black")
        self.latin = fonts.face("fonts/serif_regular")
        rng = np.random.default_rng(seed)
        from scipy.ndimage import gaussian_filter
        g = gaussian_filter(rng.standard_normal((H // 2, W // 2)).astype(np.float32), 0.7)
        self.grain = np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None] * 0.012

    # ── 取景 ──────────────────────────────────────────────
    def pen_x(self, tau):
        """下地平線的筆尖、綠地的刷頭，在版面上的 x。"""
        if tau < 2.6:
            return float(np.interp(ease(ramp(tau, 0.3, 2.4)), np.linspace(0, 1, len(LOWER)), LOWER[:, 0]))
        if tau < 7.2:
            return float(P(0.3 + 0.7 * ease(ramp(tau, 5.1, 6.9)), 0)[0])
        return 960.0

    def camera(self, tau):
        for a, b, v0, v1 in SHOTS:
            if a <= tau < b or (b == DUR and tau >= a):
                k = ease((tau - a) / (b - a))
                z = float(np.exp(np.log(v0[0]) + (np.log(v1[0]) - np.log(v0[0])) * k))
                rot = v0[3] + (v1[3] - v0[3]) * k
                if v0[1] is None:
                    cx = self.pen_x(tau)
                    cy = float(P(0, v0[2])[1])
                else:
                    c0, c1 = P(v0[1], v0[2]), P(v1[1], v1[2])
                    cx, cy = (c0 + (c1 - c0) * k).tolist()
                return z, cx, cy, rot
        return 1.0, 960.0, 540.0, 0.0

    # ── 畫 ────────────────────────────────────────────────
    def _poly(self, pts, close=True):
        c = self.ctx
        c.move_to(float(pts[0, 0]), float(pts[0, 1]))
        for x, y in pts[1:]:
            c.line_to(float(x), float(y))
        if close:
            c.close_path()

    def _fill(self, polys, rgb):
        c = self.ctx
        c.new_path()
        for q in polys:
            self._poly(q)
        c.set_source_rgb(*rgb)
        c.fill()

    def _line(self, pts, frac, w, rgb):
        """畫到 frac 的一道線（筆尖微微變粗）。"""
        n = int(len(pts) * frac)
        if n < 2:
            return
        c = self.ctx
        c.new_path()
        self._poly(pts[:n], close=False)
        c.set_source_rgb(*rgb)
        c.set_line_width(w)
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        c.set_line_join(cairo.LINE_JOIN_ROUND)
        c.stroke()

    def _blade(self, mid, w, frac, rgb):
        n = max(2, int(len(mid) * frac))
        if frac <= 0:
            return
        Q = resample(mid[:n], 2.0)
        if len(Q) < 2:
            return
        u = np.linspace(0, 1, len(Q)) * frac
        d = np.gradient(Q, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        N = np.stack([-d[:, 1], d[:, 0]], 1)
        half = (w / 2 * (1 - u) ** 0.9)[:, None]
        self._fill([np.concatenate([Q + N * half, (Q - N * half)[::-1]])], rgb)

    def _sweep_clip(self, x0, x1, frac, y0=-200.0, y1=1400.0, slant=0.25):
        """刷子由左往右刷開：前緣斜斜的、有一點毛。"""
        xf = x0 + (x1 - x0) * frac
        ys = np.linspace(y0, y1, 40)
        edge = xf + (ys - (y0 + y1) / 2) * slant + 6 * np.sin(ys / 13.0) + 4 * np.sin(ys / 5.0)
        poly = np.concatenate([np.array([[x0 - 400, y0]]), np.stack([edge, ys], 1), np.array([[x0 - 400, y1]])])
        c = self.ctx
        c.new_path()
        self._poly(poly.astype(np.float32))
        c.clip()

    def _glyphs(self, face, text, u, v, size_v, rgb, frac, gap=1.08):
        """直排的字：一個字一個字由上往下寫出來（字內由上到下，大致是筆順）；frac 0..1 是整串字的進度。"""
        size = size_v * PH
        x0, y0 = P(u, v)
        chars = []
        row = 0
        for ch in text:
            if ch != "　":
                g = face.glyph(ch)
                ox, oy = x0 - size * g.advance / 2, y0 + size * (row * gap + 0.88)
                chars.append([q * size + (ox, oy) for q in g.contours])
            row += 1
        m = len(chars)
        c = self.ctx
        for k, polys in enumerate(chars):
            a = frac * m - k
            if a <= 0 or not polys:
                break
            allp = np.concatenate(polys)
            lo, hi = allp.min(0), allp.max(0)
            c.save()
            if a < 1:
                yy = lo[1] + (hi[1] - lo[1] + 6) * ease(a)
                c.new_path()
                c.rectangle(float(lo[0]) - 20, float(lo[1]) - 20, float(hi[0] - lo[0]) + 40, float(yy - lo[1]) + 20)
                c.clip()
            c.new_path()
            for q in polys:
                self._poly(q)
            c.set_fill_rule(cairo.FILL_RULE_WINDING)
            c.set_source_rgb(*rgb)
            c.fill()
            c.restore()

    def render(self, tau):
        c = self.ctx
        z, cx, cy, rot = self.camera(tau)
        c.identity_matrix()
        c.reset_clip()
        c.set_source_rgb(*BG)
        c.paint()
        c.translate(self.W / 2, self.H / 2)
        c.rotate(rot)
        c.scale(z, z)
        c.translate(-cx, -cy)
        # 封面本身：一張微微有影子的深褐色紙
        x0, y0 = P(0, 0)
        c.rectangle(float(x0) + 10, float(y0) + 14, PW, PH)
        c.set_source_rgba(0, 0, 0, 0.5)
        c.fill()
        c.rectangle(float(x0), float(y0), PW, PH)
        c.set_source_rgb(*PAGE)
        c.fill()
        c.save()
        c.rectangle(float(x0), float(y0), PW, PH)
        c.clip()
        # 1. 兩道地平線
        self._line(LOWER, ease(ramp(tau, 0.3, 2.4)), 2.2, WHITE)
        self._line(UPPER, ease(ramp(tau, 1.5, 2.6)), 2.0, WHITE)
        # 3. 綠地：刷開
        g = ease_in(ramp(tau, 5.1, 6.9), 1.3)
        if g > 0:
            c.save()
            self._sweep_clip(float(P(0.27, 0)[0]), float(P(1.02, 0)[0]), g)
            self._fill([GREEN_POLY], GREEN)
            c.restore()
        # 4. 白的河岸：由左上往右下刷開
        r = ease(ramp(tau, 7.3, 8.9))
        if r > 0:
            c.save()
            self._sweep_clip(float(P(-0.02, 0)[0]), float(P(0.82, 0)[0]), r, slant=-0.6)
            self._fill([RIVER], WHITE)
            c.restore()
        # Wild Grass
        wf = ramp(tau, 8.4, 9.1)
        if wf > 0:
            c.save()
            gx0, gx1 = float(P(0.14, 0)[0]), float(P(0.48, 0)[0])
            self._sweep_clip(gx0, gx1, wf, slant=0.0)
            self._latin("Wild Grass", 0.145, 0.835, 0.034, INKD)
            c.restore()
        c.restore()
        c.save()
        c.rectangle(float(x0), float(y0), PW, PH)
        c.clip()
        # 2. 野草：一葉一葉長出來（大叢先、小叢、右邊那叢在綠地之後）
        for tuft, t0, t1 in ((BIG, 2.75, 4.9), (SMALL, 4.1, 5.0), (RIGHT, 6.0, 7.1)):
            n = len(tuft)
            for k, (mid, w) in enumerate(tuft):
                a = t0 + (t1 - t0) * k / n
                self._blade(mid, w, ease_out(ramp(tau, a, a + (t1 - t0) / n * 2.2), 2), GOLD)
        # 右邊草叢旁的兩粒種子
        for k, (u, v) in enumerate(((0.80, 0.574), (0.83, 0.579))):
            s = ramp(tau, 7.0 + 0.1 * k, 7.2 + 0.1 * k)
            if s > 0:
                q = P(u, v)
                self._fill([self._dash(q, 0.004 * PH, 0.012 * PH, 1.1)], GOLD)
        # 5. 書名、作者、〈希望〉的一句
        self._glyphs(self.title, "野草", 0.535, 0.155, 0.062, WHITE, ramp(tau, 9.5, 11.5))
        self._glyphs(self.small, "絕望之為虛妄　正與希望相同", 0.605, 0.15, 0.0165, WHITE, ramp(tau, 11.4, 12.5), gap=1.12)
        self._glyphs(self.small, "魯迅　著", 0.535, 0.305, 0.0135, WHITE, ramp(tau, 12.2, 12.8), gap=1.15)
        # 6. 兩粒種子落下，停在書名旁
        for k, (u, v) in enumerate(((0.478, 0.245), (0.497, 0.35))):
            s = ease_out(ramp(tau, 12.4 + 0.25 * k, 13.2 + 0.25 * k), 2)
            if s > 0:
                q = P(u, v) + np.array([12 * (1 - s), -160 * (1 - s)])
                self._fill([self._dash(q, 0.0045 * PH, 0.016 * PH, 0.35 + 1.4 * (1 - s))], SEED)
        c.restore()
        self.surf.flush()
        a = np.ndarray((self.H, self.W, 4), np.uint8, self.surf.get_data())
        img = a[..., [2, 1, 0]].astype(np.float32) / 255
        return np.clip(img + self.grain, 0, 1)

    def _dash(self, q, w, L, ang):
        th = np.linspace(0, 2 * np.pi, 20)
        e = np.stack([L / 2 * np.cos(th), w / 2 * np.sin(th)], 1)
        c, s = np.cos(ang), np.sin(ang)
        return (np.stack([e[:, 0] * c - e[:, 1] * s, e[:, 0] * s + e[:, 1] * c], 1) + q).astype(np.float32)

    def _latin(self, text, u, v, size_v, rgb):
        size = size_v * PH
        x, y = P(u, v)
        polys = []
        for ch in text:
            g = self.latin.glyph(ch)
            polys += [q * size + (x, y) for q in g.contours]
            x += size * (g.advance if g.contours or ch == " " else 0.3)
        c = self.ctx
        c.new_path()
        for q in polys:
            self._poly(q)
        c.set_source_rgb(*rgb)
        c.fill()
