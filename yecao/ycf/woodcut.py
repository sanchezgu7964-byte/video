"""木刻版畫：中國新興木刻的語言——只有黑與白。

- 明暗不是灰階：先把畫面分成大塊的黑與白，邊緣是刀口的碎裂感。
- 中間調用刀痕表現：兩頭尖、中間粗、長短不一的刀線。暗處在黑底上刻出白線，亮處在白底上留下黑線，
  越接近另一邊，刀痕越粗越密。
- 刀向：背景是橫向的長刀（像肖像背後那一道道白線）；物件依自己的輪廓分成幾個面，每一面換一個刀向。
- 物件有立體：每個物件依自己的形狀生出柔和的起伏，光從左上來，亮的一側被刻開。
- 墨不勻：大片黑裡有一點點露白的斑，白裡偶爾帶一層淡墨。
"""
import cairo
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .gfx import Layer, path_polys, resize
from .noise import fbm

PAPER = np.array([0.90, 0.875, 0.81], np.float32)
INK = np.array([0.055, 0.05, 0.048], np.float32)


class Plate:
    """一塊版：tone（明暗，0 黑 1 白）、sel（物件＝1）。red 圖層保留介面，印成深色。"""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.tone = Layer(W, H)
        self.sel = Layer(W, H)
        self.red = Layer(W, H)
        self.tc, self.sc, self.rc = self.tone.ctx, self.sel.ctx, self.red.ctx
        self.flat = False

    def begin(self, tone_top, tone_bottom=None, flat=False):
        self.flat = flat
        for L in (self.tone, self.sel, self.red):
            L.clear()
            L.ctx.identity_matrix()
            L.ctx.new_path()
        tb = tone_top if tone_bottom is None else tone_bottom
        g = cairo.LinearGradient(0, 0, 0, self.H)
        g.add_color_stop_rgb(0, tone_top, tone_top, tone_top)
        g.add_color_stop_rgb(1, tb, tb, tb)
        self.tc.set_source(g)
        self.tc.paint()

    def fill(self, polys, tone, obj=True, alpha=1.0, close=True):
        self.tc.new_path()
        path_polys(self.tc, polys, close)
        self.tc.set_fill_rule(cairo.FILL_RULE_WINDING)
        self.tc.set_source_rgba(tone, tone, tone, alpha)
        self.tc.fill()
        if obj is not None:
            self.sc.new_path()
            path_polys(self.sc, polys, close)
            self.sc.set_fill_rule(cairo.FILL_RULE_WINDING)
            v = 1.0 if obj else 0.0
            self.sc.set_source_rgba(v, v, v, alpha)
            self.sc.fill()

    def grad_fill(self, polys, p0, p1, t0, t1, a0=1.0, a1=1.0, obj=None):
        self.tc.new_path()
        path_polys(self.tc, polys)
        g = cairo.LinearGradient(*p0, *p1)
        g.add_color_stop_rgba(0, t0, t0, t0, a0)
        g.add_color_stop_rgba(1, t1, t1, t1, a1)
        self.tc.set_source(g)
        self.tc.fill()
        if obj is not None:
            self.sc.new_path()
            path_polys(self.sc, polys)
            v = 1.0 if obj else 0.0
            self.sc.set_source_rgba(v, v, v, 1)
            self.sc.fill()

    def stroke(self, polys, tone, width, obj=None, close=False, alpha=1.0):
        self.tc.new_path()
        path_polys(self.tc, polys, close)
        self.tc.set_source_rgba(tone, tone, tone, alpha)
        self.tc.set_line_width(width)
        self.tc.set_line_cap(cairo.LINE_CAP_ROUND)
        self.tc.set_line_join(cairo.LINE_JOIN_ROUND)
        self.tc.stroke()
        if obj is not None:
            self.sc.new_path()
            path_polys(self.sc, polys, close)
            v = 1.0 if obj else 0.0
            self.sc.set_source_rgba(v, v, v, 1)
            self.sc.set_line_width(width)
            self.sc.stroke()

    def circle(self, x, y, r, tone, obj=True, alpha=1.0):
        th = np.linspace(0, 2 * np.pi, 40)
        self.fill([np.stack([x + r * np.cos(th), y + r * np.sin(th)], 1)], tone, obj, alpha)

    def redfill(self, polys, alpha=1.0):
        self.rc.new_path()
        path_polys(self.rc, polys)
        self.rc.set_source_rgba(1, 1, 1, alpha)
        self.rc.fill()

    def redstroke(self, polys, width, alpha=1.0):
        self.rc.new_path()
        path_polys(self.rc, polys, close=False)
        self.rc.set_source_rgba(1, 1, 1, alpha)
        self.rc.set_line_width(width)
        self.rc.set_line_cap(cairo.LINE_CAP_ROUND)
        self.rc.stroke()

    def arrays(self):
        t = self.tone.rgba()[..., 0]
        s = self.sel.rgba()[..., 0]
        r = self.red.rgba()[..., 3]
        return t, s, r


def _hash(a, b, seed):
    h = (a.astype(np.int64) * 73856093) ^ (b.astype(np.int64) * 19349663) ^ (seed * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFFFF).astype(np.float32) / 0xFFFFFF


class Woodcut:
    ANGLES = [0.0, 0.62, 1.25, 1.88, 2.51]      # 物件的幾個刀向（弧度）
    R = 2                                       # 以半解析計算刀痕，再放大成銳利的邊

    def __init__(self, W, H, seed):
        self.W, self.H = W, H
        self.seed = seed
        w, h = W // self.R, H // self.R
        self.w, self.h = w, h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) * self.R
        lw, lh = W // 8, H // 8
        X, Y = np.meshgrid(np.linspace(0, 12, lw), np.linspace(0, 7, lh))

        def up(a, ww=w, hh=h):
            return resize(a.astype(np.float32)[..., None], ww, hh)[..., 0]
        self.wave = up(fbm(X * 0.6, Y * 0.6, 0.0, 3, seed + 600)) * 9.0              # 刀線微微起伏
        self.chip = up(fbm(X * 6, Y * 6, 1.0, 3, seed + 601))                          # 刀口的碎裂
        self.inkv = up(fbm(X * 3, Y * 3, 2.0, 3, seed + 602), W, H)                    # 墨的濃淡
        rng = np.random.default_rng(seed + 603)
        self.speck = rng.random((H, W)).astype(np.float32)
        self.paper = 1 + 0.03 * up(fbm(X * 9, Y * 30, 3.0, 2, seed + 604), W, H)
        self.coords = []
        for a in [0.0] + self.ANGLES:
            u = xx * np.cos(a) + yy * np.sin(a)
            v = -xx * np.sin(a) + yy * np.cos(a) + self.wave
            self.coords.append((u.astype(np.float32), v.astype(np.float32)))

    def _gouge(self, u, v, density, spacing, length, seed):
        """刀痕：沿 u 方向、相距 spacing 的刀線，切成長短不一、兩頭尖的一段段。
        density 0..1：越大，刀線越粗、越多（淺處只剩零星幾刀）。"""
        k = np.floor(v / spacing)
        off = _hash(k, k * 0 + 7, seed) * length
        seg_len = length * (0.4 + 0.9 * _hash(k, k * 0 + 11, seed))
        su = (u + off) / seg_len
        j = np.floor(su)
        s = su - j
        present = _hash(k, j, seed + 1) < np.clip(density * 1.6, 0, 0.95)
        taper = np.sin(np.pi * s) ** 0.6
        d = np.abs(v / spacing - k - 0.5) * 2
        w = np.clip(density, 0, 1) ** 0.8 * taper * (0.75 + 0.5 * _hash(k, j, seed + 2))
        # 回傳覆蓋率（邊緣有一點過渡），放大後才會是平順的刀口
        return present * np.clip((w - d) * spacing / 3.0 + 0.5, 0, 1)

    def __call__(self, tone, sel, red, jitter=0, flat=False):
        W, H, w, h = self.W, self.H, self.w, self.h
        if red.max() > 0:
            tone = np.minimum(tone, 1 - 0.85 * red)        # 單色木刻：紅版印成深色
        small = resize(np.stack([tone, sel], -1), W // 4, H // 4)
        st, ss = small[..., 0], small[..., 1]
        st0 = st.copy()
        if not flat:
            # 立體：依物件到邊緣的距離長出一個圓頂般的起伏，光從左上來，亮的一面整片被刻開
            inside = ss > 0.5
            dist = distance_transform_edt(inside)
            hgt = gaussian_filter(np.sqrt(dist), 2.0)
            gy, gx = np.gradient(hgt)
            nz = 0.35
            norm = np.sqrt(gx * gx + gy * gy + nz * nz)
            lam = (-gx * 0.62 - gy * 0.55 + nz * 0.55) / norm
            shade = np.clip((lam - 0.45) * 2.2, -1, 1)
            st = np.clip(st + 0.62 * np.maximum(shade, 0) * ss - 0.15 * np.maximum(-shade, 0) * ss, 0, 1)
            # 刀向：用很平滑的輪廓方向，分出大塊的「面」
            gy2, gx2 = np.gradient(gaussian_filter(ss, 3))
            c2 = gaussian_filter(gx2 * gx2 - gy2 * gy2, 10)
            s2 = gaussian_filter(2 * gx2 * gy2, 10)
            ang = (0.5 * np.arctan2(s2, c2) + np.pi / 2) % np.pi
        else:
            ang = np.zeros_like(st)
        # 明暗本身取半解析（細節清楚），只有立體的明暗變化取自低解析
        half = resize(tone[..., None], w, h)[..., 0]
        T = gaussian_filter(np.clip(half + resize((st - st0)[..., None], w, h)[..., 0], 0, 1), 0.6)
        S = resize(ss[..., None], w, h)[..., 0]
        A = resize(ang[..., None], w, h)[..., 0]
        white = (T + self.chip * 0.07) > 0.5
        dark_mid = np.clip((T - 0.15) / 0.35, 0, 1) ** 1.2      # 黑塊裡要刻掉多少
        light_mid = np.clip((0.82 - T) / 0.32, 0, 1) ** 1.4     # 白塊裡要留多少黑
        out = white.astype(np.float32)
        obj = S >= 0.5
        js = jitter % 3
        # 背景：橫向長刀
        u, v = self.coords[0]
        m = ~obj & ~white & (dark_mid > 0.01)
        if m.any():
            out[m] = np.maximum(out[m], self._gouge(u[m], v[m], dark_mid[m], 14.0, 260.0, self.seed + js))
        m = ~obj & white & (light_mid > 0.01)
        if m.any():
            out[m] = np.minimum(out[m], 1 - self._gouge(u[m], v[m] + 7, light_mid[m], 14.0, 200.0, self.seed + 50 + js))
        # 物件：依輪廓分面，每面一個刀向，刀線短而密
        if obj.any():
            bins = np.floor(A / np.pi * len(self.ANGLES)).astype(np.int32) % len(self.ANGLES)
            for bi in range(len(self.ANGLES)):
                u, v = self.coords[bi + 1]
                m = obj & (bins == bi) & ~white & (dark_mid > 0.01)
                if m.any():
                    out[m] = np.maximum(out[m], self._gouge(u[m], v[m], dark_mid[m], 9.0, 95.0, self.seed + 100 + bi))
                m = obj & (bins == bi) & white & (light_mid > 0.01)
                if m.any():
                    out[m] = np.minimum(out[m], 1 - self._gouge(u[m], v[m] + 4, light_mid[m], 9.0, 80.0, self.seed + 150 + bi))
        # 黑底上的物件：沿外緣刻一道細白線，物件才從黑裡分出來
        rim = (gaussian_filter(S, 1.2) - S > 0.1) & ~white
        out = np.where(rim, np.maximum(out, 0.85), out)
        # 放大回全解析，重新銳化成刀刻的邊
        full = resize(out[..., None], W, H)[..., 0]
        ink = np.clip((0.5 - full) / 0.12 + 0.5, 0, 1)
        # 墨不勻：黑裡露出零星的紙，白裡偶爾一層淡墨
        ink = np.where((ink > 0.5) & (self.speck > 0.988 - 0.015 * self.inkv), 0.25, ink)
        ink = np.maximum(ink, np.clip(self.inkv * 0.2 - 0.08, 0, 0.05))
        img = PAPER[None, None, :] * self.paper[..., None] * (1 - ink[..., None]) + INK * ink[..., None]
        return img.astype(np.float32)
