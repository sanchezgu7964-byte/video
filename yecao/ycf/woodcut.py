"""木刻版畫：把一張「明暗圖」刻成線條。

明暗用刀線的粗細表現：越暗，留下的黑線越粗；物件與背景用不同的刀向。
第二色（紅）是另一塊版，套印時略微錯位。紙有纖維、墨有濃淡。
魯迅提倡新興木刻，所以片中所有「意象」都用這種版畫語言出現，和字草的世界分開。
"""
import cairo
import numpy as np

from .gfx import Layer, path_polys, resize
from .noise import fbm

PAPER = np.array([0.87, 0.83, 0.74], np.float32)
INK = np.array([0.045, 0.036, 0.03], np.float32)
RED = np.array([0.70, 0.13, 0.08], np.float32)


class Plate:
    """一塊版：tone（明暗，0 黑 1 白）、sel（物件＝1，用另一個刀向）、red（紅版）。"""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.tone = Layer(W, H)
        self.sel = Layer(W, H)
        self.red = Layer(W, H)
        self.tc, self.sc, self.rc = self.tone.ctx, self.sel.ctx, self.red.ctx

    def begin(self, tone_top, tone_bottom=None):
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


class Woodcut:
    def __init__(self, W, H, seed):
        self.W, self.H = W, H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        lw, lh = W // 8, H // 8
        X, Y = np.meshgrid(np.linspace(0, 12, lw), np.linspace(0, 7, lh))
        wob = fbm(X, Y, 0.0, 3, seed + 600).astype(np.float32)
        self.wob = resize(wob[..., None], W, H)[..., 0] * 3.0
        fib = fbm(X * 9, Y * 30, 1.0, 2, seed + 601).astype(np.float32)
        self.paper = 1 + 0.05 * resize(fib[..., None], W, H)[..., 0]
        inkv = fbm(X * 4, Y * 4, 2.0, 3, seed + 602).astype(np.float32)
        self.inkvar = 0.94 + 0.06 * resize(inkv[..., None], W, H)[..., 0]
        self.p0 = (xx * np.cos(0.6) + yy * np.sin(0.6)) / 7.5
        self.p1 = (xx * np.cos(-0.95) + yy * np.sin(-0.95)) / 5.8

    def __call__(self, tone, sel, red, jitter=0):
        # 每一張印出來都有一點不同：刀線位置隨印次微微漂移
        drift = (jitter % 7) * 0.13
        ph = np.where(sel > 0.5, self.p1, self.p0) + self.wob / 6 + drift
        tri = np.abs(2 * (ph % 1.0) - 1)
        th = np.clip(1 - tone, 0, 1) ** 0.9
        ink = np.clip((th - tri) / 0.16 + 0.5, 0, 1)
        ink = np.where(th > 0.93, 1.0, ink)
        # 刻刀沿物件邊緣挖出的一道白線
        from scipy.ndimage import gaussian_filter
        # 刻在物件外側（背景那一邊），細的根、旗桿才不會被挖掉
        edge = (gaussian_filter(sel, 1.6) - sel) > 0.12
        ink = np.where(edge & (th > 0.5), 0.0, ink) * self.inkvar
        img = PAPER[None, None, :] * self.paper[..., None] * (1 - ink[..., None]) + INK * ink[..., None]
        if red.max() > 0:
            r = np.roll(np.roll(red, 3, 0), 2, 1)[..., None] * 0.92      # 套色略微錯位
            img = img * (1 - r) + img * r * (RED / PAPER) * (0.55 + 0.45 * self.inkvar[..., None])
        return img.astype(np.float32)
