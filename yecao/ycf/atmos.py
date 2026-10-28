"""環境：會飄的天空、有地層與埋藏物的土壤剖面、隨風的塵埃、光束。"""
import cairo
import numpy as np

from . import fonts
from .gfx import Layer, bloom, fill_polys, mix, resize, splat
from .noise import fbm, rand


class Sky:
    """低解析的雲霧紋理，橫向比畫面寬，隨時間緩慢平移。"""

    def __init__(self, seed, W, H):
        self.W, self.H = W, H
        w, h = 720, 135
        X, Y = np.meshgrid(np.linspace(0, 9, w), np.linspace(0, 1.7, h))
        n = fbm(X, Y, 0.0, 5, seed + 900) * 0.5 + 0.5
        n2 = fbm(X * 2.3 + 4, Y * 3, 1.0, 3, seed + 901) * 0.5 + 0.5
        self.tex = (n * 0.7 + n2 * 0.3).astype(np.float32)
        self.w = w

    def field(self, t, speed=6.0):
        """回傳 (H, W) 的 0..1 雲霧明暗。"""
        off = int(t * speed) % (self.w // 2)
        crop = self.tex[:, off:off + self.w // 2]
        return resize(crop[..., None], self.W, self.H)[..., 0]


class Soil:
    """土壤剖面（世界座標，半解析）：地層、碎石、細粒，與埋在土裡的殘筆。"""
    X0, X1, Y1 = -1500.0, 3420.0, 1300.0
    RES = 0.5

    def __init__(self, tl):
        rng = rand(tl.seed, 20, 1)
        pal = tl.pal
        w = int((self.X1 - self.X0) * self.RES)
        h = int(self.Y1 * self.RES)
        xs = np.linspace(self.X0, self.X1, w, dtype=np.float32)[None, :]
        ys = np.linspace(0, self.Y1, h, dtype=np.float32)[:, None]
        depth = np.clip(ys / 520, 0, 1)
        base = pal["soil"][None, None, :] * (1 - depth[..., None]) + pal["soil_deep"][None, None, :] * depth[..., None]
        warp = 30 * fbm(xs / 700, ys / 300, 0.0, 3, tl.seed + 910)
        strata = fbm(xs / 2200, (ys + warp) / 26, 0.5, 3, tl.seed + 911)
        grain = fbm(xs / 7, ys / 7, 2.0, 2, tl.seed + 912)
        light = 1 + 0.16 * strata + 0.07 * grain
        img = (base * light[..., None]).astype(np.float32)
        # 碎石與埋在土裡的殘筆，用 cairo 畫在同一張紋理上
        lay = Layer(w, h)
        ctx = lay.ctx
        ctx.scale(self.RES, self.RES)
        ctx.translate(-self.X0, 0)
        for _ in range(900):
            x, y = rng.uniform(self.X0, self.X1), rng.uniform(25, self.Y1) ** 1.0
            r = rng.uniform(1.5, 6.5)
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(rng.uniform(0, np.pi))
            ctx.scale(r, r * rng.uniform(0.45, 0.85))
            ctx.arc(0, 0, 1, 0, 2 * np.pi)
            ctx.restore()
            v = rng.uniform(0.6, 1.5)
            ctx.set_source_rgba(*(pal["soil"] * v).tolist(), rng.uniform(0.35, 0.8))
            ctx.fill()
        face = fonts.face(tl.raw["fonts"]["grass"])
        text = "".join(ln["text"] for ln in tl.raw["line"])
        chars = [ch for ch in text if ch not in "，。；！、：　"]
        for _ in range(140):
            g = face.glyph(chars[rng.integers(len(chars))])
            if not g.contours:
                continue
            # 只取字的一兩筆，橫躺、碎裂
            pick = [g.contours[i] for i in rng.choice(len(g.contours), min(2, len(g.contours)), replace=False)]
            size = rng.uniform(40, 120)
            x, y = rng.uniform(self.X0, self.X1), rng.uniform(60, self.Y1)
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(rng.uniform(-1.6, 1.6))
            fill_polys(ctx, [c * size for c in pick], pal["soil_deep"] * 0.55, rng.uniform(0.25, 0.55))
            ctx.restore()
        over_rgba = lay.rgba()
        a = over_rgba[..., 3:4]
        img = img * (1 - a) + over_rgba[..., :3]
        self.tex = img
        self.w, self.h = w, h

    def view(self, s, hz, W, H):
        """依鏡頭回傳地平線以下的畫面 (H−hz, W, 3)。"""
        h0 = int(np.ceil(hz))
        if h0 >= H:
            return None, h0
        wx0, wx1 = (0 - 960) / s + 960, (W - 960) / s + 960
        wy1 = (H - h0) / s
        i0 = int((wx0 - self.X0) * self.RES)
        i1 = int(np.ceil((wx1 - self.X0) * self.RES))
        j1 = int(np.ceil(wy1 * self.RES))
        crop = self.tex[0:max(j1, 1), max(i0, 0):i1]
        return resize(crop, W, H - h0), h0


class Motes:
    """空氣中的塵：小點隨風飄，少數近景的大顆是失焦的光斑。"""

    def __init__(self, seed, W, H, n=520):
        rng = rand(seed, 21, 1)
        self.W, self.H = W, H
        self.p0 = rng.random((n, 2)) * (W + 400, H * 0.82) - (200, 0)
        self.v = np.stack([rng.uniform(8, 40, n), rng.uniform(-6, 3, n)], 1)
        self.ph = rng.uniform(0, 6.28, n)
        self.w = rng.uniform(0.15, 0.6, n)
        self.big = rng.random(n) < 0.04

    def draw(self, img, t, strength, wind=1.0, view=None):
        if strength <= 0.002:
            return
        p = self.p0 + self.v * t * wind + np.stack([18 * np.sin(t * 0.4 + self.ph), 10 * np.sin(t * 0.7 + self.ph * 1.3)], 1)
        p[:, 0] = (p[:, 0] + 200) % (self.W + 400) - 200
        if view is not None:
            p = view.pts(p)
        col = np.array([1.0, 0.86, 0.66], np.float32)
        tw = 0.6 + 0.4 * np.sin(t * 2.1 + self.ph * 3)
        small = ~self.big
        buf = np.zeros_like(img)
        splat(buf, p[small], col, self.w[small] * tw[small] * strength)
        img += buf + bloom(buf, 3, down=2) * 0.8
        if self.big.any():
            bb = np.zeros_like(img)
            splat(bb, p[self.big], col, self.w[self.big] * 12 * strength)
            img += bloom(bb, 14, down=4) * 0.9


def rays(W, H, t, strength, color):
    """斜射的柔和光束（加法圖層）。"""
    if strength <= 0.002:
        return None
    lay = Layer(W, H)
    ctx = lay.ctx
    for k in range(6):
        x = -300 + k * 330 + 60 * np.sin(t * 0.13 + k * 1.9)
        w = 70 + 50 * ((k * 37) % 5) / 4
        a = strength * (0.10 + 0.06 * np.sin(t * 0.31 + k * 2.3))
        dx = 520
        quad = [(x, -50), (x + w, -50), (x + w + dx, H * 0.75), (x + dx - w * 0.3, H * 0.75)]
        g = cairo.LinearGradient(0, -50, 0, H * 0.75)
        g.add_color_stop_rgba(0, *[float(v) for v in color], a)
        g.add_color_stop_rgba(1, *[float(v) for v in color], 0)
        ctx.move_to(*quad[0])
        for q in quad[1:]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.set_source(g)
        ctx.fill()
    return bloom(lay.rgba()[..., :3], 30, down=4)
