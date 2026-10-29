"""字草世界裡的一段：被踩進泥裡的傳單，與從那片泥裡長出來的一株「頭」。

觀眾在木刻鏡頭裡看過這張紙被一隻帶鞋釘的靴子踩下去，只剩「中國的人民可以殺戮而不可以低」；
這裡換到字草的世界：同一張紙埋在土層裡，缺了「頭」的那一角；
上面長出一株字草，正是那個「頭」。它的根往下找，找到紙上的墨。
然後靴子的影子又落下來——它被壓彎，斷了一筆，撐住，慢慢直起來。

這一株會在全片裡反覆出現，觀眾記得它，後面的刪刈與火才有重量。
"""
import numpy as np

from . import atmos, fonts
from .gfx import Layer, bloom, ease, ease_out, fill_polys, fill_polys_grad, lin_gradient_v, mix, over, path_polys, ramp
from .noise import noise3, rand
from .scene_field import clip_u

HZ = 390.0          # 地平線（畫面 y）
SOIL_S = 1.8        # 土壤剖面的放大倍數


class Tou:
    """「頭」：一株有自己的傷與記憶的字草。"""

    def __init__(self, face, x, size, t_rise):
        g = face.glyph("頭")
        x0, y0, x1, y1 = g.bbox
        self.h = (y1 - y0) * size
        self.x = x
        self.base = [np.stack([(c[:, 0] - (x0 + x1) / 2) * size, (y1 - c[:, 1]) / (y1 - y0)], 1) for c in g.contours]
        self.t_rise = t_rise
        self.theta = 0.0
        self.omega = 0.0
        self.squash = 1.0
        self.sv = 0.0

    def warp(self, local, theta, squash):
        L = self.h * squash
        bulge = 1 + 0.6 * (1 - squash)
        out = []
        for c in local:
            x, u = c[:, 0] * bulge, c[:, 1]
            if abs(theta) < 1e-4:
                px, py = x, u * L
            else:
                phi = theta * u
                r = L / theta
                px = r * (1 - np.cos(phi)) + x * np.cos(phi)
                py = r * np.sin(phi) - x * np.sin(phi)
            out.append(np.stack([self.x + px, HZ - py], 1))
        return out


class LeafletGrass:
    """τ（秒）為這段自己的時間。事件：1.6 冒芽、6.0 根碰到墨、8.4 靴落下、之後撐住。"""

    RISE, TOUCH, STOMP = 1.6, 6.0, 8.4
    DUR = 13.0

    def __init__(self, tl):
        self.tl = tl
        self.W, self.H = tl.W, tl.H
        self.pal = tl.pal
        self.sky = atmos.Sky(tl.seed, self.W, self.H)
        self.soil = atmos.Soil(tl)
        self.motes = atmos.Motes(tl.seed, self.W, self.H, n=260)
        self.gface = fonts.face(tl.raw["fonts"]["grass"])
        self.serif = fonts.face("fonts/serif_regular")
        self.layer = Layer(self.W, self.H)
        self.plant = Tou(self.gface, 960.0, 300.0, self.RISE)
        self.fps = tl.fps
        self._sim_t = 0.0
        rng = rand(tl.seed, 501, 1)
        # 紙上的位置（畫面座標）：缺了「頭」的那一角，就在字草正下方
        self.cols = [("國亡了。同胞起來呀。", 1220), ("中國的人民可以殺戮而不可以低", 1000),
                     ("中國的土地可以征服而不可以斷送", 780)]
        self.size = 36
        self.y0 = 470
        tear_y = self.y0 + self.size * 14.2
        self.target = np.array([1000.0, tear_y + 18])
        # 根：主根找那個缺口，其他幾條各找一個字
        targets = [self.target, (1000, self.y0 + self.size * 13.4), (1000, self.y0 + self.size * 9.5),
                   (750, self.y0 + self.size * 12.5), (1250, self.y0 + self.size * 4.5), (880, 620)]
        self.roots = []
        for k, tg in enumerate(targets):
            tg = np.asarray(tg, np.float32)
            n = 40
            s = np.linspace(0, 1, n)[:, None]
            start = np.array([960 + rng.normal(0, 10), HZ + 4])
            mid = start + (tg - start) * s
            wob = np.stack([24 * np.sin(s[:, 0] * (5 + k) + k) * np.sin(np.pi * s[:, 0]),
                            6 * np.sin(s[:, 0] * 9 + k)], 1)
            pts = mid + wob + np.cumsum(rng.normal(0, 1.6, (n, 2)), 0) * np.sin(np.pi * s)
            self.roots.append((pts.astype(np.float32), 0.6 + 0.4 * (k == 0), rng.uniform(0, 1)))

    # ── 模擬 ──────────────────────────────────────────────
    def _step(self, t, dt):
        p = self.plant
        wind = 0.10 * np.sin(0.7 * t) + 0.07 * noise3(t * 0.5, 3.0, 0.0, 7)
        rest = 0.0
        if t >= self.STOMP:
            rest = 0.22 * (1 - 0.6 * ease(ramp(t, self.STOMP + 1.5, self.DUR)))   # 受過傷，留著一點彎
        k, c = 9.0, 2.2
        acc = -k * (p.theta - rest - wind) - c * p.omega
        if self.STOMP <= t < self.STOMP + dt * 1.5:
            p.omega += 4.5
            p.sv = -9.0
        p.omega += acc * dt
        p.theta += p.omega * dt
        sq_rest = 1.0 if t < self.STOMP else 0.55 + 0.4 * ease(ramp(t, self.STOMP + 0.6, self.DUR - 1.0))
        p.sv += (-30 * (p.squash - sq_rest) - 5 * p.sv) * dt
        p.squash = float(np.clip(p.squash + p.sv * dt, 0.35, 1.05))

    def advance(self, t):
        dt = 1 / self.fps
        while self._sim_t + dt <= t + 1e-9:
            self._sim_t += dt
            self._step(self._sim_t, dt)

    # ── 畫 ────────────────────────────────────────────────
    def _paper(self, ctx, t):
        """埋在土裡的那張宣言：撕過、皺過，土從紙的縫隙透出來。"""
        from .carve import rough
        from .plates import sole
        pal = self.pal
        paper_col = np.array([0.40, 0.365, 0.30], np.float32)
        x0, x1 = 680, 1330
        top = self.y0 - 40
        poly = rough(np.array([[x0, top + 10], [x1, top - 12], [x1 + 18, 1130], [x0 - 14, 1130]], np.float32), 6, 11)
        # 撕去的那一角：「頭」原本在那裡，現在只有土
        ty = self.target[1]
        tear = rough(np.array([[905, ty - 30], [985, ty - 46], [1075, ty - 22], [1105, ty + 90], [1050, 1140],
                               [890, 1140], [870, ty + 60]], np.float32), 10, 12)
        ctx.save()
        ctx.new_path()
        path_polys(ctx, [poly])
        ctx.clip()
        ctx.new_path()
        path_polys(ctx, [tear])
        ctx.rectangle(0, 0, self.W, self.H)
        ctx.set_fill_rule(1)
        ctx.clip()
        ctx.set_fill_rule(0)
        fill_polys(ctx, [poly], paper_col, 0.72)
        # 皺：幾道暗的摺痕
        for k in range(7):
            yk = top + 60 + k * 95 + 20 * np.sin(k * 2.1)
            ctx.move_to(x0 - 20, yk)
            ctx.curve_to(x0 + 200, yk + 30 * np.sin(k), x1 - 240, yk - 25 * np.cos(k), x1 + 20, yk + 10)
            ctx.set_source_rgba(*(pal["soil_deep"] * 0.8).tolist(), 0.35)
            ctx.set_line_width(3 + 3 * (k % 2))
            ctx.stroke()
        ink = np.array([0.06, 0.045, 0.038], np.float32)
        for text, cx in self.cols:
            row = 0
            for ch in text:
                if ch in "，。！：":
                    ctx.arc(cx + self.size * 0.55, self.y0 + self.size * (row - 0.15), 3.5, 0, 2 * np.pi)
                    ctx.set_source_rgba(*ink.tolist(), 0.85)
                    ctx.fill()
                    continue
                g = self.serif.glyph(ch)
                polys = [q * self.size + (cx - self.size * g.advance / 2, self.y0 + self.size * (row + 0.88))
                         for q in g.contours]
                fill_polys(ctx, polys, ink, 0.88)
                row += 1
        # 鞋印的泥：蓋住紙的右下，鞋釘的小坑
        sp = sole(1250, 1010, 470, 0.45)
        fill_polys(ctx, [rough(sp, 5, 13)], pal["soil_deep"] * 1.2, 0.85)
        for k in range(12):
            th = np.pi * k / 11
            q = np.array([0.12 * np.cos(th) * 470, (0.38 + 0.1 * np.sin(th)) * 470])
            c, s_ = np.cos(0.45), np.sin(0.45)
            px, py = 1250 + q[0] * c - q[1] * s_, 1010 + q[0] * s_ + q[1] * c
            ctx.arc(float(px), float(py), 4, 0, 2 * np.pi)
            ctx.set_source_rgba(*paper_col.tolist(), 0.5)
            ctx.fill()
        ctx.restore()

    def _plant_polys(self, t):
        p = self.plant
        rise = ease_out(ramp(t, p.t_rise, p.t_rise + 2.6), 2)
        if rise <= 0:
            return [], []
        local = p.base
        broken = []
        if t >= self.STOMP:
            # 最上面一筆被踩斷，掉在一旁
            local = [q for q in (clip_u(c, 0.0, 0.84) for c in p.base) if q is not None]
            piece = [q for q in (clip_u(c, 0.84, 1.0) for c in p.base) if q is not None]
            tau = t - self.STOMP
            fall = ease_out(min(tau / 0.5, 1), 2)
            for q in p.warp(piece, 0.0, 1.0):
                c0 = q.mean(0)
                ang = 1.3 * fall
                cs, sn = np.cos(ang), np.sin(ang)
                d = q - c0
                rq = np.stack([d[:, 0] * cs - d[:, 1] * sn, d[:, 0] * sn + d[:, 1] * cs], 1)
                tgt = np.array([p.x + 170, HZ - 14])
                broken.append(rq + mix(c0, tgt, fall))
        polys = []
        n = len(local)
        for i, cont in enumerate(local):
            d = 0.03 * noise3(t * 2.4 + i * 1.3, i * 0.7, 0.0, 300)
            polys += p.warp([cont], p.theta + d, p.squash)
        dy = (1 - rise) * p.h * 1.05
        return [q + (0, dy) for q in polys], broken

    def render(self, frame):
        t = frame / self.fps
        self.advance(t)
        pal = self.pal
        W, H = self.W, self.H
        top = np.array([0.045, 0.038, 0.033])
        near = np.array([0.16, 0.125, 0.095])
        img = lin_gradient_v(H, W, [(0, top), (HZ / H * 0.8, near * 0.8), (HZ / H, near), (1, near)])
        img *= (0.8 + 0.4 * self.sky.field(t, 5.0))[..., None]
        ys = np.arange(H, dtype=np.float32)
        img += np.exp(-((ys - HZ) / 120) ** 2)[:, None, None] * np.array([0.09, 0.07, 0.05], np.float32)
        soil, h0 = self.soil.view(SOIL_S, HZ, W, H)
        img[h0:] = soil
        lay = self.layer
        lay.clear()
        ctx = lay.ctx
        self._paper(ctx, t)
        over(img, lay.rgba())
        lay.clear()
        # 根
        touch = ease(ramp(t, self.TOUCH - 0.4, self.TOUCH + 1.2))
        for k, (pts, wt, ph) in enumerate(self.roots):
            g = ease_out(ramp(t, self.RISE + 0.4 + 0.25 * k, self.TOUCH + 0.4 * k), 1.6)
            n = max(2, int(len(pts) * g))
            if g <= 0:
                continue
            q = pts[:n]
            col = mix(pal["grass_dim"] * 0.8, pal["blood"] * 1.6, touch * (0.7 + 0.3 * (k == 0)))
            for j in range(len(q) - 1):
                w = (2.6 * wt + 1.0) * (1 - j / len(pts)) + 0.6
                ctx.move_to(*[float(v) for v in q[j]])
                ctx.line_to(*[float(v) for v in q[j + 1]])
                ctx.set_line_width(w)
                ctx.set_source_rgba(*[float(v) for v in col], 0.85)
                ctx.stroke()
            # 碰到墨之後，血沿根往上
            if touch > 0 and n == len(pts):
                for m in range(3):
                    u = (t * 0.45 + ph + m / 3) % 1.0
                    i = int((1 - u) * (len(pts) - 1))
                    ctx.arc(float(pts[i, 0]), float(pts[i, 1]), 2.2, 0, 2 * np.pi)
                    ctx.set_source_rgba(0.85, 0.2, 0.12, 0.75 * touch)
                    ctx.fill()
        # 墨被吸走：缺口旁的「低」慢慢淡下去（染成血色）
        over(img, lay.rgba())
        lay.clear()
        # 字草
        polys, broken = self._plant_polys(t)
        p = self.plant
        if polys:
            base = pal["grass"]
            tint = touch
            root = mix(base * 0.36, pal["blood"] * 1.1, 0.8 * tint)
            midc = mix(base * 0.8, pal["blood"] * 1.3, 0.3 * tint)
            tip = base * 1.1
            r0 = np.array([p.x, HZ])
            r1 = r0 - (0, p.h * p.squash)
            fill_polys_grad(ctx, polys, r0, r1, [(0.0, root, 1.0), (0.45, midc, 1.0), (1.0, tip, 1.0)])
        for q in broken:
            fill_polys(ctx, [q], pal["grass"] * 0.7, 1.0)
        # 地平線的亮邊
        xs = np.linspace(-20, W + 20, 120)
        ctx.new_path()
        ctx.move_to(float(xs[0]), HZ + 2 * float(noise3(0, 0, 0, 91)))
        for x in xs[1:]:
            ctx.line_to(float(x), HZ + 2.0 * float(noise3(x / 120, 0, 0, 91)))
        ctx.set_source_rgba(*[float(v) for v in pal["mud"] * 1.8], 0.8)
        ctx.set_line_width(1.5)
        ctx.stroke()
        over(img, lay.rgba())
        # 靴的影子壓下來
        if self.STOMP - 0.35 < t < self.STOMP + 0.9:
            k = ease(ramp(t, self.STOMP - 0.35, self.STOMP)) * (1 - ease(ramp(t, self.STOMP + 0.2, self.STOMP + 0.9)))
            yy, xx = np.ogrid[0:H, 0:W]
            r = np.sqrt(((xx - 990) / 640) ** 2 + ((yy - HZ + 120) / 260) ** 2)
            sh = np.clip(1.4 - r, 0, 1) ** 0.8 * k
            img *= (1 - 0.85 * sh[..., None])
        # 塵
        if self.STOMP <= t < self.STOMP + 2.5:
            rng = rand(7, 502, 1)
            tau = t - self.STOMP
            buf = np.zeros_like(img)
            from .gfx import splat
            n = 900
            ang = rng.uniform(np.pi, 2 * np.pi, n)
            sp = rng.uniform(80, 520, n)
            pos = np.stack([960 + np.cos(ang) * sp * (1 - np.exp(-tau * 2.5)) * 0.9,
                            HZ - 4 + np.sin(ang) * sp * 0.35 * (1 - np.exp(-tau * 2.5)) + 40 * tau * tau], 1)
            splat(buf, pos, np.array([0.5, 0.42, 0.33], np.float32), rng.uniform(0.2, 0.9, n) * np.exp(-tau * 1.3))
            img += bloom(buf, 2.5, down=2) * 1.4 + buf
        self.motes.draw(img, t, 0.55, wind=0.7)
        if self.STOMP <= t < self.STOMP + 0.3:
            amp = 9 * (1 - (t - self.STOMP) / 0.3)
            dy = int(round(amp * np.sin((t - self.STOMP) * 90)))
            img = np.roll(img, dy, axis=0)
        return np.clip(img, 0, 1)
