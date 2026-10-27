"""第 11–12 段：火速（生長、腐爛、重生的循環越轉越快）→ 驟停 → 去罷：全詩被風吹散 → 被大地收下。"""
import cairo
import numpy as np

from . import clump as clumpmod
from . import fonts
from .gfx import Layer, ease, ease_in, ease_out, fill_polys, lin_gradient_v, mix, over, ramp
from .noise import noise3, rand

BG = np.array([0.05, 0.045, 0.04], np.float32)


class EndScene:
    t0 = 172.0
    t1 = 1e9

    def __init__(self, tl):
        self.tl = tl
        self.W, self.H, self.fps = tl.W, tl.H, tl.fps
        self.c = tl.c
        self.seed = tl.seed
        self.layer = Layer(self.W, self.H)
        self.clump = clumpmod.make_clump(tl)
        self.face = fonts.face(tl.raw["fonts"]["serif"])
        self.sprites = {}
        self._build_cycles()
        self._build_poem()

    def advance_to(self, frame):
        pass  # 本場景全部是時間的閉式函數

    # ── 第 11 段：循環 ───────────────────────────────────────
    def _build_cycles(self):
        c = self.c
        d = 2.4
        # 第一圈從「已長成」的相位開始，接續第 10 段那一叢
        t = c("cycle") - 0.3 * d
        self.cycles = []
        while t < c("stop"):
            self.cycles.append((t, d))
            t += d
            d = max(d * 0.78, 0.1)
        rng = rand(self.seed, 11, 1)
        self.droop_dir = rng.choice([-1, 1], len(self.clump))
        self.specks = [rng.normal(0, 1, (18, 2)) * (p.size * 0.3, p.h * 0.25) - (0, p.h * 0.5) for p in self.clump]
        self.years = self.tl.raw["years"]["list"]

    def _cycle_at(self, t):
        for k, (t0, d) in enumerate(self.cycles):
            if t0 <= t < t0 + d:
                return k, (t - t0) / d
        return len(self.cycles) - 1, 1.0

    def _draw_cycle(self, ctx, t):
        pal = self.tl.pal
        k, u = self._cycle_at(t)
        grow = ease_out(ramp(u, 0.0, 0.3), 2)
        rot = ease(ramp(u, 0.5, 1.0))
        alpha = 1 - ease(ramp(u, 0.85, 1.0))
        ctx.save()
        ctx.rectangle(0, 0, self.W, clumpmod.BASE[1])
        ctx.clip()
        for i, p in enumerate(self.clump):
            theta = 0.05 * np.sin(t * 2.3 + i) * (1 - rot) + self.droop_dir[i] * 0.9 * rot
            polys = clumpmod.clump_polys(p, theta=theta, squash=1 - 0.45 * rot, rise=grow)
            col = mix(mix(pal["grass"], pal["light_warm"], 0.25), np.array([0.26, 0.19, 0.13]), rot)
            fill_polys(ctx, polys, col, alpha)
        ctx.restore()
        # 腐爛也是生長：黴點
        if rot > 0.1:
            for i, p in enumerate(self.clump):
                for sx, sy in self.specks[i]:
                    ctx.arc(p.x + sx * (1 + rot), clumpmod.BASE[1] + sy * (1 - 0.45 * rot), 1.3, 0, 2 * np.pi)
                ctx.set_source_rgba(0.85, 0.83, 0.76, 0.5 * rot * alpha)
                ctx.fill()
        # 角落淡淡浮過一個年份
        year = str(self.years[k % len(self.years)])
        ya = self.tl.raw["years"]["alpha"] * np.sin(np.pi * np.clip(u, 0, 1))
        x = 1800 - len(year) * 30 * 0.62
        for ch in year:
            self._blit(ctx, ch, x, 118, 30, (0.86, 0.82, 0.74), ya)
            x += 30 * 0.62

    # ── 第 12 段：去罷 ─────────────────────────────────────
    def _build_poem(self):
        c = self.c
        rng = rand(self.seed, 12, 1)
        text = "".join(ln["text"] for ln in self.tl.raw["line"] if ln.get("mode", "type") == "type")
        size, adv, per_row, lead = 25, 33.5, 44, 42
        rows = [text[i:i + per_row] for i in range(0, len(text), per_row)]
        y0 = 150
        glyphs = []
        for r, row in enumerate(rows):
            x0 = self.W / 2 - per_row * adv / 2
            for j, ch in enumerate(row):
                glyphs.append([ch, x0 + j * adv, y0 + r * lead, size])
        self.poem = glyphs
        n = len(glyphs)
        xs = np.array([g[1] for g in glyphs])
        # 風由右向左：右邊的先被吹走
        self.p_rel = c("farewell") + 1.6 + (1 - xs / self.W) * 5.5 + rng.uniform(0, 0.9, n)
        self.p_spin = rng.normal(0, 2.2, n)
        self.p_lift = rng.uniform(-40, 25, n)
        self.p_ph = rng.uniform(0, 6.28, n)
        self.c_rel = [c("farewell") + 1.6 + (1 - p.x / self.W) * 5.5 + 0.6 for p in self.clump]
        self.survivor = 2   # 最後只剩中間那個「草」
        self.c_rel[self.survivor] = c("colophon") + 3.2

    def _wind(self, tau):
        """被釋放後的位移（閉式）：越吹越快。"""
        return -(70 * tau + 300 * tau * tau)

    def _draw_farewell(self, ctx, t):
        c = self.c
        pal = self.tl.pal
        appear = ease(ramp(t, c("farewell") + 0.2, c("farewell") + 1.3))
        if appear <= 0:
            return
        for i, (ch, x, y, size) in enumerate(self.poem):
            tau = t - self.p_rel[i]
            dx, dy, rot = 0.0, 0.0, 0.0
            if tau > 0:
                dx = self._wind(tau)
                dy = self.p_lift[i] * tau + 18 * np.sin(tau * 3 + self.p_ph[i])
                rot = self.p_spin[i] * tau
            a = 0.5 * appear * (1 - ease(ramp(tau, 1.2, 2.8)))
            if a <= 0.004 or x + dx < -60:
                continue
            self._blit(ctx, ch, x + dx, y + dy, size, (0.84, 0.8, 0.72), a, rot)
        # 那一叢，連同最後飄著的一個字
        for i, p in enumerate(self.clump):
            tau = t - self.c_rel[i]
            polys = clumpmod.clump_polys(p, theta=0.04 * np.sin(t * 2 + i))
            a = appear
            if i == self.survivor:
                # 只剩它飄著：先離地浮起，最後也飛走
                lift = ease(ramp(t, c("colophon") - 2.5, c("colophon") + 0.5))
                bob = 14 * np.sin((t - c("colophon")) * 1.4)
                off = np.array([-60 * lift, -150 * lift + bob * lift])
                if tau > 0:
                    off = off + (self._wind(tau) * 0.8, -40 * tau)
                rot = 0.25 * lift * np.sin(t * 0.7) + (0.8 * tau if tau > 0 else 0)
            else:
                off = np.array([self._wind(tau), -30 * tau]) if tau > 0 else np.zeros(2)
                rot = 1.5 * tau if tau > 0 else 0.0
                a *= 1 - ease(ramp(tau, 1.0, 2.5))
            if a <= 0.004:
                continue
            cx, cy = p.x, clumpmod.BASE[1] - p.h / 2
            cr, sr = np.cos(rot), np.sin(rot)
            R = np.array([[cr, -sr], [sr, cr]])
            polys = [((q - (cx, cy)) @ R.T) + (cx, cy) + off for q in polys]
            fill_polys(ctx, polys, mix(pal["grass"], pal["light_warm"], 0.25), a)

    # ── 共用 ──────────────────────────────────────────────
    def _blit(self, ctx, ch, x, y, size, rgb, alpha, rot=0.0):
        """以快取貼圖繪製宋體字；(x, y) 為基線左端。"""
        S = 96
        if ch not in self.sprites:
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, S, S)
            cc = cairo.Context(surf)
            g = self.face.glyph(ch)
            fill_polys(cc, [q * S * 0.9 + (S * 0.05, S * 0.85) for q in g.contours], (1, 1, 1), 1.0)
            self.sprites[ch] = surf
        sc = size / (S * 0.9)
        ctx.save()
        # 以字心為軸旋轉，再把貼圖的基線左端對到 (x, y)
        ctx.translate(x + size / 2, y - size * 0.38)
        ctx.rotate(rot)
        ctx.translate(-size / 2, size * 0.38)
        ctx.scale(sc, sc)
        ctx.translate(-S * 0.05, -S * 0.85)
        ctx.set_source_rgba(float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))
        ctx.mask_surface(self.sprites[ch], 0, 0)
        ctx.restore()

    def render(self, frame):
        t = frame / self.fps
        c = self.c
        pal = self.tl.pal
        img = np.broadcast_to(BG, (self.H, self.W, 3)).astype(np.float32).copy()
        lay = self.layer
        lay.clear()
        ctx = lay.ctx
        if t < c("stop"):
            gl = ease(ramp(t, c("cycle"), c("cycle") + 0.6))
            ctx.move_to(0, clumpmod.BASE[1] + 0.5)
            ctx.line_to(self.W, clumpmod.BASE[1] + 0.5)
            ctx.set_source_rgba(0.42, 0.34, 0.26, 0.6 * gl)
            ctx.set_line_width(1.2)
            ctx.stroke()
            self._draw_cycle(ctx, t)
        elif t >= c("farewell"):
            self._draw_farewell(ctx, t)
        # 被大地收下：溫暖的土色由下往上沉積
        e = ease(ramp(t, c("earth"), c("music_end") - 0.3))
        if e > 0:
            edge = self.H * (1 - e * 1.25) + 0.0
            ys = np.arange(self.H, dtype=np.float32)
            m = np.clip((ys - edge) / 260 + 0.5, 0, 1)[:, None, None]
            img = mix(img, np.broadcast_to(pal["earth"], img.shape), m).astype(np.float32)
        over(img, lay.rgba())
        return img
