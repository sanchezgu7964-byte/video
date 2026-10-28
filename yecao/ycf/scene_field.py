"""第 4–8 段：字長成草 → 踐踏刪刈 → 坦然 → 自愛與網格 → 地火噴發 → 凍結。

世界座標：地平線 y=0，向上為負；鏡頭 screen = (960 + s·(x−960), Hz + s·y)。
字草的模擬（彈簧擺動、壓扁、切斷）以固定步長逐影格推進，任何影格都能由場景起點重播得到。
"""
import cairo
import numpy as np
from PIL import Image, ImageDraw

from . import atmos
from . import clump as clumpmod
from . import fonts
from . import tree as treemod
from .gfx import (Layer, bloom, ease, ease_in, ease_out, fill_polys, fill_polys_grad, lin_gradient_v, mix, over,
                  path_polys,
                  ramp, resize, splat)
from .noise import fbm, noise3, rand

G = 96.0             # 網格格子（世界單位）
WIN_I = (-9, 9)      # 野草窗口所佔的欄
WIN_J = (-3, -1)     # 野草窗口所佔的列（地平線以上三列）


def clip_u(poly, lo, hi):
    """把 (x, u) 多邊形裁到 lo ≤ u ≤ hi（Sutherland–Hodgman，兩個半平面）。"""
    def clip(p, keep, edge):
        out = []
        n = len(p)
        for k in range(n):
            a, b = p[k - 1], p[k]
            ka, kb = keep(a[1]), keep(b[1])
            if kb:
                if not ka:
                    out.append(edge(a, b))
                out.append(b)
            elif ka:
                out.append(edge(a, b))
        return out

    def isect(level):
        def f(a, b):
            t = (level - a[1]) / (b[1] - a[1])
            return (a[0] + t * (b[0] - a[0]), level)
        return f

    p = [tuple(q) for q in poly]
    if hi < 1.0:
        p = clip(p, lambda u: u <= hi, isect(hi))
    if lo > 0.0 and p:
        p = clip(p, lambda u: u >= lo, isect(lo))
    return np.array(p, np.float32) if len(p) >= 3 else None


class Plant:
    def __init__(self, k, ch, glyph, x, row, size, rng, t_rise):
        self.k, self.ch, self.x, self.row, self.size = k, ch, x, row, size
        x0, y0, x1, y1 = glyph.bbox
        self.h = (y1 - y0) * size
        # 局部座標 (x, u)：x 以字心為 0（像素），u 為 0（根）到 1（梢）
        self.base = [np.stack([(c[:, 0] - (x0 + x1) / 2) * size, (y1 - c[:, 1]) / (y1 - y0)], 1)
                     for c in glyph.contours]
        self.local = self.base
        self.cut_u = 1.0
        self.theta = 0.0
        self.omega = 0.0
        self.squash = 1.0
        self.squash_v = 0.0
        self.squash_rest = 1.0
        self.freq = 0.55 + 0.9 * (1 - min(size, 240) / 240) + 0.15 * rng.random()
        self.flex = 0.6 + 0.6 * rng.random()
        self.phase = rng.random() * 100
        self.t_rise = t_rise
        self.tint = 0.4 + 0.6 * rng.random()
        n_roots = rng.integers(3, 6)
        self.roots = []
        for _ in range(n_roots):
            L = size * (0.12 + 0.22 * rng.random())
            n = 9
            ang = rng.normal(0, 0.55)
            pts = [(rng.normal(0, size * 0.12), 0.0)]
            for i in range(n):
                ang += rng.normal(0, 0.35)
                px, py = pts[-1]
                pts.append((px + np.sin(ang) * L / n, py + np.cos(ang) * L / n))
            self.roots.append(np.array(pts, np.float32))
        self.absorb_t = None   # 被網格吸收的起始時間
        self.cell = None

    def set_cut(self, u):
        self.cut_u = u
        self.local = [q for q in (clip_u(c, 0.0, u) for c in self.base) if q is not None]

    def piece_local(self, lo, hi):
        return [q for q in (clip_u(c, lo, hi) for c in self.base) if q is not None]

    def warp(self, local, theta=None, squash=None):
        """把局部 (x, u) 輪廓彎折成世界中相對根部的座標（y 向上為負）。"""
        th = self.theta if theta is None else theta
        sq = self.squash if squash is None else squash
        L = self.h * sq
        bulge = 1 + 0.5 * (1 - sq)
        out = []
        for c in local:
            x = c[:, 0] * bulge
            u = c[:, 1]
            if abs(th) < 1e-4:
                px, py = x, u * L
            else:
                phi = th * u
                r = L / th
                cx, cy = r * (1 - np.cos(phi)), r * np.sin(phi)
                px, py = cx + x * np.cos(phi), cy - x * np.sin(phi)
            out.append(np.stack([px, -py], 1))
        return out

    def uplift(self):
        return min(abs(self.theta), 1.0) * self.h * 0.10


def shift(img, dx, dy):
    """整張畫面平移（邊緣複製），用於撞擊的震動。"""
    H, W = img.shape[:2]
    pad = max(abs(dx), abs(dy)) + 1
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return p[pad - dy:pad - dy + H, pad - dx:pad - dx + W]


def calm_k(t, c):
    """第 5 段（呼吸之後、自愛之前）的明亮程度。"""
    return ease(ramp(t, c("breath"), c("breath") + 2)) * (1 - ease(ramp(t, c("selflove") - 1, c("selflove") + 2)))


class FieldScene:
    t0 = 42.0
    t1 = 172.0

    def __init__(self, tl):
        self.tl = tl
        self.W, self.H = tl.W, tl.H
        self.fps = tl.fps
        self.dt = 1 / tl.fps
        self.seed = tl.seed
        c = tl.c
        self.c = c
        self.gface = fonts.face(tl.raw["fonts"]["grass"])
        self.sface = fonts.face(tl.raw["fonts"]["sans"])
        self.layer = Layer(self.W, self.H)
        self.glow = Layer(self.W, self.H)
        self.far = Layer(self.W, self.H)
        self.sky = atmos.Sky(self.seed, self.W, self.H)
        self.soil = atmos.Soil(tl)
        self.motes = atmos.Motes(self.seed, self.W, self.H)
        self._build_plants()
        self._build_events()
        self._build_grid()
        self._build_names()
        self.frame = int(round(self.t0 * self.fps)) - 1
        self._build_tree()
        self._build_witness()
        self._build_impacts()
        self.pieces = []      # 被切下、正在落下的字塊
        self.ash = None
        self.lava_cache = None

    # ── 建構 ──────────────────────────────────────────────
    def _build_plants(self):
        rng = rand(self.seed, 4, 1)
        text = "".join(ln["text"] for ln in self.tl.raw["line"] if 35 <= ln["at"] < 80)
        chars = [ch for ch in text if ch not in "，。；！、："]
        # 一叢一叢地長：先定叢的位置，每叢 3–6 株，前中後三列混雜
        centers = np.linspace(110, 1810, 8) + rng.normal(0, 55, 8)
        self.plants = []
        k = 0
        for cx in centers:
            for _ in range(rng.integers(3, 7)):
                row = int(rng.choice(3, p=[0.45, 0.35, 0.20]))
                smin, smax = [(46, 78), (92, 140), (170, 250)][row]
                size = rng.uniform(smin, smax)
                x = float(np.clip(cx + rng.normal(0, 70 + 30 * row), 30, 1890))
                ch = chars[(k * 7 + 3) % len(chars)]
                g = self.gface.glyph(ch)
                t_rise = self.c("grass_rise") + rng.uniform(0, 7.5) + (2 - row) * 0.6
                self.plants.append(Plant(k, ch, g, x, row, size, rng, t_rise))
                k += 1
        self.max_h = max(p.h for p in self.plants)
        # 地平線上一整排細小、模糊的字草，給田野深度
        self.fringe = []
        xs = np.sort(rng.uniform(-40, 1960, 110))
        for i, x in enumerate(xs):
            ch = chars[(i * 11 + 5) % len(chars)]
            size = rng.uniform(16, 34)
            t_rise = self.c("grass_rise") + 1.5 + rng.uniform(0, 8)
            fp = Plant(1000 + i, ch, self.gface.glyph(ch), float(x), -1, size, rng, t_rise)
            fp.roots = []
            self.fringe.append(fp)

    def _build_tree(self):
        c = self.c
        rng = rand(self.seed, 3, 2)
        segs = treemod.lsystem(self.seed, step=19.0)
        segs[:, [0, 2]] += 960
        self.tree_L = float(segs[:, 6].max()) * 0.55     # 只長到一半
        self.tree = segs[segs[:, 5] < self.tree_L]
        n = len(self.tree)
        # 崩落後每一節變成躺在地上的一小筆
        mid = (self.tree[:, 0:2] + self.tree[:, 2:4]) / 2
        self.stroke_rest = np.stack([mid[:, 0] + rng.normal(0, 60, n), -rng.uniform(2, 10, n)], 1)
        self.stroke_ang = rng.normal(0, 0.35, n) + rng.choice([0, np.pi], n)
        self.stroke_len = rng.uniform(8, 16, n)
        self.stroke_fall_delay = (-self.tree[:, 1] / 600) * 0.25 + rng.uniform(0, 0.15, n)
        # 每一筆被一株字草收去
        self.stroke_plant = rng.integers(0, len(self.plants), n)
        self.stroke_target = np.zeros((n, 2), np.float32)
        for i in range(n):
            p = self.plants[self.stroke_plant[i]]
            q = p.base[rng.integers(len(p.base))]
            pt = q[rng.integers(len(q))]
            self.stroke_target[i] = (p.x + pt[0], -pt[1] * p.h * 0.6)
        # 落地濺起的泥點
        m = 70
        ang = np.radians(rng.uniform(-170, -10, m))
        sp = rng.uniform(150, 620, m)
        self.splash = dict(v=np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1), r=rng.uniform(1.5, 5, m))

    def _draw_mud_land(self, ctx, t):
        c = self.c
        u = t - c("mud_land")
        if u < 0 or u > 1.2:
            return
        pal = self.tl.pal
        # 泥團攤開
        k = ease_out(ramp(u, 0, 0.45), 3)
        w = mix(150, 900, k)
        hgt = mix(120, 0, ease(ramp(u, 0, 0.6)))
        if hgt > 0.5:
            ctx.save()
            ctx.translate(960, 720 - hgt * 0.4)
            ctx.scale(w / 2, max(hgt / 2, 0.5))
            ctx.arc(0, 0, 1, 0, 2 * np.pi)
            ctx.restore()
            ctx.set_source_rgba(*[float(v) for v in pal["mud"] * 0.82], 1)
            ctx.fill()
        sp = self.splash
        pos = np.array([960, 700]) + sp["v"] * u + np.array([0, 1400]) * u * u / 2
        for (x, y), r in zip(pos, sp["r"]):
            if y < 720:
                ctx.arc(float(x), float(y), float(r), 0, 2 * np.pi)
                ctx.set_source_rgba(*[float(v) for v in pal["mud"]], 1)
                ctx.fill()

    def _draw_tree(self, ctx, t):
        c = self.c
        if t < c("mud_land") or t > c("grass_rise") + 9:
            return
        pal = self.tl.pal
        L = self.tree_L * ease_out(ramp(t, c("mud_land") + 0.25, c("tree_peak")), 1.6)
        wither = ease(ramp(t, c("tree_peak"), c("tree_collapse")))
        col = mix(np.array([0.36, 0.28, 0.2]), np.array([0.17, 0.13, 0.1]), wither)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        segs = self.tree
        if t < c("tree_collapse"):
            for x0, y0, x1, y1, depth, s0, s1 in segs:
                if s0 >= L:
                    continue
                f = min(1.0, (L - s0) / (s1 - s0))
                p = np.array([[x0, y0], [x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]], np.float64)
                # 萎縮：枝條向外向下垂
                if wither > 0:
                    rel = p - (960, 0)
                    sag = wither * 0.55 * (s0 / self.tree_L) * np.sign(rel[:, 0].mean() + 1e-3)
                    cs, sn = np.cos(sag), np.sin(sag)
                    rel = rel @ np.array([[cs, sn], [-sn, cs]])
                    p = rel * (1 - 0.18 * wither) + (960, 0)
                sc = self.to_screen(p, t)
                ctx.move_to(*sc[0])
                ctx.line_to(*sc[1])
                ctx.set_source_rgba(*[float(v) for v in col], 1)
                ctx.set_line_width(max(1.2, (17 - depth * 3.2) * (1 - 0.4 * wither)))
                ctx.stroke()
            return
        # 崩回地面、碎成短筆畫，再飛去組成字
        for i, (x0, y0, x1, y1, depth, s0, s1) in enumerate(segs):
            mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            u = ease_in(ramp(t, c("tree_collapse") + self.stroke_fall_delay[i], c("tree_collapse") + self.stroke_fall_delay[i] + 0.55), 2)
            pos = mix(mid, self.stroke_rest[i], u)
            a0 = np.arctan2(y1 - y0, x1 - x0)
            ang = mix(a0, self.stroke_ang[i], u)
            ln = mix(19.0, self.stroke_len[i], u)
            p = self.plants[self.stroke_plant[i]]
            fly = ease(ramp(t, p.t_rise - 0.9, p.t_rise - 0.1))
            alpha = 1 - ease(ramp(t, p.t_rise - 0.25, p.t_rise + 0.35))
            if alpha <= 0:
                continue
            pos = mix(pos, self.stroke_target[i], fly)
            d = np.array([np.cos(ang), np.sin(ang)]) * ln / 2
            q = np.array([pos - d, pos + d])
            sc = self.to_screen(q, t)
            ctx.move_to(*sc[0])
            ctx.line_to(*sc[1])
            sc_col = mix(col, pal["grass"] * 0.8, fly)
            ctx.set_source_rgba(*[float(v) for v in sc_col], float(alpha))
            ctx.set_line_width(mix(max(1.4, 9 - depth * 1.8), 2.6, u))
            ctx.stroke()

    def _build_witness(self):
        c = self.c
        rng = rand(self.seed, 10, 3)
        self.clump = clumpmod.make_clump(self.tl)
        beats = self.tl.beats_between(c("witness"), c("cycle"))
        angs = [np.radians(rng.uniform(-35, 35))]
        for _ in beats[1:]:
            a = angs[-1] + np.radians(rng.choice([-1, 1]) * rng.uniform(28, 70))
            angs.append(a)
        self.w_beats = beats
        self.w_angles = np.array(angs)
        # 三對輪廓：友與仇、人與獸、愛者與不愛者——用字本身的輪廓，放得很大、只剩一條線
        serif = fonts.face(self.tl.raw["fonts"]["serif"])
        self.pairs = []
        spec = [(-500, 175, 250, "友", "仇", False), (470, 205, 290, "人", "獸", False), (-20, 240, 200, "愛", "愛", True)]
        for k, (u0, v0, size, a, b, dashed) in enumerate(spec):
            shapes = []
            for ch in (a, b):
                g = serif.glyph(ch)
                x0, y0, x1, y1 = g.bbox
                ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
                shapes.append([((cc - ctr) * size).astype(np.float32) for cc in g.contours])
            t_in = float(beats[min(3 + 3 * k, len(beats) - 1)])
            self.pairs.append(dict(u=u0, v=v0, shapes=shapes, t=t_in, dashed=dashed))
        self.gather_n = 26000
        self.gather_tgt = clumpmod.sample_points(self.clump, self.gather_n, self.seed, self.W, self.H)

    def w_angle(self, t):
        b = self.w_beats
        i = np.searchsorted(b, t, side="right") - 1
        if i <= 0:
            return float(self.w_angles[0])
        k = ease(ramp(t, b[i], b[i] + 0.14))
        return float(mix(self.w_angles[i - 1], self.w_angles[i], k))

    def _render_witness(self, img, t):
        """第 10 段：分界線剖開畫面，每一拍旋轉重切；兩側明暗冷暖對立。"""
        c = self.c
        pal = self.tl.pal
        cx, cy = clumpmod.BASE
        ang = self.w_angle(t)
        dvec = np.array([np.cos(ang), np.sin(ang)])
        nvec = np.array([-np.sin(ang), np.cos(ang)])
        cut = ease_out(ramp(t, c("witness"), c("witness") + 0.35))
        out_k = ease(ramp(t, c("cycle") - 0.7, c("cycle")))
        if not hasattr(self, "_xx"):
            self._yy, self._xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        d = (self._xx - cx) * nvec[0] + (self._yy - cy) * nvec[1]
        side = np.clip(d / 1.2 + 0.5, 0, 1)[..., None]
        light = np.array([0.44, 0.39, 0.31], np.float32)
        dark = np.array([0.055, 0.065, 0.085], np.float32)
        split = light * side + dark * (1 - side)
        # 同一道光：以叢為中心的暖光同時照到兩側
        r2 = ((self._xx - cx) ** 2 + (self._yy - cy + 60) ** 2) / (520 ** 2)
        split = split + (np.exp(-r2) * 0.12)[..., None] * pal["light_warm"]
        k = cut * (1 - out_k)
        end_bg = np.array([0.05, 0.045, 0.04], np.float32)
        img[:] = mix(img, split, k) * (1 - out_k) + end_bg * out_k
        lay = self.layer
        lay.clear()
        ctx = lay.ctx
        # 分界線本身
        if k > 0:
            L = 2400 * cut
            p0, p1 = np.array([cx, cy]) - dvec * L / 2, np.array([cx, cy]) + dvec * L / 2
            ctx.move_to(*p0)
            ctx.line_to(*p1)
            ctx.set_source_rgba(0.9, 0.86, 0.78, 0.55 * k)
            ctx.set_line_width(1.0)
            ctx.stroke()
        # 成對輪廓：同一道光（以叢為中心的暖色徑向漸層）照在兩邊
        light = cairo.RadialGradient(cx, cy - 60, 0, cx, cy - 60, 900)
        light.add_color_stop_rgba(0, *[float(v) for v in pal["light_warm"]], 0.9)
        light.add_color_stop_rgba(1, *[float(v) for v in pal["light_warm"]], 0.0)
        for pr in self.pairs:
            a = ease(ramp(t, pr["t"], pr["t"] + 0.8)) * (1 - out_k)
            if a <= 0:
                continue
            draw = ease(ramp(t, pr["t"], pr["t"] + 1.2))
            for side, sgn in ((0, 1), (1, -1)):
                center = np.array([cx, cy]) + pr["u"] * dvec + sgn * pr["v"] * nvec
                # 字的方向跟著分界線轉
                R = np.stack([dvec, nvec], 1)
                polys = []
                for q in pr["shapes"][side]:
                    n = max(2, int(len(q) * draw))
                    polys.append(q[:n] @ R.T + center)
                on_light = (center - (cx, cy)) @ nvec > 0
                cc = (0.17, 0.14, 0.11) if on_light else (0.78, 0.75, 0.69)
                for src, width, alpha in (((*cc, 1.0), 1.8, 0.8 * a), (light, 2.4, 0.6 * a)):
                    ctx.save()
                    if pr["dashed"] and side == 1:
                        ctx.set_dash([7, 9])     # 不愛者：斷續的同一個字
                    ctx.push_group()
                    ctx.new_path()
                    path_polys(ctx, polys, close=draw >= 1)
                    if isinstance(src, tuple):
                        ctx.set_source_rgba(*src)
                    else:
                        ctx.set_source(src)
                    ctx.set_line_width(width)
                    ctx.stroke()
                    ctx.pop_group_to_source()
                    ctx.paint_with_alpha(alpha)
                    ctx.restore()
        # 灰燼聚成的一叢字草
        ca = ease(ramp(t, c("witness") + 2.2, c("witness") + 3.0))
        if ca > 0:
            sway = 0.05 * np.sin(t * 1.9)
            for p in self.clump:
                polys = clumpmod.clump_polys(p, theta=sway * (1.2 - p.size / 130))
                fill_polys(ctx, polys, mix(pal["grass"], pal["light_warm"], 0.25), ca)
        over(img, lay.rgba())
        return img

    def _build_impacts(self):
        c = self.c
        rng = rand(self.seed, 4, 12)
        # (時間, 幅度 px, 軸, 衰減秒)
        self.shakes = [(c("mud_land"), 13, 1, 0.22), (c("erupt"), 20, 1, 0.55), (c("restore"), 11, 0, 0.35)]
        self.dust = []
        for ev in self.events:
            if ev["kind"] == "stomp":
                self.shakes.append((ev["t"], 9, 1, 0.14))
                n = 220
                x = rng.uniform(ev["xc"] - ev["hw"], ev["xc"] + ev["hw"], n)
                side = np.sign(x - ev["xc"] + 1e-3)
                v = np.stack([side * rng.uniform(60, 560, n), -rng.uniform(40, 330, n)], 1)
                self.dust.append(dict(t=ev["t"], p0=np.stack([x, -rng.uniform(0, 6, n)], 1), v=v,
                                      life=rng.uniform(0.5, 1.6, n), w=rng.uniform(0.3, 1.0, n)))
            else:
                self.shakes.append((ev["t"], 3, 0, 0.1))

    def shake(self, t):
        dx = dy = 0.0
        for t0, amp, axis, dec in self.shakes:
            u = t - t0
            if 0 <= u < dec * 5:
                v = amp * np.exp(-u / dec) * np.sin(2 * np.pi * 11 * u + 0.4)
                if axis:
                    dy += v
                else:
                    dx += v
        return dx, dy

    def _draw_dust(self, img, t):
        for d in self.dust:
            u = t - d["t"]
            if u < 0 or u > 1.7:
                continue
            k = 2.2
            e = (1 - np.exp(-k * u)) / k
            pos = d["p0"] + d["v"] * e + np.array([0, 420]) * u * u / 2
            pos[:, 1] = np.minimum(pos[:, 1], 0)
            a = np.clip(1 - u / d["life"], 0, 1) * d["w"]
            sc = self.to_screen(pos, t)
            buf = np.zeros_like(img)
            splat(buf, sc, np.array([0.5, 0.42, 0.33], np.float32), a * 0.9)
            img += buf + bloom(buf, 9, down=4) * 2.5

    def _build_events(self):
        tl = self.tl
        rng = rand(self.seed, 4, 2)
        t_a, t_h, t_b = self.c("stomp_start"), self.c("stomp_halfbeat"), self.c("breath")
        beats = list(tl.beats_between(t_a, t_b))
        times = []
        for i, b in enumerate(beats):
            times.append(b)
            if b >= t_h and i + 1 < len(beats):
                times.append((b + beats[i + 1]) / 2)
        times = [t for t in times if t < t_b - 0.3]
        self.events = []
        n = len(times)
        for i, te in enumerate(times):
            prog = i / max(n - 1, 1)
            if i % 2 == 0:
                hw = rng.uniform(200, 380)
                xc = rng.uniform(150, 1770)
                self.events.append(dict(kind="stomp", t=te, xc=xc, hw=hw, done=False))
            else:
                hc = mix(0.82, 0.30, prog) * self.max_h * rng.uniform(0.85, 1.1)
                d = 1 if (i // 2) % 2 == 0 else -1
                self.events.append(dict(kind="blade", t=te, hc=hc, dir=d, dur=0.2, done=set()))

    def _build_grid(self):
        text = self.tl.raw["grid"]["text"]
        self.cells = []
        self.cell_index = {}
        k = 0
        for j in range(-15, 14):
            for i in range(-26, 27):
                ch = text[k % len(text)]
                k += 1
                in_win = WIN_I[0] <= i <= WIN_I[1] and WIN_J[0] <= j <= WIN_J[1]
                # d：離野草窗口幾列，決定網格出現的先後
                if j >= 0:
                    d = j + 1
                elif j >= WIN_J[0]:
                    d = 0
                else:
                    d = WIN_J[0] - j
                cell = dict(i=i, j=j, ch=ch, win=in_win, d=d, cx=960 + i * G, cy=(j + 0.5) * G,
                            fill_t=None, plant=None)
                self.cells.append(cell)
                self.cell_index[(i, j)] = cell
        # 網格出現：每一拍向外推兩列
        beats = self.tl.beats_between(self.c("grid_in"), self.c("grid_in") + 20)
        for cell in self.cells:
            if cell["win"]:
                continue
            b = min(cell["d"] // 2, len(beats) - 1)
            cell["t_in"] = float(beats[b]) + 0.03 * ((cell["i"] * 7 + cell["j"] * 3) % 5)
        # 吸收：部分野草依拍點被排進窗口最下一列的格子
        rng = rand(self.seed, 4, 3)
        survivors = list(range(len(self.plants)))
        rng.shuffle(survivors)
        absorb_beats = self.tl.beats_between(self.c("grid_absorb"), self.c("fire_rise") - 0.5)
        win_cells = [self.cell_index[(i, j)] for j in (-1, -2, -3) for i in range(WIN_I[0], WIN_I[1] + 1)]
        n_abs = int(len(self.plants) * 0.72)
        for n, pk in enumerate(survivors[:n_abs]):
            p = self.plants[pk]
            p.absorb_t = float(absorb_beats[min(n * len(absorb_beats) // n_abs, len(absorb_beats) - 1)])
            p.absorb_t += 0.12 * (n % 3)
        # 依 x 指派到最近的空格（優先最下一列）
        order = sorted([p for p in self.plants if p.absorb_t is not None], key=lambda p: p.x)
        free = list(win_cells)
        for p in order:
            best = min(free, key=lambda c: abs(c["cx"] - p.x) + 260 * (-1 - c["j"]))
            free.remove(best)
            p.cell = best
            best["plant"] = p
            best["fill_t"] = p.absorb_t + 1.1
        rest_beats = self.tl.beats_between(self.c("grid_absorb") + 2, self.c("fire_rise"))
        for n, cell in enumerate(free):
            cell["fill_t"] = float(rest_beats[n % len(rest_beats)]) + 0.04 * n % 0.3
        # 網格字的貼圖快取
        self.sprites = {}

    def _build_names(self):
        rng = rand(self.seed, 4, 4)
        names = self.tl.raw["names"]["list"]
        self.names = []
        t0, t1 = self.c("names_start"), self.c("names_end")
        n = max(len(names), 6)
        for k in range(n):
            strokes = []
            if names:
                strokes = None
                label = names[k % len(names)]
            else:
                label = None
                # 無法辨認的筆跡：2–4 個「字」，每字 3–6 筆平滑隨機筆畫
                for ci in range(rng.integers(2, 5)):
                    for _ in range(rng.integers(3, 7)):
                        p0 = np.array([ci * 26 + rng.uniform(0, 20), rng.uniform(-22, 0)])
                        pts = [p0]
                        ang = rng.uniform(0, 2 * np.pi)
                        for _ in range(6):
                            ang += rng.normal(0, 0.5)
                            pts.append(pts[-1] + 3.2 * np.array([np.cos(ang), np.sin(ang)]))
                        strokes.append(np.array(pts, np.float32))
            self.names.append(dict(t=rng.uniform(t0, t1), x=rng.uniform(150, 1700),
                                   y=rng.uniform(70, 260), strokes=strokes, label=label))

    # ── 鏡頭 ──────────────────────────────────────────────
    def camera(self, t):
        z = ease(ramp(t, self.c("zoom_start"), self.c("zoom_end")))
        s = mix(1.0, 0.42, z)
        hz = mix(720.0, 560.0, z)
        return s, hz

    def to_screen(self, pts, t):
        s, hz = self.camera(t)
        return np.stack([960 + s * (pts[..., 0] - 960), hz + s * pts[..., 1]], -1)

    # ── 模擬 ──────────────────────────────────────────────
    def wind(self, t, x):
        c = self.c
        base = 0.18
        if t < c("stomp_start"):
            base = 0.16 + 0.1 * ramp(t, c("blood"), c("stomp_start"))
        elif t < c("breath"):
            base = 0.30
        elif t < c("selflove"):
            base = 0.10
        else:
            base = mix(0.03, 0.05, ramp(t, c("zoom_start"), c("zoom_end")))
        gust = noise3(x / 520, t * 0.35, 1.0, 11) * 0.8 + noise3(x / 160, t * 1.3, 2.0, 12) * 0.35
        # 一陣一陣的風由左向右掃過，每 4.5 秒一道
        front = (t * 520) % 2340 - 210
        wave = np.exp(-((x - front) / 230) ** 2) * (1.6 if t < c("breath") else 0.7)
        return base * (0.55 + gust + wave)

    def advance_to(self, frame):
        while self.frame < frame:
            self.frame += 1
            self._step(self.frame / self.fps)

    def _step(self, t):
        dt = self.dt
        c = self.c
        sing = c("sing") <= t < c("selflove") - 0.6
        for p in self.plants + self.fringe:
            if t < p.t_rise:
                continue
            target = self.wind(t, p.x) * p.flex
            if sing:
                b0, ph = self.tl.beat_phase(t - p.x / 9000)
                target += 0.16 * np.sin(2 * np.pi * ph) * ease(ramp(t, c("sing"), c("sing") + 1.5))
            if p.absorb_t is not None and t > p.absorb_t:
                target = 0.0
            k = (2 * np.pi * p.freq) ** 2
            for _ in range(2):
                acc = k * (target - p.theta) - 2 * 0.16 * np.sqrt(k) * p.omega
                p.omega += acc * dt / 2
                p.theta += p.omega * dt / 2
            p.theta = float(np.clip(p.theta, -1.25, 1.25))
            ks = (2 * np.pi * 1.6) ** 2
            acc = ks * (p.squash_rest - p.squash) - 2 * 0.35 * np.sqrt(ks) * p.squash_v
            p.squash_v += acc * dt
            p.squash += p.squash_v * dt
            p.squash = float(np.clip(p.squash, 0.18, 1.1))
        for ev in self.events:
            if ev["kind"] == "stomp" and not ev["done"] and t >= ev["t"]:
                ev["done"] = True
                for p in self.plants + self.fringe:
                    d = abs(p.x - ev["xc"])
                    if d < ev["hw"] and t >= p.t_rise:
                        f = 1 - (d / ev["hw"]) ** 2
                        p.squash = min(p.squash, mix(1.0, 0.24, f))
                        p.squash_v = 0
                        p.squash_rest = max(0.42, p.squash_rest * mix(1.0, 0.80, f))
                        p.omega += np.sign(p.x - ev["xc"] + 1e-3) * 3.5 * f
            elif ev["kind"] == "blade":
                t_a = ev["t"] - ev["dur"] / 2
                if t_a <= t <= ev["t"] + ev["dur"] / 2 + dt:
                    bx = self._blade_x(ev, t)
                    for p in self.plants:
                        if p.k in ev["done"] or t < p.t_rise:
                            continue
                        crossed = (bx >= p.x) if ev["dir"] > 0 else (bx <= p.x)
                        if crossed:
                            ev["done"].add(p.k)
                            self._try_cut(p, ev, t)
        for pc in self.pieces:
            pc["vel"][1] += 1500 * dt
            pc["pos"] += pc["vel"] * dt
            pc["rot"] += pc["spin"] * dt
            if pc["pos"][1] > -6:
                pc["pos"][1] = -6
                pc["vel"] *= 0.3
                pc["spin"] *= 0.3
        if t >= c("erupt") and self.ash is None:
            self._spawn_ash(t)
        if self.ash is not None:
            self._step_ash(t)

    def _blade_x(self, ev, t):
        u = (t - (ev["t"] - ev["dur"] / 2)) / ev["dur"]
        x = mix(-120, 2040, u)
        return x if ev["dir"] > 0 else 1920 - x

    def _try_cut(self, p, ev, t):
        top = p.h * p.squash * p.cut_u
        if top <= ev["hc"] + 6:
            return
        u = ev["hc"] / (p.h * p.squash)
        if u < 0.12:
            return
        # 切下的字塊：取切點以上的部分，在切的瞬間定格成世界座標
        loc = p.piece_local(u, p.cut_u)
        world = [w + (p.x, -p.uplift()) for w in p.warp(loc)]
        p.set_cut(u)
        if not world:
            return
        allp = np.concatenate(world)
        pivot = allp.mean(0)
        rng = rand(self.seed, 40, p.k, int(t * 1000))
        self.pieces.append(dict(
            polys=[w - pivot for w in world], pos=pivot.astype(np.float64),
            vel=np.array([ev["dir"] * rng.uniform(250, 600), rng.uniform(-420, -150)]),
            rot=0.0, spin=ev["dir"] * rng.uniform(2, 7), t=t, k=p.k,
            cut=np.array([p.x, -ev["hc"]]), dir=ev["dir"],
            slivers=np.stack([ev["dir"] * rng.uniform(150, 900, 9), rng.uniform(-500, 120, 9)], 1)))

    # ── 灰燼 ─────────────────────────────────────────────
    def _spawn_ash(self, t):
        # 以噴發前一刻的字形遮罩取樣粒子位置
        mask = self._glyph_mask(self.c("erupt") - 0.05)
        ys, xs = np.nonzero(mask > 0.25)
        rng = rand(self.seed, 7, 1)
        n = 90000
        idx = rng.choice(len(xs), n, replace=True)
        pos = np.stack([xs[idx], ys[idx]], 1).astype(np.float32) * 2 + rng.random((n, 2)).astype(np.float32) * 2
        vel = np.stack([rng.normal(0, 55, n), -rng.uniform(60, 420, n) * (0.5 + 0.5 * (pos[:, 1] / self.H))],
                       1).astype(np.float32)
        # 火線由下往上掃過：被燒到之前，字形仍以發光的塵停在原處
        release = (self.c("erupt") + 0.05 + (1 - pos[:, 1] / self.H) * 1.1
                   + rng.uniform(0, 0.12, n)).astype(np.float32)
        self.ash = dict(pos=pos, vel=vel, release=release, temp=np.full(n, 0.15, np.float32),
                        cool=rng.uniform(0.15, 0.6, n).astype(np.float32),
                        w=rng.uniform(0.4, 1.0, n).astype(np.float32),
                        big=(rng.random(n) < 0.025))

    def _step_ash(self, t):
        a = self.ash
        c = self.c
        dt = self.dt
        if c("freeze") <= t < c("restore"):
            return
        speed = 1.0
        if t > c("freeze") - 0.45 and t < c("freeze"):
            speed = 1 - ease(ramp(t, c("freeze") - 0.45, c("freeze")))
        churn = t >= c("restore")
        p, v = a["pos"], a["vel"]
        live = t >= a["release"]
        if not live.all() and not churn:
            # 還沒被火線掃到的：不動，溫度隨火線逼近上升
            near = np.clip(1 - (a["release"] - t) / 0.4, 0, 1)
            a["temp"] = np.where(live, a["temp"], np.maximum(a["temp"], 0.15 + 0.85 * near).astype(np.float32))
        tz = t * (0.6 if not churn else 1.6)
        nx = noise3(p[:, 0] / 210, p[:, 1] / 210, tz, 31)
        ny = noise3(p[:, 0] / 210, p[:, 1] / 210, tz, 32)
        if churn:
            # 「天地即不如此靜穆」：凍結的粒子劇烈翻騰
            v += np.stack([nx, ny], 1) * 2600 * dt
            v *= 0.94
            # 向畫面中央的柔性回拉，避免散出畫面
            v += (np.array([960, 540]) - p) * 0.9 * dt
        else:
            v[:, 0] += nx * 260 * dt
            v[:, 1] += ny * 200 * dt - 25 * dt
            v *= (1 - 0.55 * dt)
        m = live[:, None] if not churn else True
        p += np.where(m, v * dt * speed, 0)
        a["temp"] = np.where(live | churn, np.maximum(0, a["temp"] - a["cool"] * dt * speed), a["temp"]).astype(np.float32)

    def _glyph_mask(self, t):
        """回傳半解析的字形遮罩（植物＋網格），供灰燼取樣。"""
        lay = Layer(self.W // 2, self.H // 2)
        lay.ctx.scale(0.5, 0.5)
        self._draw_grid(lay.ctx, t, mono=True)
        self._draw_plants(lay.ctx, t, mono=True)
        return lay.rgba()[..., 3]

    # ── 繪製 ──────────────────────────────────────────────
    def _sprite(self, ch):
        if ch not in self.sprites:
            S = 128
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, S, S)
            ctx = cairo.Context(surf)
            g = self.sface.glyph(ch)
            polys = [cc * S * 0.86 + (S * 0.07, S * 0.82) for cc in g.contours]
            fill_polys(ctx, polys, (1, 1, 1), 1.0)
            self.sprites[ch] = surf
        return self.sprites[ch]

    def _blit_char(self, ctx, ch, cx, cy, size, rgb, alpha):
        S = 128
        surf = self._sprite(ch)
        ctx.save()
        ctx.translate(cx - size / 2, cy - size / 2)
        ctx.scale(size / S, size / S)
        ctx.set_source_rgba(float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))
        ctx.mask_surface(surf, 0, 0)
        ctx.restore()

    def _grid_alpha(self, cell, t):
        if cell["win"]:
            if cell["fill_t"] is None:
                return 0.0
            return ease(ramp(t, cell["fill_t"], cell["fill_t"] + 0.35))
        if "t_in" not in cell:
            return 0.0
        return ease(ramp(t, cell["t_in"], cell["t_in"] + 0.3))

    def _draw_grid(self, ctx, t, mono=False, emissive=None):
        if t < self.c("grid_in"):
            return
        s, hz = self.camera(t)
        col = self.tl.pal["grid"]
        size = G * 0.62 * s
        for cell in self.cells:
            a = self._grid_alpha(cell, t)
            if a <= 0:
                continue
            sx = 960 + s * (cell["cx"] - 960)
            sy = hz + s * cell["cy"]
            if sx < -size or sx > self.W + size or sy < -size or sy > self.H + size:
                continue
            if mono:
                self._blit_char(ctx, cell["ch"], sx, sy, size, (1, 1, 1), a)
            elif emissive is not None:
                h = emissive(sx, sy)
                if h > 0.01:
                    self._blit_char(ctx, cell["ch"], sx, sy, size, self.tl.pal["lava"], a * h)
            else:
                pop = 1 + 0.08 * (1 - a)
                self._blit_char(ctx, cell["ch"], sx, sy, size * pop, col, a * 0.62)

    def _plant_polys(self, p, t):
        """回傳 (世界座標多邊形, 是否在吸收中的混合係數)。"""
        rise = ease_out(ramp(t, p.t_rise, p.t_rise + 2.2), 2)
        # 每一筆（輪廓）在整株的彎曲之外再各自顫動，像葉片
        n = len(p.local)
        if n:
            ks = np.arange(n, dtype=np.float32)
            still = 1.0 if p.absorb_t is None else 1 - ramp(t, p.absorb_t, p.absorb_t + 0.6)
            amp = (0.05 + 0.25 * abs(p.omega) / 3) * still
            d = amp * noise3(t * 2.6 + ks * 1.37, p.phase + ks * 0.7, 0.0, 300)
            polys = []
            for i, cont in enumerate(p.local):
                polys += p.warp([cont], p.theta + float(d[i]))
        else:
            polys = []
        dy = (1 - rise) * p.h * 1.05 - p.uplift()
        polys = [q + (p.x, dy) for q in polys]
        m = 0.0
        if p.absorb_t is not None and t > p.absorb_t:
            m = ease(ramp(t, p.absorb_t, p.absorb_t + 1.1))
            # 縮放、扶正並移到格子中央
            target_h = G * 0.62
            allp = np.concatenate(polys) if polys else np.zeros((1, 2))
            cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
            scale = target_h / max(p.h, 1)
            tx, ty = p.cell["cx"], p.cell["cy"]
            polys = [np.stack([mix(q[:, 0], tx + (q[:, 0] - cx) * scale, m),
                               mix(q[:, 1], ty + (q[:, 1] - cy) * scale, m)], 1) for q in polys]
        return polys, m

    DEPTH = {-1: 0.30, 0: 0.55, 1: 0.78, 2: 1.0}

    def _draw_plants(self, ctx, t, mono=False, emissive=None, rows=(-1, 0, 1, 2)):
        pal = self.tl.pal
        c = self.c
        tint_k = ease(ramp(t, c("blood") + 2, c("stomp_start") + 2))
        warm_k = ease(ramp(t, c("selflove"), c("selflove") + 0.8)) if t >= c("selflove") else 0.0
        haze = np.array([0.15, 0.125, 0.10])
        ctx.save()
        if t < c("zoom_start"):
            s, hz = self.camera(t)
            ctx.rectangle(0, 0, self.W, hz)
            ctx.clip()
        pool = (self.fringe if -1 in rows else []) + [p for p in self.plants if p.row in rows]
        fringe_fade = 1 - ease(ramp(t, c("grid_absorb"), c("grid_absorb") + 4))
        for p in sorted(pool, key=lambda p: p.row):
            if t < p.t_rise:
                continue
            polys, m = self._plant_polys(p, t)
            if not polys:
                continue
            fade = 1.0
            if p.row < 0:
                fade = fringe_fade
            elif p.cell is not None and p.cell["fill_t"] is not None:
                fade = 1 - ease(ramp(t, p.cell["fill_t"] - 0.2, p.cell["fill_t"] + 0.3))
            if fade <= 0:
                continue
            scr = [self.to_screen(q, t) for q in polys]
            depth = self.DEPTH[p.row]
            if mono:
                fill_polys(ctx, scr, (1, 1, 1), fade)
                continue
            if emissive is not None:
                cx = np.concatenate(scr).mean(0)
                h = emissive(*cx)
                if h > 0.01:
                    fill_polys(ctx, scr, pal["lava"], h * fade)
                continue
            base = mix(pal["grass"], pal["grid"], m)
            base = mix(base, pal["light_warm"], 0.28 * warm_k * (1 - m))
            # 根部暗、梢部亮；血從根往上染
            root = base * 0.36
            root = mix(root, pal["blood"] * 1.1, 0.75 * tint_k * p.tint * (1 - m))
            midc = mix(base * 0.8, pal["blood"] * 1.3, 0.28 * tint_k * p.tint * (1 - m))
            tip = base * (1.08 + 0.25 * warm_k)
            k = mix(0.30, 0.86, depth)
            stops = []
            for off, col in ((0.0, root), (0.45, midc), (1.0, tip)):
                col = mix(haze, col * k, 0.35 + 0.65 * depth)   # 越遠越溶進空氣
                stops.append((off, col, fade))
            s_, hz_ = self.camera(t)
            r0 = self.to_screen(np.array([p.x, 0.0]), t)
            r1 = r0 - (0, p.h * s_ * p.squash * p.cut_u)
            if m > 0:
                r0, r1 = r0 + (0, 1), r0 - (0, 1)
                stops = [(0.0, base * k, fade), (1.0, base * k, fade)]
            fill_polys_grad(ctx, scr, r0, r1, stops)
        ctx.restore()

    def _draw_roots(self, ctx, t):
        pal = self.tl.pal
        red = ease(ramp(t, self.c("blood"), self.c("blood") + 6))
        for p in self.plants:
            if t < p.t_rise or p.absorb_t is not None and t > p.absorb_t:
                continue
            rise = ease_out(ramp(t, p.t_rise, p.t_rise + 2.2), 2)
            up = p.uplift()
            depth = self.DEPTH[p.row]
            col = mix(pal["grass_dim"], pal["blood"] * 1.5, red * 0.8) * depth
            for r in p.roots:
                q = r.copy()
                q[:, 0] += p.x
                q[:, 1] += -up + (1 - rise) * p.h * 1.05
                q[:, 1] = q[:, 1] * mix(0.2, 1.0, rise)
                sc = self.to_screen(q, t)
                ctx.new_path()
                path_polys(ctx, [sc], close=False)
                ctx.set_source_rgba(*[float(v) for v in col], 0.75 * rise)
                ctx.set_line_width(1.1 + 0.6 * max(p.row, 0))
                ctx.stroke()
                # 血沿根往上流的光點
                if red > 0:
                    for k in range(2):
                        u = ((t * 0.7 + p.phase + k * 0.5) % 1.0)
                        i = int((1 - u) * (len(sc) - 1))
                        ctx.arc(float(sc[i, 0]), float(sc[i, 1]), 1.6, 0, 2 * np.pi)
                        ctx.set_source_rgba(0.75, 0.18, 0.12, 0.6 * red)
                        ctx.fill()

    def _draw_pieces(self, ctx, t, glow_ctx=None):
        pal = self.tl.pal
        for pc in self.pieces:
            age = t - pc["t"]
            if age < 0:
                continue
            if age < 0.5:
                # 刀過之處：一點冷光與幾片飛出去的碎屑
                c0 = self.to_screen(pc["cut"], t)
                fa = 1 - age / 0.5
                for v in pc["slivers"]:
                    q0 = pc["cut"] + v * age + np.array([0, 800]) * age * age / 2
                    q1 = q0 - v * 0.018
                    a, b = self.to_screen(q0, t), self.to_screen(q1, t)
                    ctx.move_to(*a)
                    ctx.line_to(*b)
                    ctx.set_source_rgba(0.88, 0.9, 0.86, 0.8 * fa)
                    ctx.set_line_width(1.3)
                    ctx.stroke()
                if glow_ctx is not None and age < 0.18:
                    glow_ctx.arc(float(c0[0]), float(c0[1]), 10, 0, 2 * np.pi)
                    glow_ctx.set_source_rgba(0.8, 0.86, 0.9, 1 - age / 0.18)
                    glow_ctx.fill()
            a = 1 - ease(ramp(age, 0.5, 1.6))
            if a <= 0:
                continue
            cr, sr = np.cos(pc["rot"]), np.sin(pc["rot"])
            R = np.array([[cr, -sr], [sr, cr]])
            polys = [q @ R.T + pc["pos"] for q in pc["polys"]]
            scr = [self.to_screen(q, t) for q in polys]
            fill_polys(ctx, scr, pal["grass"] * 0.75, a)

    def _stomp_y(self, u):
        if u < 0:
            return mix(-900, -14, ease_in(ramp(u, -0.2, 0.0), 2))
        if u < 0.1:
            return -14
        return mix(-14, -900, ease(ramp(u, 0.1, 0.42)))

    def _draw_stomp(self, ctx, t):
        for ev in self.events:
            if ev["kind"] != "stomp":
                continue
            u = t - ev["t"]
            if u < -0.2 or u > 0.42:
                continue
            x0, x1 = ev["xc"] - ev["hw"] * 1.25, ev["xc"] + ev["hw"] * 1.25
            # 地面先被它的陰影壓暗
            shade = 0.6 * ease(ramp(u, -0.2, 0.0)) if u < 0.1 else 0.6 * (1 - ease(ramp(u, 0.1, 0.42)))
            g0 = self.to_screen(np.array([[x0 - 120, -500], [x1 + 120, 0]]), t)
            ctx.rectangle(float(g0[0, 0]), float(g0[0, 1]), float(g0[1, 0] - g0[0, 0]), float(g0[1, 1] - g0[0, 1]))
            g2 = cairo.LinearGradient(0, float(g0[0, 1]), 0, float(g0[1, 1]))
            g2.add_color_stop_rgba(0, 0, 0, 0, 0)
            g2.add_color_stop_rgba(1, 0, 0, 0, shade)
            ctx.set_source(g2)
            ctx.fill()
            # 巨大的口號印版：本體往上、往兩側溶進黑暗；下落時疊印出動態模糊
            taps = 6
            gtxt = self.tl.raw["grid"]["text"]
            for k in range(taps):
                yb = self._stomp_y(u - k * 0.012)
                a = 0.96 if k == 0 else 0.35 / taps
                q = self.to_screen(np.array([[x0, yb - 900], [x1, yb]]), t)
                (sx0, sy0), (sx1, sy1) = q
                body = cairo.LinearGradient(0, sy0, 0, sy1)
                body.add_color_stop_rgba(0, 0.02, 0.018, 0.016, 0)
                body.add_color_stop_rgba(0.55, 0.035, 0.031, 0.028, a * 0.85)
                body.add_color_stop_rgba(1, 0.07, 0.062, 0.055, a)
                edge = cairo.LinearGradient(sx0 - 40, 0, sx1 + 40, 0)
                edge.add_color_stop_rgba(0, 0, 0, 0, 0)
                edge.add_color_stop_rgba(0.12, 0, 0, 0, 1)
                edge.add_color_stop_rgba(0.88, 0, 0, 0, 1)
                edge.add_color_stop_rgba(1, 0, 0, 0, 0)
                ctx.save()
                ctx.rectangle(sx0 - 40, sy0, sx1 - sx0 + 80, sy1 - sy0)
                ctx.clip()
                ctx.set_source(body)
                ctx.mask(edge)
                ctx.restore()
            # 印面：一條透視的亮帶，上面凸起反寫的標語字
            yb = self._stomp_y(u)
            band = self.to_screen(np.array([[x0 + 30, yb - 20], [x1 - 30, yb - 20], [x1 + 20, yb], [x0 - 20, yb]]), t)
            ctx.new_path()
            path_polys(ctx, [band])
            face = cairo.LinearGradient(0, float(band[0, 1]), 0, float(band[2, 1]))
            face.add_color_stop_rgba(0, 0.10, 0.09, 0.08, 0.95)
            face.add_color_stop_rgba(1, 0.22, 0.2, 0.17, 0.95)
            ctx.set_source(face)
            ctx.fill()
            n = int((x1 - x0) / 34)
            for j in range(n):
                cxw = x0 + (j + 0.5) * (x1 - x0) / n
                sc = self.to_screen(np.array([cxw, yb - 9]), t)
                ch = gtxt[(j + int(ev["xc"])) % len(gtxt)]
                ctx.save()
                ctx.translate(float(sc[0]), float(sc[1]))
                ctx.scale(-1, 0.42)   # 印版上的字是反的，而且被透視壓扁
                self._blit_char(ctx, ch, 0, 0, 30, (0.04, 0.035, 0.03), 0.9)
                ctx.restore()

    def _draw_blades(self, ctx, glow_ctx, t):
        for ev in self.events:
            if ev["kind"] != "blade":
                continue
            t_a = ev["t"] - ev["dur"] / 2
            if t < t_a or t > ev["t"] + ev["dur"] / 2 + 0.35:
                continue
            bx = self._blade_x(ev, min(t, ev["t"] + ev["dur"] / 2))
            y = -ev["hc"]
            after = max(0.0, t - (ev["t"] + ev["dur"] / 2))
            x_from = -120 if ev["dir"] > 0 else 2040
            p = self.to_screen(np.array([[x_from, y], [bx, y]]), t)
            fa = 1 - ease(after / 0.35)
            # 刀痕殘影
            lg = cairo.LinearGradient(float(p[0, 0]), 0, float(p[1, 0]), 0)
            lg.add_color_stop_rgba(0, 0.85, 0.88, 0.9, 0.0)
            lg.add_color_stop_rgba(1, 0.85, 0.88, 0.9, 0.55 * fa)
            for c_, w in ((ctx, 1.2), (glow_ctx, 3.0)):
                c_.new_path()
                c_.move_to(float(p[0, 0]), float(p[0, 1]))
                c_.line_to(float(p[1, 0]), float(p[1, 1]))
                c_.set_source(lg)
                c_.set_line_width(w)
                c_.stroke()

    def _draw_names(self, ctx, t):
        a0 = self.tl.raw["names"]["alpha"]
        for nm in self.names:
            u = t - nm["t"]
            if u < 0 or u > 1.6:
                continue
            a = a0 * min(1, u / 0.15) * (1 - ease(ramp(u, 0.3, 1.6)))
            dy = 45 * ease(ramp(u, 0.2, 1.6))
            if nm["strokes"] is None:
                continue  # 指定了真名時，由字幕字體繪製（見 _draw_name_labels）
            for st in nm["strokes"]:
                q = st + (nm["x"], nm["y"] + dy)
                sc = self.to_screen(q, t)
                ctx.new_path()
                path_polys(ctx, [sc], close=False)
                ctx.set_source_rgba(0.82, 0.55, 0.48, a)
                ctx.set_line_width(1.3)
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.stroke()

    def _background(self, t):
        pal = self.tl.pal
        c = self.c
        s, hz = self.camera(t)
        calm = ease(ramp(t, c("breath"), c("breath") + 1.5)) * (1 - ease(ramp(t, c("selflove"), c("selflove") + 3)))
        top = mix(np.array([0.040, 0.034, 0.030]), np.array([0.12, 0.105, 0.088]), calm)
        near = mix(np.array([0.12, 0.095, 0.072]), np.array([0.27, 0.22, 0.165]), calm)
        hzf = hz / self.H
        img = lin_gradient_v(self.H, self.W, [(0, top), (hzf * 0.8, near * 0.8), (hzf, near), (1, near)])
        # 雲霧：讓天空有空氣感
        cloud = self.sky.field(t, 9.0 if t < c("breath") else 4.0)
        img *= (0.78 + 0.45 * cloud)[..., None]
        # 地平線上一道低低的光
        ys = np.arange(self.H, dtype=np.float32)
        glow = np.exp(-((ys - hz) / (90 * s + 30)) ** 2)[:, None, None]
        img += glow * np.array([0.10, 0.075, 0.05], np.float32) * (0.6 + 0.8 * calm)
        soil, h0 = self.soil.view(s, hz, self.W, self.H)
        if soil is not None:
            if t < c("mud_land") + 0.7:
                spread = 960 * 1.15 * ease_out(ramp(t, c("mud_land"), c("mud_land") + 0.6), 3)
                xs = np.abs(np.arange(self.W) - 960)
                m = np.clip((xs - spread) / 90 + 0.5, 0, 1)[None, :, None]
                img[h0:] = mix(soil, img[h0:], m)
            else:
                img[h0:] = soil
        return img

    def _underground(self, img, t):
        """地底暗紅細流（低解析噪聲場，上採樣）。"""
        c = self.c
        k = ease(ramp(t, c("blood") - 1, c("blood") + 3))
        if k <= 0:
            return
        s, hz = self.camera(t)
        h0 = int(hz)
        if h0 >= self.H:
            return
        w, h = 640, max(4, (self.H - h0) // 2)
        xs = np.linspace(0, self.W, w)[None, :]
        ys = np.linspace(0, (self.H - h0) / s, h)[:, None]
        wx = (xs - 960) / s + 960
        field = np.zeros((h, w), np.float32)
        rng = rand(self.seed, 4, 9)
        for vi in range(11):
            y0 = rng.uniform(25, 330)
            yc = y0 + 38 * fbm(wx / 520, vi * 3.1, t * 0.06, 3, 50 + vi)
            width = rng.uniform(1.0, 2.6)
            speed = rng.uniform(0.6, 1.6)
            pulse = np.sin(wx / rng.uniform(90, 220) - t * speed * 2 + vi * 1.7)
            pulse = np.clip(pulse * 1.6 - 0.4, 0, 1) * (0.4 + 0.6 * np.clip(fbm(wx / 300, vi, t * 0.2, 2, 80 + vi) + 0.5, 0, 1))
            field += np.exp(-((ys - yc) / width) ** 2) * pulse * rng.uniform(0.4, 1.0)
        field = np.clip(field, 0, 1.5)
        up = resize(field[..., None], self.W, self.H - h0)[..., 0]
        img[h0:] += up[..., None] * self.tl.pal["blood"] * 0.8 * k

    def _heat(self, t):
        """回傳 emissive(x, y) → 0..1：地火從畫面底部與字的內部透出。"""
        c = self.c
        k = ramp(t, c("fire_rise"), c("dim"))
        if k <= 0 or t >= c("erupt") + 0.2:
            return None
        flick = 1.0
        if t >= c("dim"):
            # 2:01 暗一下；2:02.1 的小 onset 透出一絲光；然後蓄勢
            flick = 0.12 + 0.35 * np.exp(-((t - 122.12) / 0.12) ** 2) + 0.9 * ease(ramp(t, 122.9, c("erupt")))

        def f(x, y):
            v = (k * 1.6 - (1 - y / self.H) * 1.1 + 0.3 * float(noise3(x / 300, y / 300, t * 0.9, 61)))
            return float(np.clip(v, 0, 1)) * flick
        return f

    def _lava(self, t):
        """回傳 (mul, add, heat)：煙的遮蔽、熔岩與火光、熱浪強度（皆為整張畫面）。"""
        c = self.c
        te = min(t, c("freeze"))
        u = te - c("erupt")
        if u < 0:
            return None
        if self.lava_cache is not None and self.lava_cache[0] == te:
            return self.lava_cache[1]
        pal = self.tl.pal
        w, h = 480, 270
        xs = np.linspace(0, 1, w, dtype=np.float32)[None, :] * 1.78
        ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        height = 0.95 * ease_out(ramp(u, 0.0, 1.0), 2) - 0.6 * ease_out(ramp(u, 1.0, 8), 2)
        surf = 1 - height + 0.06 * fbm(xs * 3, te * 0.4, 0.0, 3, 70)
        below = ys - surf
        # 地殼與裂縫：ridged noise，裂縫隨時間緩慢流動
        wx = xs * 3 + 0.3 * fbm(xs * 2, ys * 2 - te * 0.25, te * 0.2, 3, 71)
        wy = ys * 3 + 0.3 * fbm(xs * 2 + 5, ys * 2, te * 0.2, 3, 72)
        ridge = 1 - np.abs(fbm(wx * 2.2, wy * 2.2, te * 0.15, 4, 73))
        crack = np.clip((ridge - 0.86) / 0.13, 0, 1) ** 2
        fine = np.clip((1 - np.abs(fbm(wx * 7, wy * 7, te * 0.3, 2, 74)) - 0.9) / 0.1, 0, 1) ** 2
        energy = mix(1.0, 0.45, ease(ramp(u, 0.6, 12)))
        inside = np.clip(below * 40 + 0.5, 0, 1)
        rim = np.exp(-(below / 0.025) ** 2)
        heatv = (crack * 1.15 + fine * 0.35 + 0.05 + rim * 0.8) * energy
        lava = np.zeros((h, w, 3), np.float32)
        stops = [(0, np.array([0.05, 0.02, 0.012])), (0.25, pal["blood"] * 0.9), (0.6, pal["lava"]),
                 (1.0, pal["ember"]), (1.6, np.array([1.0, 0.92, 0.75]))]
        for ci in range(3):
            lava[..., ci] = np.interp(heatv, [q[0] for q in stops], [q[1][ci] for q in stops])
        crust = np.array([0.045, 0.02, 0.012], np.float32)
        lava = mix(crust, lava, np.clip(heatv * 1.4, 0, 1)[..., None])
        # 煙：從表面升起，底下被火照亮
        above = np.clip(-below, 0, 1)
        smoke = np.clip(fbm(xs * 2.2, ys * 2.6 + te * 0.9, te * 0.25, 4, 75) * 1.4 + 0.25, 0, 1)
        smoke *= np.exp(-above / 0.45) * ease(ramp(u, 0.1, 1.2))
        firelight = np.exp(-above / 0.18) * energy
        mul = (1 - inside) * (1 - 0.65 * smoke) + 0
        add = lava * inside[..., None] + (firelight * (1 - inside))[..., None] * pal["lava"] * 0.55 \
            + (smoke * firelight)[..., None] * pal["ember"] * 0.25
        heat = np.exp(-np.clip(-below, 0, None) / 0.3) * (1 - inside) * energy
        out = (resize(mul[..., None], self.W, self.H), resize(add, self.W, self.H),
               resize(heat[..., None], self.W, self.H)[..., 0])
        self.lava_cache = (te, out)
        return out

    def _shimmer(self, img, heat, t):
        """熱浪：依熱度把每一列像素橫向擾動幾個像素。"""
        te = min(t, self.c("freeze"))
        w, h = 240, 135
        X, Y = np.meshgrid(np.linspace(0, 24, w), np.linspace(0, 14, h))
        d = noise3(X, Y + te * 3.5, te * 1.3, 77)
        d = resize(d[..., None].astype(np.float32), self.W, self.H)[..., 0] * 5.0 * heat
        xi = np.clip(np.arange(self.W)[None, :] + d.astype(np.int32), 0, self.W - 1)
        return np.take_along_axis(img, np.broadcast_to(xi[..., None], img.shape), axis=1)

    def _draw_ash(self, img, t):
        a = self.ash
        c = self.c
        pos, temp = a["pos"], a["temp"]
        vis = 1.0
        if t >= c("hush"):
            # 2:30 靜穆：灰燼溶成極緩慢呼吸的灰色平面
            vis = 1 - ease(ramp(t, c("hush"), c("hush") + 1.6))
            if t >= c("restore"):
                vis = ease_out(ramp(t, c("restore"), c("restore") + 0.12))
        if vis <= 0:
            return
        ash_col = np.array([0.42, 0.40, 0.37], np.float32)
        hot = np.array([0.95, 0.5, 0.18], np.float32)
        tt = temp[:, None]
        col = mix(ash_col, mix(self.tl.pal["blood"] * 2.2, hot, np.clip(tt * 1.6 - 0.6, 0, 1)), np.clip(tt * 2, 0, 1))
        w = a["w"] * (0.8 + 1.4 * temp) * vis
        if t >= c("witness"):
            n = self.gather_n
            g = ease(ramp(t, c("witness"), c("witness") + 2.4))
            pos = pos.copy()
            pos[:n] = mix(pos[:n], self.gather_tgt, g)
            fade_rest = 1 - ease(ramp(t, c("witness"), c("witness") + 1.6))
            fade_g = 1 - ease(ramp(t, c("witness") + 2.4, c("witness") + 3.2))
            w = w.copy()
            w[:n] *= fade_g * 1.4
            w[n:] *= fade_rest
            col = np.broadcast_to(col, (len(pos), 3)).copy()
            col[:n] = mix(col[:n], self.tl.pal["light_warm"], g * 0.6)
        splat(img, pos, col, w)
        big = a["big"]
        if big.any():
            glow = np.zeros_like(img)
            splat(glow, pos[big], col[big], w[big] * 6)
            img += bloom(glow, 6, down=2) * 1.5

    def subtitle_backing(self, t):
        """網格出現後字幕底下需要一條柔和的暗帶。"""
        c = self.c
        grid = ease(ramp(t, c("grid_in"), c("grid_in") + 2)) * (1 - ease(ramp(t, c("erupt"), c("erupt") + 0.5)))
        lava = ease(ramp(t, c("erupt") + 0.3, c("erupt") + 1.0)) * (1 - ease(ramp(t, c("hush"), c("hush") + 1.6)))
        return max(grid, lava)

    def render(self, frame):
        t = frame / self.fps
        c = self.c
        pal = self.tl.pal
        img = self._background(t)
        s, hz = self.camera(t)

        # 噴發前的世界
        pre = t < c("erupt") + 0.15
        if pre:
            self._underground(img, t)
            lay = self.layer
            lay.clear()
            glow = self.glow
            glow.clear()
            ctx = lay.ctx
            self._draw_names(ctx, t)
            self._draw_roots(ctx, t)
            # 地平線：細微起伏的亮邊
            xs = np.linspace(-20, 1940, 90)
            ys = 2.0 * noise3(xs / 120, 0.0, 0.0, 91)
            edge = np.stack([xs, ys], 1)
            sc = self.to_screen(edge, t)
            ctx.new_path()
            path_polys(ctx, [sc], close=False)
            ctx.set_source_rgba(*[float(v) for v in pal["mud"] * 1.8], 0.8 * ramp(t, c("mud_land") + 0.2, c("mud_land") + 0.7))
            ctx.set_line_width(1.5)
            ctx.stroke()
            self._draw_grid(ctx, t)
            self._draw_mud_land(ctx, t)
            self._draw_tree(ctx, t)
            if t < c("fire_rise"):
                far = self.far
                far.clear()
                self._draw_plants(far.ctx, t, rows=(-1, 0))
                fr = far.rgba()
                blur_k = 1 - ease(ramp(t, c("zoom_start"), c("zoom_end")))
                if fr[..., 3].max() > 0:
                    fb = bloom(fr, 3.2 * blur_k + 0.01, down=2)
                    over(img, lay.rgba())
                    lay.clear()
                    over(img, fb)
                self._draw_plants(ctx, t, rows=(1, 2))
            else:
                self._draw_plants(ctx, t)
            ctx.save()
            ctx.rectangle(0, 0, self.W, hz)
            ctx.clip()
            self._draw_pieces(ctx, t, glow.ctx)
            ctx.restore()
            self._draw_stomp(ctx, t)
            self._draw_blades(ctx, glow.ctx, t)
            over(img, lay.rgba())
            lay.clear()
            self._draw_dust(img, t)
            heat = self._heat(t)
            if heat is not None:
                self._draw_grid(glow.ctx, t, emissive=heat)
                self._draw_plants(glow.ctx, t, emissive=heat)
                # 格子縫隙透光
                gk = ramp(t, c("fire_rise"), c("dim"))
                for cell in self.cells[::1]:
                    if self._grid_alpha(cell, t) <= 0:
                        continue
                    sx = 960 + s * (cell["cx"] - 960 - G / 2)
                    sy = hz + s * (cell["cy"] + G / 2)
                    if not (-G < sx < self.W + G and -G < sy < self.H + G):
                        continue
                    h = heat(sx, sy)
                    if h < 0.02:
                        continue
                    glow.ctx.rectangle(sx, sy - s * G, s * G, s * G)
                    glow.ctx.set_source_rgba(*[float(v) for v in pal["lava"]], 0.55 * h * gk)
                    glow.ctx.set_line_width(1.2)
                    glow.ctx.stroke()
            # 空氣中的塵：第 5 段的明亮安靜裡最清楚
            mk = (0.5 * ramp(t, c("grass_rise"), c("grass_rise") + 4)
                  + 0.9 * calm_k(t, c)) * (1 - ease(ramp(t, c("zoom_start"), c("fire_rise"))) * 0.8)
            over(img, lay.rgba())
            lay.clear()
            self.motes.draw(img, t, mk, wind=1.8 if t < c("breath") else 0.6)
            rk = calm_k(t, c) + 0.6 * ease(ramp(t, c("selflove"), c("selflove") + 1)) * (1 - ease(ramp(t, c("zoom_start"), c("zoom_end"))))
            r = atmos.rays(self.W, self.H, t, rk, pal["light_warm"])
            if r is not None:
                img += r
            if t >= c("selflove"):
                # 「自愛」：一道暖光照著字草
                k = ease(ramp(t, c("selflove"), c("selflove") + 0.8)) * (1 - 0.6 * ease(ramp(t, c("zoom_start"), c("zoom_end"))))
                yy, xx = np.ogrid[0:self.H, 0:self.W]
                r = np.sqrt(((xx - 960) / (900 * s + 200)) ** 2 + ((yy - (hz - 140 * s)) / (420 * s + 120)) ** 2)
                light = np.clip(1 - r, 0, 1) ** 1.8
                img += light[..., None] * pal["light_warm"] * 0.22 * k
            over(img, lay.rgba())
            dimk = 1.0
            if t >= c("dim"):
                dimk = mix(1.0, 0.1, ease(ramp(t, c("dim"), c("dim") + 0.2)))
                dimk = mix(dimk, 1.0, ease(ramp(t, 122.95, c("erupt"))))
            if t >= c("erupt"):
                dimk *= 1 - ease(ramp(t, c("erupt"), c("erupt") + 0.15))
            img *= dimk
            gl = glow.rgba()[..., :3]
            if gl.max() > 0:
                img += gl * 0.8 + bloom(gl, 22) * 2.2
        if t >= c("erupt"):
            k_after = ease(ramp(t, c("erupt"), c("erupt") + 0.08))
            base = np.array([0.05, 0.04, 0.035], np.float32)
            img = mix(img, np.broadcast_to(base, img.shape), k_after * 0.85)
            lava = self._lava(t)
            hushk = 1.0
            if t >= c("hush"):
                hushk = 1 - ease(ramp(t, c("hush"), c("hush") + 1.6))
            if lava is not None:
                mul, add, heat = lava
                # 凍結之後，熔岩慢慢冷成暗紅（顏色在變，形狀不動）
                cool = 1 - 0.45 * ease(ramp(t, c("freeze"), c("hush")))
                img = img * mix(1.0, mul, hushk) + add * hushk * cool
                if t < c("freeze"):
                    img = self._shimmer(img, heat, t)
            flash = np.exp(-max(0.0, t - c("erupt")) / 0.12) if t >= c("erupt") else 0
            img += flash * np.array([1.0, 0.8, 0.55], np.float32) * 0.45
            if self.ash is not None:
                self._draw_ash(img, t)
            if t >= c("hush"):
                # 靜穆：極緩慢呼吸的灰色平面
                breathe = 0.5 + 0.5 * np.sin(2 * np.pi * (t - c("hush")) / 7.0 - np.pi / 2)
                gray = pal["ash_gray"] * (0.85 + 0.15 * breathe)
                k = ease(ramp(t, c("hush"), c("hush") + 1.6))
                if t >= c("restore"):
                    k *= 1 - ease_out(ramp(t, c("restore"), c("restore") + 0.1))
                img = mix(img, np.broadcast_to(gray, img.shape), k * 0.9)
        if t >= c("witness"):
            # 分界線在灰燼底下
            ash_layer = img.copy()
            base = np.array([0.05, 0.04, 0.035], np.float32)
            img = self._render_witness(np.broadcast_to(base, img.shape).copy(), t)
            img += np.clip(ash_layer - base, 0, None)
        dx, dy = self.shake(t)
        if abs(dx) >= 0.5 or abs(dy) >= 0.5:
            img = shift(img, int(round(dx)), int(round(dy)))
        return img
