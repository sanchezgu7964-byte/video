"""第 4–8 段：字長成草 → 踐踏刪刈 → 坦然 → 自愛與網格 → 地火噴發 → 凍結。

世界座標：地平線 y=0，向上為負；鏡頭 screen = (960 + s·(x−960), Hz + s·y)。
字草的模擬（彈簧擺動、壓扁、切斷）以固定步長逐影格推進，任何影格都能由場景起點重播得到。
"""
import cairo
import numpy as np
from PIL import Image, ImageDraw

from . import clump as clumpmod
from . import fonts
from . import tree as treemod
from .gfx import (Layer, bloom, ease, ease_in, ease_out, fill_polys, lin_gradient_v, mix, over, path_polys,
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
        self._build_plants()
        self._build_events()
        self._build_grid()
        self._build_names()
        self.frame = int(round(self.t0 * self.fps)) - 1
        self._build_tree()
        self._build_witness()
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
        # 三對抽象輪廓：友與仇、人與獸、愛者與不愛者
        self.pairs = []
        for k, (u0, v0, size, aspect) in enumerate([(-440, 230, 95, 1.0), (380, 300, 120, 1.7), (130, 170, 62, 0.8)]):
            th = np.linspace(0, 2 * np.pi, 90, endpoint=False)
            rr = size * (1 + 0.22 * noise3(np.cos(th) * 1.3, np.sin(th) * 1.3, k * 5.0, 140 + k))
            shape = np.stack([np.cos(th) * rr * aspect ** 0.5, np.sin(th) * rr / aspect ** 0.5], 1)
            t_in = float(beats[min(3 + 3 * k, len(beats) - 1)])
            self.pairs.append(dict(u=u0, v=v0, shape=shape.astype(np.float32), t=t_in))
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
        # 成對輪廓
        for pr in self.pairs:
            a = ease(ramp(t, pr["t"], pr["t"] + 0.6)) * (1 - out_k)
            if a <= 0:
                continue
            for sgn in (1, -1):
                loc = pr["shape"] + (pr["u"], pr["v"])
                pts = np.array([cx, cy]) + loc[:, :1] * dvec + sgn * loc[:, 1:2] * nvec
                n = max(2, int(len(pts) * ease(ramp(t, pr["t"], pr["t"] + 0.6))))
                ctx.new_path()
                path_polys(ctx, [pts[:n]], close=n == len(pts))
                # 亮側用暗線，暗側用亮線；判斷所在側
                on_light = (pts.mean(0) - (cx, cy)) @ nvec > 0
                cc = (0.16, 0.13, 0.1) if on_light else (0.8, 0.77, 0.7)
                ctx.set_source_rgba(*cc, 0.85 * a)
                ctx.set_line_width(1.6)
                ctx.stroke()
                # 被同一道光照到的一側邊緣
                ctx.new_path()
                path_polys(ctx, [pts[: max(2, n // 3)]], close=False)
                ctx.set_source_rgba(*[float(v) for v in pal["light_warm"]], 0.6 * a)
                ctx.set_line_width(2.2)
                ctx.stroke()
        # 灰燼聚成的一叢字草
        ca = ease(ramp(t, c("witness") + 2.2, c("witness") + 3.0))
        if ca > 0:
            sway = 0.05 * np.sin(t * 1.9)
            for p in self.clump:
                polys = clumpmod.clump_polys(p, theta=sway * (1.2 - p.size / 130))
                fill_polys(ctx, polys, mix(pal["grass"], pal["light_warm"], 0.25), ca)
        over(img, lay.rgba())
        return img

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
        return base * (0.55 + gust)

    def advance_to(self, frame):
        while self.frame < frame:
            self.frame += 1
            self._step(self.frame / self.fps)

    def _step(self, t):
        dt = self.dt
        c = self.c
        sing = c("sing") <= t < c("selflove") - 0.6
        for p in self.plants:
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
                for p in self.plants:
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
            rot=0.0, spin=ev["dir"] * rng.uniform(2, 7), t=t, k=p.k))

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
        polys = p.warp(p.local)
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

    def _draw_plants(self, ctx, t, mono=False, emissive=None):
        pal = self.tl.pal
        c = self.c
        tint_k = ease(ramp(t, c("blood") + 2, c("stomp_start") + 2)) * 0.30
        ctx.save()
        if t < c("zoom_start"):
            s, hz = self.camera(t)
            ctx.rectangle(0, 0, self.W, hz)
            ctx.clip()
        for p in sorted(self.plants, key=lambda p: p.row):
            if t < p.t_rise:
                continue
            polys, m = self._plant_polys(p, t)
            if not polys:
                continue
            fade = 1.0
            if p.cell is not None and p.cell["fill_t"] is not None:
                fade = 1 - ease(ramp(t, p.cell["fill_t"] - 0.2, p.cell["fill_t"] + 0.3))
            if fade <= 0:
                continue
            scr = [self.to_screen(q, t) for q in polys]
            depth = [0.55, 0.78, 1.0][p.row]
            if mono:
                fill_polys(ctx, scr, (1, 1, 1), fade)
                continue
            if emissive is not None:
                cx = np.concatenate(scr).mean(0)
                h = emissive(*cx)
                if h > 0.01:
                    fill_polys(ctx, scr, pal["lava"], h * fade)
                continue
            col = mix(pal["grass"], pal["blood"] * 1.25, tint_k * p.tint)
            col = mix(col, pal["grid"], m)
            if t >= c("selflove"):
                col = mix(col, pal["light_warm"], 0.25 * ease(ramp(t, c("selflove"), c("selflove") + 0.8))
                          * (1 - m))
            col = col * mix(0.32, 0.82, depth)
            fill_polys(ctx, scr, col, fade)
        ctx.restore()

    def _draw_roots(self, ctx, t):
        pal = self.tl.pal
        red = ease(ramp(t, self.c("blood"), self.c("blood") + 6))
        for p in self.plants:
            if t < p.t_rise or p.absorb_t is not None and t > p.absorb_t:
                continue
            rise = ease_out(ramp(t, p.t_rise, p.t_rise + 2.2), 2)
            up = p.uplift()
            depth = [0.5, 0.75, 1.0][p.row]
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
                ctx.set_line_width(1.1 + 0.6 * p.row)
                ctx.stroke()
                # 血沿根往上流的光點
                if red > 0:
                    for k in range(2):
                        u = ((t * 0.7 + p.phase + k * 0.5) % 1.0)
                        i = int((1 - u) * (len(sc) - 1))
                        ctx.arc(float(sc[i, 0]), float(sc[i, 1]), 1.6, 0, 2 * np.pi)
                        ctx.set_source_rgba(0.75, 0.18, 0.12, 0.6 * red)
                        ctx.fill()

    def _draw_pieces(self, ctx, t):
        pal = self.tl.pal
        for pc in self.pieces:
            age = t - pc["t"]
            if age < 0:
                continue
            a = 1 - ease(ramp(age, 0.5, 1.6))
            if a <= 0:
                continue
            cr, sr = np.cos(pc["rot"]), np.sin(pc["rot"])
            R = np.array([[cr, -sr], [sr, cr]])
            polys = [q @ R.T + pc["pos"] for q in pc["polys"]]
            scr = [self.to_screen(q, t) for q in polys]
            fill_polys(ctx, scr, pal["grass"] * 0.75, a)

    def _draw_stomp(self, ctx, t):
        s, hz = self.camera(t)
        for ev in self.events:
            if ev["kind"] != "stomp":
                continue
            u = t - ev["t"]
            if u < -0.2 or u > 0.42:
                continue
            if u < 0:
                yb = mix(-900, -14, ease_in(ramp(u, -0.2, 0.0), 2))
            elif u < 0.1:
                yb = -14
            else:
                yb = mix(-14, -900, ease(ramp(u, 0.1, 0.42)))
            a = 0.92
            x0, x1 = ev["xc"] - ev["hw"] * 1.25, ev["xc"] + ev["hw"] * 1.25
            # 巨大平面：下緣銳利，往上溶進黑暗
            sk = 60
            quad = np.array([[x0 - sk, yb - 700], [x1 + sk, yb - 700], [x1, yb], [x0, yb]])
            sc = self.to_screen(quad, t)
            grad = cairo.LinearGradient(0, float(sc[0, 1]), 0, float(sc[2, 1]))
            grad.add_color_stop_rgba(0, 0.03, 0.026, 0.022, 0)
            grad.add_color_stop_rgba(0.7, 0.04, 0.035, 0.03, a * 0.8)
            grad.add_color_stop_rgba(1, 0.06, 0.052, 0.045, a)
            ctx.new_path()
            path_polys(ctx, [sc])
            ctx.set_source(grad)
            ctx.fill()
            # 下緣一道冷光的稜
            ctx.new_path()
            ctx.move_to(float(sc[3, 0]), float(sc[3, 1]))
            ctx.line_to(float(sc[2, 0]), float(sc[2, 1]))
            ctx.set_source_rgba(0.55, 0.52, 0.47, 0.35)
            ctx.set_line_width(1.4)
            ctx.stroke()
            # 落下前地面先被它的陰影壓暗
            shade = 0.55 * ease(ramp(u, -0.2, 0.0)) if u < 0.1 else 0.55 * (1 - ease(ramp(u, 0.1, 0.42)))
            g0 = self.to_screen(np.array([[x0, -400], [x1, 0]]), t)
            ctx.rectangle(float(g0[0, 0]), float(g0[0, 1]), float(g0[1, 0] - g0[0, 0]), float(g0[1, 1] - g0[0, 1]))
            g2 = cairo.LinearGradient(0, float(g0[0, 1]), 0, float(g0[1, 1]))
            g2.add_color_stop_rgba(0, 0, 0, 0, 0)
            g2.add_color_stop_rgba(1, 0, 0, 0, shade)
            ctx.set_source(g2)
            ctx.fill()

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
        top = mix(np.array([0.045, 0.038, 0.032]), np.array([0.13, 0.11, 0.09]), calm)
        near = mix(np.array([0.11, 0.09, 0.07]), np.array([0.24, 0.20, 0.155]), calm)
        hzf = hz / self.H
        img = lin_gradient_v(self.H, self.W, [(0, top), (hzf, near), (min(hzf + 0.002, 1), pal["soil"]),
                                               (1, pal["soil_deep"])])
        if t < c("mud_land") + 0.7:
            spread = 960 * 1.15 * ease_out(ramp(t, c("mud_land"), c("mud_land") + 0.6), 3)
            sky = lin_gradient_v(self.H, self.W, [(0, top), (hzf, near), (1, near * 0.85)])
            xs = np.abs(np.arange(self.W) - 960)
            m = np.clip((xs - spread) / 90 + 0.5, 0, 1)[None, :, None]
            img = mix(img, sky, m).astype(np.float32)
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
        c = self.c
        te = min(t, c("freeze"))
        u = te - c("erupt")
        if u < 0:
            return None
        if self.lava_cache is not None and self.lava_cache[0] == te:
            return self.lava_cache[1]
        w, h = 320, 180
        xs = np.linspace(0, 1, w)[None, :]
        ys = np.linspace(0, 1, h)[:, None]
        height = 0.95 * ease_out(ramp(u, 0.0, 1.0), 2) - 0.62 * ease_out(ramp(u, 1.0, 8), 2)
        wx = xs * 3.2 + 0.35 * fbm(xs * 2, ys * 2 - te * 0.7, te * 0.3, 3, 71)
        wy = ys * 2.4 + 0.35 * fbm(xs * 2 + 5, ys * 2 - te * 0.7, te * 0.3, 3, 72)
        n = fbm(wx * 1.6, wy * 1.6 + te * 1.8, te * 0.5, 4, 73)
        v = (ys - (1 - height)) * 2.2 + 0.55 * n
        inten = np.clip(v, 0, 1) * mix(0.78, 0.5, ease(ramp(u, 0.4, 10)))
        pal = self.tl.pal
        stops = [(0, np.zeros(3)), (0.25, pal["blood"] * 0.9), (0.55, pal["lava"] * 0.9),
                 (0.85, pal["lava"]), (1.0, pal["ember"] * 0.9)]
        rgb = np.zeros((h, w, 3), np.float32)
        for ci in range(3):
            rgb[..., ci] = np.interp(inten, [s_[0] for s_ in stops], [s_[1][ci] for s_ in stops])
        up = resize(rgb, self.W, self.H)
        self.lava_cache = (te, up)
        return up

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
        return ease(ramp(t, c("grid_in"), c("grid_in") + 2)) * (1 - ease(ramp(t, c("erupt"), c("erupt") + 0.5)))

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
            self._draw_plants(ctx, t)
            ctx.save()
            ctx.rectangle(0, 0, self.W, hz)
            ctx.clip()
            self._draw_pieces(ctx, t)
            ctx.restore()
            self._draw_stomp(ctx, t)
            self._draw_blades(ctx, glow.ctx, t)
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
                img += lava * hushk
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
        return img
