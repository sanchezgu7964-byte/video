"""第 1–3 段：充實的黑 → 裂縫 → 死與生（葉脈）→ 生命的泥。"""
import cairo
import numpy as np

from .gfx import Layer, bloom, ease, ease_in, ease_out, lin_gradient_v, mix, over, path_polys, ramp, resize, splat
from . import leaf as leafmod
from .noise import curl_grid, fbm, noise3, rand


class OpeningScene:
    t0 = 0.0
    t1 = 42.0

    def __init__(self, tl):
        self.tl = tl
        self.W, self.H, self.fps = tl.W, tl.H, tl.fps
        self.dt = 1 / tl.fps
        self.c = tl.c
        self.seed = tl.seed
        rng = rand(self.seed, 1, 1)
        n = 140000
        self.p = rng.random((n, 2)).astype(np.float32) * (self.W, self.H)
        self.v = np.zeros((n, 2), np.float32)
        self.w = (rng.random(n) ** 2 * 0.9 + 0.1).astype(np.float32)
        self.alive = np.ones(n, bool)
        self.layer = Layer(self.W, self.H)
        # 裂縫：從畫面中心向上下延伸的鋸齒線
        ys = np.linspace(-20, self.H + 20, 160)
        steps = rng.normal(0, 3.2, len(ys))
        steps[len(ys) // 2] = 0
        xs = np.cumsum(steps)
        xs -= xs[len(ys) // 2]
        self.crack = np.stack([self.W / 2 + xs, ys], 1).astype(np.float32)
        self.frame = -1
        self._build_leaf()

    def crack_x(self, y):
        return np.interp(y, self.crack[:, 1], self.crack[:, 0])

    def crack_width(self, t):
        c = self.c
        if t < c("crack"):
            return 0.0
        u = t - c("crack")
        # 先迸開成一道窄縫，再慢慢撐滿整個畫面
        return 26 * ease_out(ramp(u, 0, 0.18)) + (self.W * 1.6) * ease_in(ramp(u, 0.3, 2.4), 2.2)

    def advance_to(self, frame):
        while self.frame < frame:
            self.frame += 1
            self._step(self.frame / self.fps)

    def _step(self, t):
        c = self.c
        if t > c("leaf_in") + 1:
            return
        p, v = self.p, self.v
        dt = self.dt
        # 音樂歸零前，微粒像屏住呼吸一樣放慢
        hold = 1 - 0.85 * ease(ramp(t, c("crack") - 1.2, c("crack") - 0.2))
        if t >= c("crack"):
            hold = 1.0
        z = t * 0.05
        ax, ay = curl_grid(self.W, self.H, z, 410.0, 1, 24, 160.0)(p[:, 0], p[:, 1])
        bx, by = curl_grid(self.W, self.H, z * 3, 120.0, 2, 12, 60.0)(p[:, 0], p[:, 1])
        target = np.stack([ax * 16 + bx * 6, ay * 9 + by * 4], 1)
        target *= hold
        if t >= c("crack"):
            u = t - c("crack")
            cx = self.crack_x(p[:, 1])
            d = cx - p[:, 0]
            pull = 260 + 4200 * ease_in(ramp(u, 0, 1.9), 1.5)
            target[:, 0] += np.sign(d) * pull * (0.4 + 0.6 * np.exp(-np.abs(d) / 600))
            # 落進裂縫的微粒就洩出去了
            w = self.crack_width(t)
            self.alive &= np.abs(d) > w / 2 + 2
        v += (target - v) * min(1, dt * (2.5 if t < c("crack") else 7.0))
        p += v * dt
        if t < c("crack"):
            # 離開畫面的微粒在隨機位置重生（非週期流場若直接環繞，邊緣會堆積成線）
            out = (p[:, 0] < 0) | (p[:, 0] >= self.W) | (p[:, 1] < 0) | (p[:, 1] >= self.H)
            if out.any():
                rng = rand(self.seed, 1, 2, self.frame)
                p[out] = rng.random((int(out.sum()), 2)).astype(np.float32) * (self.W, self.H)
                v[out] = 0

    def render(self, frame):
        t = frame / self.fps
        c = self.c
        pal = self.tl.pal
        if t < c("leaf_in") + 1:
            return self._render_void(t)
        if t < c("mud_hang"):
            return self._render_leaf(t)
        return self._render_mud(t)

    # ── 第 2 段：死與生 ─────────────────────────────────────
    def _build_leaf(self):
        c = self.c
        segs, margin = leafmod.leaf(self.seed)
        rng = rand(self.seed, 2, 2)
        self.lcx, self.lcy = 520.0, 470.0
        self.segs = segs
        self.margin = margin
        n = len(segs)
        d = segs[:, 5]
        # 從梢往柄熄滅
        self.die = (c("leaf_in") + 1.2 + 2.6 * (1 - d / d.max()) + rng.uniform(0, 0.35, n)).astype(np.float32)
        # 每條脈碎成兩段
        frags = []
        for i, (x0, y0, x1, y1, order, _) in enumerate(segs):
            for h in (0, 1):
                a = np.array([x0, y0]) + (np.array([x1, y1]) - np.array([x0, y0])) * (h * 0.5)
                b = np.array([x0, y0]) + (np.array([x1, y1]) - np.array([x0, y0])) * (h * 0.5 + 0.5)
                mid = (a + b) / 2
                hairs = []
                for _ in range(rng.integers(1, 4) if order < 2 else rng.integers(0, 2)):
                    ang = rng.uniform(0, 2 * np.pi)
                    pts = [np.zeros(2)]
                    for _ in range(14):
                        ang += rng.normal(0, 0.45)
                        pts.append(pts[-1] + 2.6 * np.array([np.cos(ang), np.sin(ang)]))
                    hairs.append(np.array(pts, np.float32))
                frags.append(dict(i=i, a=a - mid, b=b - mid, mid=mid, order=int(order),
                                  v=np.array([rng.normal(0, 26), rng.uniform(-10, 20)]),
                                  fall=rng.uniform(14, 34), rest=rng.uniform(330, 470) - self.lcy + 540,
                                  spin=rng.normal(0, 1.2), hairs=hairs, hd=rng.uniform(0, 1.5)))
        self.frags = frags

    def _render_leaf(self, t):
        c = self.c
        pal = self.tl.pal
        k = ramp(t, c("crack") + 1, c("mud_hang"))
        bg = mix(pal["ash_gray"], mix(pal["cold_gray"], pal["warm_gray"], ease(ramp(t, c("leaf_in") + 2, c("mud_hang")))),
                 ease(ramp(t, c("leaf_in") - 0.5, c("leaf_in") + 1.5)))
        img = np.broadcast_to(bg, (self.H, self.W, 3)).astype(np.float32).copy()
        lay = self.layer
        lay.clear()
        ctx = lay.ctx
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        glow = np.zeros_like(img)
        gl = Layer(self.W, self.H)
        gctx = gl.ctx
        gctx.set_line_cap(cairo.LINE_CAP_ROUND)
        cold = np.array([0.66, 0.86, 0.9])
        appear = ease(ramp(t, c("leaf_in"), c("leaf_in") + 0.9))
        widths = [3.2, 1.7, 0.9]
        ox, oy = self.lcx, self.lcy
        mx = self.W - ox  # 右邊的鏡像中心
        fade_all = 1 - ease(ramp(t, c("mud_hang") - 1.5, c("mud_hang") + 0.2))
        # 活著的網路（左）與它的殘影（右）
        for i, (x0, y0, x1, y1, order, d) in enumerate(self.segs):
            die = self.die[i]
            b = appear * (1 - 0.85 * ease(ramp(t, die, die + 0.4)))
            if t < die + 0.45 and b > 0.01:
                gctx.move_to(ox + x0, oy + y0)
                gctx.line_to(ox + x1, oy + y1)
                gctx.set_source_rgba(*cold, float(b))
                gctx.set_line_width(widths[int(order)])
                gctx.stroke()
            g = 0.36 * ease(ramp(t, die, die + 0.6)) * fade_all
            if g > 0.005:
                ctx.move_to(mx - x0, oy + y0)
                ctx.line_to(mx - x1, oy + y1)
                ctx.set_source_rgba(0.86, 0.83, 0.76, float(g))
                ctx.set_line_width(max(0.7, widths[int(order)] * 0.5))
                ctx.stroke()
        # 右側葉緣：隨左側死亡逐步勾出
        mk = ease(ramp(t, c("leaf_in") + 1.2, c("decay") + 0.5))
        if mk > 0:
            m = self.margin
            n = max(2, int(len(m) * mk))
            q = np.stack([mx - m[:n, 0], oy + m[:n, 1]], 1)
            ctx.new_path()
            path_polys(ctx, [q], close=False)
            ctx.set_source_rgba(0.86, 0.83, 0.76, 0.3 * fade_all)
            ctx.set_line_width(1.0)
            ctx.stroke()
        # 碎片：飄落，「朽腐」之後長出黴絲
        warm = np.array([0.52, 0.47, 0.38])
        for f in self.frags:
            t0 = self.die[f["i"]] + 0.45
            if t < t0:
                continue
            tau = t - t0
            kd = 1.4
            e = (1 - np.exp(-kd * tau)) / kd
            pos = f["mid"] + f["v"] * e + np.array([0, f["fall"] * tau])
            pos[1] = min(pos[1], f["rest"])
            rot = f["spin"] * e
            cr, sr = np.cos(rot), np.sin(rot)
            R = np.array([[cr, -sr], [sr, cr]])
            a = R @ f["a"] + pos + (ox, oy)
            b = R @ f["b"] + pos + (ox, oy)
            al = (0.75 - 0.35 * ramp(tau, 0, 3)) * fade_all
            ctx.move_to(float(a[0]), float(a[1]))
            ctx.line_to(float(b[0]), float(b[1]))
            ctx.set_source_rgba(*warm, float(al))
            ctx.set_line_width(widths[f["order"]] * 0.8)
            ctx.stroke()
            if t > c("decay") and f["hairs"]:
                grow = ease(ramp(t, c("decay") + f["hd"], c("decay") + f["hd"] + 3.8))
                for hpts in f["hairs"]:
                    n = int(len(hpts) * grow)
                    if n < 2:
                        continue
                    q = (hpts[:n] @ R.T) + pos + (ox, oy)
                    ctx.new_path()
                    path_polys(ctx, [q], close=False)
                    ctx.set_source_rgba(0.88, 0.86, 0.8, float(0.45 * fade_all))
                    ctx.set_line_width(0.6)
                    ctx.stroke()
        g = gl.rgba()[..., :3]
        img += g * 0.9 + bloom(g, 10) * 1.6
        over(img, lay.rgba())
        return img

    # ── 第 3 段：生命的泥 ───────────────────────────────────
    def _sky(self, t):
        """與第 4 段開頭一致的天色（地面尚未出現）。"""
        top = np.array([0.045, 0.038, 0.032])
        near = np.array([0.11, 0.09, 0.07])
        return lin_gradient_v(self.H, self.W, [(0, top), (720 / self.H, near), (1, near * 0.85)])

    def _render_mud(self, t):
        c = self.c
        pal = self.tl.pal
        k = ease(ramp(t, c("mud_hang"), c("mud_drop")))
        warm = np.broadcast_to(pal["warm_gray"], (self.H, self.W, 3))
        img = mix(warm, self._sky(t), k).astype(np.float32)
        # 黏稠的泥：metaball 場（低解析）
        w, h = 960, 540
        sx = self.W / w
        X = np.arange(w, dtype=np.float32)[None, :] * sx
        Y = np.arange(h, dtype=np.float32)[:, None] * sx
        blobs = []
        hang = ramp(t, c("mud_hang"), c("mud_drop"))
        wob = 14 * float(noise3(t * 0.4, 0, 0, 21))
        if t < c("mud_drop"):
            # 從畫面頂端垂下、越積越重
            cy = mix(20, 330, ease(hang))
            r = mix(70, 185, ease(hang))
            main = (960 + wob, cy)
            for j in range(1, 7):
                u = j / 7
                blobs.append((960 + wob * u, mix(-60, cy, u), mix(95, 46, u) * mix(1.0, 0.75, hang)))
        else:
            u = ramp(t, c("mud_drop"), c("mud_land"))
            cy = mix(330, 720 - 95, ease_in(u, 2.2))
            r = 185 * (1 + 0.1 * ease_in(u, 3))
            main = (960 + wob, cy)
            blobs.append((main[0], cy - r * 1.1 * (1 - u), r * 0.6 * (1 - u) + 1))
            back = ease_out(ramp(t, c("mud_drop"), c("mud_drop") + 0.8))
            for j in range(1, 5):
                blobs.append((960, mix(-60 + j * 55, -90, back), 60 * (1 - back) + 12))
        # 主體由幾個緩慢蠕動的小團組成，看起來黏稠
        blobs.append((main[0], main[1], r * 0.85))
        for j in range(6):
            ang = j * np.pi / 3 + t * 0.7 + float(noise3(j, t * 0.5, 0, 22))
            blobs.append((main[0] + np.cos(ang) * r * 0.45, main[1] + np.sin(ang) * r * 0.35, r * 0.42))
        # 垂掛時零星滴下的小泥點
        drng = rand(self.seed, 3, 7)
        for k in range(7):
            td = c("mud_hang") + 1.2 + k * 0.7 + drng.uniform(0, 0.4)
            if td < t < td + 1.2 and t < c("mud_drop"):
                uu = t - td
                hh = mix(10, 330, ease(ramp(td, c("mud_hang"), c("mud_drop"))))
                blobs.append((960 + drng.uniform(-60, 60), hh + 120 + 900 * uu * uu, 14))
        f = np.zeros((h, w), np.float32)
        for bx, by, br in blobs:
            f += br * br / ((X - bx) ** 2 + (Y - by) ** 2 + 1)
        a = np.clip((f - 1.6) / 0.25, 0, 1)
        from scipy.ndimage import gaussian_filter
        ab = gaussian_filter(a, 4.0)
        gy, gx = np.gradient(ab)
        shade = 0.82 + np.clip(gy * 3.0 + gx * 0.8, -0.25, 0.45)
        col = pal["mud"][None, None, :] * shade[..., None]
        # 濕亮的高光：在朝上、朝左的邊緣
        spec = np.clip((gy * 0.9 + gx * 0.5) * 4 - 0.35, 0, 1) ** 2 * a
        col = col + spec[..., None] * np.array([0.55, 0.45, 0.34], np.float32)
        rgba = np.concatenate([col * a[..., None], a[..., None]], -1)
        up = resize(rgba, self.W, self.H)
        img *= 1 - up[..., 3:4]
        img += up[..., :3]
        return img

    def _render_void(self, t):
        c = self.c
        pal = self.tl.pal
        a = self.alive
        # 微粒就是那「充實的黑」；洩出多少，底下的灰就露出多少
        gone = 1 - a.mean()
        bg = mix(pal["void"], pal["ash_gray"], ease(gone) if t >= c("crack") else 0.0)
        img = np.broadcast_to(bg, (self.H, self.W, 3)).astype(np.float32).copy()
        speed = np.linalg.norm(self.v[a], axis=1)
        col = np.array([0.62, 0.58, 0.52], np.float32)
        intensity = 0.95
        fade_in = ease(ramp(t, 0.0, 3.0))
        dens = np.zeros_like(img)
        splat(dens, self.p[a], col, self.w[a] * intensity * fade_in)
        if t >= c("crack"):
            # 洩出時拉出流線
            for k in (0.33, 0.66):
                splat(dens, self.p[a] - self.v[a] * self.dt * k * 1.5, col,
                      self.w[a] * intensity * 0.6 * np.clip(speed / 400, 0, 1))
        img += dens + bloom(dens, 10) * 0.45
        lay = self.layer
        lay.clear()
        ctx = lay.ctx
        if t >= c("crack_hairline"):
            # 「我將開口」：髮絲般的裂縫，從中心慢慢延伸
            L = mix(0, self.H * 0.32, ease(ramp(t, c("crack_hairline"), c("crack") - 0.3)))
            if t >= c("crack"):
                L = mix(L, self.H, ease_out(ramp(t, c("crack"), c("crack") + 0.12)))
            seg = self.crack[np.abs(self.crack[:, 1] - self.H / 2) <= L / 2 + 1]
            w = self.crack_width(t)
            if len(seg) > 1:
                if w < 1:
                    ctx.new_path()
                    path_polys(ctx, [seg], close=False)
                    a = 0.18 + 0.25 * ramp(t, c("crack") - 4, c("crack"))
                    ctx.set_source_rgba(0.75, 0.72, 0.66, a)
                    ctx.set_line_width(0.9)
                    ctx.stroke()
                else:
                    left = seg - (w / 2, 0)
                    right = (seg + (w / 2, 0))[::-1]
                    ctx.new_path()
                    path_polys(ctx, [np.concatenate([left, right])])
                    g = pal["ash_gray"]
                    ctx.set_source_rgba(float(g[0]), float(g[1]), float(g[2]), 1)
                    ctx.fill()
        over(img, lay.rgba())
        if t >= c("crack"):
            # 迸開的一瞬，縫裡的光暈
            w = self.crack_width(t)
            flash = np.exp(-(t - c("crack")) / 0.5) * 0.35
            if flash > 0.01:
                xx = np.arange(self.W, dtype=np.float32)
                prof = np.exp(-np.maximum(np.abs(xx - self.W / 2) - w / 2, 0) / 70) * flash
                img += prof[None, :, None] * np.array([0.8, 0.78, 0.72], np.float32)
        return img
