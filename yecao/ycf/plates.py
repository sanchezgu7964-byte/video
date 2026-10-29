"""刻版的意象鏡頭：具體的人、物與處境。

每一張是 f(b, p, t, **kw)：b 是版（carve.Block），p 是這個鏡頭自己的進度 0..1，t 是片子的時間。
每張版都是一筆一筆刻的（見 carve.py）；`Plates.render(..., plain=True)` 印出不帶任何印痕的純黑白，
用來檢查造型本身。
"""
import numpy as np

from . import fonts
from .carve import Block, Print, _h, ellipse, offset, rect, resample, rough, spline, xform
from .figures import Profile, head, shoulders
from .gfx import ease, ease_in, ease_out, ramp

W, H = 1920, 1080
REG = {}


def plate(name):
    def deco(f):
        REG[name] = f
        return f
    return deco


def smooth(e0, e1, x):
    u = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0, 1)
    return u * u * (3 - 2 * u)


def neck(b, base, top, w0, w1, s, seed, light=1.0):
    """頸：一條黑帶，迎光的前緣兩三刀，伸長時頸側的筋是一道長刀。"""
    base, top = np.asarray(base, np.float32), np.asarray(top, np.float32)
    d = top - base
    L = np.linalg.norm(d)
    u = d / (L + 1e-6)
    n = np.array([-u[1], u[0]])                      # 指向左（背後）
    poly = np.array([base - n * w0 * 0.5 * -1, top - n * w1 * 0.5 * -1, top + n * w1 * 0.5 * -1,
                     base + n * w0 * 0.5 * -1])
    poly = np.array([base + n * w0 / 2, top + n * w1 / 2, top - n * w1 / 2, base - n * w0 / 2])
    b.ink([poly])
    items = []
    for k in range(3):
        a = base - n * (w0 / 2 - (k + 0.6) * 9) + u * 10
        c = top - n * (w1 / 2 - (k + 0.6) * 8) - u * 6
        P = resample(np.array([a, (a + c) / 2 - n * 3, c]), 2.0)
        w = (1 - k / 3.2) * light * 7.5 * np.clip(np.sin(np.linspace(0, np.pi, len(P))) * 1.8, 0, 1)
        items.append((P, w))
    b.strokes(items, 1.0, wmin=0.8, seed=seed)
    # 胸鎖乳突肌：從耳後斜到頸根前方
    if L > 30:
        b.knife(np.array([top + n * w1 * 0.3, (top + base) / 2 - n * 2, base - n * w0 * 0.25]), 5.0 * s, 1.0,
                taper=(0.5, 0.4), seed=seed + 1)


def onlooker(b, base, s, crane, prof, mouth=0.0, seed=0, lean0=0.25, chin0=0.06, hunch=0.4, light=1.0, eye=1.0):
    """一個看客：聳起的肩、往前伸的頸、抬起的下巴，眼睛盯著同一個方向。base 是頸根。"""
    bx, by = base
    lean = lean0 + 0.55 * crane                      # 頸往前伸的角度
    L = (26 + 26 * crane) * s
    shoulders(b, bx - 8 * s, by + 8 * s, s * 0.82, hunch=hunch + 0.4 * crane, seed=seed, px=10.0, light=light,
              rim=max(1.6, 0.9 * s))
    top = np.array([bx + np.sin(lean) * L, by - np.cos(lean) * L])
    rot = -(chin0 + 0.16 * crane)
    neck(b, (bx, by + 10 * s), top, 40 * s, 30 * s, s / 3.0, seed + 5, light)
    # 頭：耳朵在頸頂往上、往後一點
    ear = top + xform(np.array([[-6.0, -50.0]]), 0, 0, s, rot)[0]
    head(b, ear[0], ear[1], s, prof, rot=rot, mouth=mouth, eye=eye, seed=seed + 10, px=max(7.0, 2.6 * s), light=light,
         neck=62, rim=max(1.6, 0.9 * s))


# ── 看客 ───────────────────────────────────────────────────
@plate("onlookers")
def onlookers(b, p, t, **kw):
    """《吶喊》自序：幻燈片裡「圍著看的」——一樣強壯的體格，顯出麻木的神情。
    全都側著臉朝右，頸越伸越長；被看的那一邊是刺眼的白，什麼都看不見。"""
    crane = ease(p)
    # 背景：左邊沉黑，向右漸漸被橫刀刻開，到被看的那一邊成一片白
    b.ink([rect(-200, -200, 2200, 1300)])
    tone = lambda x, y: smooth(700, 1700, x + 60 * np.sin(y / 140)) ** 1.1
    b.rules([rect(-200, -200, 2200, 1300)], -40, 1120, tone, spacing=15, seed=11, seg=(380, 1300), gap=(8, 40))
    # 被看的那一邊：刻空的一片。邊緣是一刀一刀刻出來的，每一行停在不同的地方
    clear_right(b, lambda y: 1600 + 50 * np.sin(y / 170.0), 15, seed=12)
    breathe = lambda k: 0.012 * np.sin(t * 1.4 + k * 1.7)
    people = [  # (頸根, 尺度, 臉, 嘴, 頸的起始前傾, 啟動延遲)
        ((1330, 430), 0.95, Profile(cap="skull", nose=1.1), 0.1, 0.25, 0.3),
        ((1530, 455), 0.9, Profile(cap="shaved", nose=0.9, age=0.4), 0.2, 0.3, 0.2),
        ((1120, 470), 1.0, Profile(cap="shaved", nose=1.3, brow=1.3), 0.05, 0.2, 0.4),
        ((1420, 560), 1.3, Profile(cap="skull", nose=1.2, brow=1.4, chin=1.5), 0.05, 0.30, 0.25),
        ((1170, 560), 1.45, Profile(cap="shaved", nose=0.8, age=0.6, jaw=1.05), 0.12, 0.20, 0.10),
        ((900, 620), 1.7, Profile(cap="bare", nose=1.0, skull=1.04, age=0.3), 0.18, 0.15, 0.0),
        ((520, 840), 3.2, Profile(cap="skull", nose=1.1, brow=1.2, jaw=1.08, age=0.4, moustache=True), 0.12, 0.22, 0.12),
        ((1200, 1060), 4.4, Profile(cap="shaved", nose=1.25, brow=1.6, age=0.8, gaunt=0.6, chin=1.0), 0.22, 0.18, 0.05),
    ]
    for k, (base, s, prof, mouth, lean0, delay) in enumerate(people):
        c = ease(np.clip((crane - delay) / (1 - delay + 1e-6), 0, 1))
        onlooker(b, base, s, c + breathe(k), prof, mouth=mouth + 0.1 * c * (k % 2), seed=100 * k, lean0=lean0,
                 hunch=0.5, light=1.0)


def clear_right(b, xfn, spacing, seed=0, y0=-60, y1=1140):
    """右邊整片刻白：一行一行從 xfn(y) 附近開始，左端是刀尖。"""
    y = y0
    k = 0
    while y < y1:
        x = xfn(y) + (_h(seed, k) - 0.5) * 120
        L = 2400 - x
        b.knife(np.array([[x, y], [x + 200, y], [2400, y]]), spacing * 1.25, 1.0, taper=(min(0.5, 160 / L), 0),
                smooth=False, seed=seed + k, swell=0.0)
        y += spacing
        k += 1


# ── 布旗 ───────────────────────────────────────────────────
class Cloth:
    """兩根竿子之間的一塊布。(u, v) ∈ [0,1]²：u 沿著布、v 往下。風從左邊來，褶沿著 u 往右走。"""

    def __init__(self, left, right, height, t, wind=1.0, seed=0):
        self.L, self.R, self.Hb, self.t, self.wind, self.seed = np.asarray(left, np.float32), \
            np.asarray(right, np.float32), height, t, wind, seed

    def z(self, u, v):
        t, w = self.t, self.wind
        env = np.maximum(np.sin(np.pi * np.clip(u, 0, 1)), 0) ** 0.7
        z = (26 * np.sin(2 * np.pi * (2.3 * u - 0.55 * t)) + 12 * np.sin(2 * np.pi * (4.1 * u - 0.9 * t) + 1.3)
             + 7 * np.sin(2 * np.pi * (6.7 * u + 0.4 * v - 1.3 * t) + 0.4))
        return w * env * z * (0.75 + 0.5 * v)

    def __call__(self, u, v):
        u, v = np.asarray(u, np.float32), np.asarray(v, np.float32)
        top = self.L + (self.R - self.L) * u[..., None]
        z = self.z(u, v)
        sag = 34 * np.sin(np.pi * u) * (0.6 + 0.4 * v)
        x = top[..., 0] - 0.35 * z
        y = top[..., 1] + v * self.Hb * (1 - 0.04 * np.sin(np.pi * u)) + sag + 0.55 * z + 9 * v * np.sin(
            2 * np.pi * (3 * u - 1.1 * self.t)) * self.wind
        return np.stack([x, y], -1)

    def shade(self, u, v):
        e = 0.004
        dz = (self.z(u + e, v) - self.z(u - e, v)) / (2 * e)
        return dz / 240.0                     # 正：迎光（風從左吹、光從左上）


def fist(b, x, y, s, seed=0, angle=0.0):
    """舉起的拳，指節朝前：四節手指三道刀，拇指橫壓在前面，下緣一道光。"""
    T = lambda P: xform(P, x, y, s, angle)
    body = spline([(-13, -8), (-11, -13), (-4, -14.5), (4, -14.5), (11, -13), (13.5, -7), (13, 6), (9, 13), (-8, 13),
                   (-13.5, 5)], closed=True, step=0.4)
    b.halo(T(body), 1.8, seed)
    b.ink([T(body)])
    for k, xx in enumerate((-6.5, 0.0, 6.5)):
        b.knife(T(np.array([[xx, -14], [xx + 0.3, -9], [xx, -3.5]])), 1.3 * s, 1.0, taper=(0.3, 0.5), seed=seed + k)
    for k, xx in enumerate((-9.8, -3.2, 3.2, 9.8)):
        b.knife(T(np.array([[xx - 2.2, -11.5], [xx, -12.6], [xx + 2.2, -11.5]])), 0.9 * s, 1.0, taper=(0.4, 0.4),
                seed=seed + 10 + k)
    b.knife(T(np.array([[-12, 1.5], [-2, -0.5], [9, 0.5]])), 1.4 * s, 1.0, taper=(0.15, 0.5), seed=seed + 5)
    b.knife(T(np.array([[-11, 9], [0, 11], [10, 9.5]])), 1.0 * s, 1.0, taper=(0.4, 0.4), seed=seed + 6)


def arm(b, shoulder, hand, s, seed=0, bend=0.25, w0=58, w1=40):
    """袖：從肩到腕，一條粗而略收的黑，迎光的一側一道刀，袖口一圈白，肘彎兩刀。"""
    sh, hd = np.asarray(shoulder, np.float32), np.asarray(hand, np.float32)
    d = hd - sh
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
    elbow = sh + d * 0.5 + n * np.linalg.norm(d) * bend
    C = spline([sh, elbow, hd], step=2)
    wv = np.linspace(w0, w1, len(C)) * s / 3.0
    N = np.gradient(C, axis=0)[:, ::-1] * np.array([-1, 1])
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    poly = np.concatenate([C + N * wv[:, None] / 2, (C - N * wv[:, None] / 2)[::-1]])
    b.halo(poly, 1.8, seed)
    b.ink([poly])
    hi = C - N * (wv[:, None] / 2 - 6 * s / 3)
    b.knife(hi[len(hi) // 10: -len(hi) // 8], 4.5 * s / 3, 1.0, taper=(0.4, 0.5), seed=seed + 1)
    hi2 = C - N * (wv[:, None] / 2 - 14 * s / 3)
    b.knife(hi2[len(hi2) // 3: -len(hi2) // 5], 2.5 * s / 3, 1.0, taper=(0.5, 0.5), seed=seed + 3)
    i = len(C) // 2
    for j in (0, 4):
        b.knife(np.array([C[i + j] + N[i] * wv[i] * 0.45, C[i + j + 2] - N[i] * wv[i] * 0.05]), 3.0 * s / 3, 1.0,
                taper=(0.5, 0.5), seed=seed + 2 + j)
    cuff = np.array([C[-3] + N[-3] * wv[-3] / 2, C[-3] - N[-3] * wv[-3] / 2])
    b.knife(cuff, 4.0 * s / 3, 1.0, taper=(0.15, 0.15), smooth=False, seed=seed + 9)


def pole(b, x0, y0, x1, y1, w, seed=0):
    P = np.array([[x0, y0], [x1, y1]], np.float32)
    b.halo(np.array([[x0 - w / 2, y0], [x1 - w / 2, y1], [x1 + w / 2, y1], [x0 + w / 2, y0]]), 1.6, seed)
    b.knife(P, w, 0.0, taper=(0, 0), smooth=False, seed=seed, swell=0.02)
    b.knife(P + (-w * 0.18, 0), w * 0.18, 1.0, taper=(0.1, 0.1), smooth=False, seed=seed + 1)


def crowd_line(b, y, n, s, t, seed=0, x0=-60, x1=1980, light_rim=True):
    """遠處的人頭：一排剪影，帽、頭、肩，偶爾一根旗竿。"""
    xs = np.linspace(x0, x1, n)
    polys = []
    for k, x in enumerate(xs):
        x = x + (_h(seed, k) - 0.5) * 60
        yy = y + (_h(seed, k, 1) - 0.5) * 14 * s + 3 * s * np.sin(t * 2.2 + k)
        r = (11 + 4 * _h(seed, k, 2)) * s
        polys.append(ellipse(x, yy, r * 0.85, r))
        if _h(seed, k, 3) < 0.5:
            polys.append(np.array([[x - r * 1.1, yy - r * 0.55], [x + r * 1.1, yy - r * 0.55], [x + r * 0.8, yy - r * 1.05],
                                   [x - r * 0.8, yy - r * 1.05]], np.float32))
        polys.append(spline([(x - 2.9 * r, yy + 6 * r), (x - 2.6 * r, yy + 2.2 * r), (x - 1.6 * r, yy + 1.2 * r),
                             (x, yy + 0.9 * r), (x + 1.6 * r, yy + 1.2 * r), (x + 2.6 * r, yy + 2.2 * r),
                             (x + 2.9 * r, yy + 6 * r)], step=3))
    if light_rim:
        for q in polys:
            b.halo(q, 1.3, seed, breaks=0.4)
    b.ink(polys)


def small_flag(b, x, y, h, t, seed=0, sway=0.0):
    """遠處的一面小三角旗。"""
    P = xform(np.array([[0, 0], [0, -h]]), x, y, 1, sway)
    b.knife(P, max(2.5, h / 70), 0.0, taper=(0, 0.05), smooth=False, seed=seed, swell=0.0)
    top = P[1]
    w = h * 0.45
    k = np.linspace(0, 1, 12)
    up = np.stack([top[0] + w * k, top[1] + h * 0.02 * np.sin(6 * k - t * 4 - seed) * k + h * 0.08 * k], 1)
    lo = np.stack([top[0] + w * k, top[1] + h * 0.28 - h * 0.14 * k + h * 0.02 * np.sin(6 * k - t * 4 - seed - 0.7) * k], 1)
    fl = np.concatenate([up, lo[::-1]])
    b.halo(fl, 1.3, seed)
    b.ink([fl])


@plate("banner")
def banner(b, p, t, text="外爭主權　內除國賊", **kw):
    """一九一九年五月四日，天安門前：學生舉著白布橫幅，風把布吹出褶，字跟著褶走。仰角。"""
    kai = fonts.face("fonts/kai_bold")
    # 天：上沉下亮，橫刀
    b.ink([rect(-200, -200, 2200, 1300)])
    tone = lambda x, y: smooth(330, 1000, y + 40 * np.sin(x / 210))
    b.rules([rect(-200, -200, 2200, 1300)], -40, 1120, tone, spacing=14, seed=21, seg=(400, 1300), gap=(8, 40))
    # 遠處：一排排的人頭與小旗
    for k in range(7):
        small_flag(b, 120 + k * 290 + 50 * np.sin(k * 2.1), 790 + 10 * np.sin(k), 210 + 30 * np.sin(k * 1.7), t, seed=k,
                   sway=0.05 * np.sin(t * 1.3 + k))
    crowd_line(b, 790, 30, 1.0, t, seed=3)
    crowd_line(b, 850, 22, 1.5, t, seed=4)

    # 布
    cl = Cloth((190, 150), (1730, 132), 290, t, wind=0.55 + 0.3 * ease(p))
    U = np.linspace(0, 1, 140)
    edge_t, edge_b = cl(U, U * 0), cl(U, U * 0 + 1)
    vv = np.linspace(0, 1, 30)
    outline = np.concatenate([edge_t, cl(vv * 0 + 1, vv), edge_b[::-1], cl(vv * 0, vv[::-1])])
    b.halo(outline, 2.5, 7, breaks=0.1)
    b.carve([outline])
    # 字：楷書，一個字一個字貼在布上
    chars = [c for c in text]
    n = len(chars)
    cw = 0.9 / n
    for j, ch in enumerate(chars):
        g = kai.glyph(ch)
        if not g.contours:
            continue
        u0 = 0.05 + j * cw
        polys = []
        for q in g.contours:
            qq = resample(np.concatenate([q, q[:1]]), 0.008)
            uv = np.stack([u0 + qq[:, 0] * cw * 0.95, 0.5 + (qq[:, 1] + 0.38) * 0.62], 1)
            polys.append(cl(uv[:, 0], uv[:, 1]))
        b.ink(polys)
    # 褶：沿著 v 走的黑刀，寬度跟著背光的程度
    with b.clip([outline]):
        items = []
        for k, u in enumerate(np.linspace(0.004, 0.996, 160)):
            sv = np.linspace(-0.02, 1.02, 40)
            P = cl(np.full_like(sv, u), sv)
            dark = np.clip(-cl.shade(np.full_like(sv, u), sv) * 1.1 - 0.12, 0, 1.0) ** 1.3
            items.append((P, dark * 9.0))
        b.strokes(items, 0.0, wmin=0.9, seed=31, taper=0.1, swell=0.15)
    b.knife(cl(U, U * 0 + 0.035), 2.0, 0.0, taper=(0.02, 0.02), smooth=False, seed=8)
    b.knife(cl(U, U * 0 + 0.965), 2.0, 0.0, taper=(0.02, 0.02), smooth=False, seed=9)
    lp, rp = cl(np.array([0.0]), np.array([0.0]))[0], cl(np.array([1.0]), np.array([0.0]))[0]
    pole(b, lp[0] - 4, lp[1] - 50, lp[0] - 30, 1200, 22, seed=40)
    pole(b, rp[0] + 4, rp[1] - 50, rp[0] + 26, 1200, 22, seed=41)

    # 學生：仰著臉喊
    shout = 0.5 + 0.5 * np.abs(np.sin(np.pi * (t * 0.9)))
    studs = [  # (頸根, 尺度, 臉, 嘴)
        ((330, 900), 2.5, Profile(cap="student", nose=0.9, brow=1.3, jaw=1.0), 0.35 + 0.6 * shout),
        ((1560, 930), 2.6, Profile(cap="bare", nose=1.1, brow=1.5, chin=0.5), 0.85 - 0.5 * shout),
        ((900, 960), 3.1, Profile(cap="student", nose=1.0, brow=1.0), 1.0 - 0.35 * shout),
    ]
    for k, (base, s, prof, mouth) in enumerate(studs):
        bx, by = base
        shoulders(b, bx - 8 * s, by + 8 * s, s * 0.82, hunch=0.0, seed=200 + k, px=10, rim=max(1.6, 0.9 * s))
        for j in range(3):                                   # 學生裝：前襟一排扣
            b.carve([ellipse(bx + 100 * s * 0.82, by + (70 + 60 * j) * s * 0.82, 2.6 * s / 2.3, 2.6 * s / 2.3)])
        rot = -0.24
        top = np.array([bx + 4 * s, by - 26 * s])
        neck(b, (bx, by + 10 * s), top, 40 * s, 32 * s, s / 3.0, 300 + k)
        ear = top + xform(np.array([[-6.0, -50.0]]), 0, 0, s, rot)[0]
        head(b, ear[0], ear[1], s, prof, rot=rot, mouth=mouth, seed=400 + 37 * k, px=max(7.0, 2.6 * s),
             neck=62, rim=max(1.6, 0.9 * s))
        if k == 2:
            # 舉起的拳頭
            lift = np.sin(np.pi * t * 0.9) ** 2
            hx, hy = bx + 150 * s, by - 175 * s - 12 * s * lift
            arm(b, (bx + 95 * s, by + 40 * s), (hx, hy + 12 * s), s, seed=530, bend=0.1, w0=110, w1=72)
            fist(b, hx, hy, s * 1.45, seed=540, angle=0.12)


# ── 鐵屋子 ─────────────────────────────────────────────────
def sleeper(b, x, y, s, prof, t, seed=0, legs=1.0, rim=2.0, eye=0.0, lift=0.0):
    """一個仰躺熟睡的人：頭在左、臉朝上，身上蓋著一條被，被的褶隨呼吸起伏。(x, y) 是耳朵。"""
    breath = 1 + 0.035 * np.sin(t * 1.1 + seed * 0.7)
    # 被：從頸到腳的一道起伏
    L = 520 * s / 3.0 * legs
    T = lambda P: xform(P, x, y, s / 3.0)
    top = [(40, -60), (120, -95 * breath), (220, -100 * breath), (330, -70), (430, -82), (560, -88), (700, -66),
           (L / (s / 3.0), -50)]
    mound = spline([(30, 40)] + top + [(L / (s / 3.0) + 30, 10), (L / (s / 3.0) + 30, 50), (30, 50)], closed=True, step=3)
    b.halo(T(mound), rim, seed + 1)
    b.ink([T(mound)])
    ctop = spline(top, step=2)
    items = []
    for k in range(4):
        C = resample(offset(ctop, k * 13.0), 2.0)
        u = np.linspace(0, 1, len(C))
        w = (1 - k / 3.6) ** 1.6 * np.clip(np.sin(np.pi * u) * 3.0, 0, 1)
        if k:
            w *= 0.5 + 0.5 * np.sin(u * 9 + seed)
        items.append((T(C), w * (9.0 if k == 0 else 6.5) * s / 3.0))
    b.strokes(items, 1.0, wmin=0.9, seed=seed + 2, breaks=0.08)
    # 褶：從被面往下的幾刀
    for j in range(5):
        u0 = 0.1 + 0.18 * j + 0.05 * _h(seed, j)
        i = int(u0 * (len(ctop) - 1))
        q = ctop[i]
        b.knife(T(np.array([q + (0, 16), q + (14 * (_h(seed, j, 1) - 0.5), 60), q + (6, 105)])), 4.0 * s / 3.0, 1.0,
                taper=(0.3, 0.7), seed=seed + 10 + j)
    # 頭：仰著，閉眼
    head(b, x, y - 18 * s * lift, s, prof, rot=-np.pi / 2 + 0.55 * lift, mouth=0.06, eye=eye, seed=seed + 20,
         px=max(7.0, 2.6 * s), neck=58, rim=rim, comb=0.45)


def dozer(b, base, s, prof, droop, t, seed=0, eye=0.0, rim=2.0, lean=0.0):
    """一個坐著睡著的人：肩垮下來，頭垂到胸前（droop 1＝垂到底，0＝抬起），隨呼吸微微起伏。"""
    bx, by = base
    br = 0.012 * np.sin(t * 0.9 + seed)
    shoulders(b, bx - 8 * s, by + 8 * s, s * 0.82, hunch=0.9, seed=seed, px=10.0, rim=rim, rot=lean)
    rot = 0.42 * droop - 0.18 * (1 - droop) + br
    top = np.array([bx + (6 + 16 * droop) * s, by - (34 - 10 * droop) * s])
    neck(b, (bx, by + 10 * s), top, 40 * s, 32 * s, s / 3.0, seed + 5)
    ear = top + xform(np.array([[-6.0, -50.0]]), 0, 0, s, rot)[0]
    head(b, ear[0], ear[1], s, prof, rot=rot, mouth=0.04 + 0.06 * droop, eye=eye, seed=seed + 10, px=max(7.0, 2.6 * s),
         neck=62, rim=rim)


@plate("iron_house")
def iron_house(b, p, t, **kw):
    """《吶喊》自序：一間鐵屋子，絕無窗戶而萬難破毀，裡面有許多熟睡的人們，不久都要悶死了。
    天花板壓在頭頂；人挨著人坐著睡，頭垂在胸前。中間一個，睜開了眼，慢慢抬起頭。"""
    b.ink([rect(-200, -200, 2200, 1300)])
    # 後牆：鋼板接縫與鉚釘，幾乎全黑
    for row, yy in enumerate((470, 700)):
        b.knife(np.array([[-100, yy], [2100, yy + 6]]), 4, 1.0, taper=(0, 0), smooth=False, seed=row)
        for i in range(26):
            xx = -60 + i * 84 + (row % 2) * 42
            b.carve([ellipse(xx, yy + 18, 5.0, 4.5)])
            b.ink([ellipse(xx + 1.5, yy + 19.5, 3.0, 2.6)])
    for i in range(5):
        xx = 120 + i * 430
        b.knife(np.array([[xx, 390], [xx + 3, 800]]), 3, 1.0, taper=(0.02, 0.4), smooth=False, seed=10 + i)
    # 天花板：一整塊壓下來的黑，下緣一道冷光，一排鉚釘
    ceil_y = 380.0
    b.ink([np.array([[-200, -200], [2200, -200], [2200, ceil_y - 8], [-200, ceil_y + 8]], np.float32)])
    b.knife(np.array([[-100, ceil_y + 6], [2100, ceil_y - 8]]), 8, 1.0, taper=(0, 0), smooth=False, seed=3)
    for i in range(30):
        b.carve([ellipse(-40 + i * 70, ceil_y - 34 - 0.008 * i * 70, 6, 5.5)])
    b.rules([rect(-200, -200, 2200, ceil_y - 50)], 40, ceil_y - 50, lambda x, y: 0.0 + 0.24 * smooth(120, 330, y),
            spacing=28, seed=52, seg=(600, 1800), gap=(10, 30))
    # 後排：小，擠在一起
    for i in range(9):
        x = -20 + i * 240 + 30 * np.sin(i * 1.9)
        prof = Profile(cap="bare" if i % 3 else "shaved", nose=0.8 + 0.4 * _h(1, i), brow=1 + 0.5 * _h(1, i, 1),
                       age=0.6 * _h(1, i, 2))
        dozer(b, (x, 640 + 12 * np.sin(i * 2.7)), 1.45, prof, 0.85 + 0.15 * _h(1, i, 3), t, seed=1000 + 31 * i, rim=1.6)
    # 前排：大
    wake = ease(ramp(p, 0.15, 0.45))
    rise = ease(ramp(p, 0.5, 1.0))
    front = [((150, 860), 2.7, Profile(cap="skull", nose=1.1, brow=1.3, age=0.5), 1.0),
             ((560, 900), 2.9, Profile(cap="shaved", nose=0.9, age=0.7, jaw=1.05), 0.95),
             ((1020, 880), 3.0, Profile(cap="bare", nose=1.05, brow=1.5, age=0.4, gaunt=0.5, chin=0.6), None),
             ((1470, 910), 2.8, Profile(cap="bare", nose=1.2, brow=1.1, age=0.2), 0.9),
             ((1880, 870), 2.7, Profile(cap="skull", nose=1.0, brow=1.2), 1.0)]
    for k, (base, s, prof, droop) in enumerate(front):
        if droop is None:
            dozer(b, base, s, prof, 0.95 * (1 - rise), t * (1 - 0.8 * wake), seed=2000 + 53 * k,
                  eye=0.0 if wake < 0.5 else 1.5 + 0.2 * rise, rim=2.4)
        else:
            dozer(b, base, s, prof, droop, t, seed=2000 + 53 * k, rim=2.4)


# ── 傳單 ───────────────────────────────────────────────────
MANIFESTO = ["北京學界全體宣言",
             "現在日本在萬國和會要求併吞青島，", "管理山東一切權利，就要成功了！",
             "所以我們學界今天排隊到各公使館去，", "外爭主權，內除國賊，", "中國存亡，就在此一舉了！",
             "今與全國同胞立兩個信條道：", "中國的土地可以征服而不可以斷送！", "中國的人民可以殺戮而不可以低頭！",
             "國亡了！同胞起來呀！"]
PUNCT = "，。！：、"


def sole(cx, cy, L, ang):
    """靴底的輪廓（俯視）：腳尖圓、腰窄、後跟方。"""
    P = [(0, -0.5), (0.2, -0.47), (0.26, -0.3), (0.24, -0.1), (0.16, 0.08), (0.17, 0.3), (0.16, 0.5), (-0.16, 0.5),
         (-0.17, 0.3), (-0.13, 0.08), (-0.2, -0.1), (-0.22, -0.3), (-0.17, -0.46)]
    return xform(spline(P, closed=True, step=0.01), cx, cy, L, ang)


@plate("leaflet")
def leaflet(b, p, t, **kw):
    """一九一九年五月四日散發的《北京學界全體宣言》，掉在泥地上（俯拍）。
    p 0–0.4：紙；0.4–0.55：靴的影子壓下來；0.55–1：抬起，紙上一個帶鞋釘的泥印，只剩半句看得清。"""
    serif = fonts.face("fonts/serif_regular")
    black = fonts.face("fonts/serif_black")
    stomped = p >= 0.55
    # 泥地：黑，一道道車轍與水痕
    b.ink([rect(-200, -200, 2200, 1300)])
    tone = lambda x, y: 0.12 + 0.12 * np.sin(x / 260 + y / 400) + 0.08 * np.sin(y / 70)
    b.rules([rect(-200, -200, 2200, 1300)], -60, 1140, tone, spacing=16, seed=61, seg=(120, 520), gap=(20, 90),
            angle=0.35, wobble=6)
    for k in range(14):
        x, y = 80 + _h(62, k) * 1760, 60 + _h(62, k, 1) * 960
        r = 12 + 26 * _h(62, k, 2)
        st = ellipse(x, y, r, r * 0.7, _h(62, k, 3) * 3)
        b.halo(st, 2.5, 60 + k)
        b.ink([st])
    # 紙
    ang = -0.07
    cx, cy = 960, 560
    pw, ph = 760, 1000
    corners = xform(np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]]), cx, cy, 1, ang)
    paper = rough(corners, 2.5, 3)
    if stomped:
        # 左下角被撕去
        tear = np.array([[cx - pw / 2 - 40, cy + 120], [cx - 160, cy + 330], [cx - 250, cy + 600], [cx - pw / 2 - 60, cy + 640]])
        tear = rough(tear, 9, 4)
    b.halo(paper, 3.0, 5, breaks=0.05)
    b.carve([paper])
    if stomped:
        b.ink([tear])

    def crump(P):
        if not stomped:
            return P
        d = P - (cx - 60, cy + 260)
        r = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        return P + 9 * np.sin(r / 23) * d / r * np.exp(-r / 380)

    loc = lambda P: crump(xform(np.asarray(P, np.float32) - (cx, cy), cx, cy, 1, ang))
    # 直排，由右至左；句讀用旁邊的圈點
    size = 44
    colx = cx + pw / 2 - 80
    mark = (cx, cy)
    for j, line in enumerate(MANIFESTO):
        face, sz = (black, 60) if j == 0 else (serif, size)
        x = colx - (0 if j == 0 else 36 + j * 60)
        y = cy - ph / 2 + 70
        row = 0
        for ch in line:
            if ch in PUNCT:
                b.ink([loc(ellipse(x + sz * 0.55, y + sz * (row - 0.15), 4.5, 4.5))])
                continue
            g = face.glyph(ch)
            polys = [loc(q * sz + (x - sz * g.advance / 2, y + sz * (row + 0.88))) for q in g.contours]
            b.ink(polys)
            if j == 8 and ch == "頭":
                mark = loc(np.array([[x, y + sz * (row + 0.5)]]))[0]
            row += 1
        if j == 0:
            b.knife(loc(np.array([[x - 44, y - 20], [x - 44, y + 60 * 8.4]])), 2.5, 0.0, taper=(0, 0), smooth=False)
    # 摺痕
    for k, (u0, v0, u1, v1) in enumerate([(-pw / 2, -ph / 6, pw / 2, -ph / 6 + 8), (-pw / 2, ph / 6, pw / 2, ph / 6 - 6)]):
        P = loc(resample(np.array([[cx + u0, cy + v0], [cx + u1, cy + v1]]), 8))
        b.knife(P, 2.2, 0.0, taper=(0.1, 0.1), smooth=False, seed=70 + k)
        b.knife(P + (0, 3), 1.5, 1.0, taper=(0.1, 0.1), smooth=False, seed=72 + k)
    # 靴的影子壓下來
    if 0.38 < p < 0.6:
        k = ease_in(ramp(p, 0.38, 0.55), 2.2)
        L = 900 + 1900 * k
        sx, sy = cx - 300 - 260 * (1 - k), cy + 340 - 420 * (1 - k)
        b.ink([sole(sx, sy, L, 1.05)])
    # 泥印：鞋底的形狀，前掌一道道橫紋，後跟一圈鞋釘
    if stomped:
        # 腳尖正好踩在「頭」上
        L, sa = 500, 1.05
        toe = xform(np.array([[0.0, -0.36]]), 0, 0, L, sa)[0]
        sx, sy = mark[0] - toe[0], mark[1] - toe[1]
        sp = sole(sx, sy, L, sa)
        b.ink([rough(sp, 4, 9)])
        # 壓得輕的地方，泥沒有印實：留幾塊紙白
        for k in range(7):
            q = xform(np.array([[(_h(95, k) - 0.5) * 0.36, -0.42 + 0.84 * _h(95, k, 1)]]), sx, sy, L, sa)[0]
            gap = rough(ellipse(q[0], q[1], 10 + 30 * _h(95, k, 2), 6 + 14 * _h(95, k, 3), _h(95, k, 4) * 3), 4, 96 + k)
            with b.clip([sp]):
                b.carve([gap])
        with b.clip([sp]):
            for k in range(7):
                v = -0.4 + k * 0.06
                q = xform(np.array([[-0.3, v], [0.3, v + 0.01]]), sx, sy, L, sa)
                b.knife(q, 7, 1.0, taper=(0.15, 0.15), smooth=False, seed=80 + k)
            for k in range(14):
                th = np.pi * k / 13
                q = xform(np.array([[0.12 * np.cos(th), 0.38 + 0.1 * np.sin(th)]]), sx, sy, L, sa)[0]
                b.carve([ellipse(q[0], q[1], 7, 7)])
                b.ink([ellipse(q[0] + 2, q[1] + 2, 3, 3)])
        # 濺出的泥點
        for k in range(26):
            th = _h(90, k) * 2 * np.pi
            r = L * (0.3 + 0.25 * _h(90, k, 1))
            q = (sx + r * np.cos(th) * 0.7, sy + r * np.sin(th))
            b.ink([ellipse(q[0], q[1], 4 + 14 * _h(90, k, 2), 3 + 9 * _h(90, k, 3), th)])


# ── 渲染 ───────────────────────────────────────────────────
class Plates:
    def __init__(self, seed, W=W, H=H):
        from . import plates2  # noqa: F401  其餘的版在這裡登記
        self.block = Block(W, H)
        self.press = Print(W, H, seed)

    def render(self, name, p, t, view=(1.0, 960, 540), plain=False, **kw):
        self.block.begin(view)
        REG[name](self.block, float(np.clip(p, 0, 1)), t, **kw)
        return self.press(self.block.mask(), plain=plain)
