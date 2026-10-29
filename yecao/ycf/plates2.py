"""其餘的刻版意象：嘴、燭、燈下寫字的人、牢窗、日曆、石縫的草、根與骨、牌坊、靴、方陣、散場的街、呈文。

和 plates.py 同一套刀法：先定黑白大塊，刀線交代結構，背景是橫刀。
"""
import numpy as np

from . import fonts
from .carve import _h, ellipse, offset, rect, resample, rough, spline, xform
from .figures import Profile, head, shoulders
from .gfx import ease, ease_in, ease_out, ramp
from .plates import MANIFESTO, PUNCT, Cloth, arm, crowd_line, fist, neck, plate, small_flag, smooth, sole

W, H = 1920, 1080
FULL = [rect(-300, -300, 2300, 1400)]


def rings(b, cx, cy, r0, r1, spacing, wfn, seed=0, ry=1.0):
    """光：以 (cx, cy) 為中心的一圈圈白刀，寬度隨距離遞減。木刻裡畫光的老辦法。"""
    r = r0
    k = 0
    while r < r1:
        th = np.linspace(0, 2 * np.pi, max(24, int(r / 3)))
        P = np.stack([cx + r * np.cos(th), cy + r * ry * np.sin(th)], 1)
        w = wfn(r) * spacing * (0.9 + 0.2 * _h(seed, k))
        b.strokes([(P, np.full(len(P), w))], 1.0, wmin=0.6, seed=seed + k, taper=0.0, breaks=0.15, swell=0.1)
        r += spacing * (0.9 + 0.25 * _h(seed, k, 1))
        k += 1


# ── 嘴 ─────────────────────────────────────────────────────
@plate("mouth")
def mouth(b, p, t, **kw):
    """「我將開口，同時感到空虛。」側臉的大特寫：嘴慢慢張開，裡面是黑的，什麼也沒有。"""
    b.ink(FULL)
    tone = lambda x, y: 0.55 * smooth(900, 1900, x) ** 1.2
    b.rules(FULL, -40, 1120, tone, spacing=15, seed=301, seg=(400, 1300), gap=(10, 40))
    m = ease(p) * 1.15
    head(b, 520, 560, 10.5, Profile(cap="bare", nose=1.1, brow=1.4, age=0.5, gaunt=0.5, chin=0.4), rot=-0.05,
         mouth=m, eye=1.0, seed=311, px=24, neck=80, rim=6)


# ── 燭 ─────────────────────────────────────────────────────
@plate("candle")
def candle(b, p, t, **kw):
    """一支燭：火在 p≈0.7 時熄滅，光圈一圈圈收回去，剩一縷煙。"""
    b.ink(FULL)
    life = 1 - ease(ramp(p, 0.62, 0.74))
    cx, top = 960, 600
    fl = 1 + 0.08 * np.sin(t * 13) + 0.05 * np.sin(t * 29 + 1)
    if life > 0:
        rings(b, cx, top - 70, 90, 90 + 620 * life, 18, lambda r: life * np.clip(1 - (r - 90) / (620 * life + 1), 0, 1) ** 1.3,
              seed=321, ry=1.1)
    # 桌面：一道橫的亮邊，下面是黑
    b.knife(np.array([[-100, 905], [2100, 905]]), 6, 1.0, taper=(0, 0), smooth=False)
    b.rules([rect(-200, 910, 2200, 1300)], 912, 1120, lambda x, y: 0.25 * life * np.exp(-((x - cx) / 420) ** 2),
            spacing=13, seed=322, seg=(300, 900))
    # 燭身：白，右側背光的幾道黑刀；頂上凹下的蠟與一道流淚
    body = rough(np.array([[cx - 62, top], [cx + 62, top], [cx + 66, 905], [cx - 66, 905]]), 1.5, 323)
    b.halo(body, 3, 324)
    b.carve([body])
    items = []
    for k in range(5):
        x = cx + 18 + k * 10
        P = resample(np.array([[x, top + 24], [x + 2, 905]]), 4)
        items.append((P, np.full(len(P), 1.2 + 1.4 * k)))
    b.strokes(items, 0.0, wmin=0.5, seed=325, taper=0.08)
    b.ink([ellipse(cx, top + 6, 60, 13)])
    b.carve([ellipse(cx, top + 2, 56, 9)])
    b.knife(np.array([[cx - 40, top + 6], [cx - 44, top + 60], [cx - 41, top + 120]]), 10, 1.0, taper=(0.1, 0.8))
    b.knife(np.array([[cx - 34, top + 8], [cx - 30, top + 60], [cx - 33, top + 110]]), 3, 0.0, taper=(0.2, 0.8))
    # 芯
    wick = np.array([[cx, top + 4], [cx + 1, top - 26], [cx + 5, top - 38]])
    b.knife(wick, 5, 0.0, taper=(0, 0.4), smooth=False)
    if life > 0:
        h = 150 * life * fl
        u = np.linspace(0, 1, 30)
        wv = 3 * np.sin(t * 9 + u * 4) * u
        right = np.stack([cx + 2 + wv + 30 * np.sin(np.pi * u ** 0.8) * (1 - u) ** 0.5 * life, top - 22 - h * u], 1)
        left = np.stack([cx + 2 + wv - 30 * np.sin(np.pi * u ** 0.8) * (1 - u) ** 0.5 * life, top - 22 - h * u], 1)[::-1]
        flame = np.concatenate([right, left])
        b.carve([flame])
        b.knife(np.array([[cx, top - 26], [cx + 1, top - 26 - 0.35 * h]]), 9 * life, 0.0, taper=(0.1, 0.9))
    smoke = ramp(p, 0.7, 1.0)
    if smoke > 0:
        ys = np.linspace(top - 40, top - 40 - 520 * ease_out(smoke, 2), 60)
        xs = cx + 6 + 34 * np.sin((top - ys) / 70 + t * 1.3) * (top - ys) / 380
        b.knife(np.stack([xs, ys], 1), 7 * (1 - 0.5 * smoke), 1.0, taper=(0.05, 0.7), seed=326, swell=0.4)


# ── 燈下寫字的人（白雲樓，1927 年 4 月 26 日夜） ─────────────────────
@plate("lamp_window")
def lamp_window(b, p, t, **kw):
    """夜。樓的一角是一大塊黑；一扇窗亮著，窗裡一個人伏在桌上寫字，燈罩下一圈光。"""
    b.ink(FULL)
    # 天：極少的橫刀；月一圈圈的光
    b.rules(FULL, -40, 320, lambda x, y: 0.18 * smooth(0, 320, y) * smooth(300, 1500, x), spacing=17, seed=331,
            seg=(500, 1500))
    rings(b, 260, 170, 52, 230, 16, lambda r: np.clip(1 - (r - 52) / 180, 0, 1) ** 1.6, seed=332)
    b.carve([ellipse(260, 170, 50, 50)])
    # 樓：磚縫是稀疏的橫刀，在亮窗附近才看得見
    wx0, wy0, wx1, wy1 = 980, 360, 1380, 760
    tone = lambda x, y: 0.42 * np.exp(-(((x - 1180) / 520) ** 2 + ((y - 560) / 360) ** 2))
    b.rules([rect(560, 260, 2200, 1300)], 270, 1120, tone, spacing=14, seed=333, seg=(70, 160), gap=(4, 12))
    b.knife(np.array([[560, 262], [2100, 248]]), 7, 1.0, taper=(0, 0), smooth=False)
    b.knife(np.array([[566, 262], [570, 1100]]), 4, 1.0, taper=(0.0, 0.6), smooth=False)
    # 其他的窗：黑，只有窗台一道白
    for (x, y) in ((680, 380), (680, 820), (1560, 380), (1560, 820), (1180, 870)):
        b.ink([rect(x, y, x + 230, y + 300)])
        b.knife(np.array([[x - 14, y + 304], [x + 244, y + 304]]), 5, 1.0, taper=(0.05, 0.05), smooth=False)
    # 亮的窗：一圈光從窗口往外，窗格
    rings(b, (wx0 + wx1) / 2, (wy0 + wy1) / 2, 240, 560, 15,
          lambda r: 0.55 * np.clip(1 - (r - 240) / 320, 0, 1) ** 1.5, seed=334, ry=0.95)
    b.carve([rect(wx0, wy0, wx1, wy1)])
    b.knife(np.array([[wx0 - 20, wy1 + 8], [wx1 + 20, wy1 + 8]]), 9, 1.0, taper=(0.02, 0.02), smooth=False)
    # 窗裡：燈、桌、伏著寫字的人（側臉朝右，低著頭）
    lx, ly = 1270, 470
    b.ink([np.array([[lx - 58, ly + 30], [lx + 58, ly + 30], [lx + 26, ly - 16], [lx - 26, ly - 16]])])
    b.knife(np.array([[lx, ly + 30], [lx, wy1]]), 6, 0.0, taper=(0, 0), smooth=False)
    b.ink([rect(wx0, 650, wx1, wy1)])                                      # 桌
    b.knife(np.array([[wx0, 652], [wx1, 650]]), 3, 1.0, taper=(0.02, 0.02), smooth=False)
    pen = 0.5 + 0.5 * np.sin(t * 7.3)
    with b.clip([rect(wx0, wy0, wx1, wy1)]):
        shoulders(b, 1095, 600, 0.55, hunch=0.8, seed=335, px=6, light=0.4, rot=0.15)
        neck(b, (1100, 606), (1128, 578), 22, 18, 0.2, 336, light=0.3)
        head(b, 1128, 548, 1.05, Profile(cap="bare", nose=1.0, brow=1.4, age=0.6, moustache=True), rot=0.55,
             mouth=0.0, eye=0.0, seed=337, px=4.0, neck=55, light=0.5)
        arm(b, (1120, 620), (1190 + 8 * pen, 640), 0.55, seed=338, bend=0.1, w0=70, w1=46)
        b.knife(np.array([[1188 + 8 * pen, 642], [1204 + 8 * pen, 612]]), 3, 0.0, taper=(0.1, 0.1), smooth=False)
        b.carve([rect(1150, 643, 1260, 649)])                              # 紙
    # 窗格
    b.knife(np.array([[(wx0 + wx1) / 2, wy0], [(wx0 + wx1) / 2, wy1]]), 12, 0.0, taper=(0, 0), smooth=False)
    b.knife(np.array([[wx0, wy0 + 120], [wx1, wy0 + 120]]), 10, 0.0, taper=(0, 0), smooth=False)


# ── 牢窗 ───────────────────────────────────────────────────
@plate("bars")
def bars(b, p, t, **kw):
    """牢房的高窗：鐵條、斜進來的光，地上一格格的影。什麼都不動。"""
    b.ink(FULL)
    wx0, wy0, wx1, wy1 = 1200, 150, 1520, 430
    # 牆：石塊的縫，斷斷續續、粗細不一
    for j in range(9):
        y = 40 + j * 120
        P = resample(np.array([[-100, y], [2100, y + 4]]), 6)
        b.strokes([(P, 2.5 + 2.5 * (0.5 + 0.5 * np.sin(P[:, 0] / 90 + j)))], 1.0, wmin=1.2, seed=j, breaks=0.35)
        for i in range(9):
            x = -60 + i * 260 + (j % 2) * 130 + 30 * _h(341, i, j)
            b.knife(np.array([[x, y + 6], [x + 3, y + 114]]), 3.5, 1.0, taper=(0.3, 0.3), smooth=False, seed=i + 9 * j)
    # 光束：從窗斜到地上，一道道白刀
    beam = np.array([[wx0, wy0], [wx1, wy0], [1160, 1100], [420, 1100]], np.float32)
    with b.clip([beam]):
        b.ink([beam])
        tone = lambda x, y: 0.75 - 0.5 * smooth(150, 1100, y)
        b.rules([beam], 140, 1110, tone, spacing=12, seed=342, seg=(800, 1600), angle=-0.0)
    b.carve([rect(wx0, wy0, wx1, wy1)])
    # 地上的影：窗與鐵條的形狀
    floor = np.array([[540, 860], [1100, 860], [1170, 1080], [430, 1080]], np.float32)
    b.carve([floor])
    for k in range(5):
        u = (k + 0.5) / 5
        xt, xb = 540 + u * 560, 430 + u * 740
        b.ink([np.array([[xt - 10, 860], [xt + 10, 860], [xb + 14, 1080], [xb - 14, 1080]])])
    for k in range(5):
        x = wx0 + 30 + k * 65
        b.ink([rect(x - 10, wy0, x + 10, wy1)])
        b.knife(np.array([[x - 6, wy0], [x - 6, wy1]]), 2.5, 1.0, taper=(0, 0), smooth=False)
    b.ink([rect(wx0, 280, wx1, 296)])


# ── 日曆 ───────────────────────────────────────────────────
CAL = [("一九一一", "十月十日"), ("一九一三", "三月二十日"), ("一九一五", "五月九日"), ("一九一七", "七月一日"),
       ("一九一九", "五月四日"), ("一九二六", "三月十八日"), ("一九二七", "四月十二日")]


@plate("calendar")
def calendar(b, p, t, year=0, **kw):
    """牆上的撕頁日曆：每一頁是一個日子——武昌、宋教仁遇刺、二十一條、復辟、五四、三一八、四一二。
    p 讓這一頁被撕下、翻走，露出下一頁。"""
    serif = fonts.face("fonts/serif_black")
    small = fonts.face("fonts/serif_regular")
    b.ink(FULL)
    b.rules(FULL, -40, 1120, lambda x, y: 0.2 * np.exp(-(((x - 960) / 700) ** 2 + ((y - 540) / 500) ** 2)), spacing=16,
            seed=351, seg=(300, 1100))
    x0, y0, x1, y1 = 700, 150, 1220, 980
    b.knife(np.array([[960, 60], [960, 150]]), 5, 1.0, taper=(0.3, 0), smooth=False)
    b.carve([ellipse(960, 60, 9, 9)])

    def page(k, M=None):
        yr, md = CAL[min(k, len(CAL) - 1)]
        T = (lambda P: P) if M is None else M
        sheet = T(rect(x0, y0, x1, y1))
        b.halo(sheet, 3, 352 + k)
        b.carve([sheet])
        b.ink([T(rect(x0, y0, x1, y0 + 34))])
        for j, ch in enumerate(yr):
            g = serif.glyph(ch)
            b.ink([T(q * 150 + (960 - 75 * g.advance, y0 + 70 + j * 150 + 132)) for q in g.contours])
        for j, ch in enumerate(md):
            g = small.glyph(ch)
            b.ink([T(q * 44 + (1150 - 22 * g.advance, y0 + 90 + j * 48 + 40)) for q in g.contours])
        # 紙邊的陰影刀：右側
        P = T(resample(np.array([[x1 - 8, y0 + 40], [x1 - 8, y1]]), 4))
        b.strokes([(P, np.full(len(P), 4.0))], 0.0, wmin=0.5, seed=360 + k, taper=0.02)

    page(year + 1)
    k = ease_in(ramp(p, 0.1, 1.0), 1.7)
    if k < 0.98:
        ang = -1.3 * k
        def M(P):
            Q = np.asarray(P, np.float32) - (x1, y0)
            c, s = np.cos(ang), np.sin(ang)
            Q = np.stack([Q[:, 0] * c - Q[:, 1] * s, Q[:, 0] * s + Q[:, 1] * c], 1)
            return Q + (x1 + 520 * k, y0 - 260 * k * k)
        page(year, M)
    b.ink([rect(x0 - 10, y0 - 14, x1 + 10, y0 + 20)])
    b.knife(np.array([[x0 - 10, y0 + 18], [x1 + 10, y0 + 18]]), 3, 1.0, taper=(0, 0), smooth=False)


# ── 石縫的草 ───────────────────────────────────────────────
def blade(b, x, y, L, lean, w, seed=0, v=0.0, curl=0.3):
    """一片草葉：從根往上漸細，梢彎下來。"""
    u = np.linspace(0, 1, 14)
    ang = lean + curl * u ** 2
    dx = np.cumsum(np.sin(ang)) * L / 14
    dy = -np.cumsum(np.cos(ang)) * L / 14
    P = np.stack([x + dx, y + dy], 1)
    b.knife(P, w * (1 - 0.85 * u), v, taper=(0.0, 0.25), smooth=False, seed=seed, swell=0.15)


@plate("grass_crack")
def grass_crack(b, p, t, **kw):
    """俯看石板地：石板是白的，縫是黑的；野草從縫裡一株株鑽出來（縮時）。"""
    b.ink(FULL)
    rows, cw = 6, 330
    cracks = []
    for j in range(rows):
        y0 = -40 + j * 200
        off = (j % 2) * 160
        for i in range(-1, 7):
            x0 = i * cw + off
            k = j * 10 + i
            slab = rough(np.array([[x0 + 9, y0 + 9], [x0 + cw - 9, y0 + 7], [x0 + cw - 8, y0 + 192], [x0 + 8, y0 + 193]]),
                         3.5, 400 + k)
            b.carve([slab])
            # 石面的紋：幾道短黑刀，右下側較密
            for q in range(4):
                xx = x0 + 40 + _h(401, k, q) * (cw - 80)
                yy = y0 + 30 + _h(402, k, q) * 140
                b.knife(np.array([[xx, yy], [xx + 50 + 40 * _h(403, k, q), yy + 6]]), 2.5, 0.0, taper=(0.4, 0.4),
                        smooth=False, seed=k * 7 + q)
            b.knife(np.array([[x0 + cw - 16, y0 + 20], [x0 + cw - 15, y0 + 185]]), 5, 0.0, taper=(0.1, 0.1), smooth=False)
            b.knife(np.array([[x0 + 20, y0 + 186], [x0 + cw - 16, y0 + 185]]), 5, 0.0, taper=(0.1, 0.1), smooth=False)
            cracks.append((x0, y0 + 200))
    # 裂縫
    rng_pts = []
    for c in range(5):
        x, y = 260 + 340 * c + 60 * _h(404, c), 140 + 180 * _h(405, c)
        pts = [(x, y)]
        ang = _h(406, c) * np.pi
        for s in range(16):
            ang += (_h(407, c, s) - 0.5) * 1.1
            x, y = x + 30 * np.cos(ang), y + 30 * np.sin(ang)
            pts.append((x, y))
        P = np.array(pts, np.float32)
        b.knife(P, 9, 0.0, taper=(0.3, 0.5), smooth=True, seed=410 + c)
        rng_pts += [P[k] for k in range(2, 16, 3)]
    roots = rng_pts + [np.array([x0 + 60 + 100 * _h(411, i), y]) for i, (x0, y) in enumerate(cracks[::3])]
    for k, (x, y) in enumerate(roots):
        t0 = _h(412, k) * 0.55
        g = ease_out(np.clip((p - t0) / 0.45, 0, 1), 2)
        if g <= 0:
            continue
        for j in range(2 + int(_h(413, k) * 4)):
            L = (90 + 190 * _h(414, k, j)) * g
            lean = (_h(415, k, j) - 0.5) * 1.3 + 0.08 * np.sin(t * 1.6 + k + j)
            blade(b, x, y, L, lean, 14 * (0.5 + 0.5 * g), seed=k * 13 + j, curl=0.5 * (_h(416, k, j) - 0.5))


# ── 根與骨 ─────────────────────────────────────────────────
@plate("roots_bones")
def roots_bones(b, p, t, **kw):
    """地底：顱骨與肋骨是白的；根從上面鑽下來，穿過骨頭。根裡有一粒粒往上走的汁液。"""
    b.ink(FULL)
    b.rules(FULL, -40, 1120, lambda x, y: 0.16 + 0.1 * np.sin(y / 90 + x / 600), spacing=22, seed=421, seg=(200, 700),
            gap=(20, 80), wobble=6)
    # 顱骨：側面，朝左
    sk = rough(spline([(470, 640), (520, 560), (640, 530), (760, 570), (800, 660), (780, 760), (720, 800), (690, 850),
                       (600, 860), (560, 820), (500, 800), (470, 730)], closed=True, step=4), 3, 422)
    b.halo(sk, 4, 423)
    b.carve([sk])
    b.ink([rough(ellipse(560, 680, 48, 40, 0.2), 3, 424)])                  # 眼窩
    b.ink([rough(np.array([[505, 745], [530, 735], [528, 785]]), 2, 425)])    # 鼻
    for k in range(6):
        x = 560 + k * 22
        b.knife(np.array([[x, 822], [x + 2, 858]]), 5, 0.0, taper=(0.1, 0.4), smooth=False)
    b.knife(np.array([[640, 600], [720, 620], [760, 700]]), 4, 0.0, taper=(0.4, 0.6))
    for k in range(4):                                                      # 顱頂背光的刀
        c = offset(spline([(780, 650), (740, 570), (650, 545)], step=3), 10 + k * 12)
        b.knife(c, 6 - k, 0.0, taper=(0.4, 0.4), seed=426 + k)
    # 肋骨：一道道彎的白
    for k in range(7):
        xs = np.linspace(1050, 1560, 30)
        y = 560 + k * 62
        ys = y + (80 + 6 * k) * np.sin(np.pi * (xs - 1050) / 510)
        P = np.stack([xs, ys], 1)
        b.halo(np.concatenate([offset(P, 12), offset(P, -12)[::-1]]), 3, 430 + k)
        b.knife(P, 24, 1.0, taper=(0.15, 0.15), smooth=False, seed=440 + k)
        b.knife(offset(P, 7), 5, 0.0, taper=(0.3, 0.3), smooth=False, seed=450 + k)
    spine = np.array([[1300, 500], [1306, 1000]])
    b.knife(spine, 40, 1.0, taper=(0.02, 0.02), smooth=False)
    for k in range(9):
        b.knife(np.array([[1282, 520 + k * 54], [1326, 520 + k * 54]]), 5, 0.0, taper=(0.1, 0.1), smooth=False)
    # 根：從上往下鑽；在黑土裡是白的，在白骨上是黑的
    for k in range(10):
        x = 300 + 150 * k + 60 * _h(460, k)
        pts = [(x, -20.0)]
        ang = np.pi / 2
        for i in range(36):
            ang = np.clip(ang + (_h(461, k, i) - 0.5) * 0.45, 0.7, 2.4)
            px, py = pts[-1]
            pts.append((px + 30 * np.cos(ang), py + 30 * np.sin(ang)))
        P = np.array(pts, np.float32)
        n = max(2, int(len(P) * ease_out(np.clip(p * 1.25 - k * 0.025, 0, 1), 2)))
        Q = P[:n]
        w = np.linspace(9, 3, n)
        b.strokes([(Q, w)], 1.0, wmin=0.5, seed=470 + k, taper=0.04)
        with b.clip([sk]):
            b.strokes([(Q, w)], 0.0, wmin=0.5, seed=480 + k, taper=0.04)
        # 汁液：沿根往上
        if p > 0.35 and n > 4:
            for m in range(3):
                u = (t * 0.5 + m / 3 + k * 0.17) % 1.0
                i = int((1 - u) * (n - 1))
                b.ink([ellipse(Q[i, 0], Q[i, 1], 5, 5)])


# ── 節烈牌坊 ───────────────────────────────────────────────
@plate("gate")
def gate(b, p, t, **kw):
    """〈我之節烈觀〉（1918）裡的牌坊：仰角。p 先讓石柱裂開，然後整座塌下來，塵土揚起。"""
    serif = fonts.face("fonts/serif_black")
    # 天：亮的，橫刀在上方
    b.ink(FULL)
    b.rules(FULL, -40, 1120, lambda x, y: 0.06 + 0.42 * smooth(-40, 950, y), spacing=15, seed=491, seg=(500, 1500))
    gy = 960
    b.ink([rect(-200, gy, 2200, 1400)])
    b.rules([rect(-200, gy + 6, 2200, 1400)], gy + 8, 1100, lambda x, y: 0.2, spacing=14, seed=492, seg=(200, 600))

    def pillar(x0, y0, x1, y1):
        polys = [rect(x0, y0, x1, y1)]
        return polys

    pieces = []      # (多邊形們, 種類, 何時開始掉, 方向)
    for k, px in enumerate((520, 800, 1120, 1400)):
        pieces.append((pillar(px - 46, 560, px + 46, gy), "stone", 0.62 + 0.15 * _h(493, k), 1 if k % 2 else -1))
        pieces.append((pillar(px - 40, 330, px + 40, 560), "stone", 0.42 + 0.12 * _h(494, k), -1 if k % 2 else 1))
    pieces.append(([rect(440, 520, 1480, 580)], "beam", 0.45, -1))
    pieces.append(([rect(500, 300, 1420, 352)], "beam", 0.36, 1))

    def roof(x0, x1, y, h):
        xs = np.linspace(x0, x1, 40)
        u = (xs - x0) / (x1 - x0)
        curl = 34 * ((2 * u - 1) ** 8)
        top = np.stack([xs, y - h + 6 * np.sin(np.pi * u) - curl * 0.3], 1)
        bot = np.stack([xs, y - curl], 1)[::-1]
        return np.concatenate([top, bot]).astype(np.float32)
    pieces.append(([roof(420, 1500, 300, 64)], "roof", 0.3, -1))
    pieces.append(([roof(560, 1360, 520, 48)], "roof", 0.4, 1))
    pieces.append(([roof(690, 1230, 170, 74)], "roof", 0.22, 1))
    plaque = rect(800, 360, 1120, 505)
    pieces.append(([plaque], "plaque", 0.34, -1))
    for k, (polys, kind, t0, d) in enumerate(pieces):
        tau = max(0.0, p - t0) * 2.6
        dy = 0.5 * 2200 * tau * tau
        ang = d * 0.9 * tau * tau
        allp = np.concatenate(polys)
        cx, cy = allp[:, 0].mean(), allp[:, 1].mean()

        def M(P):
            Q = np.asarray(P, np.float32) - (cx, cy)
            c, s = np.cos(ang), np.sin(ang)
            Q = np.stack([Q[:, 0] * c - Q[:, 1] * s, Q[:, 0] * s + Q[:, 1] * c], 1) + (cx + d * 120 * tau, cy + dy)
            Q[:, 1] = np.minimum(Q[:, 1], gy + 50)
            return Q
        moved = [M(q) for q in polys]
        if kind == "roof":
            b.halo(moved[0], 3, 500 + k)
            b.ink(moved)
            x0, x1 = allp[:, 0].min(), allp[:, 0].max()
            for j in range(int((x1 - x0) / 34)):                       # 瓦楞：一道道白
                x = x0 + 20 + j * 34
                yb = allp[:, 1].max()
                b.knife(M(np.array([[x, yb - 8], [x + 2, yb - 0.75 * (yb - allp[:, 1].min())]])), 5, 1.0,
                        taper=(0.1, 0.5), smooth=False)
        else:
            b.halo(moved[0], 3, 500 + k)
            b.carve(moved)
            x0, x1, y0, y1 = allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max()
            if kind == "stone":                                         # 柱的右側背光
                for j in range(4):
                    x = x1 - 6 - j * 9
                    b.knife(M(np.array([[x, y0 + 6], [x, y1 - 6]])), 6 - 1.3 * j, 0.0, taper=(0.05, 0.05), smooth=False)
            elif kind == "beam":
                b.knife(M(np.array([[x0 + 8, y1 - 8], [x1 - 8, y1 - 8]])), 8, 0.0, taper=(0.02, 0.02), smooth=False)
            else:
                b.ink([M(rect(x0 + 14, y0 + 14, x1 - 14, y1 - 14))])
                b.carve([M(rect(x0 + 24, y0 + 24, x1 - 24, y1 - 24))])
                for j, ch in enumerate("節烈"):
                    g = serif.glyph(ch)
                    b.ink([M(q * 100 + (850 + j * 120, 478)) for q in g.contours])
    # 裂紋
    if p < 0.62:
        for k, px in enumerate((520, 800, 1120, 1400)):
            n = int(10 * ramp(p, 0.05, 0.4))
            if n > 1:
                ys = np.linspace(gy - 10, gy - 40 * n, n)
                xs = px + np.cumsum([(_h(510, k, i) - 0.5) * 14 for i in range(n)])
                b.knife(np.stack([xs, ys], 1), 5, 0.0, taper=(0.2, 0.6), seed=520 + k)
    # 塵：一團團短刀
    dust = ramp(p, 0.45, 1.0)
    if dust > 0:
        for k in range(70):
            x = 300 + 1350 * _h(530, k) + (_h(531, k) - 0.5) * 300 * dust
            h = _h(532, k) ** 1.5 * 640 * ease_out(dust, 2)
            r = (30 + 80 * _h(533, k)) * (0.6 + dust)
            cl = rough(ellipse(x, gy - h, r, r * 0.7), r * 0.15, 540 + k)
            b.ink([cl])
            for j in range(3):
                yy = gy - h - r * 0.4 + j * r * 0.35
                b.knife(np.array([[x - r * 0.6, yy], [x + r * 0.5, yy + 3]]), 3 + 2 * dust, 1.0, taper=(0.4, 0.4),
                        smooth=False, seed=k * 3 + j)


# ── 靴 ─────────────────────────────────────────────────────
BOOT = [(-58, 0), (-60, -30), (-48, -92), (-14, -98), (12, -92), (20, -60), (44, -40), (78, -30), (96, -16), (98, 0)]


def leg(b, hip, foot, s, lift, seed=0):
    """一條腿：褲管、打了綁腿的小腿、短靴。foot 是靴底中心；lift>0 時腳跟抬起、靴底翻出一排鞋釘。"""
    fx, fy = foot
    ang = -0.55 * lift                                   # 抬腳時腳尖朝下
    T = lambda P: xform(np.asarray(P, np.float32), fx, fy, s, ang)
    ankle = T([(-10, -95)])[0]
    knee = np.array([(hip[0] + ankle[0]) / 2 + 60 * s * (0.3 + lift), (hip[1] + ankle[1]) / 2])
    # 褲管：粗的黑，膝前一道光
    C = spline([hip, knee, ankle + (0, -60 * s)], step=4)
    wv = np.linspace(150, 82, len(C)) * s
    N = np.gradient(C, axis=0)[:, ::-1] * np.array([-1, 1])
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    poly = np.concatenate([C + N * wv[:, None] / 2, (C - N * wv[:, None] / 2)[::-1]])
    b.halo(poly, 2.5, seed)
    b.ink([poly])
    hi = C - N * (wv[:, None] / 2 - 10 * s)
    b.knife(hi[len(hi) // 6:], 9 * s, 1.0, taper=(0.4, 0.6), seed=seed + 1)
    for j in range(3):                                   # 膝後的褶
        i = len(C) // 2 + j * 3
        b.knife(np.array([C[i] + N[i] * wv[i] * 0.45, C[i] + N[i] * wv[i] * 0.05 + (0, 20 * s)]), 6 * s, 1.0,
                taper=(0.5, 0.5), smooth=False, seed=seed + 2 + j)
    # 綁腿：小腿上一圈圈斜的布
    top = ankle + (0, -64 * s)
    shin = np.array([top + (-38 * s, 0), top + (38 * s, 0), ankle + (34 * s, 0), ankle + (-34 * s, 0)])
    b.halo(shin, 2.5, seed + 9)
    b.ink([shin])
    for j in range(5):
        y = top[1] + (j + 0.5) * (ankle[1] - top[1]) / 5
        b.knife(np.array([[top[0] - 34 * s, y + 8 * s], [top[0] + 34 * s, y - 8 * s]]), 4 * s, 1.0, taper=(0.2, 0.2),
                smooth=False, seed=seed + 10 + j)
    # 靴
    boot = T(spline(BOOT, closed=True, step=2))
    b.halo(boot, 2.5, seed + 20)
    b.ink([boot])
    b.knife(T(np.array([[50, -34], [80, -24], [92, -12]])), 7 * s, 1.0, taper=(0.4, 0.4), seed=seed + 21)   # 鞋頭的光
    b.knife(T(np.array([[-56, -6], [96, -6]])), 4 * s, 1.0, taper=(0.05, 0.1), smooth=False, seed=seed + 22)  # 鞋底邊
    b.knife(T(np.array([[-44, -80], [-40, -30]])), 4 * s, 1.0, taper=(0.3, 0.3), smooth=False, seed=seed + 23)
    if lift > 0.15:
        sole_pts = T(np.array([[x, 0] for x in np.linspace(-50, 90, 9)]))
        for q in sole_pts:
            b.carve([ellipse(q[0], q[1] + 6 * s, 5 * s, 5 * s)])


@plate("boots")
def boots(b, p, t, step=None, crush=True, **kw):
    """貼地的低角度：一排打綁腿的兵在行進，靴子落下，一株草被踩倒。"""
    step = p if step is None else step
    gy = 900
    b.ink(FULL)
    b.rules(FULL, -40, gy, lambda x, y: 0.12 + 0.55 * smooth(200, gy, y), spacing=15, seed=561, seg=(600, 1600))
    b.carve([rect(-200, gy, 2200, 1400)])
    b.rules([rect(-200, gy, 2200, 1400)], gy + 4, 1100, lambda x, y: 0.75 - 0.4 * smooth(gy, 1100, y), spacing=11,
            seed=562, v=0.0, seg=(80, 300), gap=(10, 60))
    s = 1.45
    for i in range(4):
        x0 = 150 + i * 470
        for lg in (1, 0):
            ph = (step * 2 + lg * 0.5 + i * 0.13) % 1.0
            lift = max(0.0, np.sin(np.pi * ph * 2)) if ph < 0.5 else 0.0
            fx = x0 + lg * 160 + (ph - 0.25) * 90 * (1 if ph < 0.5 else -1)
            fy = gy - 130 * lift
            leg(b, (x0 + 80, -60), (fx, fy), s, lift, seed=600 + 10 * i + lg)
    # 一株草：在一隻靴子落下時被踩扁
    bx = 690
    crushed = ease(ramp(step % 1.0, 0.45, 0.55)) if crush else 0.0
    for k in range(6):
        L = 230 * (1 - 0.85 * crushed)
        lean = (k - 2.5) * 0.22 + crushed * 1.35
        b.halo(np.array([[bx + k * 9, gy], [bx + k * 9 + np.sin(lean) * L, gy - np.cos(lean) * L]]), 2, 650 + k,
               closed=False)
        blade(b, bx + k * 9, gy, L, lean, 13, seed=660 + k, curl=0.2 * (k - 2.5) / 2.5)


# ── 方陣 ───────────────────────────────────────────────────
@plate("formation")
def formation(b, p, t, beat=None, text="打倒列強　除軍閥", **kw):
    """被收編的那一刻：一模一樣的人排成一模一樣的列，舉著一模一樣的橫幅，嘴在同一拍上開合。"""
    kai = fonts.face("fonts/kai_bold")
    ph = (t * 1.1) % 1.0 if beat is None else beat
    b.ink(FULL)
    b.rules(FULL, -40, 1120, lambda x, y: 0.3 * smooth(100, 700, y) * (1 - smooth(700, 1000, y)), spacing=14, seed=571,
            seg=(1600, 2400), gap=(2, 4), wobble=0.5)
    mouth = 0.15 + 0.6 * max(0.0, np.sin(2 * np.pi * ph)) ** 2
    prof = Profile(cap="student", nose=1.0, brow=1.2, jaw=1.0)
    rows = [(430, 0.9, 13), (600, 1.25, 9), (820, 1.7, 7), (1080, 2.3, 5)]
    for r, (yb, s, n) in enumerate(rows):
        # 橫幅：每一列一模一樣
        x0, x1 = -40, 1960
        top = yb - 300 * s
        cl = rect(x0, top, x1, top + 46 * s)
        b.carve([cl])
        b.knife(np.array([[x0, top + 46 * s], [x1, top + 46 * s]]), 2 * s, 0.0, taper=(0, 0), smooth=False)
        reps = max(1, int((x1 - x0) / (len(text) * 46 * s * 1.25)))
        for k in range(reps + 1):
            b.text(kai, text, x0 + 30 * s + k * len(text) * 46 * s * 1.2, top + 40 * s, 38 * s, v=0.0)
        for i in range(n):
            x = 60 + i * (1800 / max(n - 1, 1))
            b.knife(np.array([[x + 70 * s, yb + 40 * s], [x + 70 * s, top + 46 * s]]), 5 * s, 0.0, taper=(0, 0),
                    smooth=False)
            shoulders(b, x - 8 * s, yb + 8 * s, s * 0.82, hunch=0.0, seed=700, px=10, rim=max(1.4, 0.9 * s))
            top_n = np.array([x + 4 * s, yb - 30 * s])
            neck(b, (x, yb + 10 * s), top_n, 40 * s, 32 * s, s / 3.0, 701)
            ear = top_n + xform(np.array([[-6.0, -50.0]]), 0, 0, s, -0.08)[0]
            head(b, ear[0], ear[1], s, prof, rot=-0.08, mouth=mouth, seed=710, px=max(6.0, 2.6 * s), neck=62,
                 rim=max(1.4, 0.9 * s))


# ── 散場的街 ───────────────────────────────────────────────
@plate("square")
def square(b, p, t, **kw):
    """散場以後的街：一條白布橫幅倒在地上，「外爭主權」只剩半截；傳單、一頂學生帽、一隻鞋。沒有人。"""
    kai = fonts.face("fonts/kai_bold")
    serif = fonts.face("fonts/serif_regular")
    hz = 330
    b.ink(FULL)
    b.rules([rect(-200, -200, 2200, hz)], -40, hz, lambda x, y: 0.2 + 0.5 * smooth(0, hz, y), spacing=16, seed=581)
    # 兩側的牆
    b.ink([np.array([[-200, -200], [520, -200], [760, hz], [-200, hz + 260]], np.float32)])
    b.ink([np.array([[2200, -200], [1420, -200], [1180, hz], [2200, hz + 260]], np.float32)])
    for k in range(6):
        y = hz - 40 - k * 60
        b.knife(np.array([[-100, y + 170 + k * 25], [760 - 14 * k, hz - 30 - k * 55]]), 3, 1.0, taper=(0.1, 0.3),
                smooth=False, seed=k)
        b.knife(np.array([[2100, y + 170 + k * 25], [1180 + 14 * k, hz - 30 - k * 55]]), 3, 1.0, taper=(0.1, 0.3),
                smooth=False, seed=k + 9)
    # 地面：透視的石板縫
    floor = np.array([[-200, hz + 260], [760, hz], [1180, hz], [2200, hz + 260], [2200, 1400], [-200, 1400]], np.float32)
    b.rules([floor], hz, 1120, lambda x, y: 0.5 - 0.25 * smooth(hz, 1100, y), spacing=12, seed=597, seg=(100, 400),
            gap=(6, 30))
    for k in range(-9, 10):
        b.knife(np.array([[960 + k * 25, hz + 2], [960 + k * 330, 1120]]), 4, 0.0, taper=(0.6, 0.0), smooth=False,
                seed=k + 20)
    # 倒在地上的橫幅：布皺起來，字只剩半截
    cl = Cloth((300, 820), (1500, 760), 170, t * 0.0 + 3.0, wind=0.35)
    U = np.linspace(0, 1, 100)
    vv = np.linspace(0, 1, 20)
    outline = np.concatenate([cl(U, U * 0), cl(vv * 0 + 1, vv), cl(U, U * 0 + 1)[::-1], cl(vv * 0, vv[::-1])])
    outline = rough(outline, 4, 590)
    b.halo(outline, 4, 591)
    b.carve([outline])
    for j, ch in enumerate("外爭主"):
        g = kai.glyph(ch)
        for q in g.contours:
            qq = resample(np.concatenate([q, q[:1]]), 0.01)
            uv = np.stack([0.08 + j * 0.13 + qq[:, 0] * 0.11, 0.5 + (qq[:, 1] + 0.38) * 0.62], 1)
            b.ink([cl(uv[:, 0], uv[:, 1])])
    with b.clip([outline]):
        items = []
        for u in np.linspace(0.005, 0.995, 120):
            sv = np.linspace(-0.02, 1.02, 30)
            P = cl(np.full_like(sv, u), sv)
            dark = np.clip(-cl.shade(np.full_like(sv, u), sv) * 1.4, 0, 1.0) ** 1.2
            items.append((P, dark * 9.0))
        b.strokes(items, 0.0, wmin=0.9, seed=592, taper=0.1)
        # 布被踩過：一個泥鞋印
        b.ink([rough(sole(1180, 830, 260, 1.2), 4, 593)])
    # 竿子斷在一旁
    b.knife(np.array([[260, 840], [40, 1010]]), 16, 0.0, taper=(0, 0.05), smooth=False)
    b.knife(np.array([[1530, 770], [1700, 690]]), 14, 0.0, taper=(0, 0.3), smooth=False)
    # 傳單
    for k in range(9):
        x, y = 200 + 1500 * _h(594, k), 520 + 520 * _h(595, k)
        if 280 < x < 1520 and 700 < y < 960:
            continue
        sc = 0.5 + 0.6 * (y - hz) / 700
        sh = xform(rect(-60, -80, 60, 80), x, y, sc, _h(596, k) * 3)
        b.halo(sh, 2, 600 + k)
        b.carve([sh])
        for c in range(5):
            P = xform(np.array([[40 - c * 18, -62], [40 - c * 18, 58]]), x, y, sc, _h(596, k) * 3)
            b.knife(P, 4 * sc, 0.0, taper=(0.1, 0.1), smooth=False, seed=610 + k * 5 + c)
    # 一頂學生帽
    cx, cy = 1600, 940
    b.halo(ellipse(cx, cy, 110, 40), 3, 620)
    b.ink([ellipse(cx, cy, 110, 40)])
    b.ink([np.array([[cx - 90, cy - 10], [cx - 80, cy - 70], [cx + 85, cy - 75], [cx + 95, cy - 12]])])
    b.knife(np.array([[cx - 86, cy - 30], [cx + 90, cy - 34]]), 6, 1.0, taper=(0.1, 0.1), smooth=False)
    b.knife(np.array([[cx - 60, cy + 18], [cx + 80, cy + 12]]), 5, 1.0, taper=(0.3, 0.3), smooth=False)


# ── 公車上書 ───────────────────────────────────────────────
@plate("petition")
def petition(b, p, t, **kw):
    """一八九五年：各省舉人聯名的呈文。直排的訴求，後面是密密的、一個個簽上去的名字；
    一枝筆正在簽下一個。"""
    kai = fonts.face("fonts/kai_bold")
    b.ink(FULL)
    b.rules(FULL, -40, 1120, lambda x, y: 0.14, spacing=18, seed=631, seg=(500, 1400))
    paper = rough(np.array([[-60, 70], [1980, 50], [1990, 1020], [-60, 1035]], np.float32), 4, 632)
    b.carve([paper])
    cols = [("上清帝第二書", 1.0), ("下詔鼓天下之氣", 0.0), ("遷都定天下之本", 0.0), ("練兵強天下之勢", 0.0),
            ("變法成天下之治", 0.0)]
    x = 1820
    for text, title in cols:
        b.text(kai, text, x, 110, 74 if title else 62, v=0.0, vertical=True, track=1.1)
        x -= 115 if title else 92
    b.knife(np.array([[x + 40, 100], [x + 40, 1000]]), 3, 0.0, taper=(0.05, 0.05), smooth=False)
    # 名字：一欄一欄，筆跡各不相同（認不出來的字），隨 p 一個個簽上去
    names = int(ease(p) * 70) + 12
    k = 0
    xc = x - 40
    while xc > 40 and k < names:
        y = 120
        while y < 940 and k < names:
            n_ch = 2 + int(_h(633, k) * 2)
            for c in range(n_ch):
                cx, cy = xc, y + c * 52
                for st in range(3 + int(_h(634, k, c) * 4)):
                    a = _h(635, k, c, st) * np.pi
                    L = 14 + 26 * _h(636, k, c, st)
                    ox, oy = (_h(637, k, c, st) - 0.5) * 30, (_h(638, k, c, st) - 0.5) * 30
                    b.knife(np.array([[cx + ox - L / 2 * np.cos(a), cy + oy - L / 2 * np.sin(a)],
                                      [cx + ox + L / 2 * np.cos(a), cy + oy + L / 2 * np.sin(a)]]),
                            3.5 + 2.5 * _h(639, k, c, st), 0.0, taper=(0.2, 0.5), smooth=False, seed=k * 31 + c * 7 + st)
            y += n_ch * 52 + 40
            k += 1
        xc -= 70
    # 筆：從右上伸進來，筆尖落在正在簽的那一個名字上
    tip = np.array([xc + 70 + 10 * np.sin(t * 6), min(y, 900) - 10 + 6 * np.sin(t * 9)])
    shaft = np.array([tip + (8, -30), tip + (360, -620)])
    b.halo(np.array([tip, tip + (360, -620)]), 3, 640, closed=False)
    b.knife(shaft, 26, 0.0, taper=(0, 0), smooth=False)
    b.knife(shaft + (-6, 0), 5, 1.0, taper=(0.1, 0.1), smooth=False)
    brush = np.array([tip, tip + (-4, -10), tip + (2, -36), tip + (16, -38), tip + (16, -20)])
    b.ink([spline(brush, closed=True, step=1)])
    sleeve = np.array([tip + (300, -560), tip + (520, -500), tip + (700, -900), tip + (420, -1000)])
    b.halo(sleeve, 3, 641)
    b.ink([sleeve])
    for j in range(3):
        b.knife(np.array([tip + (330 + 40 * j, -600 - 30 * j), tip + (470 + 40 * j, -900)]), 5, 1.0, taper=(0.4, 0.4),
                smooth=False, seed=642 + j)
