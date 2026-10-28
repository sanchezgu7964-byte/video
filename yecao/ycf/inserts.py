"""意象：穿插在字草世界裡的木刻版畫鏡頭。

每個意象是一個函式 f(pl, p, t, rng, **kw)：pl 是版（Plate），p 是這個意象自己的進度 0..1，
t 是片子的時間（給擺動、呼吸之類的細節用）。鏡頭可以完整播出一段 p，也可以只閃出某一個 p。
"""
import numpy as np

from . import fonts
from .gfx import ease, ease_in, ease_out, ramp
from .noise import noise3, rand
from .woodcut import Plate, Woodcut

W, H = 1920, 1080
REG = {}


def insert(name):
    def deco(f):
        REG[name] = f
        return f
    return deco


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def ellipse(cx, cy, rx, ry, n=48, rot=0.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(th), ry * np.sin(th)
    c, s = np.cos(rot), np.sin(rot)
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], 1).astype(np.float32)


def rotate(poly, ang, cx, cy):
    c, s = np.cos(ang), np.sin(ang)
    q = poly - (cx, cy)
    return np.stack([q[:, 0] * c - q[:, 1] * s, q[:, 0] * s + q[:, 1] * c], 1) + (cx, cy)


def figure(cx, base, s, arm=None, mouth=0.0):
    """面向鏡頭的半身人形（頭、肩），可舉起一隻握拳的手臂。回傳 (身體多邊形們, 嘴)。"""
    head = ellipse(cx, base - 150 * s, 30 * s, 37 * s)
    neck = rect(cx - 12 * s, base - 120 * s, cx + 12 * s, base - 98 * s)
    body = np.array([[cx - 70 * s, base], [cx - 62 * s, base - 80 * s], [cx - 30 * s, base - 102 * s],
                     [cx + 30 * s, base - 102 * s], [cx + 62 * s, base - 80 * s], [cx + 70 * s, base]], np.float32)
    polys = [head, neck, body]
    if arm is not None:
        # 從右肩舉起的手臂與拳頭
        sx, sy = cx + 48 * s, base - 88 * s
        L = 120 * s
        ex, ey = sx + np.sin(arm) * L * 0.55, sy - np.cos(arm) * L * 0.55
        hx, hy = sx + np.sin(arm * 0.6) * L, sy - np.cos(arm * 0.6) * L
        w = 11 * s
        polys.append(np.array([[sx - w, sy], [ex - w, ey], [hx - w * 0.8, hy], [hx + w * 0.8, hy], [ex + w, ey],
                               [sx + w, sy]], np.float32))
        polys.append(ellipse(hx, hy - 6 * s, 15 * s, 17 * s))
    m = ellipse(cx, base - 132 * s, 9 * s, (2 + 9 * mouth) * s) if mouth > 0 else None
    return polys, m


# ── 意象 ───────────────────────────────────────────────────
@insert("iron_house")
def iron_house(pl, p, t, rng, **kw):
    """鐵屋子：沒有窗戶、萬難破毀，裡面有許多熟睡的人。p 過半之後，後牆裂開一道光。"""
    pl.begin(0.1)
    vx, vy = 960, 470
    bx0, by0, bx1, by1 = 600, 270, 1320, 680
    pl.fill([np.array([[0, 0], [W, 0], [bx1, by0], [bx0, by0]])], 0.06, False)            # 天花板
    pl.fill([np.array([[0, 0], [bx0, by0], [bx0, by1], [0, H]])], 0.14, False)             # 左牆
    pl.fill([np.array([[W, 0], [W, H], [bx1, by1], [bx1, by0]])], 0.17, False)             # 右牆
    pl.grad_fill([np.array([[0, H], [bx0, by1], [bx1, by1], [W, H]])], (0, by1), (0, H), 0.22, 0.34)
    pl.fill([rect(bx0, by0, bx1, by1)], 0.2, False)                                          # 後牆
    # 鉚釘與鋼板接縫
    for k in range(1, 6):
        x = bx0 + k * (bx1 - bx0) / 6
        pl.stroke([np.array([[x, by0], [x, by1]])], 0.12, 2)
        for j in range(5):
            pl.circle(x - 8, by0 + 30 + j * 85, 3.5, 0.42, None)
    for k in range(6):
        a = k / 6
        pl.stroke([np.array([[0, a * H * 0.9], [bx0, by0 + a * (by1 - by0)]])], 0.09, 2)
        pl.stroke([np.array([[W, a * H * 0.9], [bx1, by0 + a * (by1 - by0)]])], 0.1, 2)
    crack = ramp(p, 0.55, 1.0)
    if crack > 0:
        w = 4 + 46 * ease_in(crack, 1.5)
        ys = np.linspace(by0, by1, 24)
        xs = vx + np.cumsum(rng.normal(0, 5, 24))
        left = np.stack([xs - w / 2, ys], 1)
        right = np.stack([xs + w / 2, ys], 1)[::-1]
        pl.fill([np.concatenate([left, right])], 1.0, False)
        spill = np.array([[xs[-1] - w, by1], [xs[-1] + w, by1], [vx + 520 * crack + 100, H], [vx - 520 * crack - 100, H]])
        pl.grad_fill([spill], (0, by1), (0, H), 0.95, 0.55, 0.9 * crack, 0.0)
    # 熟睡的人：橫躺在地上，一排一排
    for row, (yb, s) in enumerate([(730, 0.55), (820, 0.75), (950, 1.0)]):
        n = [4, 3, 3][row]
        for i in range(n):
            cx = 960 + (i - (n - 1) / 2) * 470 * s + (row % 2) * 110 * s
            breathe = 1 + 0.04 * np.sin(t * 1.3 + i + row)
            body = ellipse(cx, yb, 170 * s, 30 * s * breathe)
            head = ellipse(cx - 190 * s, yb - 8 * s, 34 * s, 30 * s)
            lit = 0.04 + 0.35 * crack * np.exp(-((cx - vx) / 500) ** 2)
            pl.fill([body, head], lit, True)


@insert("mouth")
def mouth(pl, p, t, rng, **kw):
    """側臉：嘴慢慢張開，裡面是一片白——「我將開口，同時感到空虛」。"""
    pl.begin(0.9, 0.78)
    upper = np.array([[700, 1080], [690, 800], [640, 620], [660, 380], [760, 230], [930, 200], [1060, 260],
                      [1100, 360], [1112, 430], [1150, 500], [1215, 580], [1172, 610], [1160, 632], [1186, 652],
                      [1166, 672]], np.float32)
    lower = np.array([[1166, 672], [1182, 692], [1160, 720], [1172, 762], [1120, 822], [1050, 852], [1000, 880],
                      [990, 1080]], np.float32)
    a = 0.42 * ease(p)
    piv = (1015, 615)
    lw = lower.copy()
    lw[:6] = rotate(lower[:6], a, *piv)
    head = np.concatenate([upper, lw[1:]])
    pl.fill([head], 0.04, True)
    if a > 0.01:
        cavity = np.concatenate([np.array([[1166, 672], [1090, 650]], np.float32),
                                 rotate(np.array([[1095, 676], [1182, 692]], np.float32), a, *piv)])
        pl.fill([cavity], 1.0, False)
    pl.circle(1068, 418, 9, 0.6, None)


@insert("candle")
def candle(pl, p, t, rng, **kw):
    """一支蠟燭，火苗在 p 過七成時熄滅，剩一縷煙。"""
    pl.begin(0.06)
    pl.grad_fill([rect(880, 560, 1040, 1080)], (880, 0), (1040, 0), 0.75, 0.35, obj=True)
    pl.fill([ellipse(960, 560, 80, 16)], 0.85, True)
    life = 1 - ease(ramp(p, 0.68, 0.8))
    if life > 0:
        fl = 1 + 0.12 * noise3(t * 7, 0, 0, 5)
        h = 120 * life * fl
        flame = np.array([[960 + 26 * np.sin(np.pi * u) * (1 - u) ** 0.4 * life * (1 if side else -1), 545 - h * u]
                          for side in (True, False) for u in (np.linspace(0, 1, 16) if side else np.linspace(1, 0, 16))])
        pl.grad_fill([ellipse(960, 470, 360 * life, 300 * life)], (960, 260), (960, 900), 0.5, 0.06, 0.6 * life, 0.0)
        pl.fill([flame], 1.0, False)
    pl.stroke([np.array([[960, 548], [960, 525]])], 0.02, 4)
    smoke = ramp(p, 0.72, 1.0)
    if smoke > 0:
        ys = np.linspace(520, 520 - 420 * smoke, 50)
        xs = 960 + 40 * np.sin((520 - ys) / 55 + t * 1.5) * (520 - ys) / 300
        pl.stroke([np.stack([xs, ys], 1)], 0.6, 3, None, alpha=0.8 * (1 - 0.4 * smoke))


@insert("grass_crack")
def grass_crack(pl, p, t, rng, **kw):
    """石板地的裂縫裡，野草一株一株鑽出來（縮時）。"""
    r = rand(7, 70, 1)
    pl.begin(0.55)
    # 石板
    rows = 7
    for j in range(rows):
        y0 = j * H / rows
        off = (j % 2) * 140
        for i in range(-1, 8):
            x0 = i * 280 + off
            tone = 0.58 + r.uniform(-0.1, 0.1)
            pl.fill([rect(x0 + 5, y0 + 5, x0 + 275, y0 + H / rows - 5)], tone, False)
    # 裂縫
    cracks = []
    for k in range(5):
        x, y = r.uniform(200, 1700), r.uniform(150, 950)
        pts = [(x, y)]
        ang = r.uniform(0, np.pi)
        for _ in range(14):
            ang += r.normal(0, 0.5)
            x, y = x + 28 * np.cos(ang), y + 28 * np.sin(ang)
            pts.append((x, y))
        cracks.append(np.array(pts, np.float32))
        pl.stroke([cracks[-1]], 0.04, 5, False)
    # 從裂縫與石縫長出的草葉
    roots = [c[r.integers(len(c))] for c in cracks for _ in range(6)]
    roots += [(r.uniform(0, W), (r.integers(1, rows)) * H / rows) for _ in range(18)]
    grow_spd = kw.get("speed", 1.0)
    for k, (x, y) in enumerate(roots):
        t0 = r.uniform(0, 0.55)
        g = ease_out(np.clip((p - t0) / 0.45 * grow_spd, 0, 1), 2)
        if g <= 0:
            continue
        for b in range(r.integers(2, 5)):
            L = r.uniform(90, 260) * g
            lean = r.uniform(-0.6, 0.6) + 0.12 * np.sin(t * 1.7 + k + b)
            n = 12
            u = np.linspace(0, 1, n)
            cx = x + np.sin(lean) * L * u + lean * L * 0.25 * u * u
            cy = y - np.cos(lean) * L * u
            w = 9 * (1 - u) * (0.6 + 0.4 * g)
            nx, ny = np.cos(lean), np.sin(lean)
            blade = np.concatenate([np.stack([cx - nx * w, cy - ny * w], 1), np.stack([cx + nx * w, cy + ny * w], 1)[::-1]])
            pl.fill([blade], 0.03, True)


@insert("roots_bones")
def roots_bones(pl, p, t, rng, **kw):
    """地底：根穿過肋骨與顱骨，把血往上吸（紅版）。"""
    r = rand(7, 71, 1)
    pl.begin(0.3, 0.18)
    for k in range(9):
        y = 120 + k * 110 + r.uniform(-20, 20)
        xs = np.linspace(0, W, 40)
        pl.stroke([np.stack([xs, y + 18 * np.sin(xs / 200 + k)], 1)], 0.2 + r.uniform(-0.05, 0.05), 6, False)
    # 顱骨與肋骨（骨頭是亮的）
    pl.fill([ellipse(620, 700, 150, 130)], 0.86, True)
    pl.fill([rect(560, 760, 690, 840)], 0.86, True)
    for ex in (565, 675):
        pl.fill([ellipse(ex, 690, 34, 30)], 0.05, True)
    pl.fill([np.array([[620, 735], [600, 780], [640, 780]])], 0.05, True)
    for k in range(7):
        y = 560 + k * 58
        xs = np.linspace(1000, 1500, 30)
        ys = y + 90 * np.sin(np.pi * (xs - 1000) / 500) * (0.6 + 0.05 * k)
        pl.stroke([np.stack([xs, ys], 1)], 0.84, 16, True)
    pl.stroke([np.array([[1250, 520], [1250, 1000]])], 0.84, 24, True)
    # 根：由上往下鑽，穿過骨頭
    for k in range(9):
        x = r.uniform(300, 1650)
        pts = [(x, 0)]
        ang = np.pi / 2
        for i in range(30):
            ang += r.normal(0, 0.18)
            ang = np.clip(ang, 0.6, 2.5)
            x, y = pts[-1]
            pts.append((x + 34 * np.cos(ang), y + 34 * np.sin(ang)))
        pts = np.array(pts, np.float32)
        n = max(2, int(len(pts) * ease_out(np.clip(p * 1.3 - k * 0.03, 0, 1), 2)))
        pl.stroke([pts[:n]], 0.03, 7 - k % 3, True)
        # 血沿根往上
        blood = ramp(p, 0.4, 1.0)
        if blood > 0 and n > 2:
            m = max(2, int(n * blood))
            pl.redstroke([pts[n - m:n]], 6)


@insert("boots")
def boots(pl, p, t, rng, **kw):
    """貼地的低角度：一排靴子踏步，踩倒一株草。"""
    pl.begin(0.8, 0.62)
    gy = 860
    pl.fill([rect(0, gy, W, H)], 0.4, False)
    for k in range(10):
        pl.stroke([np.array([[0, gy + 20 + k * 22], [W, gy + 20 + k * 22]])], 0.3, 2, False)
    step = kw.get("step", p)
    for i in range(4):
        x0 = 280 + i * 430
        for leg in (0, 1):
            ph = (step * 2 + leg * 0.5 + i * 0.25) % 1.0
            lift = max(0.0, np.sin(np.pi * ph * 2)) if ph < 0.5 else 0.0
            fx = x0 + leg * 130 + (ph - 0.25) * 60
            fy = gy - 140 * lift
            knee = (fx - 20, fy - 330 - 40 * lift)
            hip = (x0 + 65, -40)
            pl.stroke([np.array([hip, knee, (fx, fy - 170)])], 0.18, 110, True)
            bootp = np.array([[fx - 52, fy - 230], [fx + 48, fy - 230], [fx + 52, fy - 40], [fx + 120, fy - 22],
                              [fx + 124, fy], [fx - 56, fy]], np.float32)
            pl.fill([bootp], 0.03, True)
    # 一株草：在一隻靴子落下時被踩扁
    bx = 760
    crushed = ease(ramp(step % 1.0, 0.45, 0.55)) if kw.get("crush", True) else 0
    for k in range(5):
        L = 170 * (1 - 0.85 * crushed)
        lean = (k - 2) * 0.25 + crushed * 1.3
        u = np.linspace(0, 1, 10)
        cx = bx + k * 10 + np.sin(lean) * L * u
        cy = gy - np.cos(lean) * L * u
        w = 7 * (1 - u)
        blade = np.concatenate([np.stack([cx - w, cy], 1), np.stack([cx + w, cy], 1)[::-1]])
        pl.fill([blade], 0.05, True)


@insert("onlookers")
def onlookers(pl, p, t, rng, **kw):
    """看客：一排排伸長脖子的頭，全都朝著同一個方向看。被看的那一邊是空白。"""
    r = rand(7, 72, 1)
    pl.begin(0.84, 0.7)
    pl.fill([rect(1500, 0, W, H)], 0.98, False)
    crane = ease(p)
    for row, (yb, s, n) in enumerate([(560, 0.55, 12), (760, 0.8, 9), (1020, 1.15, 6)]):
        for i in range(n):
            x = 60 + i * 1450 / n + r.uniform(-30, 30) + row * 40
            lean = 0.35 + 0.25 * crane + r.uniform(-0.05, 0.05)
            neck_len = (90 + 50 * crane) * s
            nx, ny = x + np.sin(lean) * neck_len, yb - 140 * s - np.cos(lean) * neck_len
            body = np.array([[x - 90 * s, yb + 200 * s], [x - 70 * s, yb - 90 * s], [x - 20 * s, yb - 140 * s],
                             [x + 40 * s, yb - 130 * s], [x + 85 * s, yb - 60 * s], [x + 95 * s, yb + 200 * s]], np.float32)
            neck = np.array([[x - 10 * s, yb - 130 * s], [nx - 14 * s, ny + 10 * s], [nx + 16 * s, ny + 14 * s],
                             [x + 34 * s, yb - 120 * s]], np.float32)
            head = ellipse(nx + 18 * s, ny - 8 * s, 46 * s, 40 * s, rot=lean * 0.6)
            pl.fill([body, neck, head], 0.05 + 0.04 * row, True)


@insert("crowd")
def crowd(pl, p, t, rng, variant="surge", beat=None, **kw):
    """搖旗吶喊。surge：人潮、舉拳、紅旗翻飛；formation：一模一樣的方陣、同步的旗；empty：散場後的廣場。"""
    r = rand(7, 73, {"surge": 1, "formation": 2, "empty": 3}[variant])
    pl.begin(0.9, 0.7)
    gy = 700
    pl.fill([rect(0, gy, W, H)], 0.6 if variant == "empty" else 0.35, False)
    if variant == "empty":
        for k in range(-8, 9):
            pl.stroke([np.array([[960 + k * 60, gy], [960 + k * 400, H]])], 0.48, 2, False)
        for k in range(12):
            pl.stroke([np.array([[0, gy + 8 + k * k * 3], [W, gy + 8 + k * k * 3]])], 0.5, 2, False)
        # 倒在地上的旗、散落的紙
        flag = np.array([[700, 900], [980, 870], [1060, 930], [900, 990], [760, 960]], np.float32)
        pl.redfill([flag])
        pl.fill([flag], 0.55, True)
        pl.stroke([np.array([[640, 950], [1180, 860]])], 0.05, 9, True)
        for k in range(14):
            x, y = r.uniform(100, 1800), r.uniform(gy + 30, H - 40)
            pl.fill([rotate(rect(x, y, x + 60, y + 40), r.uniform(0, 3), x + 30, y + 20)], 0.95, True)
        return
    beat_ph = beat if beat is not None else (t * 1.1) % 1.0
    flags = []
    rows = [(gy - 40, 0.45, 15), (gy + 80, 0.65, 11), (gy + 230, 0.9, 8), (gy + 420, 1.3, 5)]
    for ri, (yb, s, n) in enumerate(rows):
        for i in range(n):
            if variant == "formation":
                x = 140 + i * (1640 / (n - 1))
                ph, up = 0.0, True
            else:
                x = r.uniform(-40, 1960)
                ph, up = r.uniform(0, 1), r.random() < 0.6
            pump = 0.5 + 0.5 * np.sin(2 * np.pi * (beat_ph + ph * (0 if variant == "formation" else 0.35)))
            arm = (0.25 + 0.35 * pump) if up else None
            polys, m = figure(x, yb, s, arm, mouth=pump if variant == "surge" else 0.15)
            pl.fill(polys, 0.04 + 0.05 * (3 - ri) / 3, True)
            if m is not None:
                pl.fill([m], 0.95, False)
            if (variant == "formation" and i % 2 == 0) or (variant == "surge" and r.random() < 0.28):
                flags.append((x + 40 * s, yb - 120 * s, s, ph))
    for (fx, fy, s, ph) in flags:
        pole_top = fy - 330 * s
        pl.stroke([np.array([[fx, fy], [fx, pole_top]])], 0.05, 6 * s, True)
        fw, fh = 220 * s, 130 * s
        xs = np.linspace(0, 1, 16)
        wave_t = t * (5 if variant == "surge" else 3) + (ph * 6 if variant == "surge" else 0)
        top = np.stack([fx + xs * fw, pole_top + 18 * s * np.sin(xs * 5 - wave_t) * xs], 1)
        bot = np.stack([fx + xs * fw, pole_top + fh + 18 * s * np.sin(xs * 5 - wave_t - 0.6) * xs], 1)[::-1]
        cloth = np.concatenate([top, bot])
        pl.redfill([cloth])
        pl.fill([cloth], 0.6, True)


@insert("gate")
def gate(pl, p, t, rng, **kw):
    """節烈牌坊：仰角。p 先讓石柱裂開，然後整座塌下來，塵土揚起。"""
    r = rand(7, 74, 1)
    pl.begin(0.95, 0.78)
    gy = 960
    pl.fill([rect(0, gy, W, H)], 0.22, False)
    serif = fonts.face("fonts/serif_regular")
    pieces = []      # (多邊形們, 明暗, 何時開始掉, 旋轉方向)
    for px in (520, 800, 1120, 1400):
        pieces.append(([rect(px - 34, 560, px + 34, gy)], 0.28, 0.62 + r.uniform(0, 0.15), r.choice([-1, 1])))
        pieces.append(([rect(px - 30, 330, px + 30, 560)], 0.3, 0.42 + r.uniform(0, 0.12), r.choice([-1, 1])))
    pieces.append(([rect(440, 520, 1480, 580)], 0.18, 0.45, -1))
    pieces.append(([rect(500, 300, 1420, 350)], 0.18, 0.36, 1))

    def roof(x0, x1, y, h):
        xs = np.linspace(x0, x1, 30)
        u = (xs - x0) / (x1 - x0)
        curl = 26 * ((2 * u - 1) ** 8)
        top = np.stack([xs, y - h + 6 * np.sin(np.pi * u) - curl * 0.2], 1)
        bot = np.stack([xs, y - curl], 1)[::-1]
        return np.concatenate([top, bot]).astype(np.float32)
    pieces.append(([roof(420, 1500, 300, 60)], 0.04, 0.3, -1))
    pieces.append(([roof(560, 1360, 520, 46)], 0.04, 0.4, 1))
    pieces.append(([roof(700, 1220, 170, 70)], 0.03, 0.22, 1))
    plaque = rect(820, 360, 1100, 500)
    pieces.append(([plaque], 0.1, 0.34, -1))
    for k, (polys, tone, t0, d) in enumerate(pieces):
        tau = max(0.0, p - t0) * 2.6
        dy = 0.5 * 2200 * tau * tau
        ang = d * 0.9 * tau * tau
        cx = np.mean([q[:, 0].mean() for q in polys])
        cy = np.mean([q[:, 1].mean() for q in polys])
        moved = [rotate(q, ang, cx, cy) + (d * 120 * tau, dy) for q in polys]
        moved = [np.stack([q[:, 0], np.minimum(q[:, 1], gy + 40)], 1) for q in moved]
        pl.fill(moved, tone, True)
        if polys[0] is plaque:
            # 匾上兩個字：節烈
            for j, ch in enumerate("節烈"):
                g = serif.glyph(ch)
                gp = [q * 100 + (850 + j * 120, 470) for q in g.contours]
                gp = [rotate(q, ang, cx, cy) + (d * 120 * tau, dy) for q in gp]
                pl.fill(gp, 0.9, False)
    # 裂紋，然後塵土
    if p < 0.6:
        for px in (520, 800, 1120, 1400):
            n = int(8 * ramp(p, 0.05, 0.4))
            if n > 1:
                ys = np.linspace(gy, gy - 50 * n, n)
                xs = px + np.cumsum(r.normal(0, 6, n))
                pl.stroke([np.stack([xs, ys], 1)], 0.95, 3)
    dust = ramp(p, 0.45, 1.0)
    if dust > 0:
        for k in range(160):
            x = r.uniform(250, 1700) + r.normal(0, 40) * dust
            h = r.uniform(0.0, 1.0) ** 1.5 * 620 * ease_out(dust, 2)
            rad = r.uniform(25, 90) * (0.5 + dust)
            pl.fill([ellipse(x, gy - h, rad, rad * 0.7)], r.uniform(0.75, 0.95), False, alpha=0.22 * (1 - 0.4 * dust))


@insert("calendar")
def calendar(pl, p, t, rng, year=0, **kw):
    """日曆被撕下：一九一一……一九二七。"""
    years = ["一九一一", "一九一三", "一九一五", "一九一七", "一九一九", "一九二六", "一九二七"]
    serif = fonts.face("fonts/serif_regular")
    pl.begin(0.45, 0.35)

    def page(label, poly_xform, tone=0.94):
        base = rect(700, 180, 1220, 900)
        pl.fill([poly_xform(base)], tone, True)
        # 年份直排
        for j, ch in enumerate(label):
            g = serif.glyph(ch)
            gp = [poly_xform(q * 150 + (885, 340 + j * 150)) for q in g.contours]
            pl.fill(gp, 0.04, False)
    nxt = years[min(year + 1, len(years) - 1)]
    page(nxt, lambda q: q)
    k = ease_in(ramp(p, 0.15, 1.0), 1.6)
    page(years[year], lambda q: rotate(q, -1.2 * k, 1220, 180) + (500 * k, -200 * k * k))
    pl.fill([rect(690, 160, 1230, 200)], 0.1, True)


@insert("bars")
def bars(pl, p, t, rng, **kw):
    """牢房的窗：鐵條、斜進來的光，什麼都不動。"""
    pl.begin(0.08)
    win = rect(1180, 180, 1520, 520)
    pl.fill([win], 0.97, False)
    beam = np.array([[1180, 180], [1520, 180], [1100, 1080], [380, 1080]], np.float32)
    pl.grad_fill([beam], (1300, 200), (700, 1080), 0.8, 0.12, 0.85, 0.2)
    for k in range(5):
        x = 1220 + k * 70
        pl.fill([rect(x - 9, 180, x + 9, 520)], 0.03, True)
    pl.fill([rect(1180, 340, 1520, 356)], 0.03, True)


@insert("lamp_window")
def lamp_window(pl, p, t, rng, **kw):
    """夜裡白雲樓上的一扇窗：燈下有人在寫。"""
    r = rand(7, 75, 1)
    pl.begin(0.05, 0.1)
    pl.circle(300, 180, 70, 0.92, False)
    pl.fill([rect(420, 260, 1500, H)], 0.16, True)
    pl.fill([np.array([[380, 270], [960, 120], [1540, 270]])], 0.1, True)
    for j in range(4):
        for i in range(5):
            x, y = 500 + i * 200, 360 + j * 180
            lit = (i, j) == (2, 1)
            pl.fill([rect(x, y, x + 120, y + 120)], 0.95 if lit else 0.07, not lit)
            if lit:
                # 燈與伏案的人
                pl.grad_fill([ellipse(x + 60, y + 70, 140, 120)], (x + 60, y), (x + 60, y + 160), 0.99, 0.75, 0.9, 0.0)
                hx = x + 52 + 2 * np.sin(t * 2.2)
                pl.fill([ellipse(hx, y + 66, 13, 15)], 0.05, True)
                pl.fill([np.array([[x + 30, y + 120], [x + 36, y + 84], [hx, y + 78], [x + 80, y + 92],
                                   [x + 86, y + 120]])], 0.05, True)
                pen = x + 78 + 6 * np.sin(t * 9)
                pl.stroke([np.array([[x + 70, y + 92], [pen, y + 104]])], 0.05, 4, True)


# ── 渲染 ───────────────────────────────────────────────────
class Inserts:
    def __init__(self, seed):
        self.plate = Plate(W, H)
        self.cut = Woodcut(W, H, seed)
        self.seed = seed

    def render(self, name, p, t, frame, **kw):
        rng = rand(self.seed, 80, frame)
        REG[name](self.plate, float(np.clip(p, 0, 1)), t, rng, **kw)
        tone, sel, red = self.plate.arrays()
        return self.cut(tone, sel, red, jitter=frame)
