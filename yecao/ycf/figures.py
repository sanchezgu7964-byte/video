"""木刻裡的人：側臉、頸、肩。

造型照木刻的做法：整個頭先是一塊黑，迎光的臉被整片刻白，留下骨相；
明暗交界處是一組順著臉頰弧度的刀線，向後收進黑裡。眉、眼、鼻翼、嘴、下巴只用幾刀。

局部座標：側臉朝 +x，y 向下；頭頂到下巴約 100 單位，原點在耳朵。
"""
import numpy as np

from .carve import ellipse, offset, resample, spline, xform


def _hh(*k):
    from .carve import _h
    return _h(*k)


def _rot(P, ang, cx, cy):
    c, s = np.cos(ang), np.sin(ang)
    Q = np.asarray(P, np.float32) - (cx, cy)
    return np.stack([Q[:, 0] * c - Q[:, 1] * s, Q[:, 0] * s + Q[:, 1] * c], 1) + (cx, cy)


class Profile:
    """一張側臉的形狀參數。"""

    def __init__(self, nose=1.0, brow=1.0, jaw=1.0, chin=0.0, lips=1.0, skull=1.0, age=0.0, cap="skull",
                 moustache=False, gaunt=0.0):
        self.nose, self.brow, self.jaw, self.chin, self.lips = nose, brow, jaw, chin, lips
        self.skull, self.age, self.cap, self.moustache, self.gaunt = skull, age, cap, moustache, gaunt

    def front(self, mouth=0.0):
        """臉的正面輪廓（由頭頂往下到喉嚨），與張嘴時下巴繞著的樞紐。"""
        n, b, j, c, l = self.nose, self.brow, self.jaw, self.chin, self.lips
        up = [(-8, -57), (18, -53), (33, -40), (39, -25), (42 + 1.5 * b, -13), (39.5, -7),
              (44 + 4 * n, 3 + 2 * n), (49 + 5 * n, 10 + 1.5 * n), (45 + 3 * n, 14), (41, 15.5),
              (42.5 + 1.2 * l, 20), (41, 22.5)]
        lo = [(41.5 + 1.0 * l, 26), (37.5 - c, 30.5), (39.5 - 1.5 * c, 37), (34 - c, 44 * j), (18, 46 * j)]
        piv = (-2.0, 2.0)
        a = -0.28 * mouth
        lo = [tuple(q) for q in _rot(lo, -a, *piv)]
        return up, lo, piv, a

    def outline(self, mouth=0.0, neck=120):
        up, lo, piv, a = self.front(mouth)
        sk = self.skull
        throat = [(8, 52), (6.5, 66), (7, neck)]
        back = [(-26, neck), (-25, min(64, neck - 4)), (-29, 28), (-38 * sk, 12), (-46 * sk, -10), (-43 * sk, -34), (-28 * sk, -52)]
        pts = up + lo + throat + back
        return spline(pts, closed=True, step=1.2), up, lo, piv, a


def head(b, ox, oy, s, prof, rot=0.0, flip=False, mouth=0.0, eye=1.0, seed=0, px=11.0, light=1.0, neck=120, rim=0.0, comb=1.0):
    """畫一個側臉的頭與頸。(ox, oy) 是耳朵在版上的位置，s 是每單位多少版面單位。
    px：刀線間距（版面單位）。light：迎光的程度（0..1），影響明暗交界的位置。"""
    T = lambda P: xform(P, ox, oy, s, rot, flip)
    sp = px / s                                    # 局部座標下的刀距
    out, up, lo, piv, a = prof.outline(mouth, neck)
    sil = T(out)
    if rim:
        sk = prof.skull
        back = spline([(-29, 28), (-38 * sk, 12), (-46 * sk, -10), (-43 * sk, -34), (-28 * sk, -52), (-8, -57), (18, -53)],
                      step=1.0)
        b.halo(T(back), rim, seed + 99, closed=False)
    b.ink([sil])

    # 迎光的臉：由正面輪廓往內，到明暗交界。交界在眼窩、顴骨下凹進來——那些暗面和頭側的黑連成一片
    g = prof.gaunt
    sh = 6 * (light - 0.7)
    term = [(24, -49), (19, -34), (19.5, -21), (25, -16.5), (32, -15.8), (37.2, -12.5), (38.4, -7.5), (36.5, -3.2), (27, -1.0),
            (17, 3), (15.5, 9), (20 + 6 * g, 13), (26 + 4 * g, 16.5), (22 + 3 * g, 20.5), (18, 26), (22, 36),
            (28, 44 * prof.jaw)]
    if prof.cap == "shaved":
        term = [(-14, -57), (0, -47), (12, -34)] + term[2:]
    term = [(x - sh, y) for x, y in term]
    tcurve = spline(term, step=0.5)
    lit = spline(up + lo + [(14, 47), (term[-1][0] - 2, term[-1][1])], step=1.2)
    lit = np.concatenate([lit, tcurve[::-1]])
    with b.clip([sil]):
        b.carve([T(lit)])
        # 交界：一道道橫過頭側的短刀，順著頭的圓往後彎，從亮面伸進黑裡，長短不一，收成刀尖
        ty, tx = tcurve[:, 1], tcurve[:, 0]
        order = np.argsort(ty)
        items = []
        y = -46.0
        j = 0
        while y < 44:
            x0 = float(np.interp(y, ty[order], tx[order]))
            L = (7 + 13 * _hh(seed, j)) * (1.0 if -30 < y < 30 else 0.6) * comb
            bow = 1.5 + 0.06 * abs(y)
            P = spline([(x0 + 2.5, y), (x0 - L * 0.5, y - bow * 0.6 + 0.08 * y), (x0 - L, y - bow + 0.16 * y)], step=0.6)
            u = np.linspace(0, 1, len(P))
            w = (1 - u) ** 0.9
            items.append((T(P), w * sp * s * (0.55 + 0.25 * _hh(seed, j, 2))))
            y += sp * (0.85 + 0.3 * _hh(seed, j, 3))
            j += 1
        b.strokes(items, 1.0, wmin=0.5, seed=seed + 1, taper=0.02)

        # 頸：喉嚨迎光的兩刀
        items = []
        for k in range(1, 2):
            C = resample(np.array([[7.5 - k * sp * 0.9, 50], [6.5 - k * sp * 0.9, 70], [7 - k * sp, neck - 5]]), 0.8)
            w = np.full(len(C), max(0, 1 - k / 3.0)) * np.clip(np.linspace(0, 1, len(C)) * 3, 0, 1)
            items.append((T(C), w * sp * s * 0.55))
        b.strokes(items, 1.0, wmin=0.6, seed=seed + 2)

        # 骨相：鼻側的暗面
        b.knife(T(np.array([[37.5, -5.5], [40, 1.5], [42.5, 7.5]])), 1.4 * s, 0.0, taper=(0.4, 0.3), seed=seed + 8)

        # 下巴底下的影
        jawline = spline([(-1, 16), (8, 30), (20, 42 * prof.jaw), (32 - prof.chin, 45 * prof.jaw)], step=1.0)
        jaw = _rot(jawline, -a, *piv)
        b.knife(T(jaw), 1.6 * s, 0.0, taper=(0.5, 0.15), seed=seed + 3)
        under = np.concatenate([jaw, np.array([[34, 47], [10, 52], [0, 30]])])
        b.ink([T(_rot(under, 0, 0, 0))])

        if prof.cap not in ("student", "skull"):
            _hair(b, T, prof, sp, s, seed)
        _ear(b, T, s, seed)
        _features(b, T, prof, mouth, piv, a, eye, s, seed)
    if prof.cap in ("student", "skull"):
        _hair(b, T, prof, sp, s, seed)                   # 帽子超出頭的輪廓，在剪裁之外畫
    return sil


def _hair(b, T, prof, sp, s, seed):
    """頭髮或帽：一塊沉實的黑，迎光的頂上幾刀。剃光的前額是一片白，後面拖著辮子。"""
    cap = prof.cap
    if cap == "shaved":
        # 清末：前半剃光，頂上留髮。剃光的部分是亮的
        stub = [resample(spline([(-18 + k * 2.4, -56 + k * 0.4), (-8 + k * 2.6, -45 + k * 0.6),
                                 (2 + k * 2.6, -33 + k * 0.6)], step=1), 0.7) for k in range(4)]
        b.strokes([(T(c), np.full(len(c), (0.9 - 0.18 * k) * s)) for k, c in enumerate(stub)], 0.0, wmin=0.3,
                  seed=seed + 9, breaks=0.3)
        _hair_cuts(b, T, sp, s, seed, front=-16)
        # 辮子：從後腦垂下
        braid = spline([(-36, -20), (-48, 10), (-50, 50), (-46, 110)], step=1)
        bw = np.linspace(9, 5, len(braid))
        b.knife(T(braid), bw * s, 0.0, taper=(0.05, 0.2), seed=seed + 11)
        b.knife(T(offset(braid, bw * 0.55)), 1.2 * s, 1.0, taper=(0.3, 0.3), seed=seed + 12)
        for k in range(14):
            i = int((k + 0.5) / 14 * (len(braid) - 1))
            q = braid[i]
            n = np.array([1.0, 0.25])
            c = np.array([q - n * bw[i] * 0.45 + (0, -2), q + (0, 2.2), q + n * bw[i] * 0.45 + (0, -2)])
            b.knife(T(c), 1.0 * s, 1.0, taper=(0.4, 0.4), smooth=False, seed=seed + 20 + k)
        return
    if cap == "skull":
        # 瓜皮帽：半個球，頂上一顆結，帽沿一道
        rim_y = -38
        capp = spline([(-36, rim_y + 6), (-38, -52), (-14, -66), (16, -62), (34, -46), (37, rim_y)], step=1)
        poly = np.concatenate([capp, [[37, rim_y + 3], [-36, rim_y + 10]]])
        b.ink([T(poly)])
        b.ink([T(ellipse(-4, -67, 5, 3.5))])
        b.knife(T(np.array([[-35, rim_y + 7], [10, rim_y + 3.5], [36, rim_y + 1]])), 1.0 * s, 1.0,
                taper=(0.3, 0.2), seed=seed + 12)
        # 帽瓣的接縫：幾道順著球面的弧
        for k, x in enumerate((-18, 2, 20)):
            c = spline([(x - 4, rim_y + 2), (x, -56), (-4, -66)], step=1)
            b.knife(T(c), 0.7 * s, 1.0, taper=(0.2, 0.5), seed=seed + 13 + k)
        return
    if cap == "student":
        # 學生帽：平頂、硬帽簷
        crown = np.array([[-42, -34], [-45, -63], [32, -66], [38, -38]])
        brim = spline([(30, -42), (52, -41), (62, -36), (36, -33)], closed=True, step=1)
        b.halo(T(crown), 0.5 * s, seed + 17)
        b.halo(T(brim), 0.5 * s, seed + 18)
        b.ink([T(crown), T(brim)])
        b.knife(T(np.array([[-40, -38], [0, -40], [34, -39]])), 0.8 * s, 1.0, taper=(0.2, 0.2), seed=seed + 14)
        b.knife(T(np.array([[-38, -58], [0, -61], [28, -60]])), 0.7 * s, 1.0, taper=(0.3, 0.3), seed=seed + 15)
        b.knife(T(np.array([[34, -38], [55, -37]])), 0.6 * s, 1.0, taper=(0.2, 0.6), seed=seed + 16)
        return
    # 光頭髮：頭頂迎光的幾刀
    _hair_cuts(b, T, sp, s, seed, front=30)


def _hair_cuts(b, T, sp, s, seed, front):
    curves = []
    for k in range(4):
        r = k * sp * 1.1
        c = spline([(-44 + r, -14), (-40 + r * 0.8, -40 + r), (-12, -56 + r), (front - r * 0.5, -50 + r * 1.2)], step=1)
        w = np.clip(np.sin(np.linspace(0, np.pi, len(c))) * 1.4, 0, 1) * (1 - k / 4.2)
        curves.append((T(c), w * sp * s * 0.45))
    b.strokes(curves, 1.0, wmin=0.6, seed=seed + 30, breaks=0.25)


def _ear(b, T, s, seed):
    c = spline([(-3, -8), (4, -11), (7, -2), (4, 8), (0, 13), (-4, 12)], step=0.6)
    b.knife(T(c), 1.4 * s, 1.0, taper=(0.3, 0.4), seed=seed + 40)
    b.knife(T(spline([(-1, -4), (3, -3), (2, 4)], step=0.6)), 0.9 * s, 1.0, taper=(0.4, 0.4), seed=seed + 41)


def _features(b, T, prof, mouth, piv, a, eye, s, seed):
    k = s                                                # 刀寬以局部單位計
    by = -13 - 1.5 * prof.brow
    if eye > 0.15:
        # 眼白：眼珠前一小片光；下眼瞼一道刻白
        e = max(1.0, eye)
        ew = spline([(33.5, -6.0 - 1.4 * e), (37.0, -5.6 - e), (38.6, -5.6), (37.2, -4.4), (34.0, -4.6)], closed=True,
                    step=0.3)
        b.carve([T(ew)])
        b.ink([T(ellipse(35.4, -5.8, 1.2 * e ** 0.5, 1.4 + 0.25 * e))])
        b.knife(T(np.array([[30.5, -2.6], [34.5, -3.1], [38.2, -4.6]])), 0.9 * s, 1.0, taper=(0.5, 0.2), seed=seed + 52)
    else:
        b.knife(T(np.array([[29, -3.6], [33.5, -2.6], [37.8, -4.0]])), 0.9 * s, 1.0, taper=(0.4, 0.4), seed=seed + 51)
    # 眉：一刀，重
    b.knife(T(np.array([[24, by - 1], [33, by - 2.5], [41, by + 0.5]])), 3.6 * k, 0.0, taper=(0.25, 0.45),
            seed=seed + 50)
    # 眼：眉下的眼窩是一塊黑，眼珠留一點光，下眼瞼一道刻白
    b.knife(T(np.array([[30, -0.5], [34, 0.5], [38, -0.6]])), 1.1 * k, 0.0, taper=(0.5, 0.5), seed=seed + 52)
    if prof.age > 0.3:
        b.knife(T(np.array([[26, 2], [31, 5], [36, 4]])), 1.0 * k, 0.0, taper=(0.5, 0.5), seed=seed + 53)
    # 鼻翼與鼻孔
    n = prof.nose
    b.knife(T(spline([(40.5 + 2 * n, 6), (38.5 + 2 * n, 10.5), (41.5 + 2 * n, 13.5)], step=0.5)), 2.0 * k, 0.0,
            taper=(0.4, 0.3), seed=seed + 54)
    b.ink([T(ellipse(43.5 + 2.4 * n, 13.6, 2.2, 1.0, 0.3))])
    # 法令紋：從鼻翼往嘴角
    nl = spline([(39 + 2 * n, 13), (36.5, 19), (37, 26)], step=0.5)
    b.knife(T(nl), (1.9 + 1.6 * prof.age) * k, 0.0, taper=(0.3, 0.6), seed=seed + 55)
    # 嘴
    up, lo, _, _ = prof.front(mouth)
    if mouth > 0.08:
        ul = np.array(up[-2:])
        ll = np.array(lo[:1])
        hole = np.concatenate([ul, ll, _rot(np.array([[36.5, 24.5]]), -a, *piv), np.array([[36, 21]])])
        b.ink([T(spline(hole, closed=True, step=0.4))])
        if mouth > 0.4:
            b.carve([T(np.array([[40.5, 21.8], [37.4, 21.6], [38, 23.0], [41.0, 22.8]]))])     # 上牙
    else:
        b.knife(T(np.array([[42.5, 22.8], [38.5, 23.1], [35.5, 22.2]])), 2.2 * k, 0.0, taper=(0.15, 0.5),
                seed=seed + 56)
    # 下巴的窩
    cw = _rot(np.array([[38 - prof.chin, 31], [35.5 - prof.chin, 31.8]]), -a, *piv)
    b.knife(T(cw), 2.0 * k, 0.0, taper=(0.4, 0.4), smooth=False, seed=seed + 57)
    # 額頭的紋
    if prof.age > 0.2:
        for j in range(int(1 + prof.age * 3)):
            y = -22 - j * 4.5
            b.knife(T(np.array([[27, y + 1], [33, y - 0.4], [38, y + 0.4]])), 1.2 * k, 0.0, taper=(0.5, 0.5),
                    seed=seed + 60 + j)
    if prof.moustache:
        b.ink([T(spline([(44.0, 15.8), (38, 15.6), (33.5, 18.0), (32.5, 22.5), (36, 19.6), (42.0, 18.4)], closed=True,
                        step=0.4))])


def shoulders(b, ox, oy, s, rot=0.0, flip=False, hunch=0.0, seed=0, px=11.0, collar=True, light=1.0, rim=0.0):
    """肩與上身：一塊黑的長袍，迎光的肩頭被幾道順著肩弧的刀刻開。(ox, oy) 為頸根。"""
    T = lambda P: xform(P, ox, oy, s, rot, flip)
    sp = px / s
    h = hunch * 14
    body = spline([(-160, 400), (-158, 170), (-142, 76 - h * 0.5), (-104, 34 - h), (-40, 6 - h * 0.5), (8, 0), (70, 14),
                   (128, 44), (154, 110), (162, 400)], step=2)
    if rim:
        b.halo(T(spline([(-160, 400), (-158, 170), (-142, 76 - h * 0.5), (-104, 34 - h), (-40, 6 - h * 0.5), (4, 0)],
                        step=2)), rim,
               seed + 98, closed=False)
    b.ink([T(body)])
    top = spline([(-120, 50 - h), (-60, 14 - h * 0.6), (0, 6), (70, 20), (122, 50), (146, 110)], step=1)
    items = []
    for k in range(3):
        C = resample(offset(top, k * sp * 1.4), 1.0)
        u = np.linspace(0, 1, len(C))
        # 光從前上方：只有前肩迎光
        w = (1 - k / 3.4) ** 1.3 * np.clip(-0.5 + 1.6 * u, 0, 1) * light
        w *= np.clip(np.sin(np.pi * u) * 2.2, 0, 1)
        items.append((T(C), w * sp * s * 0.6))
    b.strokes(items, 1.0, wmin=0.7, seed=seed + 1, breaks=0.12)
    # 衣褶：幾道從肩往下的刀
    # 袖與身的分界、胸前一道褶：短，只交代結構
    for j, (pts, w) in enumerate([([(112, 70), (118, 150), (116, 240)], 2.4), ([(-112, 74), (-118, 160), (-114, 230)], 1.6),
                                  ([(52, 44), (58, 110), (54, 170)], 1.8)]):
        b.knife(T(spline(pts, step=1)), w * s * sp / 3, 1.0, taper=(0.4, 0.6), seed=seed + 10 + j)
    if collar:
        # 立領：兩道刀交代領口
        b.knife(T(spline([(-34, 2), (-8, 14), (20, 14), (36, 4)], step=1)), 2.4 * sp * s / 3, 1.0,
                taper=(0.3, 0.3), seed=seed + 20)
        b.knife(T(spline([(-30, -10), (-6, 0), (22, 0), (34, -8)], step=1)), 1.8 * sp * s / 3, 1.0,
                taper=(0.4, 0.4), seed=seed + 21)
    return T
