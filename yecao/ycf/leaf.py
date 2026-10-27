"""葉脈網路：主脈、側脈、細脈與葉緣。座標以葉片中心為原點，y 向下。"""
import numpy as np

from .noise import rand


def leaf(seed, L=720.0, Wd=250.0):
    rng = rand(seed, 2, 1)
    segs = []   # (x0, y0, x1, y1, order, d) d：離葉柄的路徑長（正規化）

    def half_width(s):
        return Wd * np.sin(np.pi * np.clip(s, 0, 1)) ** 0.8 * (1 - 0.25 * s)

    def mid(s):
        return np.array([18 * np.sin(s * 2.4), L / 2 - s * L])

    n = 48
    for i in range(n):
        a, b = mid(i / n), mid((i + 1) / n)
        segs.append((*a, *b, 0, i / n))
    margin_r = [np.array([half_width(s), mid(s)[1]]) + (mid(s)[0], 0) for s in np.linspace(0, 1, 60)]
    margin_l = [np.array([-half_width(s), mid(s)[1]]) + (mid(s)[0], 0) for s in np.linspace(0, 1, 60)]
    margin = np.array(margin_r + margin_l[::-1], np.float32)
    for side in (-1, 1):
        for k in range(13):
            s0 = 0.06 + k * 0.068 + rng.uniform(-0.01, 0.01) + (0.03 if side > 0 else 0)
            if s0 > 0.93:
                continue
            p = mid(s0)
            ang = np.radians(-90 + side * (58 - 18 * s0))
            reach = half_width(s0 + 0.08) * 0.92
            pts = [p]
            m = 12
            for j in range(m):
                ang += side * np.radians(-3.0) + rng.normal(0, 0.03)
                pts.append(pts[-1] + reach / m * 1.15 * np.array([np.cos(ang), np.sin(ang)]))
            for j in range(m):
                segs.append((*pts[j], *pts[j + 1], 1, s0 * 0.6 + 0.4 * (j + 1) / m))
                # 細脈
                if j % 3 == 1:
                    q = pts[j + 1]
                    a2 = ang + side * rng.choice([-1, 1]) * np.radians(rng.uniform(40, 75))
                    l2 = rng.uniform(22, 55)
                    q2 = q + l2 * np.array([np.cos(a2), np.sin(a2)])
                    segs.append((*q, *q2, 2, s0 * 0.6 + 0.4 * (j + 1) / m + 0.05))
    return np.array(segs, np.float32), margin
