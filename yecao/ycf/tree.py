"""喬木：L-system（X → F[+X][-X]FX，F → FF），只長到一半就萎縮崩回，碎成短筆畫。"""
import numpy as np

from .noise import rand


def lsystem(seed, iters=5, angle=24.0, step=13.0):
    s = "X"
    rules = {"X": "F[+X][-X]FX", "F": "FF"}
    for _ in range(iters):
        s = "".join(rules.get(ch, ch) for ch in s)
    rng = rand(seed, 3, 1)
    pos = np.array([0.0, 0.0])
    ang = -90.0
    depth = 0
    dist = 0.0
    stack = []
    segs = []   # (x0, y0, x1, y1, depth, s0, s1)
    for ch in s:
        if ch == "F":
            a = np.radians(ang + rng.normal(0, 4))
            nxt = pos + step * np.array([np.cos(a), np.sin(a)])
            segs.append((*pos, *nxt, depth, dist, dist + step))
            pos = nxt
            dist += step
        elif ch == "+":
            ang += angle + rng.normal(0, 5)
        elif ch == "-":
            ang -= angle + rng.normal(0, 5)
        elif ch == "[":
            stack.append((pos.copy(), ang, depth, dist))
            depth += 1
        elif ch == "]":
            pos, ang, depth, dist = stack.pop()
    return np.array(segs, np.float32)
