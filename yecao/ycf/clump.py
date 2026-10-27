"""「一叢野草」：第 10 段由灰燼聚成，第 11 段生長腐爛循環，第 12 段被風吹散。三段共用同一叢。"""
import numpy as np

from . import fonts
from .gfx import fill_polys
from .noise import rand

BASE = (960.0, 600.0)   # 叢的根部（螢幕座標），第 10 段的分界線也穿過這一點
SPEC = [  # (字, 尺寸, x 偏移)
    ("草", 56, -190),
    ("野", 82, -112),
    ("草", 122, 0),
    ("叢", 76, 104),
    ("野", 52, 176),
]


def make_clump(tl):
    from .scene_field import Plant
    face = fonts.face(tl.raw["fonts"]["grass"])
    rng = rand(tl.seed, 10, 1)
    plants = []
    for k, (ch, size, dx) in enumerate(SPEC):
        p = Plant(k, ch, face.glyph(ch), BASE[0] + dx, 1, size, rng, 0.0)
        plants.append(p)
    return plants


def clump_polys(p, theta=0.0, squash=1.0, rise=1.0, lift=0.0):
    """回傳螢幕座標多邊形；rise < 1 時從地面下冒出（呼叫端負責裁切）。"""
    polys = p.warp(p.local, theta, squash)
    dy = (1 - rise) * p.h * 1.05 - lift
    return [q + (p.x, BASE[1] + dy) for q in polys]


def sample_points(plants, n, seed, W, H):
    """在叢的字形內部取樣 n 個點（供灰燼聚攏）。"""
    import cairo
    from .gfx import Layer
    lay = Layer(W // 2, H // 2)
    lay.ctx.scale(0.5, 0.5)
    for p in plants:
        fill_polys(lay.ctx, clump_polys(p), (1, 1, 1), 1.0)
    m = lay.rgba()[..., 3]
    ys, xs = np.nonzero(m > 0.4)
    rng = rand(seed, 10, 2)
    i = rng.choice(len(xs), n, replace=True)
    return (np.stack([xs[i], ys[i]], 1) * 2 + rng.random((n, 2)) * 2).astype(np.float32)
