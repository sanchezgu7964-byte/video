"""23 比巴爾扎克人物更多的日子 — more days than Balzac has characters.
A time-lapse crowd seen from above at dusk: tiny figures streaming in from
everywhere and merging into a single river of people.
"""
from functools import lru_cache

import numpy as np

from . import segment
from ..fx import Canvas, blur, smoothstep


def river_y(x, W, H):
    return H * (0.55 + 0.12 * np.sin(x / W * 3.0 + 0.6))


@lru_cache(maxsize=2)
def crowd(seed, n, W, H):
    rng = np.random.default_rng([seed, 2301])
    start = np.stack([rng.uniform(-0.1, 1.1, n) * W, rng.uniform(-0.1, 1.1, n) * H], 1)
    delay = rng.uniform(0.0, 0.55, n)
    speed = rng.uniform(0.6, 1.4, n)
    lane = rng.normal(0, 0.03, n) * H
    tone = rng.uniform(0.2, 1.0, n)
    return start, delay, speed, lane, tone


def pos(start, delay, speed, lane, t_u, W, H):
    """Each walker drifts toward the river, then flows along it to the right."""
    k = smoothstep(delay, delay + 0.4, t_u)
    flow_x = start[:, 0] + speed * W * 0.35 * t_u
    target = np.stack([flow_x, river_y(flow_x, W, H) + lane], 1)
    wander = start + np.stack([speed * W * 0.06 * t_u, np.zeros_like(speed)], 1)
    p = wander * (1 - k[:, None]) + target * k[:, None]
    p[:, 0] = (p[:, 0] + W * 0.1) % (W * 1.2) - W * 0.1
    return p


@segment("days")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    n = int(ctx.params.get("people", 1400))
    start, delay, speed, lane, tone = crowd(ctx.tl.seed, n, W, H)
    u = ctx.u
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = np.array([0.16, 0.12, 0.1]) * (0.6 + 0.4 * (1 - y / H))[..., None]
    p1 = pos(start, delay, speed, lane, u, W, H)
    p0 = pos(start, delay, speed, lane, max(0.0, u - 0.015), W, H)
    cv = Canvas(W, H, ctx.ss)
    for (ax, ay), (bx, by), tn in zip(p0, p1, tone):
        if abs(ax - bx) > W * 0.5:
            continue
        cv.line([(ax, ay), (bx, by)], tn * 0.7, 2.0 * sc)  # time-lapse streak
        cv.dot(bx, by, 2.8 * sc, tn)
    m = cv.array()
    mass = blur(m, 30 * sc)
    lit = np.array([1.0, 0.85, 0.6])
    img = ground * (1 - 0.6 * mass[..., None]) + lit * (m * 0.55)[..., None] + lit * 0.3 * mass[..., None]
    return np.clip(img, 0, 1)
