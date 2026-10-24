"""32 消磨了歲月 — and you have spent the years. Back to the viaduct of 01,
black & white, the same high view; a train crosses as trains always have, and
a very small human figure stands below in the field.
"""
import numpy as np

from . import segment
from ..fx import Canvas, blur
from .s01_viaduct import LIGHT, draw_viaduct, viaduct_camera


def box(cam, x0, x1, y0, y1, z0, z1):
    """Projected top face and near (y0) face of a box, as screen polygons."""
    top = np.array([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
    near = np.array([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)])
    out = []
    for P in (near, top):
        sx, sy, z = cam.project(P)
        if np.all(z > 1):
            out.append(list(zip(sx, sy)))
        else:
            out.append(None)
    return out


@segment("wasted_years")
def render(ctx):
    W, H, sc = ctx.W, ctx.H, ctx.scale
    vp = ctx.tl.segment("viaduct").params
    cam = viaduct_camera(0.2 + 0.25 * ctx.u, float(vp.get("drift", 1.0)), W, H)
    img = draw_viaduct(ctx, cam, vp)
    height = float(vp.get("height", 34.0))
    # the train on the deck
    near_c, top_c = Canvas(W, H, ctx.ss), Canvas(W, H, ctx.ss)
    head = 10.0 + 14.0 * ctx.t
    for k in range(10):
        x1 = head - k * 15.8
        x0 = x1 - 15.0
        near, top = box(cam, x0, x1, -1.5, 1.5, height + 0.5, height + 4.0)
        if near:
            near_c.polygon(near, 1.0)
        if top:
            top_c.polygon(top, 1.0)
    nm, tm = near_c.array(), top_c.array()
    img = img * (1 - nm) + 0.22 * nm
    img = img * (1 - tm) + 0.5 * tm
    # a very small figure standing in the field, with its long shadow
    fig = Canvas(W, H, ctx.ss)
    gx, gy = 15.0, -85.0
    k = 2.4                                               # still tiny at this distance, but findable
    head_ = [(0.12 * np.cos(a), 1.62 + 0.14 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12)]
    body = [(-0.22, 1.45), (0.22, 1.45), (0.16, 0.0), (-0.16, 0.0)]
    to_cam = cam.pos[:2] - np.array([gx, gy])
    to_cam /= np.linalg.norm(to_cam)
    right = np.array([-to_cam[1], to_cam[0]])                # the figure faces the camera
    for poly in (head_, body):
        P = np.array([(gx + right[0] * px * k, gy + right[1] * px * k, pz * k) for px, pz in poly])
        fx, fy, _ = cam.project(P)
        fig.polygon(list(zip(fx, fy)), 1.0)
    shadow = Canvas(W, H, ctx.ss)
    tip = np.array([gx, gy]) + LIGHT[:2] * (1.75 * k / -LIGHT[2])     # same sun as the viaduct's shadows
    w = right * 0.35 * k
    S = np.array([(gx - w[0], gy - w[1], 0), (gx + w[0], gy + w[1], 0), (tip[0] + w[0], tip[1] + w[1], 0),
                  (tip[0] - w[0], tip[1] - w[1], 0)])
    sx, sy, _ = cam.project(S)
    shadow.polygon(list(zip(sx, sy)), 1.0)
    img = img * (1 - 0.6 * shadow.array())
    fm = fig.array()
    return np.clip(img * (1 - fm) + 0.02 * fm, 0, 1)
