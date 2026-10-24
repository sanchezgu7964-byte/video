"""Pinhole camera, ground-plane ray casting, Perlin noise and world textures."""
from functools import lru_cache

import numpy as np


def normalize(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


class Camera:
    """Right-handed world with z up. fov is horizontal, in degrees."""

    def __init__(self, pos, target, fov, W, H, up=(0, 0, 1)):
        self.pos = np.asarray(pos, np.float64)
        self.W, self.H = W, H
        self.f = normalize(np.asarray(target, np.float64) - self.pos)
        self.r = normalize(np.cross(self.f, up))
        self.u = np.cross(self.r, self.f)
        self.focal = (W / 2) / np.tan(np.radians(fov) / 2)

    def project(self, P):
        """P: (..., 3) -> sx, sy, depth (all (...))"""
        d = np.asarray(P, np.float64) - self.pos
        zc = d @ self.f
        xc = d @ self.r
        yc = d @ self.u
        zs = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
        return self.W / 2 + self.focal * xc / zs, self.H / 2 - self.focal * yc / zs, zc

    def ground(self, w, h, z=0.0):
        """Ray-cast a w x h pixel grid (covering the full frame) onto plane z.

        Returns X, Y, dist, hit (bool)."""
        sx = (np.arange(w) + 0.5) / w * self.W
        sy = (np.arange(h) + 0.5) / h * self.H
        px, py = np.meshgrid((sx - self.W / 2) / self.focal, -(sy - self.H / 2) / self.focal)
        d = (self.f[None, None, :] + px[..., None] * self.r + py[..., None] * self.u)
        dz = d[..., 2]
        hit = dz < -1e-4
        t = np.where(hit, (z - self.pos[2]) / np.where(hit, dz, -1), 1e9)
        X = self.pos[0] + t * d[..., 0]
        Y = self.pos[1] + t * d[..., 1]
        dist = t * np.linalg.norm(d, axis=-1)
        return X.astype(np.float32), Y.astype(np.float32), dist.astype(np.float32), hit


# --------------------------------------------------------------------------
# Perlin gradient noise (2D and 3D), vectorised
# --------------------------------------------------------------------------
class Perlin:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256)
        self.p = np.concatenate([p, p]).astype(np.int64)
        a = rng.uniform(0, 2 * np.pi, 256)
        self.g2 = np.stack([np.cos(a), np.sin(a)], -1)
        v = rng.standard_normal((256, 3))
        self.g3 = v / np.linalg.norm(v, axis=1, keepdims=True)

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def noise2(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        xi, yi = np.floor(x), np.floor(y)
        xf, yf = x - xi, y - yi
        xi = xi.astype(np.int64) & 255
        yi = yi.astype(np.int64) & 255
        p, g = self.p, self.g2
        u, v = self._fade(xf), self._fade(yf)

        def gd(ix, iy, dx, dy):
            h = p[p[ix] + iy]
            return g[h, 0] * dx + g[h, 1] * dy

        n00 = gd(xi, yi, xf, yf)
        n10 = gd(xi + 1, yi, xf - 1, yf)
        n01 = gd(xi, yi + 1, xf, yf - 1)
        n11 = gd(xi + 1, yi + 1, xf - 1, yf - 1)
        a = n00 + u * (n10 - n00)
        b = n01 + u * (n11 - n01)
        return (a + v * (b - a)) * 1.41

    def noise3(self, x, y, z):
        x, y, z = (np.asarray(c, np.float64) for c in (x, y, z))
        xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
        xf, yf, zf = x - xi, y - yi, z - zi
        xi, yi, zi = (c.astype(np.int64) & 255 for c in (xi, yi, zi))
        p, g = self.p, self.g3
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)

        def gd(ix, iy, iz, dx, dy, dz):
            h = p[p[p[ix] + iy] + iz]
            return g[h, 0] * dx + g[h, 1] * dy + g[h, 2] * dz

        def lerp(a, b, t):
            return a + t * (b - a)

        x0 = lerp(gd(xi, yi, zi, xf, yf, zf), gd(xi + 1, yi, zi, xf - 1, yf, zf), u)
        x1 = lerp(gd(xi, yi + 1, zi, xf, yf - 1, zf), gd(xi + 1, yi + 1, zi, xf - 1, yf - 1, zf), u)
        x2 = lerp(gd(xi, yi, zi + 1, xf, yf, zf - 1), gd(xi + 1, yi, zi + 1, xf - 1, yf, zf - 1), u)
        x3 = lerp(gd(xi, yi + 1, zi + 1, xf, yf - 1, zf - 1),
                  gd(xi + 1, yi + 1, zi + 1, xf - 1, yf - 1, zf - 1), u)
        return lerp(lerp(x0, x1, v), lerp(x2, x3, v), w) * 1.2

    def fbm2(self, x, y, octaves=4, lac=2.0, gain=0.5):
        s, a, f = 0.0, 1.0, 1.0
        for i in range(octaves):
            s = s + a * self.noise2(x * f + i * 17.3, y * f - i * 9.1)
            a *= gain
            f *= lac
        return s


# --------------------------------------------------------------------------
# precomputed world textures sampled bilinearly
# --------------------------------------------------------------------------
class WorldTexture:
    def __init__(self, data, x0, y0, size):
        """data: (n, n) covering world square [x0, x0+size] x [y0, y0+size]."""
        self.data = data.astype(np.float32)
        self.x0, self.y0, self.size = x0, y0, size
        self.n = data.shape[0]

    def sample(self, X, Y):
        n = self.n
        fx = np.clip((X - self.x0) / self.size * (n - 1), 0, n - 1.001)
        fy = np.clip((Y - self.y0) / self.size * (n - 1), 0, n - 1.001)
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        tx, ty = fx - ix, fy - iy
        d = self.data
        a = d[iy, ix] * (1 - tx) + d[iy, ix + 1] * tx
        b = d[iy + 1, ix] * (1 - tx) + d[iy + 1, ix + 1] * tx
        return a * (1 - ty) + b * ty


@lru_cache(maxsize=4)
def perlin(seed):
    return Perlin(seed)
