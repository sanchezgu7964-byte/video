"""可重現的向量化值噪聲（value noise）與 fbm。"""
import numpy as np


@np.errstate(over="ignore")
def _hash(ix, iy, iz, seed):
    u = lambda a: (a & 0xffffffff).astype(np.uint32)
    h = (u(ix) * np.uint32(374761393)
         + u(iy) * np.uint32(668265263)
         + u(iz) * np.uint32(2246822519)
         + np.uint32((seed * 3266489917) & 0xffffffff))
    h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
    h = h ^ (h >> np.uint32(16))
    return h.astype(np.float32) / np.float32(4294967295.0)


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


# 12 個立方體邊中點方向（Perlin 改良版）
_GRAD = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1],
                  [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], np.float32)


def noise3(x, y, z, seed=0):
    """3D Perlin 梯度噪聲，約落在 [-1, 1]；x, y, z 可廣播。"""
    x, y, z = np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(y, np.float32),
                                  np.asarray(z, np.float32))
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    rx, ry, rz = x - xi, y - yi, z - zi
    fx, fy, fz = _fade(rx), _fade(ry), _fade(rz)
    xi, yi, zi = xi.astype(np.int64), yi.astype(np.int64), zi.astype(np.int64)
    out = 0
    for dx in (0, 1):
        wx = fx if dx else 1 - fx
        for dy in (0, 1):
            wy = fy if dy else 1 - fy
            for dz in (0, 1):
                wz = fz if dz else 1 - fz
                h = (_hash(xi + dx, yi + dy, zi + dz, seed) * 12).astype(np.int64) % 12
                g = _GRAD[h]
                dot = g[..., 0] * (rx - dx) + g[..., 1] * (ry - dy) + g[..., 2] * (rz - dz)
                out = out + wx * wy * wz * dot
    return out * 1.1


def curl2(x, y, z, scale, seed=0):
    """二維 curl noise：無散度的流場（粒子不會擠成線）。"""
    e = 0.5
    psi = lambda xx, yy: noise3(xx / scale, yy / scale, z, seed)
    dpdy = (psi(x, y + e) - psi(x, y - e)) / (2 * e)
    dpdx = (psi(x + e, y) - psi(x - e, y)) / (2 * e)
    return dpdy * scale, -dpdx * scale


def fbm(x, y, z, octaves=4, seed=0, lac=2.0, gain=0.5):
    amp, freq, s, norm = 1.0, 1.0, 0, 0
    for o in range(octaves):
        s = s + amp * noise3(x * freq, y * freq, z * freq, seed + 101 * o)
        norm += amp
        amp *= gain
        freq *= lac
    return s / norm


def rand(seed: int, *key) -> np.random.Generator:
    """以種子與鍵值產生獨立的隨機數產生器。"""
    return np.random.default_rng([seed, *[int(k) & 0xffffffff for k in key]])


def curl_grid(W, H, z, scale, seed=0, step=20, edge=0.0):
    """在粗網格上算 curl noise，回傳可對任意點取樣的函式（雙線性內插）。
    edge > 0 時，流函數在邊界附近乘上窗函數，流動與邊界相切，不會有東西流進流出。"""
    gx = np.arange(-1, W // step + 3) * step
    gy = np.arange(-1, H // step + 3) * step
    X, Y = np.meshgrid(gx.astype(np.float32), gy.astype(np.float32))
    psi = noise3(X / scale, Y / scale, z, seed) * scale
    if edge > 0:
        sm = lambda d: np.clip(d / edge, 0, 1) ** 2 * (3 - 2 * np.clip(d / edge, 0, 1))
        psi = psi * sm(X) * sm(W - X) * sm(Y) * sm(H - Y)
    dpdy, dpdx = np.gradient(psi, step)
    vx, vy = dpdy, -dpdx

    def sample(px, py):
        fx = np.clip((px - gx[0]) / step, 0, len(gx) - 1.001)
        fy = np.clip((py - gy[0]) / step, 0, len(gy) - 1.001)
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        ux, uy = fx - ix, fy - iy
        out = []
        for F in (vx, vy):
            a = F[iy, ix] * (1 - ux) + F[iy, ix + 1] * ux
            b = F[iy + 1, ix] * (1 - ux) + F[iy + 1, ix + 1] * ux
            out.append(a * (1 - uy) + b * uy)
        return out
    return sample
