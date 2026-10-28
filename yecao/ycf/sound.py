"""程序生成的聲音設計。

所有聲音都從場景物件本身讀取事件（葉脈何時熄滅、字草何時冒出、刀往哪個方向掃……），
因此和畫面不可能錯開。全部以 numpy / scipy 合成，不使用任何外部音檔。
"""
import numpy as np
from scipy.signal import butter, lfilter, oaconvolve, sosfilt

from .gfx import ease, ramp
from .noise import rand

SR = 48000


# ── 基本元件 ──────────────────────────────────────────────
def sos(kind, f, order=2):
    f = np.atleast_1d(np.clip(f, 10, SR / 2 - 100))
    return butter(order, f if len(f) > 1 else f[0], btype=kind, fs=SR, output="sos")


def filt(x, kind, f, order=2):
    return sosfilt(sos(kind, f, order), x).astype(np.float32)


def sweep_filter(x, kind, f_curve, block=1024, order=2):
    """隨時間改變截止頻率的濾波（分塊處理，沿用濾波器狀態）。"""
    out = np.empty_like(x)
    zi = None
    for i in range(0, len(x), block):
        f = float(f_curve[min(i, len(f_curve) - 1)])
        s = sos(kind, f, order)
        if zi is None:
            zi = np.zeros((s.shape[0], 2))
        out[i:i + block], zi = sosfilt(s, x[i:i + block], zi=zi)
    return out


def white(rng, n):
    return rng.standard_normal(n).astype(np.float32)


def brown(rng, n):
    return lfilter([1.0], [1.0, -0.997], white(rng, n)).astype(np.float32) * 0.06


def pink(rng, n):
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, white(rng, n)).astype(np.float32)


def env_ad(n, attack, decay_tau):
    t = np.arange(n) / SR
    return (np.minimum(t / max(attack, 1e-4), 1) * np.exp(-np.maximum(t - attack, 0) / decay_tau)).astype(np.float32)


def tone(freqs, dur, amps=None, detune=0.0, rng=None):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    amps = amps or [1.0] * len(freqs)
    for f, a in zip(freqs, amps):
        ph = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
        out += a * np.sin(2 * np.pi * f * (1 + detune * np.sin(2 * np.pi * 0.13 * t + ph)) * t + ph)
    return out


def glide(f0, f1, dur, curve=2.0):
    n = int(dur * SR)
    u = np.linspace(0, 1, n) ** curve
    f = f0 * (f1 / f0) ** u
    return np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32)


class Bus:
    """立體聲匯流排；add 時可給固定或隨時間變化的聲像（-1 左 … 1 右）。"""

    def __init__(self, dur):
        self.n = int(dur * SR) + SR
        self.L = np.zeros(self.n, np.float32)
        self.R = np.zeros(self.n, np.float32)

    def add(self, x, t0, gain=1.0, pan=0.0):
        i0 = int(round(t0 * SR))
        if i0 >= self.n or len(x) == 0:
            return
        if i0 < 0:
            x = x[-i0:]
            if isinstance(pan, np.ndarray):
                pan = pan[-i0:]
            i0 = 0
        x = x[: self.n - i0] * gain
        p = pan[: len(x)] if isinstance(pan, np.ndarray) else pan
        ang = (np.clip(p, -1, 1) + 1) * np.pi / 4
        self.L[i0:i0 + len(x)] += x * np.cos(ang)
        self.R[i0:i0 + len(x)] += x * np.sin(ang)

    def stereo(self):
        return np.stack([self.L, self.R], 1)


def curve(points, n):
    """由 [(秒, 值), ...] 線性內插出長度 n 的取樣率包絡。"""
    t = np.arange(n) / SR
    return np.interp(t, [p[0] for p in points], [p[1] for p in points]).astype(np.float32)


def xpan(x):
    """螢幕 x（0..1920）→ 聲像。"""
    return float(np.clip((x - 960) / 960, -1, 1)) * 0.85


# ── 音色 ──────────────────────────────────────────────────
def thud(rng, f0=58, f1=30, dur=1.2, body=1.0, grit=0.4):
    n = int(dur * SR)
    s = glide(f0, f1, dur, 0.5) * env_ad(n, 0.003, dur * 0.25)
    nz = filt(white(rng, n), "lowpass", 380) * env_ad(n, 0.001, 0.08) * grit * 3
    return (s * body + nz).astype(np.float32)


def crackle(rng, dur, density, f_lo=900, f_hi=7000, amp=1.0):
    """隨機的細小爆裂聲（density：每秒個數，可為陣列）。"""
    n = int(dur * SR)
    d = np.broadcast_to(np.asarray(density, np.float32), (n,))
    clicks = (rng.random(n) < d / SR).astype(np.float32) * rng.uniform(0.2, 1.0, n).astype(np.float32)
    k = np.exp(-np.arange(int(0.004 * SR)) / (0.0006 * SR)).astype(np.float32)
    x = np.convolve(clicks, k)[:n] * white(rng, n)
    return filt(x, "bandpass", [f_lo, f_hi]) * amp


def knock(rng, f=420, dur=0.12, q=0.6):
    """木頭敲擊：短促的共鳴脈衝。"""
    n = int(dur * SR)
    ex = white(rng, n) * env_ad(n, 0.0005, 0.004)
    lo, hi = f * (1 - q * 0.3), f * (1 + q * 0.3)
    return filt(ex, "bandpass", [lo, hi], 2) * 6 * env_ad(n, 0.0005, dur * 0.4)


def glass(rng, f, dur=0.35):
    n = int(dur * SR)
    partials = [1, 2.76, 5.4]
    x = sum(np.sin(2 * np.pi * f * p * np.arange(n) / SR + rng.uniform(0, 6)) / (i + 1)
            for i, p in enumerate(partials))
    return (x * env_ad(n, 0.001, dur * 0.25)).astype(np.float32)


def bell(rng, f, dur=6.0):
    n = int(dur * SR)
    ratios = [0.5, 1, 1.19, 1.56, 2.0, 2.74, 3.76]
    amps = [0.5, 1, 0.5, 0.35, 0.25, 0.18, 0.1]
    t = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    for r, a in zip(ratios, amps):
        x += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (dur * 0.35 / r ** 0.5))
    return (x * np.minimum(t / 0.004, 1)).astype(np.float32)


def swish(rng, dur, f0, f1, width=0.5):
    n = int(dur * SR)
    fc = np.geomspace(f0, f1, n)
    out = np.empty(n, np.float32)
    zi = None
    w = white(rng, n)
    for i in range(0, n, 512):
        f = fc[i]
        s = butter(2, [f * (1 - width / 2), f * (1 + width / 2)], btype="bandpass", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((s.shape[0], 2))
        out[i:i + 512], zi = sosfilt(s, w[i:i + 512], zi=zi)
    return out * np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5


def pad(rng, freqs, dur, attack=2.0, release=2.0, bright=0.3):
    n = int(dur * SR)
    x = np.zeros(n, np.float32)
    for f in freqs:
        for h, a in ((1, 1.0), (2, bright), (3, bright * 0.5)):
            x += a * tone([f * h], dur, detune=0.0015, rng=rng)
    t = np.arange(n) / SR
    e = np.minimum(t / attack, 1) * np.minimum((dur - t) / release, 1)
    return (x * np.clip(e, 0, 1) / len(freqs)).astype(np.float32)


def reverb_ir(rng, seconds, damp=5000):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    decay = np.exp(-t / (seconds / 6.9))
    ir = np.stack([filt(white(rng, n), "lowpass", damp) * decay, filt(white(rng, n), "lowpass", damp) * decay], 1)
    ir[: int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))[:, None]
    return (ir / np.sqrt((ir ** 2).sum(0))).astype(np.float32)


# ── 依場景寫聲音 ───────────────────────────────────────────
def build(tl):
    from .scene_end import EndScene
    from .scene_field import FieldScene
    from .scene_opening import OpeningScene

    c = tl.c
    dur = tl.duration
    rng = rand(tl.seed, 99, 1)
    dry = Bus(dur)
    wet = Bus(dur)   # 送進殘響

    def both(x, t0, g, pan=0.0, send=0.35):
        dry.add(x, t0, g, pan)
        wet.add(x, t0, g * send, pan)

    op = OpeningScene(tl)
    field = FieldScene(tl)
    end = EndScene(tl)

    # ── 第 1 段：充實的黑 ──
    n = int(c("crack") * SR)
    press = filt(brown(rng, n), "lowpass", 90) * 2.2
    press *= curve([(0, 0), (3, 0.8), (c("crack") - 1.6, 1.0), (c("crack") - 0.35, 0.0), (c("crack"), 0.0)], n)
    both(press, 0, 0.42, 0.0, 0.2)
    grain = filt(white(rng, n), "highpass", 5000) * 0.012
    grain *= curve([(0, 0), (3, 1), (c("crack") - 1.5, 1), (c("crack") - 0.4, 0), (c("crack"), 0)], n)
    both(grain, 0, 1.0, 0.0, 0.1)
    cr = crackle(rng, c("crack") - c("crack_hairline"),
                 np.linspace(1.5, 14, int((c("crack") - c("crack_hairline")) * SR)), 600, 4000, 0.5)
    cr *= curve([(0, 0.2), (c("crack") - c("crack_hairline") - 1.3, 1), (c("crack") - c("crack_hairline") - 0.5, 0),
                 (99, 0)], len(cr))
    both(cr, c("crack_hairline"), 0.8, 0.0, 0.5)
    # 裂開：撕裂與一陣往外洩的氣流
    burst = crackle(rng, 0.6, 900, 300, 9000, 1.0) * env_ad(int(0.6 * SR), 0.002, 0.18)
    both(burst, c("crack"), 2.6, 0.0, 0.6)
    both(thud(rng, 70, 34, 1.4, 0.8, 0.6), c("crack"), 1.8, 0.0, 0.5)
    rush_n = int(3.2 * SR)
    rush = sweep_filter(pink(rng, rush_n), "lowpass", np.geomspace(500, 9000, rush_n)) * 0.25
    rush *= curve([(0, 0), (0.15, 1), (1.6, 0.8), (3.2, 0)], rush_n)
    dry.add(rush, c("crack"), 1.0, np.sin(np.linspace(0, 3, rush_n)) * 0.6)
    wet.add(rush, c("crack"), 0.5)

    # ── 第 2 段：死與生 ──
    t0 = c("leaf_in")
    shimmer = pad(rng, [1318.5, 1975.5, 2960.0], c("mud_hang") - t0, 0.8, 2.5, 0.1) * 0.12
    shimmer *= curve([(0, 1), (2.5, 1), (5.5, 0.15), (99, 0.1)], len(shimmer))
    both(shimmer, t0, 1.0, -0.55, 0.8)
    for i in range(len(op.segs)):
        die = float(op.die[i])
        x = op.lcx + op.segs[i, 0]
        if rng.random() < 0.35:
            both(glass(rng, rng.uniform(2400, 4800), 0.25), die + 0.4, 0.05, xpan(x), 0.6)
    # 右邊的殘影：一口一口的氣息
    for k in range(10):
        tt = t0 + 1.4 + k * 0.33 + rng.uniform(0, 0.2)
        bn = int(0.9 * SR)
        both(filt(white(rng, bn), "bandpass", [700, 2400]) * np.hanning(bn) * 0.06, tt, 1.0, 0.6, 0.8)
    mold = crackle(rng, c("mud_hang") - c("decay"), 40, 1500, 8000, 0.18)
    mold *= curve([(0, 0), (1.5, 1), (c("mud_hang") - c("decay") - 1, 1), (99, 0)], len(mold))
    both(mold, c("decay"), 1.0, -0.4, 0.5)
    warm = pad(rng, [146.8, 220.0], c("mud_land") - c("decay"), 3.0, 1.0, 0.25) * 0.16
    both(warm, c("decay"), 1.0, 0.0, 0.6)

    # ── 第 3 段：泥 ──
    hang = c("mud_drop") - c("mud_hang")
    hn = int(hang * SR)
    tt = np.arange(hn) / SR
    goo = np.sin(2 * np.pi * np.cumsum(70 + 18 * np.sin(2 * np.pi * 0.35 * tt) - 20 * tt / hang) / SR)
    goo = filt(goo.astype(np.float32) + 0.3 * filt(brown(rng, hn), "lowpass", 200), "lowpass", 260)
    both(goo * curve([(0, 0), (1.5, 0.5), (hang, 1.0)], hn) * 0.35, c("mud_hang"), 1.0, 0.0, 0.3)
    glop = glide(320, 70, 0.35, 0.6) * env_ad(int(0.35 * SR), 0.002, 0.09)
    both(glop, c("mud_drop"), 0.5, 0.0, 0.5)
    fall = c("mud_land") - c("mud_drop")
    both(swish(rng, fall, 150, 900, 0.8) * 0.5, c("mud_drop"), 1.0, 0.0, 0.3)
    both(thud(rng, 62, 26, 2.0, 1.4, 1.0), c("mud_land"), 1.0, 0.0, 0.5)
    both(filt(white(rng, int(0.5 * SR)), "bandpass", [300, 2500]) * env_ad(int(0.5 * SR), 0.002, 0.07) * 0.6,
         c("mud_land") + 0.02, 1.0, 0.0, 0.5)

    # ── 第 4 段：樹，與字草 ──
    for k in range(26):
        tt = c("mud_land") + 0.3 + k * (c("tree_collapse") - c("mud_land") - 0.3) / 26 + rng.uniform(0, 0.08)
        f = rng.uniform(180, 380) * (0.75 if tt > c("tree_peak") else 1.0)
        both(knock(rng, f, 0.25, 0.25) * 0.5, tt, 1.0, rng.uniform(-0.15, 0.15), 0.4)
    for i in range(len(field.tree)):
        tland = c("tree_collapse") + field.stroke_fall_delay[i] + 0.55
        if rng.random() < 0.6:
            both(knock(rng, rng.uniform(500, 1400), 0.08, 0.5) * 0.35, tland, 1.0,
                 xpan(field.stroke_rest[i][0]), 0.3)
    for p in field.plants:
        both(swish(rng, 0.7, 2500, 700, 0.6) * 0.18, p.t_rise - 0.6, 1.0, xpan(p.x), 0.4)
    # 風：跟著畫面上的陣風（強度與位置）
    w0, w1 = c("grass_rise"), c("fire_rise") + 2
    cr_t = np.arange(w0, w1, 0.02)
    xs = np.linspace(100, 1820, 9)
    amp, pan = [], []
    for t in cr_t:
        v = np.array([abs(float(field.wind(t, x))) for x in xs])
        amp.append(v.mean())
        pan.append(float(((xs - 960) / 960 * v).sum() / (v.sum() + 1e-6)))
    wn = int((w1 - w0) * SR)
    ts = np.arange(wn) / SR
    a_curve = np.interp(ts, cr_t - w0, amp).astype(np.float32)
    p_curve = np.interp(ts, cr_t - w0, pan).astype(np.float32)
    wind = sweep_filter(pink(rng, wn), "lowpass", 400 + 2600 * np.clip(a_curve, 0, 1)) * 0.5
    wind *= np.clip(a_curve * 1.6, 0, 1.2) * curve([(0, 0), (3, 1), (w1 - w0 - 4, 1), (w1 - w0, 0)], wn)
    both(wind, w0, 3.0, p_curve, 0.3)
    # 大地的低音：從泥落地一直到地火
    dn = c("fire_rise") - c("mud_land")
    drone = pad(rng, [36.7, 55.0], dn, 4.0, 3.0, 0.5) * 0.35
    both(drone, c("mud_land"), 1.0, 0.0, 0.3)
    # 地底的脈搏（血）
    for b in tl.beats_between(c("blood"), c("breath")):
        k = ramp(b, c("blood"), c("stomp_start"))
        both(thud(rng, 48, 38, 0.35, 0.8, 0.0), b, 0.25 + 0.3 * k, 0.0, 0.2)
        both(thud(rng, 44, 36, 0.3, 0.6, 0.0), b + 0.22, 0.15 + 0.2 * k, 0.0, 0.2)
    for nm in field.names:
        bn = int(1.4 * SR)
        breath = filt(white(rng, bn), "bandpass", [500, 1800]) * np.hanning(bn) * 0.05
        both(breath, nm["t"], 1.0, xpan(nm["x"]), 0.9)
    # 踐踏與刪刈
    for ev in field.events:
        if ev["kind"] == "stomp":
            both(swish(rng, 0.22, 200, 900, 1.0) * 0.4, ev["t"] - 0.22, 1.0, xpan(ev["xc"]), 0.2)
            both(thud(rng, 64, 30, 1.1, 1.6, 1.2), ev["t"], 1.0, xpan(ev["xc"]) * 0.5, 0.45)
            dn = int(1.2 * SR)
            both(filt(white(rng, dn), "bandpass", [400, 3000]) * env_ad(dn, 0.01, 0.3) * 0.12, ev["t"] + 0.02,
                 1.0, xpan(ev["xc"]), 0.4)
        else:
            sn = int(ev["dur"] * SR * 1.6)
            sw = swish(rng, ev["dur"] * 1.6, 1500, 7000, 0.4) * 0.45
            pan_c = np.linspace(-0.9, 0.9, sn) * ev["dir"]
            both(sw, ev["t"] - ev["dur"] * 0.8, 1.0, pan_c, 0.3)
            ring = tone([1230, 2970, 4410], 0.6, [1, 0.6, 0.35], rng=rng) * env_ad(int(0.6 * SR), 0.001, 0.12)
            both(ring * 0.08, ev["t"], 1.0, 0.0, 0.6)

    # ── 第 5、6 段：坦然；自愛；網格 ──
    calm0, calm1 = c("breath"), c("selflove") - 0.3
    open5 = pad(rng, [73.4, 110.0, 146.8, 220.0, 329.6], calm1 - calm0 + 2, 2.5, 2.5, 0.35) * 0.32
    on = len(open5)
    tt = np.arange(on) / SR + calm0
    # 「大笑」「歌唱」：隨拍起伏
    phase = np.array([tl.beat_phase(t)[1] for t in tt[::480]])
    swell = np.interp(np.arange(on), np.arange(0, on, 480)[: len(phase)], 0.5 - 0.5 * np.cos(2 * np.pi * phase))
    k_sing = np.clip((tt - c("sing")) / 1.5, 0, 1)
    open5 *= (1 - 0.6 * k_sing + 0.6 * k_sing * swell).astype(np.float32)
    both(open5, calm0, 1.0, 0.0, 0.7)
    twinkle = crackle(rng, calm1 - calm0, 6, 5000, 11000, 0.08)
    both(twinkle, calm0, 1.0, 0.0, 0.9)
    held = pad(rng, [220.0, 440.0], 5.2, 0.6, 1.4, 0.2) * 0.22     # 自愛：停 4 秒，只有一個音
    both(held, c("selflove"), 1.0, 0.0, 0.8)
    revealed = sorted({round(cell["t_in"], 3) for cell in field.cells if "t_in" in cell})
    for tt_ in revealed:
        both(knock(rng, 2400, 0.03, 0.2) * 1.4, tt_, 1.0, 0.0, 0.15)
    for p in field.plants:
        if p.absorb_t is not None:
            px = xpan(960 + 0.42 * (p.cell["cx"] - 960))
            both(knock(rng, 1800, 0.04, 0.2) * 1.2, p.absorb_t + 1.1, 1.0, px, 0.15)
            both(knock(rng, 140, 0.15, 0.4) * 0.9, p.absorb_t + 1.1, 1.0, px, 0.15)
    for b in tl.beats_between(c("grid_in"), c("dim")):
        both(knock(rng, 3200, 0.02, 0.1) * 0.8, b, 1.0, 0.0, 0.1)

    # ── 第 7 段：地火 ──
    rn = int((c("dim") - c("fire_rise")) * SR)
    rumble = filt(brown(rng, rn), "lowpass", 110) * 2.5
    rumble *= curve([(0, 0), (c("dim") - c("fire_rise"), 1)], rn) ** 1.5
    both(rumble, c("fire_rise"), 1.0, 0.0, 0.3)
    pops = crackle(rng, c("dim") - c("fire_rise"), np.linspace(1, 30, rn), 300, 3000, 0.4)
    both(pops, c("fire_rise"), 1.0, 0.0, 0.4)
    both(thud(rng, 50, 40, 0.5, 1.0, 0.0), 122.12, 0.7, 0.0, 0.3)       # 暗下去之後的一下心跳
    inh = int((c("erupt") - 122.6) * SR)
    both(sweep_filter(pink(rng, inh), "lowpass", np.geomspace(200, 6000, inh)) * np.linspace(0, 1, inh) ** 2 * 0.35,
         122.6, 1.0, 0.0, 0.2)
    both(thud(rng, 70, 22, 4.0, 2.2, 2.0), c("erupt"), 1.0, 0.0, 0.7)
    blast = filt(white(rng, int(2.5 * SR)), "lowpass", 3000) * env_ad(int(2.5 * SR), 0.003, 0.5)
    both(blast * 0.8, c("erupt"), 1.0, 0.0, 0.7)
    # 轟鳴：凍結之後仍然轟鳴，直到靜穆
    ro0, ro1 = c("erupt"), c("witness") + 0.5
    rn = int((ro1 - ro0) * SR)
    roarL = filt(brown(rng, rn), "lowpass", 240) * 2.2 + filt(pink(rng, rn), "lowpass", 1400) * 0.25
    roarR = filt(brown(rng, rn), "lowpass", 240) * 2.2 + filt(pink(rng, rn), "lowpass", 1400) * 0.25
    lvl = curve([(0, 0.0), (0.15, 1.0), (6, 0.85), (c("freeze") - ro0, 0.75), (c("hush") - ro0, 0.75),
                 (c("hush") - ro0 + 1.5, 0.03), (c("restore") - ro0 - 0.01, 0.03), (c("restore") - ro0, 1.0),
                 (c("witness") - ro0, 0.6), (ro1 - ro0, 0.0)], rn)
    sub_lvl = curve([(0, 0), (0.2, 1), (c("hush") - ro0, 0.9), (c("hush") - ro0 + 1.5, 0.09),
                     (c("restore") - ro0 - 0.01, 0.09), (c("restore") - ro0, 1.0), (ro1 - ro0, 0.6)], rn)
    sub = filt(brown(rng, rn), "lowpass", 60) * 3.0 * sub_lvl
    # 凍結之後：轟鳴慢慢變暗，底下浮出一個持續的低和弦
    dark = curve([(0, 1400), (c("freeze") - ro0, 1400), (c("hush") - ro0, 420), (c("restore") - ro0 - 0.01, 420),
                  (c("restore") - ro0, 2600), (ro1 - ro0, 900)], rn)
    roarL = sweep_filter(roarL, "lowpass", dark)
    roarR = sweep_filter(roarR, "lowpass", dark)
    frozen = pad(rng, [36.7, 55.0, 73.4], c("hush") - c("freeze") + 0.8, 3.0, 0.8, 0.4) * 0.5
    both(frozen, c("freeze"), 1.0, 0.0, 0.5)
    fire = crackle(rng, ro1 - ro0, 60, 600, 6000, 0.5) * lvl
    dry.add(roarL * lvl, ro0, 1.0, -0.8)
    dry.add(roarR * lvl, ro0, 1.0, 0.8)
    wet.add((roarL + roarR) * lvl, ro0, 0.15)
    both(sub, ro0, 1.0, 0.0, 0.1)
    both(fire, ro0, 1.0, 0.0, 0.4)
    hiss = filt(white(rng, rn), "highpass", 6000) * 0.03 * curve(
        [(0, 0), (0.5, 1), (c("freeze") - ro0, 1), (c("freeze") - ro0 + 0.01, 0), (99, 0)], rn)
    both(hiss, ro0, 1.0, 0.0, 0.2)
    # 靜穆：只剩地底低頻，隨畫面緩慢呼吸
    hn = int((c("restore") - c("hush")) * SR)
    t_h = np.arange(hn) / SR
    breathe = 0.5 + 0.5 * np.sin(2 * np.pi * t_h / 7.0 - np.pi / 2)
    both(filt(brown(rng, hn), "lowpass", 70) * 0.55 * (0.4 + 0.6 * breathe), c("hush"), 1.0, 0.0, 0.2)
    # 2:38 天地即不如此靜穆
    both(thud(rng, 66, 28, 2.0, 1.6, 1.5), c("restore"), 1.0, 0.0, 0.6)
    churn = int(4.3 * SR)
    both(sweep_filter(pink(rng, churn), "lowpass",
                      2000 + 1500 * np.sin(np.linspace(0, 18, churn))) * 0.3 * env_ad(churn, 0.05, 2.0),
         c("restore"), 1.0, np.sin(np.linspace(0, 9, churn)) * 0.8, 0.4)

    # ── 第 10 段：作證 ──
    gn = int(2.4 * SR)
    gather = sweep_filter(pink(rng, gn), "lowpass", np.geomspace(8000, 600, gn)) * np.linspace(0.2, 1, gn) * 0.12
    both(gather, c("witness"), 1.0, 0.0, 0.6)
    both(bell(rng, 49.0, 8.0) * 0.45, c("witness") + 0.3, 1.0, 0.0, 0.6)          # 銘文刻出
    for i, b in enumerate(field.w_beats):
        ang = field.w_angles[i]
        both(bell(rng, 98.0 * (1.0 if i % 2 == 0 else 0.749), 3.0) * 0.28, b, 1.0, float(np.cos(ang)) * 0.4, 0.6)
        sl = int(0.12 * SR)
        both(filt(white(rng, sl), "highpass", 2500) * env_ad(sl, 0.001, 0.03) * 0.25, b, 1.0, 0.0, 0.3)
    for k, pr in enumerate(field.pairs):
        base = [196.0, 174.6, 220.0][k]
        other = base * ([1.5, 1.414, 1.0][k])       # 友仇：五度；人獸：三全音；愛／不愛：同一個音
        both(pad(rng, [base], 4.5, 0.4, 3.0, 0.2) * 0.16, pr["t"], 1.0, -0.7, 0.8)
        detune = 1.0 if k < 2 else 1.012
        both(pad(rng, [other * detune], 4.5, 0.4, 3.0, 0.2) * 0.16, pr["t"] + 0.1, 1.0, 0.7, 0.8)

    # ── 第 11 段：火速 ──
    for k, (tc, d) in enumerate(end.cycles):
        if tc + d * 0.3 < c("cycle"):
            continue
        g = min(1.0, 0.35 + k * 0.06)
        both(glide(300, 600 + 40 * k, max(d * 0.3, 0.05)) * env_ad(int(max(d * 0.3, 0.05) * SR), 0.005, d * 0.2) * 0.1 * g,
             tc, 1.0, -0.3, 0.4)
        rd = max(d * 0.5, 0.04)
        both(crackle(rng, rd, 150, 1200, 8000, 0.35) * np.linspace(1, 0, int(rd * SR)), tc + d * 0.5, g, 0.3, 0.4)
        both(knock(rng, 2600, 0.03, 0.2) * 0.12, tc, 1.0, 0.8, 0.3)                 # 角落的年份
    cyc_n = int((c("stop") - c("cycle")) * SR)
    rise = np.sin(2 * np.pi * np.cumsum(np.geomspace(55, 110, cyc_n)) / SR).astype(np.float32)
    both(rise * np.linspace(0.05, 0.3, cyc_n), c("cycle"), 1.0, 0.0, 0.3)

    # ── 第 12 段：去罷 ──
    both(pad(rng, [73.4, 110.0, 185.0, 277.2], c("silence") - c("farewell") - 0.5, 3.0, 4.0, 0.2) * 0.22,
         c("farewell"), 1.0, 0.0, 0.7)
    wn = int(10 * SR)
    gale = sweep_filter(pink(rng, wn), "lowpass", 600 + 2500 * np.sin(np.linspace(0, np.pi, wn))) * 0.4
    gale *= np.sin(np.linspace(0, np.pi, wn)) ** 1.5
    both(gale, c("farewell") + 1.4, 1.0, np.linspace(0.9, -0.9, wn), 0.4)
    for i, (ch, x, y, size) in enumerate(end.poem):
        if rng.random() < 0.45:
            rn_ = int(0.18 * SR)
            rustle = filt(white(rng, rn_), "bandpass", [2000, 7000]) * env_ad(rn_, 0.005, 0.05) * 0.05
            both(rustle, float(end.p_rel[i]), 1.0, xpan(x), 0.4)
    both(bell(rng, 65.4, 9.0) * 0.4, c("colophon"), 1.0, 0.0, 0.7)              # 落款
    en = int((c("silence") - c("earth")) * SR)
    hum = tone([55.0, 82.4, 110.0], c("silence") - c("earth"), [1, 0.4, 0.2], detune=0.002, rng=rng)
    both(hum * curve([(0, 0), (2.5, 0.25), (c("silence") - c("earth") - 2.5, 0.2), (c("silence") - c("earth"), 0)], en),
         c("earth"), 1.0, 0.0, 0.5)
    return dry, wet, field, end


def insert_sound(name, dur, shot, tl, rng):
    """每個意象自己的聲音（意象鏡頭切進來時，主聲軌壓低，這個聲音浮上來）。"""
    n = max(int(dur * SR), 64)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    kw = shot.kw
    if name == "iron_house":
        out += tone([55.0, 83.2, 131.0, 197.5], dur, [1, 0.6, 0.4, 0.25], detune=0.004, rng=rng) * 0.18
        out += filt(white(rng, n), "bandpass", [200, 900]) * 0.03
        if shot.p[1] > 0.8:
            k = int(max(0.0, (0.85 - shot.p[0]) / max(shot.p[1] - shot.p[0], 1e-3)) * n)
            cr = crackle(rng, dur, 900, 400, 8000, 1.0) * env_ad(n, 0.001, 0.15)
            out[k:] += cr[: n - k] * 0.8
    elif name == "mouth":
        out += sweep_filter(white(rng, n), "lowpass", np.geomspace(600, 3500, n)) * np.linspace(0.02, 0.18, n)
    elif name == "candle":
        out += filt(white(rng, n), "lowpass", 500) * (0.5 + 0.5 * np.sin(2 * np.pi * 9 * t)) * 0.08
        k = int(n * 0.55)
        out[k:] += filt(white(rng, n - k), "bandpass", [300, 1500]) * env_ad(n - k, 0.005, 0.08) * 0.3
    elif name == "grass_crack":
        out += crackle(rng, dur, 70, 1500, 9000, 0.35)
        for k in range(int(dur * 6) + 1):
            kk = knock(rng, rng.uniform(250, 600), 0.15, 0.3) * 0.25
            i0 = int(rng.uniform(0, max(n - len(kk), 1)))
            out[i0:i0 + len(kk)] += kk[: n - i0]
    elif name == "roots_bones":
        out += filt(brown(rng, n), "lowpass", 300) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) * 0.8
        out[: min(n, int(0.35 * SR))] += thud(rng, 48, 38, 0.35, 0.8, 0.0)[: min(n, int(0.35 * SR))] * 0.6
    elif name == "boots":
        th = thud(rng, 70, 40, 0.5, 1.4, 1.6)
        i0 = max(0, n - int(0.04 * SR))   # 落地在剪回來的那一刻附近
        i0 = min(i0, n - 1)
        out[i0:] += th[: n - i0]
        out += filt(white(rng, n), "bandpass", [800, 4000]) * env_ad(n, 0.001, 0.05) * 0.2
    elif name == "onlookers":
        mur = sum(filt(white(rng, n), "bandpass", [f * 0.85, f * 1.15]) * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 5) * t + rng.uniform(0, 6)))
                  for f in (320, 520, 760))
        out += mur * 0.12
    elif name == "crowd":
        v = kw.get("variant", "surge")
        if v == "empty":
            out += sweep_filter(pink(rng, n), "lowpass", 300 + 400 * np.sin(np.linspace(0, 3, n)) ** 2) * 0.1
        else:
            syl = 4.5 if v == "surge" else 0.0
            roar = sum(filt(pink(rng, n), "bandpass", [f * 0.7, f * 1.3]) for f in (420, 900, 1800))
            am = 0.6 + 0.4 * np.sin(2 * np.pi * syl * t) if syl else 0.25
            out += roar * am * (1.8 if v == "surge" else 0.8)
            # 口號：落在拍上的齊聲吶喊（方陣是機械的整齊；人潮是參差的）
            for b in tl.beats_between(shot.t0 - 0.3, shot.t1):
                i0 = int((b - shot.t0) * SR)
                for j in range(1 if v == "formation" else 3):
                    jit = 0 if v == "formation" else int(rng.uniform(-0.04, 0.04) * SR)
                    ch = filt(white(rng, int(0.22 * SR)), "bandpass", [500, 1600]) * env_ad(int(0.22 * SR), 0.01, 0.07)
                    k0 = i0 + jit
                    if 0 <= k0 < n:
                        out[k0:k0 + len(ch)] += ch[: n - k0] * (3.2 if v == "formation" else 1.8)
                if v == "formation" and 0 <= i0 < n:
                    dr = thud(rng, 90, 60, 0.25, 1.0, 0.4)
                    out[i0:i0 + len(dr)] += dr[: n - i0] * 1.3
    elif name == "gate" and shot.p[1] <= 0.05:
        out += bell(rng, 65.4, max(dur, 0.5))[:n] * 0.3     # 還立著的牌坊：一聲低沉的鐘
    elif name == "gate":
        out += filt(brown(rng, n), "lowpass", 180) * 1.6 * np.clip(t / max(dur, 1e-3) * 2, 0, 1)
        out += crackle(rng, dur, 120, 200, 4000, 0.7)
        if shot.p[0] < 0.6 < shot.p[1] or shot.p[0] > 0.6:
            th = thud(rng, 60, 24, min(dur, 2.0), 1.6, 1.4)
            out[: len(th)] += th[:n]
    elif name == "calendar":
        out += swish(rng, dur, 2000, 7000, 0.5) * 0.5
        out += crackle(rng, dur, 400, 2500, 9000, 0.25)
    elif name == "bars":
        out += np.sin(2 * np.pi * 6200 * t).astype(np.float32) * 0.004
    elif name == "lamp_window":
        for k in range(int(dur * 3) + 1):
            c0 = int(rng.uniform(0, n))
            chirp = (np.sin(2 * np.pi * 4600 * np.arange(int(0.12 * SR)) / SR) * (np.sin(2 * np.pi * 30 * np.arange(int(0.12 * SR)) / SR) > 0)).astype(np.float32)
            out[c0:c0 + len(chirp)] += chirp[: n - c0] * 0.02
        out += crackle(rng, dur, 30, 3000, 9000, 0.08)
    f = min(int(0.004 * SR), n // 2)
    out[:f] *= np.linspace(0, 1, f)
    out[-f:] *= np.linspace(1, 0, f)
    return out


def apply_edit(mix, edit, rng):
    """聲音跟著剪輯：閃回帶著原處的聲音、蒙太奇的剪接點有一聲切音、黑畫面的靜音。"""
    src = mix.copy()
    n = len(mix)
    fade = int(0.004 * SR)
    for s in edit.shots:
        i0, i1 = int(s.t0 * SR), int(s.t1 * SR)
        if s.kind == "flash":
            # 那一刻的聲音碎片，疊進現在
            j0 = int(edit.recall[s.src] * SR)
            L = min(i1 - i0 + fade, n - j0, n - i0)
            g = np.ones(L, np.float32)
            g[:fade] = np.linspace(0, 1, fade)
            g[-fade:] = np.linspace(1, 0, fade)
            k = 0.5 if s.overlay else 1.0
            mix[i0:i0 + L] = mix[i0:i0 + L] * (1 - 0.6 * k * g[:, None]) + src[j0:j0 + L] * (k * g[:, None])
        if s.kind == "insert":
            # 切到另一個世界：主聲軌壓低，意象自己的聲音浮上來
            L = min(i1, n) - i0
            if L > 0:
                deep = {"bars": 0.12, "lamp_window": 0.3, "iron_house": 0.35, "mouth": 0.4}
                if s.insert == "crowd" and s.kw.get("variant") == "empty":
                    deep_k = 0.1
                else:
                    deep_k = deep.get(s.insert, 0.5)
                duck = deep_k if (s.t1 - s.t0) > 0.15 else 0.5 + 0.5 * deep_k
                g = np.full(L, duck, np.float32)
                g[:fade] = np.linspace(1, duck, min(fade, L))[: min(fade, L)] if L > fade else duck
                mix[i0:i0 + L] *= g[:, None]
                sig = insert_sound(s.insert, s.t1 - s.t0, s, edit.tl, rng)[:L]
                mix[i0:i0 + len(sig)] += sig[:, None]
        if s.tick:
            cl = knock(rng, 3400, 0.025, 0.15) * 0.22
            cl[: int(0.012 * SR)] += filt(white(rng, int(0.012 * SR)), "highpass", 6000) * 0.05
            mix[i0:i0 + len(cl)] += cl[: n - i0, None]
    gate = np.ones(n, np.float32)
    for s in edit.shots:
        if s.mute:
            i0, i1 = int(s.t0 * SR), min(int(s.t1 * SR), n)
            gate[i0:i1] = 0
            gate[max(i0 - fade, 0):i0] = np.linspace(1, 0, i0 - max(i0 - fade, 0))
    return mix * gate[:, None]


def write(tl, path):
    import wave
    from .edit import Edit
    dry, wet, field, end = build(tl)
    edit = Edit(tl, field, end)
    rng = rand(tl.seed, 99, 2)
    ir = reverb_ir(rng, tl.raw["sound"]["reverb_seconds"])
    W = wet.stereo()
    rev = np.stack([oaconvolve(W[:, 0], ir[:, 0])[: len(W)], oaconvolve(W[:, 1], ir[:, 1])[: len(W)]], 1)
    mix = dry.stereo() + rev * 0.9
    c = tl.c
    mix = apply_edit(mix, edit, rng)
    # 3:01 驟停：連殘響也一起切掉，真正的寂靜；去罷時再回來
    t = np.arange(len(mix)) / SR
    gate = np.interp(t, [c("stop"), c("stop") + 0.004, c("farewell"), c("farewell") + 0.5,
                         c("silence") + 0.3, c("silence") + 0.5], [1, 0, 0, 1, 1, 0]).astype(np.float32)
    mix *= gate[:, None]
    mix = np.stack([filt(mix[:, 0], "highpass", 22), filt(mix[:, 1], "highpass", 22)], 1)
    n = int(tl.duration * SR)
    mix = mix[:n]
    peak = np.abs(mix).max()
    mix *= 10 ** (tl.raw["sound"]["master_db"] / 20) / max(peak, 1e-6)
    pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
