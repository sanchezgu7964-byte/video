"""Synthesised soundtrack. Every layer is a function (tl) -> stereo float array
covering the whole film; render_mix sums them and applies a soft ceiling.

All filtering is done in the frequency domain with numpy (no scipy needed).
"""
import os
import wave

import numpy as np


def db(x):
    return 10 ** (x / 20.0)


def fft_filter(x, sr, lo=None, hi=None, order=2):
    """Zero-phase Butterworth-shaped band filter (magnitude only)."""
    n = len(x)
    N = 1 << int(np.ceil(np.log2(n + 1)))
    X = np.fft.rfft(x, N)
    f = np.fft.rfftfreq(N, 1 / sr)
    H = np.ones_like(f)
    if hi:
        H /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        H /= np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X * H, N)[:n]


def convolve(x, ir):
    n = len(x) + len(ir) - 1
    N = 1 << int(np.ceil(np.log2(n)))
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:len(x)]


def reverb_ir(sr, seconds, seed, damp_hz=4000, predelay=0.02):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-t * 6.9 / seconds)
    ir = fft_filter(ir, sr, lo=120, hi=damp_hz, order=1)
    ir = np.concatenate([np.zeros(int(predelay * sr)), ir])
    return ir / np.sqrt(np.sum(ir ** 2))


def reverb(x, sr, seconds, seed, wet=0.3, damp_hz=4000):
    """Mono in -> stereo out (decorrelated left/right tails)."""
    l = convolve(x, reverb_ir(sr, seconds, seed, damp_hz))
    r = convolve(x, reverb_ir(sr, seconds, seed + 1, damp_hz))
    return np.stack([x + wet * l, x + wet * r], 1)


def pan(x, p):
    """Constant-power pan; p in -1..1 (scalar or per-sample array)."""
    a = (np.asarray(p) + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1)


def env_lin(t, points):
    """Piecewise-linear envelope through (time, value) points."""
    ts, vs = zip(*points)
    return np.interp(t, ts, vs)


def place(buf, sample, t, sr, gain=1.0):
    i = int(round(t * sr))
    if i >= len(buf) or i + len(sample) <= 0:
        return
    a, b = max(0, i), min(len(buf), i + len(sample))
    buf[a:b] += gain * sample[a - i:b - i]


# --------------------------------------------------------------------------
# train sounds
# --------------------------------------------------------------------------
def clack_sample(rng, sr):
    """Wheel over a rail joint: a hard click, a short metallic ring and a thump."""
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    click = fft_filter(rng.standard_normal(n) * np.exp(-t / 0.0035), sr, lo=1200, hi=7000)
    ring = np.zeros(n)
    for f, tau, a in ((760, 0.06, 0.5), (1310, 0.045, 0.45), (2230, 0.03, 0.35),
                      (3480, 0.02, 0.25), (5150, 0.012, 0.2)):
        f *= 1 + rng.normal(0, 0.025)
        ring += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * np.exp(-t / tau)
    thump = np.sin(2 * np.pi * 58 * t * (1 - 0.3 * t)) * np.exp(-t / 0.05)
    body = fft_filter(rng.standard_normal(n) * np.exp(-t / 0.03), sr, lo=80, hi=600)
    s = 1.0 * click / np.abs(click).max() + 0.22 * ring + 0.9 * thump + 0.5 * body / np.abs(body).max()
    s *= np.minimum(1, t / 0.0005)
    return s / np.abs(s).max()


def whistle(sr, dur, seed):
    """Distant two-tone steam whistle with vibrato and a sagging release."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = env_lin(t, [(0, 0), (0.35, 1), (dur * 0.62, 0.9), (dur, 0)]) ** 1.3
    sag = 1 - 0.035 * np.clip((t - dur * 0.6) / (dur * 0.4), 0, 1) ** 2
    out = np.zeros(n)
    for f0, a in ((311.1, 1.0), (392.0, 0.8), (466.2, 0.35)):
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6)) + 0.002 * rng.standard_normal()
        ph = 2 * np.pi * np.cumsum(f0 * vib * sag) / sr
        for h in range(1, 7):
            out += a * (0.55 ** (h - 1)) * np.sin(h * ph + rng.uniform(0, 6))
    breath = fft_filter(rng.standard_normal(n), sr, lo=600, hi=2500) * 0.25
    out = (out / np.abs(out).max() + breath) * env
    return fft_filter(out, sr, lo=150, hi=1800, order=2)       # distance = dull


def seg_time(tl, name, edge="start", offset=0.0):
    t = tl.seg_start(name) if edge == "start" else tl.seg_end(name)
    return t + float(offset)


def envelope(tl, t, points):
    """points: [[segment, 'start'|'end', offset_s, value], ...] -> per-sample envelope."""
    pts = sorted((seg_time(tl, s, e, o), v) for s, e, o, v in points)
    return env_lin(t, [(-1, pts[0][1])] + pts + [(1e9, pts[-1][1])])


def clack_bank(tl, sr):
    return [clack_sample(np.random.default_rng([tl.seed, 502, k]), sr) for k in range(8)]


def clacks(tl, N, sr, t0, t1, period, bogie, axle, rng, start_offset=0.25):
    """da-dum ... da-dum on a regular grid."""
    out = np.zeros(N)
    bank = clack_bank(tl, sr)
    k = 0
    while True:
        tk = t0 + start_offset + k * period
        if tk > t1:
            break
        for off, a in ((0, 1.0), (axle, 0.75), (bogie, 0.9), (bogie + axle, 0.7)):
            s = bank[rng.integers(len(bank))]
            place(out, s, tk + off + rng.normal(0, 0.004), sr, a * (1 + rng.normal(0, 0.12)))
        k += 1
    return out


def train_layer(tl, key):
    c = tl.audio.get(key, {})
    sr = tl.audio.get("sample_rate", 48000)
    N = int(round(tl.duration * sr))
    rng = np.random.default_rng([tl.seed, 501, len(key)])
    t0, t1 = tl.seg_start(c["from"]), tl.seg_end(c["to"])
    beat = tl.beat
    t = np.arange(N) / sr
    prox = envelope(tl, t, c["envelope"])
    p = env_lin(t, [(t0, c.get("pan_from", -0.6)), (t1, c.get("pan_to", 0.5))])

    clk = clacks(tl, N, sr, t0, t1, c.get("clack_period_beats", 2) * beat,
                 c.get("bogie_gap_beats", 1) * beat, float(c.get("axle_gap", 0.13)), rng)
    clk = fft_filter(clk, sr, hi=4500, order=1)

    # low metallic hum + brown-noise rumble
    hz = float(c.get("hum_hz", 49))
    hum = np.zeros(N)
    for h, a in ((1, 1.0), (2, 0.5), (3, 0.28), (4.03, 0.12), (6.1, 0.06)):
        hum += a * np.sin(2 * np.pi * hz * h * t + 0.3 * np.sin(2 * np.pi * 0.11 * h * t))
    hum *= 1 + 0.25 * np.sin(2 * np.pi * 0.23 * t)
    rum = np.cumsum(rng.standard_normal(N))
    rum = fft_filter(rum, sr, lo=25, hi=170, order=2)
    rum /= np.abs(rum).max() + 1e-9
    bogie = c.get("bogie_gap_beats", 1) * beat
    swell = 1 + 0.35 * np.exp(-((t - t0 - 0.25) % bogie) / 0.18)
    low = db(c.get("hum_db", -16)) * hum / np.abs(hum).max() + db(c.get("rumble_db", -15)) * 2.2 * rum * swell

    sq = np.zeros(N)
    if "squeal_at" in c:
        d = 2.4
        n = int(d * sr)
        ts = np.arange(n) / sr
        f = 3150 + 40 * np.sin(2 * np.pi * 5.5 * ts) + 120 * np.sin(2 * np.pi * 0.4 * ts)
        s = np.sin(2 * np.pi * np.cumsum(f) / sr) + 0.3 * np.sin(4 * np.pi * np.cumsum(f) / sr)
        s *= env_lin(ts, [(0, 0), (0.5, 1), (1.6, 0.7), (d, 0)]) * (0.6 + 0.4 * np.sin(2 * np.pi * 9 * ts) ** 2)
        place(sq, s, c["squeal_at"], sr, db(c.get("squeal_db", -30)))

    train = (clk * 0.55 + low) * prox + sq
    st = pan(train, p) * 0.7 + reverb(train, sr, 1.6, tl.seed + 7, wet=0.18, damp_hz=3500) * 0.3
    if "whistle_at" in c:
        wb = np.zeros(N)
        place(wb, whistle(sr, 2.8, tl.seed + 8), c["whistle_at"], sr, db(c.get("whistle_db", -13)))
        st += pan(wb, 0.35) * 0.45 + reverb(wb, sr, 3.8, tl.seed + 9, wet=0.9, damp_hz=1600) * 0.55
    return st * db(c.get("gain_db", -4))


def layer_train(tl):
    return train_layer(tl, "train")


# --------------------------------------------------------------------------
# 04: a long tone rising with the sun, cut dead by the blackout
# --------------------------------------------------------------------------
def layer_rise(tl):
    c = tl.audio.get("rise", {})
    sr = tl.audio.get("sample_rate", 48000)
    N = int(round(tl.duration * sr))
    out = np.zeros((N, 2))
    seg = c.get("segment", "last_judgment")
    t0, t1 = tl.seg_start(seg), tl.seg_end(seg)
    n = int((t1 - t0) * sr)
    t = np.arange(n) / sr
    u = t / (t1 - t0)
    f0, f1 = float(c.get("from_hz", 73.4)), float(c.get("to_hz", 293.7))
    f = f0 * (f1 / f0) ** (u ** 1.6)
    rng = np.random.default_rng([tl.seed, 541])
    bright = 2 + 14 * u ** 1.5                                  # filter opening: more harmonics
    for side, detunes in ((0, (-0.006, 0.003)), (1, (0.005, -0.002))):
        sig = np.zeros(n)
        for dt in detunes:
            ph = 2 * np.pi * np.cumsum(f * (1 + dt)) / sr + rng.uniform(0, 6)
            for h in range(1, 17):
                sig += np.sin(h * ph) / h * np.exp(-h / bright)
        sig += 0.35 * np.sin(2 * np.pi * np.cumsum(f * 2.0) / sr) * u     # octave joins late
        out[int(t0 * sr):int(t0 * sr) + n, side] = sig
    env = env_lin(t, [(0, 0), (1.2, 0.35), ((t1 - t0) * 0.8, 0.85), ((t1 - t0) - 0.006, 1.0), (t1 - t0, 0)])
    out[int(t0 * sr):int(t0 * sr) + n] *= env[:, None]
    out /= np.abs(out).max() + 1e-9
    return out * db(c.get("gain_db", -12))


# --------------------------------------------------------------------------
# 05-06: low drone in the dark; a deep ping with each ripple
# --------------------------------------------------------------------------
def bell(sr, f, dur, seed):
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    s = np.zeros(n)
    for r, a, tau in ((1.0, 1.0, 2.2), (2.76, 0.45, 1.2), (5.4, 0.25, 0.6), (8.93, 0.12, 0.3)):
        s += a * np.sin(2 * np.pi * f * r * t + rng.uniform(0, 6)) * np.exp(-t / tau)
    return s * np.minimum(1, t / 0.004)


def layer_drone(tl):
    c = tl.audio.get("drone", {})
    sr = tl.audio.get("sample_rate", 48000)
    N = int(round(tl.duration * sr))
    t = np.arange(N) / sr
    env = envelope(tl, t, c["envelope"])
    rng = np.random.default_rng([tl.seed, 551])
    sig = np.zeros((N, 2))
    for f, a in ((36.7, 1.0), (55.0, 0.55), (73.4, 0.4), (110.0, 0.12)):
        for ch, d in ((0, -0.15), (1, 0.17)):
            sig[:, ch] += a * np.sin(2 * np.pi * (f + d) * t + rng.uniform(0, 6))
    noise = fft_filter(rng.standard_normal(N), sr, lo=30, hi=140, order=2)
    noise /= np.abs(noise).max() + 1e-9
    sig += 0.5 * noise[:, None]
    sig *= (1 + 0.2 * np.sin(2 * np.pi * 0.17 * t))[:, None]
    sig = sig / (np.abs(sig).max() + 1e-9) * env[:, None] * db(c.get("gain_db", -14))

    # pings synced to the ripples of segment 06
    from .segments.s06_sound import ring_times
    sseg = tl.segment(c.get("ping_segment", "sound"))
    pings = np.zeros(N)
    for i, rt in enumerate(ring_times(sseg.params, sseg.n_frames / tl.fps)):
        place(pings, bell(sr, 98.0, 4.0, tl.seed + 560 + i), sseg.start_frame / tl.fps + rt, sr, 0.8 ** i)
    pings = fft_filter(pings, sr, hi=1200, order=1)
    sig += reverb(pings, sr, 4.5, tl.seed + 57, wet=0.8, damp_hz=1500) * db(c.get("ping_db", -20))
    return sig


# --------------------------------------------------------------------------
# 07-12: stacked minimal patterns on the beat grid; the train as a heartbeat
# --------------------------------------------------------------------------
NOTES = {"D3": 146.83, "A3": 220.0, "F3": 174.61, "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
         "G4": 392.0, "A4": 440.0, "C5": 523.25, "D5": 587.33, "E5": 659.26, "F5": 698.46, "G5": 783.99,
         "A5": 880.0, "C6": 1046.5, "D2": 73.42}


def mallet(sr, f, seed, dur=0.9):
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.28)
         + 0.35 * np.sin(2 * np.pi * 3.93 * f * t) * np.exp(-t / 0.05)
         + 0.08 * np.sin(2 * np.pi * 9.7 * f * t) * np.exp(-t / 0.012))
    click = fft_filter(rng.standard_normal(n) * np.exp(-t / 0.002), sr, lo=2000, hi=8000) * 0.15
    return (s + click) * np.minimum(1, t / 0.0015)


def layer_minimal(tl):
    c = tl.audio.get("minimal", {})
    sr = tl.audio.get("sample_rate", 48000)
    N = int(round(tl.duration * sr))
    beat = tl.beat
    anchor = tl.anchor_time
    out = np.zeros((N, 2))
    rng = np.random.default_rng([tl.seed, 571])
    cache = {}

    def note(name, voice):
        key = (name, voice)
        if key not in cache:
            cache[key] = mallet(sr, NOTES[name], tl.seed + len(cache))
        return cache[key]

    voices = c["voices"]
    sections = c["sections"]          # segment -> {voice: gain}
    end_t = tl.seg_end(c.get("to", "gallery"))
    for vname, v in voices.items():
        pat = v["pattern"]
        step = beat * 4 / int(v["division"])                  # e.g. division 8 = eighth notes
        off = float(v.get("offset_steps", 0)) * step
        buf = np.zeros(N)
        k = 0
        while True:
            tk = anchor + off + k * step
            if tk >= end_t:
                break
            seg = next((s for s in tl.segments if s.start_frame / tl.fps <= tk < s.end_frame / tl.fps), None)
            g = sections.get(seg.name, {}).get(vname, 0.0) if seg else 0.0
            every = int(sections.get(seg.name, {}).get("every", 1)) if seg else 1
            if g > 0 and k % every == 0:
                nm = pat[k % len(pat)]
                if nm != "-":
                    accent = 1.0 if (k * step) % beat < 1e-6 else 0.78
                    place(buf, note(nm, vname), tk + rng.normal(0, 0.002), sr, g * accent)
            k += 1
        out += pan(buf, float(v.get("pan", 0.0)))

    # sustained pad for the slow section (cards / chess)
    t = np.arange(N) / sr
    pad_env = envelope(tl, t, c["pad_envelope"])
    pad = np.zeros(N)
    for nm, a in (("D3", 1.0), ("A3", 0.7), ("F4", 0.45), ("C5", 0.25)):
        f = NOTES[nm]
        pad += a * (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 1.003 * t))
    pad = fft_filter(pad, sr, hi=900) * pad_env
    out += pan(pad / 4.0, 0.0) * db(c.get("pad_db", -10))

    # cut ticks: every shot change from the anchor on lands on the grid; mark it
    ticks = np.zeros(N)
    tk_s = fft_filter(np.random.default_rng(tl.seed + 58).standard_normal(int(0.08 * sr))
                      * np.exp(-np.arange(int(0.08 * sr)) / sr / 0.012), sr, lo=300, hi=3000)
    tk_s /= np.abs(tk_s).max()
    for s in tl.segments:
        st = s.start_frame / tl.fps
        if st < anchor - 1e-6 or st >= end_t:
            continue
        for a, _ in s.shot_frames:
            place(ticks, tk_s, (s.start_frame + a) / tl.fps, sr, 1.0)
    out += pan(ticks, 0.0) * db(c.get("tick_db", -22))

    # bass on the downbeat of every bar, from the bread on
    bass = np.zeros(N)
    b0 = tl.seg_start(c.get("bass_from", "bread"))
    k = 0
    while anchor + k * 4 * beat < end_t:
        tb = anchor + k * 4 * beat
        if tb >= b0 - 1e-6:
            n = int(1.2 * sr)
            tt = np.arange(n) / sr
            place(bass, np.sin(2 * np.pi * NOTES["D2"] * tt) * np.exp(-tt / 0.45), tb, sr, 1.0)
        k += 1
    out += pan(bass, 0.0) * db(c.get("bass_db", -12))

    # the train, low and soft, as a heartbeat under everything
    hb = clacks(tl, N, sr, anchor, end_t - 0.3, beat, beat / 2, float(c.get("heart_axle", 0.11)), rng, 0.0)
    hb = fft_filter(hb, sr, hi=900, order=2)
    out += pan(hb, -0.1) * db(c.get("heartbeat_db", -24))

    out = out * 0.85 + reverb(out.mean(1), sr, 2.2, tl.seed + 59, wet=0.25, damp_hz=5000) * 0.15
    # silence falls like a blade at the end
    gate = env_lin(t, [(end_t - 0.02, 1.0), (end_t, 0.0)])
    return out * gate[:, None] * db(c.get("gain_db", -6))


# --------------------------------------------------------------------------
# 13: almost nothing — room tone and two distant drips
# --------------------------------------------------------------------------
def layer_basement(tl):
    c = tl.audio.get("basement", {})
    sr = tl.audio.get("sample_rate", 48000)
    N = int(round(tl.duration * sr))
    seg = c.get("segment", "basement")
    t0, t1 = tl.seg_start(seg), tl.seg_end(seg)
    rng = np.random.default_rng([tl.seed, 581])
    t = np.arange(N) / sr
    room = fft_filter(rng.standard_normal(N), sr, lo=60, hi=400, order=1)
    room /= np.abs(room).max() + 1e-9
    room *= env_lin(t, [(t0, 0), (t0 + 0.8, 1), (t1 - 0.3, 1), (t1, 0)]) * (t >= t0) * (t <= t1)
    drips = np.zeros(N)
    for dt in c.get("drips", [2.4, 4.6]):
        n = int(0.25 * sr)
        tt = np.arange(n) / sr
        f = 1400 * (1 + 0.6 * np.exp(-tt / 0.01))
        place(drips, np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt / 0.03), t0 + dt, sr, 1.0)
    sig = pan(room, 0.0) * db(c.get("room_db", -52)) + \
        reverb(drips, sr, 3.0, tl.seed + 59, wet=1.5, damp_hz=3000) * db(c.get("drip_db", -36))
    return sig


LAYERS = [layer_train, layer_rise, layer_drone, layer_minimal, layer_basement]


def soft_ceiling(x, ceil):
    knee = 0.7 * ceil
    a = np.abs(x)
    over = a > knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (ceil - knee) * np.tanh((a[over] - knee) / (ceil - knee)))
    return y


def render_mix(tl):
    from . import audio2
    sr = tl.audio.get("sample_rate", 48000)
    N = int(round(tl.duration * sr))
    mix = np.zeros((N, 2))
    exempt = np.zeros((N, 2))
    mz = tl.audio.get("mute_zone")
    for layer in LAYERS + audio2.LAYERS:
        if layer.__name__ in tl.audio.get("disabled_layers", []):
            continue
        if mz and layer.__name__ in mz.get("exempt", []):
            exempt += layer(tl)[:N]
        else:
            mix += layer(tl)[:N]
    # 29 flashes back through everything heard so far, so it is built from the mix itself
    mix += audio2.layer_flashback(tl, mix)[:N]
    # a hard cut to (near) silence: everything stops except the exempt layers
    if mz:
        t = np.arange(N) / sr
        a, b = seg_time(tl, mz["from"], "start"), seg_time(tl, mz["to"], "start")
        gate = 1 - env_lin(t, [(a - 0.004, 0), (a, 1), (b - 0.3, 1), (b, 0)])
        mix *= gate[:, None]
    mix += exempt
    # the film ends in true silence
    if "silence_from" in tl.audio:
        s_name, edge, off = tl.audio["silence_from"]
        t_sil = seg_time(tl, s_name, edge, off)
        t = np.arange(N) / sr
        mix *= env_lin(t, [(t_sil - 0.4, 1.0), (t_sil, 0.0)])[:, None]
    mix = fft_filter(mix[:, 0], sr, lo=22), fft_filter(mix[:, 1], sr, lo=22)
    mix = np.stack(mix, 1)
    return soft_ceiling(mix, db(tl.audio.get("ceiling_db", -1.0)))


def write_wav(path, x, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(0)
    d = (rng.random(x.shape) - rng.random(x.shape)) / 32768.0      # TPDF dither
    pcm = np.clip((x + d) * 32767, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
