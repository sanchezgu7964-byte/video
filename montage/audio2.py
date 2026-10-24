"""Soundtrack layers for v2 (segments 14–33). Same conventions as audio.py:
each layer is (tl) -> stereo array for the whole film, configured under
`audio.<block>` in timeline.json.
"""
import numpy as np

from .audio import (NOTES, bell, clacks, db, env_lin, envelope, fft_filter, mallet, pan, place,
                    reverb, seg_time, train_layer)

NOTES2 = dict(NOTES)
NOTES2.update({"F#3": 185.0, "B2": 123.47, "G2": 98.0, "A2": 110.0, "E3": 164.81, "B3": 246.94, "C#4": 277.18,
               "F#4": 369.99, "B4": 493.88, "C#5": 554.37, "F#5": 739.99, "E2": 82.41, "D1": 36.71, "E1": 41.2,
               "G3": 196.0, "D6": 1174.66, "A6": 1760.0, "B5": 987.77, "E6": 1318.5})


def _sr(tl):
    return tl.audio.get("sample_rate", 48000)


def _N(tl):
    return int(round(tl.duration * _sr(tl)))


def tone(sr, f, dur, attack, release, harmonics=((1, 1.0), (2, 0.25), (3, 0.08)), vib=0.0, seed=0, glide=None):
    """A soft sustained tone. glide: optional per-sample frequency multiplier."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    fm = 1 + vib * np.sin(2 * np.pi * rng.uniform(4.5, 5.5) * t + rng.uniform(0, 6))
    if glide is not None:
        fm = fm * glide[:n]
    ph = 2 * np.pi * np.cumsum(f * fm) / sr
    s = sum(a * np.sin(h * ph + rng.uniform(0, 6)) for h, a in harmonics)
    env = np.minimum(1, t / max(attack, 1e-3)) * np.minimum(1, (dur - t) / max(release, 1e-3))
    return s * np.clip(env, 0, 1)


# --------------------------------------------------------------------------
# 14–16: near-silence — room tone and soft footsteps
# --------------------------------------------------------------------------
def layer_quiet(tl):
    c = tl.audio.get("quiet", {})
    if not c:
        return np.zeros((_N(tl), 2))
    sr, N = _sr(tl), _N(tl)
    t = np.arange(N) / sr
    rng = np.random.default_rng([tl.seed, 1401])
    room = fft_filter(rng.standard_normal(N), sr, lo=50, hi=500, order=1)
    room /= np.abs(room).max() + 1e-9
    out = pan(room * envelope(tl, t, c["envelope"]), 0.0) * db(c.get("room_db", -50))
    from .segments.s14_body import step_times
    seg = tl.segment(c.get("steps_segment", "body"))
    steps = np.zeros(N)
    for k, st in enumerate(step_times(seg.params, seg.n_frames / tl.fps)):
        n = int(0.18 * sr)
        tt = np.arange(n) / sr
        thud = np.sin(2 * np.pi * 70 * tt) * np.exp(-tt / 0.03)
        scuff = fft_filter(rng.standard_normal(n), sr, lo=900, hi=4000) * np.exp(-tt / 0.05) * 0.25
        place(steps, (thud + scuff) * (0.8 + 0.2 * rng.random()), seg.start_frame / tl.fps + st, sr, 1.0)
    out += reverb(steps, sr, 1.2, tl.seed + 141, wet=0.4, damp_hz=2500) * db(c.get("steps_db", -30))
    return out


# --------------------------------------------------------------------------
# 17–18: the minimal patterns return and stack with the mirrors, then slide down
# --------------------------------------------------------------------------
def layer_mirror_music(tl):
    c = tl.audio.get("mirror_music", {})
    if not c:
        return np.zeros((_N(tl), 2))
    sr, N = _sr(tl), _N(tl)
    beat = tl.beat
    anchor = tl.anchor_time
    seg = tl.segment(c.get("segment", "mirrors"))
    slide = tl.segment(c.get("slide_segment", "music_slide"))
    t0 = seg.start_frame / tl.fps
    t_s0, t_s1 = slide.start_frame / tl.fps, slide.end_frame / tl.fps
    semis = float(slide.params.get("semitones", 12))
    out = np.zeros((N, 2))
    rng = np.random.default_rng([tl.seed, 1701])

    def glide_at(tk):
        u = np.clip((tk - t_s0) / (t_s1 - t_s0), 0, 1)
        return 2 ** (-semis * u ** 1.3 / 12), 1 - u ** 0.8

    # which voices play: they join as the mirrors multiply (by shot index)
    shot_starts = [t0 + a / tl.fps for a, _ in seg.shot_frames]
    for vname, v in c["voices"].items():
        step = beat * 4 / int(v["division"])
        k0 = int(np.ceil((t0 - anchor) / step))
        buf = np.zeros(N)
        k = k0
        while True:
            tk = anchor + k * step
            if tk >= t_s1:
                break
            shot = sum(1 for s in shot_starts if s <= tk + 1e-6) - 1
            if tk < t_s0 and shot < int(v["from_shot"]):
                k += 1
                continue
            nm = v["pattern"][k % len(v["pattern"])]
            if nm != "-":
                g, amp = glide_at(tk) if tk >= t_s0 else (1.0, 1.0)
                note = mallet(sr, NOTES2[nm] * g, tl.seed + k % 13)
                place(buf, note, tk + rng.normal(0, 0.002), sr, float(v.get("gain", 0.5)) * amp)
            k += 1
        out += pan(buf, float(v.get("pan", 0.0)))
    # one long tone underneath that really slides down through 18
    n = int((t_s1 - t0) * sr)
    tt = np.arange(n) / sr
    g = np.where(tt < t_s0 - t0, 1.0, 2 ** (-semis * np.clip((tt - (t_s0 - t0)) / (t_s1 - t_s0), 0, 1) ** 1.3 / 12))
    pad = tone(sr, NOTES2["D3"], n / sr, 1.5, 1.2, ((1, 1.0), (2, 0.35), (3, 0.15), (4, 0.08)), 0.002, tl.seed + 171, g)
    pad += tone(sr, NOTES2["A3"], n / sr, 2.0, 1.2, ((1, 0.6), (2, 0.2)), 0.002, tl.seed + 172, g)
    pb = np.zeros(N)
    place(pb, pad * env_lin(tt, [(0, 0), (1.5, 1), (t_s0 - t0, 1), (t_s1 - t0, 0)]), t0, sr, 1.0)
    out += pan(pb, 0.0) * db(c.get("pad_db", -14))
    hb = clacks(tl, N, sr, t0, t_s0, beat, beat / 2, 0.11, rng, 0.0)
    out += pan(fft_filter(hb, sr, hi=900), -0.1) * db(c.get("heartbeat_db", -26))
    out = out * 0.85 + reverb(out.mean(1), sr, 2.4, tl.seed + 173, wet=0.3) * 0.15
    return out * db(c.get("gain_db", -7))


# --------------------------------------------------------------------------
# 19–23: warm harmony; the charge (22) gets dense low drum pulses
# --------------------------------------------------------------------------
def layer_warm(tl):
    c = tl.audio.get("warm", {})
    if not c:
        return np.zeros((_N(tl), 2))
    sr, N = _sr(tl), _N(tl)
    out = np.zeros((N, 2))
    for name, chord in c["chords"].items():
        seg = tl.segment(name)
        t0, t1 = seg.start_frame / tl.fps, seg.end_frame / tl.fps
        dur = t1 - t0 + 0.8
        buf = np.zeros(N)
        for i, nm in enumerate(chord):
            s = tone(sr, NOTES2[nm], dur, 0.6, 0.8, ((1, 1.0), (2, 0.4), (3, 0.18), (5, 0.05)), 0.003,
                     tl.seed + 190 + i)
            place(buf, s, t0 - 0.2, sr, 1.0 / len(chord))
        out += pan(fft_filter(buf, sr, hi=2200), 0.15 * ((len(name) % 3) - 1))
    out = out * 0.8 + reverb(out.mean(1), sr, 3.0, tl.seed + 191, wet=0.4, damp_hz=3000) * 0.2
    out *= db(c.get("gain_db", -12))

    # drum-like pulses for the charge
    seg = tl.segment(c.get("drums_segment", "charge"))
    t0, t1 = seg.start_frame / tl.fps, seg.end_frame / tl.fps
    step = tl.beat / 4
    k = int(np.ceil((t0 - tl.anchor_time) / step))
    drums = np.zeros(N)
    rng = np.random.default_rng([tl.seed, 2201])
    while True:
        tk = tl.anchor_time + k * step
        if tk >= t1:
            break
        n = int(0.35 * sr)
        tt = np.arange(n) / sr
        accent = 1.0 if k % 4 == 0 else (0.6 if k % 2 == 0 else 0.42)
        f = 48 + 60 * np.exp(-tt / 0.03)
        hit = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt / (0.18 if k % 4 == 0 else 0.09))
        hit += fft_filter(rng.standard_normal(n), sr, lo=100, hi=900) * np.exp(-tt / 0.02) * 0.3
        u = (tk - t0) / (t1 - t0)
        place(drums, hit, tk, sr, accent * (0.7 + 0.5 * u))       # growing with the acceleration
        k += 1
    out += pan(drums, 0.0) * db(c.get("drums_db", -9))
    return out


# --------------------------------------------------------------------------
# 24–28: slow transparent tones; 28 pulled in pitch and filter by the dolly zoom
# --------------------------------------------------------------------------
def layer_long_tones(tl):
    c = tl.audio.get("long_tones", {})
    if not c:
        return np.zeros((_N(tl), 2))
    sr, N = _sr(tl), _N(tl)
    out = np.zeros((N, 2))
    for k, (name, notes) in enumerate(c["notes"].items()):
        seg = tl.segment(name)
        t0, t1 = seg.start_frame / tl.fps, seg.end_frame / tl.fps
        for i, item in enumerate(notes):
            nm, a, b = item if isinstance(item, list) else (item, 0.0, 1.0)
            st, en = t0 + a * (t1 - t0), t0 + b * (t1 - t0) + 0.6
            s = tone(sr, NOTES2[nm], en - st, 0.9, 1.0, ((1, 1.0), (2, 0.12)), 0.0025, tl.seed + 240 + 10 * k + i)
            buf = np.zeros(N)
            place(buf, s, st, sr, 0.5)
            out += pan(buf, (-0.35, 0.35, 0.0)[i % 3])
    # 27: the dice — light clatter, then nothing at the freeze
    ch = tl.segment(c.get("dice_segment", "chance"))
    rng = np.random.default_rng([tl.seed, 2701])
    t0 = ch.start_frame / tl.fps
    freeze = float(ch.params.get("freeze_at", 2.3))
    clat = np.zeros(N)
    for _ in range(int(c.get("dice_clicks", 14))):
        tt0 = t0 + rng.uniform(0.2, freeze - 0.05)
        n = int(0.04 * sr)
        x = np.arange(n) / sr
        click = np.sin(2 * np.pi * rng.uniform(2500, 4200) * x) * np.exp(-x / 0.006)
        place(clat, click, tt0, sr, rng.uniform(0.3, 1.0))
    out += pan(clat, 0.2) * db(c.get("dice_db", -30))
    # 28: two tones pulled apart by the dolly zoom, plus a filter sweep
    from .segments.s28_memory import dolly
    m = tl.segment(c.get("pull_segment", "memory"))
    t0, t1 = m.start_frame / tl.fps, m.end_frame / tl.fps
    n = int((t1 - t0 + 0.8) * sr)
    tt = np.arange(n) / sr
    d = np.array([dolly(min(1.0, x / (t1 - t0))) for x in tt[::240]])
    d = np.interp(tt, tt[::240], d)
    pull = float(c.get("pull_semitones", 5))
    up = tone(sr, NOTES2["A4"], n / sr, 0.6, 0.8, ((1, 1.0), (2, 0.3), (3, 0.15)), 0.002, tl.seed + 281,
              2 ** (pull * d / 12))
    dn = tone(sr, NOTES2["A3"], n / sr, 0.6, 0.8, ((1, 1.0), (2, 0.3), (3, 0.15)), 0.002, tl.seed + 282,
              2 ** (-pull * d / 12))
    saw = np.zeros(n)
    ph = 2 * np.pi * np.cumsum(NOTES2["D3"] * (1 + 0.004 * np.sin(tt * 0.7))) / sr
    for h in range(1, 20):
        bright = np.exp(-h / (1.5 + 10 * d))                        # the filter opens with the pull
        saw += np.sin(h * ph) / h * bright
    env = env_lin(tt, [(0, 0), (0.5, 1), (t1 - t0, 1), (t1 - t0 + 0.8, 0)])
    pb = np.zeros((N, 2))
    seg_sig = np.stack([up * 0.6 + saw * 0.35, dn * 0.6 + saw * 0.35], 1) * env[:, None]
    a = int(t0 * sr)
    pb[a:a + n] = seg_sig[:max(0, min(n, N - a))]
    out += pb * db(c.get("pull_db", -4))
    out = out * 0.75 + reverb(out.mean(1), sr, 3.5, tl.seed + 283, wet=0.6, damp_hz=4000) * 0.25
    return out * db(c.get("gain_db", -12))


# --------------------------------------------------------------------------
# 29: everything heard so far flashes back, grains stacked ever faster
# --------------------------------------------------------------------------
def layer_flashback(tl, premix):
    c = tl.audio.get("flashback", {})
    if not c:
        return np.zeros_like(premix)
    sr = _sr(tl)
    seg = tl.segment(c.get("segment", "all_given"))
    t0 = seg.start_frame / tl.fps
    out = np.zeros_like(premix)
    srcs = [s for s in tl.segments if s.index < seg.index]
    rng = np.random.default_rng([tl.seed, 2901])
    for i, (a, n) in enumerate(seg.shot_frames):
        src = srcs[i % len(srcs)]
        # take a grain from the loudest part of that segment, so each one is recognisable
        s0, s1 = int(src.start_frame / tl.fps * sr), int(src.end_frame / tl.fps * sr)
        seg_audio = premix[s0:s1]
        if len(seg_audio) < sr // 4:
            continue
        L = int(0.32 * sr)
        hop = max(1, (len(seg_audio) - L) // 24)
        energies = [np.sqrt((seg_audio[j:j + L] ** 2).mean()) for j in range(0, len(seg_audio) - L, hop)]
        j = int(np.argmax(energies)) * hop if energies else 0
        grain = seg_audio[j:j + L].copy()
        w = np.minimum(1, np.minimum(np.arange(L), L - np.arange(L)) / (0.01 * sr))[:, None]
        grain *= w / (np.abs(grain).max() + 1e-6)
        st = int((t0 + a / tl.fps) * sr)
        e = min(len(out), st + L)
        out[st:e] += grain[:e - st] * (0.5 + 0.5 * i / len(seg.shot_frames))
    return out * db(c.get("gain_db", -10))


# --------------------------------------------------------------------------
# 30: one very low held tone; 31: slow, transparent, near-sacred
# --------------------------------------------------------------------------
def layer_low_hold(tl):
    c = tl.audio.get("low_hold", {})
    if not c:
        return np.zeros((_N(tl), 2))
    sr, N = _sr(tl), _N(tl)
    seg = tl.segment(c.get("segment", "old_food"))
    t0, t1 = seg.start_frame / tl.fps, seg.end_frame / tl.fps
    s = tone(sr, NOTES2["D1"], t1 - t0, 1.5, 1.5, ((1, 1.0), (2, 0.5), (3, 0.1)), 0.0, tl.seed + 301)
    buf = np.zeros(N)
    place(buf, s, t0, sr, 1.0)
    return pan(buf, 0.0) * db(c.get("gain_db", -22))


def layer_sacred(tl):
    c = tl.audio.get("sacred", {})
    if not c:
        return np.zeros((_N(tl), 2))
    sr, N = _sr(tl), _N(tl)
    seg = tl.segment(c.get("segment", "sea_sun"))
    t0, t1 = seg.start_frame / tl.fps, seg.end_frame / tl.fps
    out = np.zeros((N, 2))
    for i, item in enumerate(c["notes"]):
        nm, a = item
        st = t0 + a
        s = tone(sr, NOTES2[nm], t1 - st + 1.5, 2.2, 2.0, ((1, 1.0), (2, 0.2), (3, 0.06)), 0.004, tl.seed + 310 + i)
        # choir-ish: a second, slightly detuned voice
        s += 0.6 * tone(sr, NOTES2[nm] * 1.004, t1 - st + 1.5, 2.6, 2.0, ((1, 1.0), (2, 0.15)), 0.005, tl.seed + 330 + i)
        buf = np.zeros(N)
        place(buf, s, st, sr, 0.35)
        out += pan(buf, (-0.4, 0.4, -0.15, 0.15, 0.0)[i % 5])
    out = out * 0.6 + reverb(out.mean(1), sr, 5.0, tl.seed + 340, wet=0.9, damp_hz=6000) * 0.4
    return out * db(c.get("gain_db", -10))


def layer_train_last(tl):
    if "train_last" not in tl.audio:
        return np.zeros((_N(tl), 2))
    return train_layer(tl, "train_last")


LAYERS = [layer_quiet, layer_mirror_music, layer_warm, layer_long_tones, layer_low_hold, layer_sacred,
          layer_train_last]
