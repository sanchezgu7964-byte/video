"""人聲的合成：聲門脈衝經過共振峰，加上子音的噪聲。

不是要做出聽得懂的語音，而是讓人聽見「有人在某個地方喊」：
一個人領喊，一群人跟著喊；幻燈片換了一張，講堂裡的人拍掌喊「萬歲」。
音節由（子音, 元音路徑, 聲調）組成，聲調是普通話的四聲輪廓。
"""
import numpy as np
from scipy.signal import butter, sosfilt

from .sound import SR, env_ad, filt, white

VOWEL = {  # F1, F2, F3（成年男聲，Hz）
    "a": (760, 1250, 2500), "o": (560, 880, 2400), "e": (480, 1350, 2450), "i": (300, 2250, 3000),
    "u": (330, 780, 2300), "y": (290, 1900, 2400), "n": (260, 1600, 2600), "ng": (270, 1100, 2500),
}
NASAL = {"n", "ng"}
CONS = {  # 頻帶, 長度（秒）, 強度
    "zh": ((2200, 5000), 0.06, 0.5), "ch": ((2200, 5200), 0.09, 0.6), "z": ((3800, 7500), 0.05, 0.5),
    "q": ((3000, 7000), 0.07, 0.55), "g": ((900, 2200), 0.018, 0.9), "b": ((300, 1500), 0.012, 0.8),
    "s": ((4000, 8000), 0.08, 0.5), "w": None, "n": None, "": None, "d": ((1500, 3500), 0.014, 0.8),
    "sh": ((1800, 4500), 0.09, 0.55), "t": ((2500, 6000), 0.02, 0.7),
}
TONE = {1: [(0, 1.0), (1, 1.0)], 2: [(0, 0.78), (0.3, 0.8), (1, 1.06)], 3: [(0, 0.8), (0.5, 0.7), (1, 0.86)],
        4: [(0, 1.12), (1, 0.72)], 0: [(0, 0.95), (1, 0.85)]}

# 喊的字
SYL = {
    "外": ("w", ["u", "a", "i"], 4), "爭": ("zh", ["e", "ng"], 1), "主": ("zh", ["u"], 3), "權": ("q", ["y", "a", "n"], 2),
    "內": ("n", ["e", "i"], 4), "除": ("ch", ["u"], 2), "國": ("g", ["u", "o"], 2), "賊": ("z", ["e", "i"], 2),
    "還": ("", ["u", "a", "n"], 2), "我": ("w", ["u", "o"], 3), "青": ("q", ["i", "ng"], 1), "島": ("d", ["a", "o"], 3),
    "ban": ("b", ["a", "n"], 1), "zai": ("z", ["a", "i"], 4),
}


def _formant_bank(src, path, fscale, block=240):
    """依元音路徑（隨時間內插的共振峰）濾波；每 block 個取樣換一次係數，沿用濾波狀態。"""
    n = len(src)
    out = np.zeros(n, np.float32)
    keys = [VOWEL[v] for v in path]
    m = len(keys)
    for fi, (gain, bwk) in enumerate(((1.0, 0.11), (0.55, 0.09), (0.25, 0.07))):
        zi = None
        for i in range(0, n, block):
            u = i / max(n - 1, 1)
            x = u * (m - 1)
            j = min(int(x), m - 2) if m > 1 else 0
            fr = x - j
            f = keys[j][fi] if m == 1 else keys[j][fi] * (1 - fr) + keys[j + 1][fi] * fr
            f *= fscale
            bw = max(60.0, f * bwk * 2)
            sos = butter(1, [max(40, f - bw / 2), min(SR / 2 - 200, f + bw / 2)], btype="bandpass", fs=SR, output="sos")
            if zi is None:
                zi = np.zeros((sos.shape[0], 2))
            y, zi = sosfilt(sos, src[i:i + block], zi=zi)
            out[i:i + block] += gain * y
    return out


def syllable(rng, ch, dur, f0=150.0, fscale=1.0, effort=1.0):
    """一個喊出來的音節（含子音），長 dur 秒。"""
    cons, path, tone = SYL[ch]
    cinfo = CONS[cons]
    cdur = cinfo[1] if cinfo else 0.0
    vdur = max(0.08, dur - cdur)
    n = int(vdur * SR)
    u = np.linspace(0, 1, n)
    pts = TONE[tone]
    contour = np.interp(u, [p[0] for p in pts], [p[1] for p in pts])
    jitter = 1 + 0.012 * np.cumsum(rng.normal(0, 1, n)) / np.sqrt(np.arange(1, n + 1))
    f = f0 * contour * jitter * (1 + 0.15 * (effort - 1))
    ph = np.cumsum(f) / SR
    saw = (ph % 1.0) * 2 - 1
    # 喊的時候聲門較緊：脈衝較尖，加一點氣聲
    src = (np.sign(saw) * np.abs(saw) ** (1.6 + 0.8 * effort) + 0.12 * effort * white(rng, n)).astype(np.float32)
    if cons == "w":
        path = ["u"] + path
    if cons == "n":
        path = ["n"] + path
    v = _formant_bank(src, path, fscale)
    env = np.minimum(u / 0.06, 1) * np.minimum((1 - u) / 0.25, 1) ** 1.2
    if path[-1] in NASAL:
        v *= 1 - 0.55 * np.clip((u - 0.6) / 0.4, 0, 1)
    v = (v * env).astype(np.float32)
    if cinfo:
        (lo, hi), cd, amp = cinfo
        cn = int(cd * SR)
        c = filt(white(rng, cn), "bandpass", [lo * fscale, min(hi * fscale, SR / 2 - 500)]) * env_ad(cn, cd * 0.3, cd * 0.4)
        out = np.concatenate([c * amp * 0.9, v])
    else:
        out = v
    return out / (np.abs(out).max() + 1e-6)


def phrase(rng, chars, beat, f0, fscale, effort=1.0, legato=0.85, jitter=0.0):
    """一句喊：每個字落在一拍上，回傳（音訊, 長度）。"""
    total = int((len(chars) * beat + 0.6) * SR)
    out = np.zeros(total, np.float32)
    for k, ch in enumerate(chars):
        s = syllable(rng, ch, beat * legato * (1.25 if k == len(chars) - 1 else 1.0), f0 * (1.06 if k == 0 else 1.0),
                     fscale, effort)
        i = int((k * beat + rng.normal(0, jitter)) * SR)
        i = max(0, i)
        e = min(total, i + len(s))
        out[i:e] += s[: e - i] * (1.15 if k == 0 else 1.0)
    return out


def crowd(rng, chars, beat, voices=24, f0=(120, 210), effort=1.2, jitter=0.03, spread=1.0):
    """一群人喊同一句：每個人的音高、共振峰、起拍都不一樣。回傳立體聲 (n, 2)。"""
    parts = []
    for v in range(voices):
        f = rng.uniform(*f0)
        fs = rng.uniform(0.9, 1.12)
        x = phrase(rng, chars, beat, f, fs, effort * rng.uniform(0.85, 1.1), jitter=jitter)
        d = int(abs(rng.normal(0, jitter * 1.5)) * SR)
        x = np.concatenate([np.zeros(d, np.float32), x])
        pan = rng.uniform(-1, 1) * spread
        parts.append((x * rng.uniform(0.5, 1.0), pan))
    n = max(len(p) for p, _ in parts)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    for x, pan in parts:
        a = (pan + 1) * np.pi / 4
        L[: len(x)] += x * np.cos(a)
        R[: len(x)] += x * np.sin(a)
    st = np.stack([L, R], 1)
    return st / (np.abs(st).max() + 1e-6)
