"""配樂分析：響度曲線、onset 包絡、節拍（DP beat tracking），並把設定檔的關鍵對位點吸附到最近的強 onset。"""
import json
import subprocess
from pathlib import Path

import numpy as np

SR = 22050
HOP = 256


def decode(path: Path, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def stft_mag(y: np.ndarray, n_fft: int = 2048, hop: int = HOP) -> np.ndarray:
    pad = np.pad(y, (n_fft // 2, n_fft // 2))
    n = 1 + (len(pad) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    win = np.hanning(n_fft).astype(np.float32)
    return np.abs(np.fft.rfft(pad[idx] * win, axis=1)).astype(np.float32)


def onset_envelope(mag: np.ndarray) -> np.ndarray:
    logm = np.log1p(100 * mag)
    flux = np.maximum(0, np.diff(logm, axis=0)).sum(axis=1)
    flux = np.concatenate([[0], flux])
    # 去掉慢變化的基線，只留下打擊
    k = 32
    base = np.convolve(flux, np.ones(k) / k, mode="same")
    env = np.maximum(0, flux - base)
    return env / (env.max() + 1e-9)


def estimate_period(env: np.ndarray, fps: float, bpm_hint: float) -> float:
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    bpm = 60 * fps / np.maximum(lags, 1)
    # 以提示速度為中心的 log-高斯權重，避免倍頻/半頻
    w = np.exp(-0.5 * (np.log2(bpm / bpm_hint) / 0.25) ** 2)
    w[lags < 2] = 0
    lag = int(np.argmax(ac * w))
    return lag / fps


def track_beats(env: np.ndarray, fps: float, period: float, tightness: float = 400.0) -> np.ndarray:
    """Ellis 2007 的動態規劃 beat tracking。"""
    p = period * fps
    n = len(env)
    score = env.copy()
    back = -np.ones(n, int)
    lo, hi = int(round(p / 2)), int(round(2 * p))
    for i in range(hi, n):
        prev = np.arange(i - hi, i - lo + 1)
        pen = -tightness * np.log((i - prev) / p) ** 2
        cand = score[prev] + pen
        j = int(np.argmax(cand))
        score[i] = env[i] + cand[j]
        back[i] = prev[j]
    # 從最後一個周期內的最高分往回走
    i = int(np.argmax(score[-hi:])) + n - hi
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    return np.array(beats[::-1]) / fps


def analyze(path: Path, bpm_hint: float, cues: list[float], fps_out: float = 30.0) -> dict:
    y = decode(path)
    mag = stft_mag(y)
    fps = SR / HOP
    env = onset_envelope(mag)
    period = estimate_period(env, fps, bpm_hint)
    beats = track_beats(env, fps, period)

    freqs = np.fft.rfftfreq(2048, 1 / SR)
    power = mag ** 2
    rms = np.sqrt(power.mean(axis=1))
    low = np.sqrt(power[:, freqs < 150].mean(axis=1))
    t_frames = np.arange(len(rms)) / fps
    t_out = np.arange(int(len(y) / SR * fps_out) + 1) / fps_out

    def db(x):
        return 20 * np.log10(np.maximum(x, 1e-6))

    def smooth(x, sec):
        k = max(1, int(sec * fps))
        return np.convolve(x, np.ones(k) / k, mode="same")

    loud = np.interp(t_out, t_frames, db(smooth(rms, 0.3)))
    lowdb = np.interp(t_out, t_frames, db(smooth(low, 0.3)))
    onset = np.interp(t_out, t_frames, env)

    # 把每個關鍵點吸附到 ±0.3 秒內最強的 onset（若夠強）
    snapped = {}
    for c in cues:
        m = (t_frames > c - 0.3) & (t_frames < c + 0.3)
        i = np.argmax(env * m)
        snapped[f"{c:.2f}"] = round(float(t_frames[i]), 3) if env[i] > 0.15 else c

    return {
        "duration": len(y) / SR,
        "bpm": 60 / period,
        "beats": [round(float(b), 3) for b in beats],
        "cues": snapped,
        "fps": fps_out,
        "loudness_db": [round(float(v), 2) for v in loud],
        "low_db": [round(float(v), 2) for v in lowdb],
        "onset": [round(float(v), 3) for v in onset],
    }


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def mix(tl, out: Path, t0: float = 0.0, t1: float | None = None) -> None:
    """配樂處理：2:30 壓到 −30 dB 只留低頻、2:38 拍點上恢復；3:20 淡出；結尾補無聲。"""
    a = tl.raw["audio"]
    t1 = tl.duration if t1 is None else t1
    hs, he = a["hush_start"], a["hush_end"]
    g = 10 ** (a["hush_db"] / 20)
    full = (f"if(lt(t,{hs}),1,if(lt(t,{hs + 0.5}),1-(t-{hs})/0.5*{1 - g},if(lt(t,{he}),{g},1)))")
    lg = a.get("hush_low_gain", 0.5)
    low = (f"if(lt(t,{hs}),0,if(lt(t,{hs + 0.5}),(t-{hs})/0.5*{lg},if(lt(t,{he}),{lg},0)))")
    fs, fe = a["fade_out_start"], a["fade_out_end"]
    graph = (
        f"[0:a]asplit[a][b];"
        f"[a]volume='{full}':eval=frame[f];"
        f"[b]lowpass=f={a['hush_lowpass_hz']},lowpass=f={a['hush_lowpass_hz']},volume='{low}':eval=frame[l];"
        f"[f][l]amix=inputs=2:normalize=0,afade=t=out:st={fs}:d={fe - fs},"
        f"apad=whole_dur={tl.duration},atrim={t0}:{t1},asetpts=PTS-STARTPTS[out]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(Path(tl.raw["audio"]["file"]) if Path(a["file"]).is_absolute()
                                                         else Path(__file__).resolve().parent.parent / a["file"]),
                    "-filter_complex", graph, "-map", "[out]", "-c:a", "pcm_s16le", "-ar", "48000", str(out)],
                   check=True)
