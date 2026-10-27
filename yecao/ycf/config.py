"""讀取時間軸設定與配樂分析結果。"""
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def hex2rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


@dataclass
class Timeline:
    raw: dict
    W: int
    H: int
    fps: int
    duration: float
    seed: int
    cues: dict
    pal: dict
    beats: np.ndarray
    loud: np.ndarray       # 每影格的響度（dB）
    low: np.ndarray        # 每影格低頻響度（dB）
    onset: np.ndarray

    @property
    def nframes(self) -> int:
        return int(round(self.duration * self.fps))

    def c(self, name: str) -> float:
        return float(self.cues[name])

    def beats_between(self, t0: float, t1: float) -> np.ndarray:
        return self.beats[(self.beats >= t0) & (self.beats < t1)]

    def beat_phase(self, t: float) -> tuple[float, float]:
        """回傳 (目前拍的起點, 0–1 的拍內相位)。"""
        i = np.searchsorted(self.beats, t, side="right") - 1
        if i < 0:
            return self.beats[0] - self.period, 0.0
        b0 = self.beats[i]
        b1 = self.beats[i + 1] if i + 1 < len(self.beats) else b0 + self.period
        return b0, (t - b0) / (b1 - b0)

    @property
    def period(self) -> float:
        return float(np.median(np.diff(self.beats)))

    def level(self, t: float, which: str = "loud") -> float:
        arr = self.loud if which == "loud" else self.low
        i = int(np.clip(t * 30, 0, len(arr) - 1))
        return float(arr[i])


def load(path: Path = ROOT / "config/timeline.toml") -> Timeline:
    raw = tomllib.loads(path.read_text())
    o = raw["output"]
    ana_path = ROOT / raw["audio"]["analysis"]
    if ana_path.exists():
        ana = json.loads(ana_path.read_text())
        beats = np.array(ana["beats"])
        loud, low, onset = (np.array(ana[k]) for k in ("loudness_db", "low_db", "onset"))
    else:
        # 沒有分析檔時，退回設定的 BPM 等距網格
        p = 60 / raw["audio"]["bpm_hint"]
        beats = np.arange(0, o["duration"], p)
        n = int(o["duration"] * 30) + 1
        loud, low, onset = np.zeros(n), np.zeros(n), np.zeros(n)
    pal = {k: hex2rgb(v) for k, v in raw["palette"].items()}
    return Timeline(raw, o["width"], o["height"], o["fps"], o["duration"], o["seed"],
                    raw["cues"], pal, beats, loud, low, onset)
