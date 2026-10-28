"""讀取時間軸設定，並由 [pulse] 產生全片的節拍網格。"""
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


def load(path: Path = ROOT / "config/timeline.toml") -> Timeline:
    raw = tomllib.loads(path.read_text())
    o = raw["output"]
    pu = raw["pulse"]
    period = 60 / pu["bpm"]
    k0 = -int(np.ceil(pu["anchor"] / period))
    k1 = int(np.ceil((o["duration"] - pu["anchor"]) / period))
    beats = pu["anchor"] + np.arange(k0, k1 + 1) * period
    beats = beats[(beats >= 0) & (beats <= o["duration"])]
    pal = {k: hex2rgb(v) for k, v in raw["palette"].items()}
    return Timeline(raw, o["width"], o["height"], o["fps"], o["duration"], o["seed"],
                    raw["cues"], pal, beats)
