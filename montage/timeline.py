"""Reads timeline.json and turns it into frame-exact segments and shots.

Everything that controls editing rhythm lives in the json; this module only
does the arithmetic (shot-duration curves, beat snapping, frame rounding).
"""
import json
import os
from dataclasses import dataclass, field
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TIMELINE = os.path.join(ROOT, "timeline.json")


@dataclass
class Segment:
    index: int
    name: str
    title: str
    start_frame: int
    n_frames: int
    shot_frames: list          # list of (local_start_frame, n_frames)
    spec: dict = field(repr=False)

    @property
    def end_frame(self):
        return self.start_frame + self.n_frames

    @property
    def params(self):
        return self.spec.get("params", {})

    @property
    def palette(self):
        return self.spec.get("palette", {})

    @property
    def look(self):
        return self.spec.get("look", "bw")

    def shot_at(self, local_frame):
        """-> (shot_index, frame_in_shot, shot_n_frames)"""
        for i, (s, n) in enumerate(self.shot_frames):
            if local_frame < s + n:
                return i, local_frame - s, n
        s, n = self.shot_frames[-1]
        return len(self.shot_frames) - 1, local_frame - s, n

    @property
    def slug(self):
        return f"{self.index + 1:02d}_{self.name}"


def shot_durations(spec):
    """Seconds per shot, before snapping."""
    n = int(spec.get("shots", 1))
    sd = spec.get("shot_durations")
    if sd is None:
        return [float(spec["duration"]) / n] * n
    if isinstance(sd, (int, float)):
        return [float(sd)] * n
    if isinstance(sd, list):
        return [float(x) for x in sd]
    a, b = float(sd["from"]), float(sd["to"])
    curve = sd.get("curve", "linear")
    out = []
    for i in range(n):
        u = i / (n - 1) if n > 1 else 0.0
        if curve == "ease_in":
            u = u * u
        elif curve == "ease_out":
            u = 1 - (1 - u) ** 2
        elif curve == "exponential":
            out.append(a * (b / a) ** u)
            continue
        out.append(a + (b - a) * u)
    if "duration" in spec:          # optional: stretch the curve to a fixed total
        k = float(spec["duration"]) / sum(out)
        out = [d * k for d in out]
    return out


class Timeline:
    def __init__(self, path=DEFAULT_TIMELINE):
        self.path = path
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        m = self.data["meta"]
        self.W, self.H, self.fps = int(m["width"]), int(m["height"]), int(m["fps"])
        self.seed = int(m["seed"])
        self.supersample = int(m.get("supersample", 2))
        self.look = self.data.get("look", {})
        self.audio = self.data.get("audio", {})
        mu = self.data.get("music", {})
        self.bpm = float(mu.get("bpm", 90))
        self.anchor_name = mu.get("anchor_segment")
        self.segments = []
        self._build()

    # -- grid -----------------------------------------------------------------
    def grid_seconds(self, snap):
        """'1/16' -> length of a sixteenth note in seconds."""
        num, den = (Fraction(snap) if "/" in snap else Fraction(int(snap), 1)).as_integer_ratio()
        return 240.0 / self.bpm * num / den

    @property
    def beat(self):
        return 60.0 / self.bpm

    def _build(self):
        fps = self.fps
        t = 0.0                       # running time in seconds (exact, frame aligned)
        anchor_t = None
        for i, spec in enumerate(self.data["segments"]):
            if spec["name"] == self.anchor_name:
                anchor_t = t
            durs = shot_durations(spec)
            cuts, acc = [], t
            snap = spec.get("snap")
            for d in durs:
                acc += d
                if snap and anchor_t is not None:
                    g = self.grid_seconds(snap)
                    rel = round((acc - anchor_t) / g) * g
                    prev = cuts[-1] if cuts else t
                    rel = max(rel, prev - anchor_t + g)
                    cuts.append(anchor_t + rel)
                else:
                    cuts.append(acc)
            start_f = round(t * fps)
            frames = [start_f] + [round(c * fps) for c in cuts]
            for k in range(1, len(frames)):          # never allow empty shots
                frames[k] = max(frames[k], frames[k - 1] + 1)
            shot_frames = [(frames[k] - start_f, frames[k + 1] - frames[k]) for k in range(len(durs))]
            seg = Segment(i, spec["name"], spec.get("title", spec["name"]), start_f,
                          frames[-1] - start_f, shot_frames, spec)
            self.segments.append(seg)
            t = frames[-1] / fps
        self.n_frames = self.segments[-1].end_frame
        self.anchor_time = anchor_t if anchor_t is not None else 0.0

    # -- lookup -----------------------------------------------------------------
    def segment(self, name):
        for s in self.segments:
            if s.name == name or s.slug == name or str(s.index + 1) == str(name):
                return s
        raise KeyError(f"no segment named {name!r}; have: {[s.name for s in self.segments]}")

    def seg_start(self, name):
        return self.segment(name).start_frame / self.fps

    def seg_end(self, name):
        return self.segment(name).end_frame / self.fps

    @property
    def duration(self):
        return self.n_frames / self.fps

    def summary(self):
        lines = []
        for s in self.segments:
            shots = ", ".join(f"{n / self.fps:.3f}" for _, n in s.shot_frames)
            lines.append(f"{s.slug:<20} {s.start_frame / self.fps:7.3f}s  "
                         f"{s.n_frames / self.fps:6.3f}s  {len(s.shot_frames):2d} shot(s)  [{shots}]")
        lines.append(f"total {self.duration:.3f}s, {self.n_frames} frames")
        return "\n".join(lines)
