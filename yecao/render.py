#!/usr/bin/env python3
"""《野草》題辭 —— 逐影格渲染並以 ffmpeg 合成。

  python3 render.py --analyze                      # 偵測配樂拍點，寫入 analysis/beats.json
  python3 render.py --range 14:24 --name s1        # 渲染片段（樣片）
  python3 render.py --full                         # 全片
  python3 render.py --still 65.3                   # 單張 PNG
"""
import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ycf import audio, config  # noqa: E402
from ycf.gfx import Finisher, Layer, over  # noqa: E402
from ycf.scene_end import EndScene  # noqa: E402
from ycf.scene_field import FieldScene  # noqa: E402
from ycf.scene_opening import OpeningScene  # noqa: E402
from ycf.subtitles import Subtitles  # noqa: E402

SCENES = [OpeningScene, FieldScene, EndScene]


class Film:
    def __init__(self):
        self.tl = config.load()
        self.subs = Subtitles(self.tl)
        self.fin = Finisher(self.tl.W, self.tl.H, self.tl.seed)
        self.sub_layer = Layer(self.tl.W, self.tl.H)
        self.scenes = {}
        y = np.arange(self.tl.H, dtype=np.float32)
        b = self.tl.raw["subtitle"]["baseline"] - 14
        self.backing = np.exp(-((y - b) / 42) ** 2)[:, None, None]

    def scene_for(self, t):
        for S in SCENES:
            if S.t0 <= t < S.t1:
                if S not in self.scenes:
                    self.scenes.clear()  # 舊場景不再需要，釋放記憶體
                    self.scenes[S] = S(self.tl)
                return self.scenes[S]
        return None

    def frame(self, f: int) -> np.ndarray:
        t = f / self.tl.fps
        sc = self.scene_for(t)
        if sc is None:
            img = np.zeros((self.tl.H, self.tl.W, 3), np.float32)
        else:
            sc.advance_to(f)
            img = sc.render(f)
        k = sc.subtitle_backing(t) if sc is not None and hasattr(sc, "subtitle_backing") else 0.0
        if k > 0:
            img *= 1 - 0.75 * k * self.backing
        self.sub_layer.clear()
        self.subs.draw(self.sub_layer, t)
        over(img, self.sub_layer.rgba())
        return self.fin(img, f)


def render_chunk(args):
    f0, f1, path = args
    film = Film()
    tl = film.tl
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{tl.W}x{tl.H}",
           "-r", str(tl.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
           "-tune", "grain", "-pix_fmt", "yuv420p", "-x264-params", "keyint=60", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t_start = time.time()
    for f in range(f0, f1):
        proc.stdin.write(film.frame(f).tobytes())
        if (f - f0) % 60 == 0:
            el = time.time() - t_start
            print(f"  [{path.stem}] {f - f0}/{f1 - f0}  {el:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    return path


def render_range(t0, t1, name, workers, with_audio=True):
    tl = config.load()
    f0, f1 = int(round(t0 * tl.fps)), int(round(t1 * tl.fps))
    out_dir = ROOT / "out"
    chunk_dir = out_dir / "chunks" / name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n = f1 - f0
    k = max(1, min(workers, n // 30))
    edges = [f0 + n * i // k for i in range(k + 1)]
    jobs = [(edges[i], edges[i + 1], chunk_dir / f"{i:03d}.mp4") for i in range(k)]
    t_start = time.time()
    with mp.get_context("fork").Pool(k) as pool:
        parts = pool.map(render_chunk, jobs)
    lst = chunk_dir / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    video = chunk_dir / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(video)], check=True)
    final = out_dir / f"{name}.mp4"
    preview = out_dir / f"{name}_preview.mp4"
    if with_audio:
        wav = chunk_dir / "audio.wav"
        audio.mix(tl, wav, f0 / tl.fps, f1 / tl.fps)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(final)],
                       check=True)
    else:
        video.replace(final)
    # 便於分享的輕量版
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(final), "-c:v", "libx264", "-preset", "slow",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(preview)], check=True)
    print(f"→ {final}  ({time.time() - t_start:.0f}s)")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--range")
    ap.add_argument("--name", default="clip")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()
    tl = config.load()
    if a.analyze:
        cues = [float(v) for v in tl.cues.values()]
        res = audio.analyze(ROOT / tl.raw["audio"]["file"], tl.raw["audio"]["bpm_hint"], cues)
        (ROOT / tl.raw["audio"]["analysis"]).write_text(json.dumps(res))
        print(f"BPM {res['bpm']:.2f}, {len(res['beats'])} beats")
    if a.still:
        film = Film()
        for t in a.still:
            img = film.frame(int(round(t * tl.fps)))
            p = ROOT / "out" / f"still_{t:07.2f}.png"
            Image.fromarray(img).save(p)
            print(p)
    if a.range:
        t0, t1 = (float(v) for v in a.range.split(":"))
        render_range(t0, t1, a.name, a.workers, not a.no_audio)
    if a.full:
        render_range(0, tl.duration, "yecao_tici", a.workers, not a.no_audio)


if __name__ == "__main__":
    main()
