#!/usr/bin/env python3
"""《野草》題辭 —— 逐影格渲染並以 ffmpeg 合成。

  python3 render.py --sound                        # 只生成聲音 → out/sound.wav
  python3 render.py --range 14:24 --name s1        # 渲染片段（樣片）
  python3 render.py --full                         # 全片
  python3 render.py --still 65.3                   # 單張 PNG
"""
import argparse
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ycf import config, sound  # noqa: E402
from ycf.edit import Edit  # noqa: E402
from ycf.gfx import IDENTITY, Finisher, Layer, View, over  # noqa: E402
from ycf.scene_end import EndScene  # noqa: E402
from ycf.scene_field import FieldScene  # noqa: E402
from ycf.scene_opening import OpeningScene  # noqa: E402
from ycf.subtitles import Subtitles  # noqa: E402

SCENES = [OpeningScene, FieldScene, EndScene]


class Film:
    def __init__(self, use_edit=True):
        self.tl = config.load()
        self.subs = Subtitles(self.tl)
        self.fin = Finisher(self.tl.W, self.tl.H, self.tl.seed)
        self.sub_layer = Layer(self.tl.W, self.tl.H)
        self.scenes = {}
        self.edit = Edit(self.tl) if use_edit else None
        self._inserts = None
        self._recall = {}
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

    def recall(self, i):
        if i not in self._recall:
            im = Image.open(recall_path(i))
            self._recall[i] = np.asarray(im, np.float32) / 255
        return self._recall[i]

    def picture(self, f: int) -> np.ndarray:
        """依剪輯取得這一格的畫面（不含字幕與底片處理）。場景的模擬永遠跟著片子的時間走。"""
        t = f / self.tl.fps
        sc = self.scene_for(t)
        if sc is not None:
            sc.advance_to(f)
        shot = self.edit.at(t) if self.edit is not None else None
        blank = np.zeros((self.tl.H, self.tl.W, 3), np.float32)
        if sc is None:
            return blank
        if shot is None:
            return sc.render(f)
        if shot.kind == "black":
            return blank
        if shot.kind == "white":
            return blank + np.array([0.93, 0.9, 0.84], np.float32)
        if shot.kind == "insert":
            if self._inserts is None:
                from ycf.inserts import Inserts
                self._inserts = Inserts(self.tl.seed)
            kw = dict(shot.kw)
            if kw.get("beat") == "sync":
                kw["beat"] = self.tl.beat_phase(t)[1]
            if "step" in kw and kw["step"] is None:
                # 靴子落地的那一下對齊剪接點的拍
                kw["step"] = 0.5 + (t - shot.t1) / (2 * self.tl.period)
            return self._inserts.render(shot.insert, shot.insert_p(t), t, f, **kw)
        if shot.kind == "flash":
            rec = self.recall(shot.src)
            if shot.overlay:
                return sc.render(f) * (1 - shot.alpha) + rec * shot.alpha
            return rec * shot.alpha
        view = shot.view_at(t)
        if shot.follow == "survivor":
            x, y = sc.survivor_pos(t)
            view = View(view.z, x, y)
        return sc.render(f, view)

    def frame(self, f: int) -> np.ndarray:
        t = f / self.tl.fps
        img = self.picture(f)
        sc = self.scene_for(t)
        shot = self.edit.at(t) if self.edit is not None else None
        k = sc.subtitle_backing(t) if sc is not None and hasattr(sc, "subtitle_backing") else 0.0
        if shot is not None and shot.kind in ("black", "white", "flash", "insert") and not shot.overlay:
            k = 0.0
        if k > 0:
            img *= 1 - 0.75 * k * self.backing
        self.sub_layer.clear()
        # 在亮的畫面（木刻的紙、土色）上，字幕改用墨色
        band = img[900:1040:4, 300:1620:4].mean()
        if band > 0.45:
            self.subs.draw(self.sub_layer, t, color=np.array([0.06, 0.05, 0.045]), halo=(0.9, 0.86, 0.78))
        else:
            self.subs.draw(self.sub_layer, t)
        over(img, self.sub_layer.rgba())
        return self.fin(img, f)


def recall_path(i):
    return ROOT / "out" / "cache" / f"recall_{i:02d}.png"


def ensure_recall():
    """閃回用的畫面：依 edit.toml 的 recall 時刻各拍一張全景（只缺的才拍）。"""
    tl = config.load()
    import tomllib
    times = tomllib.loads((ROOT / "config/edit.toml").read_text())["recall"]["times"]
    todo = [(i, t) for i, t in enumerate(times) if not recall_path(i).exists()]
    if not todo:
        return
    recall_path(0).parent.mkdir(parents=True, exist_ok=True)
    film = Film(use_edit=False)
    for i, t in sorted(todo, key=lambda x: x[1]):
        img = film.picture(int(round(t * tl.fps)))
        Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)).save(recall_path(i))
        print(f"  recall {i:02d} @ {t:.2f}", flush=True)


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
    # 切得比核心數多，重的段落（熔岩、灰燼）才不會拖住整體
    k = max(1, min(workers * 2, n // 30))
    edges = [f0 + n * i // k for i in range(k + 1)]
    jobs = [(edges[i], edges[i + 1], chunk_dir / f"{i:03d}.mp4") for i in range(k)]
    t_start = time.time()
    ensure_recall()
    with mp.get_context("fork").Pool(min(k, workers)) as pool:
        parts = pool.map(render_chunk, jobs)
    lst = chunk_dir / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    video = chunk_dir / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(video)], check=True)
    final = out_dir / f"{name}.mp4"
    preview = out_dir / f"{name}_preview.mp4"
    if with_audio:
        full = out_dir / "sound.wav"
        if not full.exists():
            sound.write(tl, full)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ss", f"{f0 / tl.fps:.4f}",
                        "-t", f"{(f1 - f0) / tl.fps:.4f}", "-i", str(full), "-c:v", "copy",
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
    ap.add_argument("--sound", action="store_true")
    ap.add_argument("--range")
    ap.add_argument("--name", default="clip")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()
    tl = config.load()
    (ROOT / "out").mkdir(exist_ok=True)
    if a.sound:
        sound.write(tl, ROOT / "out" / "sound.wav")
        print(ROOT / "out" / "sound.wav")
    if a.still:
        ensure_recall()
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
        if not a.no_audio:
            sound.write(tl, ROOT / "out" / "sound.wav")
        render_range(0, tl.duration, "yecao_tici", a.workers, not a.no_audio)


if __name__ == "__main__":
    main()
