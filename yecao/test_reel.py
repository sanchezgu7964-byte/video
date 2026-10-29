#!/usr/bin/env python3
"""試片：四個關鍵場景（config/reel.toml）。

  python3 test_reel.py --keyframes      # 關鍵畫面：印在紙上的版本與「純黑白」版本 → out/reel/keyframes*.png
  python3 test_reel.py --still 12.0     # 試片中某一刻
  python3 test_reel.py                  # 整支試片（含聲音）→ out/reel/yecao_reel.mp4
"""
import argparse
import multiprocessing as mp
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ycf.reel import Reel  # noqa: E402
from ycf.plates import Plates  # noqa: E402

OUT = ROOT / "out" / "reel"

KEYFRAMES = [  # (版, p, 取景, 標題)
    ("iron_house", 0.05, (1.0, 960, 540), "鐵屋：人挨著人睡"),
    ("iron_house", 0.95, (1.0, 960, 540), "中間一個醒來，抬起頭"),
    ("onlookers", 0.7, (1.0, 960, 540), "看客"),
    ("banner", 0.5, (1.0, 960, 540), "外爭主權　內除國賊"),
    ("leaflet", 0.2, (1.0, 960, 540), "北京學界全體宣言"),
    ("leaflet", 0.9, (1.0, 960, 540), "泥印：「而不可」之後被踩掉了"),
]


def save(img, path):
    Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)).save(path)


def keyframes():
    OUT.mkdir(parents=True, exist_ok=True)
    P = Plates(5)
    for plain in (False, True):
        tiles = []
        for i, (name, p, view, _) in enumerate(KEYFRAMES):
            img = P.render(name, p, 10.0, view=view, plain=plain)
            save(img, OUT / f"key_{i}_{name}{'_plain' if plain else ''}.png")
            tiles.append(np.asarray(Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).resize((960, 540))))
        rows = [np.concatenate(tiles[k:k + 2], 1) for k in range(0, len(tiles), 2)]
        Image.fromarray(np.concatenate(rows, 0)).save(OUT / f"keyframes{'_plain' if plain else ''}.png")
    print(OUT / "keyframes.png")


def render_chunk(args):
    f0, f1, path = args
    reel = Reel()
    tl = reel.tl
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{tl.W}x{tl.H}",
           "-r", str(tl.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-x264-params", "keyint=60", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f0, f1):
        img = reel.frame(f)
        proc.stdin.write(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8).tobytes())
        if (f - f0) % 60 == 0:
            print(f"  [{path.stem}] {f - f0}/{f1 - f0}  {time.time() - t0:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    return path


def write_wav(mix, path):
    pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm.tobytes())


def full(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    reel = Reel()
    t_start = time.time()
    wav = OUT / "reel.wav"
    write_wav(reel.sound(), wav)
    print(f"sound {time.time() - t_start:.0f}s", flush=True)
    # 鏡頭邊界切塊：字草那一段的模擬要從頭跑，所以整段給同一個 worker
    n = reel.nframes
    cuts = sorted({0, n} | {int(round(s["t0"] * reel.fps)) for s in reel.shots})
    chunks = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a > 0:
            chunks.append((a, b))
    jobs = [(a, b, OUT / f"chunk_{i:03d}.mp4") for i, (a, b) in enumerate(chunks)]
    jobs.sort(key=lambda j: -(j[1] - j[0]))
    with mp.get_context("fork").Pool(workers) as pool:
        pool.map(render_chunk, jobs)
    jobs.sort(key=lambda j: j[0])
    lst = OUT / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for _, _, p in jobs))
    video = OUT / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video)],
                   check=True)
    final = OUT / "yecao_reel.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(final)], check=True)
    for _, _, p in jobs:
        p.unlink()
    print(f"→ {final}  ({time.time() - t_start:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", action="store_true")
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    a = ap.parse_args()
    if a.keyframes:
        keyframes()
    elif a.still:
        OUT.mkdir(parents=True, exist_ok=True)
        reel = Reel()
        for t in a.still:
            p = OUT / f"still_{t:05.2f}.png"
            save(reel.frame(int(round(t * reel.fps))), p)
            print(p)
    else:
        full(a.workers)


if __name__ == "__main__":
    main()
