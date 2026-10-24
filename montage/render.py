"""Frame pipeline: render frames in parallel, pipe them into ffmpeg, keep thumbnails."""
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from . import segments
from .fx import finish
from .timeline import Timeline

_TL = None
_TL_FOR_THUMBS = [None]
_SCALE = 1.0


def _init(path, scale):
    global _TL, _SCALE
    _TL = Timeline(path)
    _SCALE = scale


def frame_size(tl, scale):
    W = int(round(tl.W * scale / 2)) * 2
    H = int(round(tl.H * scale / 2)) * 2
    return W, H


def render_frame(tl, seg, frame, scale=1.0):
    """Render one finished frame (uint8 HxWx3) of `seg` at local frame index."""
    W, H = frame_size(tl, scale)
    ctx = segments.Ctx(tl, seg, frame, W, H, tl.supersample)
    img = segments.get(seg.name)(ctx)
    look = dict(tl.look)
    look.update(seg.spec.get("look_override", {}))
    rng = np.random.default_rng([tl.seed, 999, seg.start_frame + frame])
    return finish(img, seg.look, look, rng, H / 1080.0)


def _work(task):
    name, frame = task
    seg = _TL.segment(name)
    return render_frame(_TL, seg, frame, _SCALE).tobytes()


def thumb_frames(seg, n_even=8):
    """Which local frames go on the contact sheet."""
    if seg.name in segments.THUMBS:
        return segments.THUMBS[seg.name](_TL_FOR_THUMBS[0], seg)
    if len(seg.shot_frames) > 1:
        return [s + n // 2 for s, n in seg.shot_frames]
    n = seg.n_frames
    return [int(round(i * (n - 1) / (n_even - 1))) for i in range(n_even)]


def encode(tl, segs, out_path, audio_path=None, scale=1.0, workers=None, crf=20, log=True):
    """Render the given segments back to back into one mp4.

    Returns {segment_name: [(local_frame, PIL thumbnail), ...]} for contact sheets."""
    W, H = frame_size(tl, scale)
    _TL_FOR_THUMBS[0] = tl
    tasks = [(s.name, f) for s in segs for f in range(s.n_frames)]
    wanted = {(s.name, f) for s in segs for f in thumb_frames(s)}
    thumbs = {s.name: [] for s in segs}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(tl.fps), "-i", "-"]
    if audio_path:
        cmd += ["-i", audio_path]
    meta = tl.data["meta"]
    crf = meta.get("crf", crf)
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf)]
    if meta.get("x264_tune"):
        cmd += ["-tune", meta["x264_tune"]]
    cmd += ["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"]
    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "256k"]
    cmd += [out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    workers = workers or max(1, (os.cpu_count() or 2))
    t0 = time.time()
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(tl.path, scale)) as pool:
        for i, buf in enumerate(pool.imap(_work, tasks, chunksize=1)):
            proc.stdin.write(buf)
            if tasks[i] in wanted:
                im = Image.frombytes("RGB", (W, H), buf)
                thumbs[tasks[i][0]].append((tasks[i][1], im.resize((480, 270), Image.LANCZOS)))
            if log and (i % 24 == 0 or i == len(tasks) - 1):
                el = time.time() - t0
                sys.stdout.write(f"\r  frame {i + 1}/{len(tasks)}  {el:6.1f}s  "
                                 f"({(i + 1) / max(el, 1e-6):.2f} fps)   ")
                sys.stdout.flush()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    if log:
        print()
    return thumbs
