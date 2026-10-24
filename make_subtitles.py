#!/usr/bin/env python3
"""Write subtitles.srt from the 'subtitles' entries in timeline.json.

Times in the json are relative to each segment's start, so the srt follows any
change you make to segment durations. Run again after editing the timeline.
"""
import os

from montage.timeline import DEFAULT_TIMELINE, ROOT, Timeline


def tc(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(tl, path=os.path.join(ROOT, "subtitles.srt")):
    cues = []
    for seg in tl.segments:
        t0, dur = seg.start_frame / tl.fps, seg.n_frames / tl.fps
        for sub in seg.spec.get("subtitles", []):
            a = t0 + min(float(sub.get("start", 0.0)), dur)
            b = t0 + min(float(sub.get("end", dur)), dur)
            if b > a:
                cues.append((a, b, sub["text"]))
    cues.sort()
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(cues, 1):
            f.write(f"{i}\n{tc(a)} --> {tc(b)}\n{text}\n\n")
    return path


if __name__ == "__main__":
    print(write_srt(Timeline(DEFAULT_TIMELINE)))
