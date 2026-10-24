#!/usr/bin/env python3
"""Render one (or several) segments on their own, with their slice of the soundtrack.

  python3 render_segment.py --segment rail_yard
  python3 render_segment.py --segment 1 2            # by number, several at once
  python3 render_segment.py --segment viaduct --preview        # half resolution, fast
  python3 render_segment.py --segment viaduct --still 3.5      # one PNG at t = 3.5 s
  python3 render_segment.py --segment rail_yard --audio-only   # only redo the sound
  python3 render_segment.py --list

Output: out/segments/NN_name.mp4 and out/contact/NN_name.png
"""
import argparse
import os
import subprocess

import numpy as np
from PIL import Image

from montage import audio, segments
from montage.contact import segment_sheet
from montage.render import encode, render_frame
from montage.timeline import DEFAULT_TIMELINE, ROOT, Timeline


def remux(mp4, wav):
    tmp = mp4 + ".tmp.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, mp4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", "-s", nargs="+", help="segment name, slug or number")
    ap.add_argument("--timeline", default=DEFAULT_TIMELINE)
    ap.add_argument("--preview", action="store_true", help="render at half resolution")
    ap.add_argument("--scale", type=float, default=None, help="resolution scale (overrides --preview)")
    ap.add_argument("--still", type=float, nargs="*", help="render PNG stills at these times (s)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--audio-only", action="store_true",
                    help="regenerate the sound and remux it into the existing mp4 (no video re-render)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    tl = Timeline(a.timeline)
    if a.list or not a.segment:
        print(tl.summary())
        for s in tl.segments:
            print(f"  {s.slug:<20} {'ready' if segments.implemented(s.name) else 'placeholder'}")
        return
    scale = a.scale if a.scale else (0.5 if a.preview else 1.0)
    suffix = "" if scale == 1.0 else f"_x{scale:g}"

    mix = None
    for name in a.segment:
        seg = tl.segment(name)
        if a.still is not None:
            os.makedirs(os.path.join(ROOT, "out", "stills"), exist_ok=True)
            for t in a.still or [seg.n_frames / tl.fps / 2]:
                f = min(seg.n_frames - 1, int(round(t * tl.fps)))
                img = render_frame(tl, seg, f, scale)
                p = os.path.join(ROOT, "out", "stills", f"{seg.slug}_f{f:04d}{suffix}.png")
                Image.fromarray(img).save(p)
                print(p)
            continue

        print(f"{seg.slug} {seg.title}: {seg.n_frames} frames ({seg.n_frames / tl.fps:.2f}s), scale {scale}")
        wav = None
        if not a.no_audio:
            if mix is None:
                mix = audio.render_mix(tl)          # once per run, shared by all requested segments
            sr = tl.audio.get("sample_rate", 48000)
            spf = sr // tl.fps
            wav = os.path.join(ROOT, "out", "audio", f"{seg.slug}.wav")
            audio.write_wav(wav, mix[seg.start_frame * spf: seg.end_frame * spf], sr)
        mp4 = os.path.join(ROOT, "out", "segments", f"{seg.slug}{suffix}.mp4")
        if a.audio_only:
            remux(mp4, wav)
            print(" ", mp4, "(audio replaced)")
            continue
        thumbs = encode(tl, [seg], mp4, wav, scale, a.workers)
        sheet = segment_sheet(tl, seg, thumbs[seg.name],
                              os.path.join(ROOT, "out", "contact", f"{seg.slug}.png"))
        print(" ", mp4)
        print(" ", sheet)


if __name__ == "__main__":
    main()
