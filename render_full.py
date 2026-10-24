#!/usr/bin/env python3
"""Render the whole film in one pass: video + mixed soundtrack + srt + contact sheets.

  python3 render_full.py                    # everything, 1920x1080
  python3 render_full.py --preview          # half resolution
  python3 render_full.py --upto rail_yard   # only segments 1..rail_yard (for checking a prefix)
  python3 render_full.py --audio-only       # only rebuild + swap the soundtrack

Output:
  out/<meta.output_name>.mp4  the film (montage_v2.mp4)
  out/audio/mix.wav           the soundtrack
  subtitles.srt               placeholder subtitles
  out/contact/NN_name.png     one contact sheet per segment
  out/contact/overview.png    all segments on one page
"""
import argparse
import os

from montage import audio
from montage.contact import overview_sheet, segment_sheet
from montage.render import encode
from montage.timeline import DEFAULT_TIMELINE, ROOT, Timeline
from make_subtitles import write_srt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=DEFAULT_TIMELINE)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--upto", default=None, help="last segment to include")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--audio-only", action="store_true",
                    help="rebuild the soundtrack and swap it into the existing mp4 (no video re-render)")
    a = ap.parse_args()

    tl = Timeline(a.timeline)
    scale = a.scale if a.scale else (0.5 if a.preview else 1.0)
    segs = tl.segments
    if a.upto:
        last = tl.segment(a.upto).index
        segs = segs[:last + 1]
    print(tl.summary())

    srt = write_srt(tl)
    sr = tl.audio.get("sample_rate", 48000)
    mix = audio.render_mix(tl)[: segs[-1].end_frame * (sr // tl.fps)]
    wav = os.path.join(ROOT, "out", "audio", "mix.wav" if not a.upto else f"mix_upto_{segs[-1].slug}.wav")
    audio.write_wav(wav, mix, sr)

    name = tl.data["meta"].get("output_name", "montage") + (f"_upto_{segs[-1].slug}" if a.upto else "") + ("" if scale == 1 else f"_x{scale:g}")
    out = a.out or os.path.join(ROOT, "out", name + ".mp4")
    if a.audio_only:
        from render_segment import remux
        remux(out, wav)
        print(out, "(audio replaced)")
        return
    thumbs = encode(tl, segs, out, wav, scale, a.workers)

    for s in segs:
        segment_sheet(tl, s, thumbs[s.name], os.path.join(ROOT, "out", "contact", f"{s.slug}.png"))
    ov = overview_sheet(tl, thumbs, os.path.join(ROOT, "out", "contact", "overview.png"))
    print("\n".join(["", out, wav, srt, ov]))


if __name__ == "__main__":
    main()
