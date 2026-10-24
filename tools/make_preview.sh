#!/bin/sh
# Small review copy of an mp4: 960x540, light denoise so the grain compresses, AAC 128k.
# usage: tools/make_preview.sh in.mp4 out.mp4 [crf]
in="$1"; out="$2"; crf="${3:-27}"
ffmpeg -y -loglevel error -i "$in" -vf "scale=960:540:flags=lanczos,hqdn3d=3:3:4:4" \
  -c:v libx264 -preset slow -crf "$crf" -pix_fmt yuv420p -profile:v main -movflags +faststart \
  -c:a aac -b:a 128k "$out"
