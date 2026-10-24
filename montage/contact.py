"""Contact sheets: a grid of thumbnails per segment, plus an overview of the whole film."""
import os

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def segment_sheet(tl, seg, thumbs, out_path, cols=4):
    tw, th, pad, head = 480, 270, 14, 64
    thumbs = sorted(thumbs, key=lambda x: x[0])
    rows = (len(thumbs) + cols - 1) // cols
    W = cols * tw + (cols + 1) * pad
    H = head + rows * (th + 30) + pad
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 16), f"{seg.slug}  {seg.title}", fill=(235, 235, 235), font=font(26))
    info = (f"{seg.n_frames / tl.fps:.3f}s · {len(seg.shot_frames)} shot(s) · "
            f"starts {seg.start_frame / tl.fps:.3f}s · look: {seg.look}")
    d.text((W - pad, 22), info, fill=(150, 150, 150), font=font(18), anchor="ra")
    for k, (f, im) in enumerate(thumbs):
        x = pad + (k % cols) * (tw + pad)
        y = head + (k // cols) * (th + 30)
        sheet.paste(im, (x, y))
        shot, _, _ = seg.shot_at(f)
        label = f"f{f:04d}  {f / tl.fps:5.2f}s"
        if len(seg.shot_frames) > 1:
            label += f"  shot {shot + 1}"
        d.text((x, y + th + 5), label, fill=(170, 170, 170), font=font(16))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    return out_path


def overview_sheet(tl, all_thumbs, out_path, per_row=6):
    """One row per segment, up to `per_row` thumbnails each, at half size."""
    tw, th, pad, lab = 240, 135, 8, 190
    segs = [s for s in tl.segments if s.name in all_thumbs]
    W = lab + per_row * (tw + pad) + pad
    H = len(segs) * (th + pad) + pad
    sheet = Image.new("RGB", (W, H), (14, 14, 14))
    d = ImageDraw.Draw(sheet)
    for r, s in enumerate(segs):
        y = pad + r * (th + pad)
        d.text((pad, y + 40), f"{s.slug}", fill=(220, 220, 220), font=font(16))
        d.text((pad, y + 64), f"{s.title}  {s.n_frames / tl.fps:.2f}s", fill=(150, 150, 150), font=font(15))
        th_list = sorted(all_thumbs[s.name], key=lambda x: x[0])
        if len(th_list) > per_row:
            idx = [round(i * (len(th_list) - 1) / (per_row - 1)) for i in range(per_row)]
            th_list = [th_list[i] for i in idx]
        for k, (_, im) in enumerate(th_list):
            sheet.paste(im.resize((tw, th), Image.LANCZOS), (lab + k * (tw + pad), y))
    sheet.save(out_path)
    return out_path
