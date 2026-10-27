"""字幕：宋體字形輪廓直接以 cairo 填色；一字一字浮現、銘文式一次刻出、落款。"""
import cairo
import numpy as np

from . import fonts
from .gfx import ease, ease_out, path_polys
from .noise import noise3

PUNCT = "，。；！、："


def wrap(text: str, max_chars: int) -> list[str]:
    """盡量在標點後斷行，每行不超過 max_chars。"""
    if len(text) <= max_chars:
        return [text]
    n_rows = int(np.ceil(len(text) / max_chars))
    target = len(text) / n_rows
    rows, cur = [], ""
    for i, ch in enumerate(text):
        cur += ch
        rest = len(text) - i - 1
        if ch in PUNCT and len(cur) >= target * 0.7 and rest > 0:
            rows.append(cur)
            cur = ""
        elif len(cur) >= max_chars and rest > 0:
            rows.append(cur)
            cur = ""
    if cur:
        rows.append(cur)
    return rows


class Subtitles:
    def __init__(self, tl):
        self.tl = tl
        s = tl.raw["subtitle"]
        self.size = s["size"]
        self.interval = s["char_interval"]
        self.fade = s["char_fade"]
        self.baseline = s["baseline"]
        self.max_chars = s["max_chars"]
        self.face = fonts.face(tl.raw["fonts"]["serif"])
        self.color = tl.pal["subtitle"]
        tr = tl.raw["tremble"]
        self.tr_every, self.tr_dur, self.tr_amp = tr["every"], tr["duration"], tr["amplitude"]
        lines = tl.raw["line"]
        self.lines = []
        for i, ln in enumerate(lines):
            nxt = lines[i + 1]["at"] if i + 1 < len(lines) else tl.duration
            fo = ln.get("fade_out", 0.5)
            # 預設在下一行開始前淡完，兩行不在同一位置重疊
            self.lines.append(dict(
                text=ln["text"], at=ln["at"], until=ln.get("until", nxt - fo - 0.05),
                mode=ln.get("mode", "type"), fade_out=fo))
        for ln in self.lines:
            ln["glyphs"] = self._layout(ln)

    def _layout(self, ln):
        mode = ln["mode"]
        if mode == "colophon":
            size, track, max_c = 26, 0.12, 40
        elif mode == "carve":
            size, track, max_c = 40, 0.14, 24
        else:
            size, track, max_c = self.size, 0.08, self.max_chars
        rows = wrap(ln["text"], max_c)
        lead = size * 1.9
        W = self.tl.W
        out = []
        n = 0
        for r, row in enumerate(rows):
            adv = [self.face.glyph(ch).advance * size * (1 + track) for ch in row]
            x = W / 2 - sum(adv) / 2 + size * track / 2
            if mode == "colophon":
                y = 1000
            elif mode == "carve":
                y = 540 - (len(rows) - 1) * lead / 2 + r * lead + size * 0.35 + 330
            else:
                y = self.baseline - (len(rows) - 1 - r) * lead
            for ch, a in zip(row, adv):
                out.append((ch, x, y, size, n))
                x += a
                n += 1
        return out

    def visible(self, t):
        return [ln for ln in self.lines if ln["at"] - 0.01 <= t < ln["until"] + ln["fade_out"] + 0.01]

    def draw(self, layer, t: float, alpha: float = 1.0, color=None):
        ctx = layer.ctx
        col = self.color if color is None else color
        k = int(t // self.tr_every)
        tr_on = (t - k * self.tr_every) < self.tr_dur and k > 0
        for ln in self.visible(t):
            mode = ln["mode"]
            fo = 1.0
            if t > ln["until"]:
                fo = 1 - ease((t - ln["until"]) / max(ln["fade_out"], 1e-3)) if ln["fade_out"] > 0 else 0.0
            g = ln["glyphs"]
            tremble_i = int((k * 2654435761) % 997) % len(g) if tr_on else -1
            for ch, x, y, size, i in g:
                if mode == "carve":
                    a = ease_out((t - ln["at"]) / 0.35, 2)
                    dy = 0.0
                elif mode == "colophon":
                    a = ease((t - ln["at"] - i * 0.14) / 1.2)
                    dy = (1 - a) * 3
                else:
                    a = ease((t - ln["at"] - i * self.interval) / self.fade)
                    dy = (1 - a) * 5
                a *= fo * alpha
                if a <= 0.003 or ch in "　 ":
                    continue
                dx = 0.0
                if i == tremble_i and a > 0.9:
                    ph = (t - k * self.tr_every) / self.tr_dur
                    env = np.sin(np.pi * ph)
                    dx = self.tr_amp * env * float(noise3(t * 38, 0.5, k, 77))
                    dy += self.tr_amp * env * float(noise3(t * 41, 3.5, k, 78))
                glyph = self.face.glyph(ch)
                polys = [c * size + (x + dx, y + dy) for c in glyph.contours]
                ctx.new_path()
                path_polys(ctx, polys)
                ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
                if mode == "carve":
                    # 銘文：先刻出暗槽，再在偏移處上亮色
                    ctx.save()
                    ctx.translate(1.2, 1.4)
                    ctx.set_source_rgba(0, 0, 0, 0.55 * a)
                    ctx.fill_preserve()
                    ctx.restore()
                    ctx.new_path()
                    path_polys(ctx, polys)
                # 柔和暗邊增加可讀性
                ctx.set_source_rgba(0.02, 0.02, 0.02, 0.28 * a)
                ctx.set_line_width(3.0)
                ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                ctx.stroke_preserve()
                ctx.set_source_rgba(float(col[0]), float(col[1]), float(col[2]), a)
                ctx.fill()
