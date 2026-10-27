"""字形輪廓：從分片 TTF 取出字形，攤平成多邊形（em 單位，基線 y=0，向上為負）。"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

from .config import ROOT


class _FlattenPen(BasePen):
    def __init__(self, glyphset, steps=10):
        super().__init__(glyphset)
        self.steps = steps
        self.contours: list[list] = []
        self.cur: list = []

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, self.steps + 1)[1:]:
            u = 1 - t
            self.cur.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                             u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, self.steps + 1)[1:]:
            u = 1 - t
            self.cur.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                             u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))

    def _closePath(self):
        if len(self.cur) > 2:
            self.contours.append(self.cur)
        self.cur = []

    _endPath = _closePath


class Glyph:
    __slots__ = ("ch", "contours", "advance", "bbox")

    def __init__(self, ch, contours, advance):
        self.ch = ch
        self.contours = contours
        self.advance = advance
        if contours:
            allp = np.concatenate(contours)
            self.bbox = (*allp.min(0), *allp.max(0))
        else:
            self.bbox = (0, 0, 0, 0)


class Face:
    def __init__(self, rel: str):
        self.dir = ROOT / rel
        self.index = json.loads((self.dir / "index.json").read_text())
        self._fonts: dict[str, TTFont] = {}

    def _font(self, fn):
        if fn not in self._fonts:
            self._fonts[fn] = TTFont(self.dir / fn, lazy=True)
        return self._fonts[fn]

    @lru_cache(maxsize=None)
    def glyph(self, ch: str) -> Glyph:
        fn = self.index.get(ch)
        if fn is None:
            return Glyph(ch, [], 1.0)
        font = self._font(fn)
        upem = font["head"].unitsPerEm
        gs = font.getGlyphSet()
        name = font.getBestCmap()[ord(ch)]
        pen = _FlattenPen(gs)
        gs[name].draw(pen)
        contours = [np.array(c, np.float32) * np.array([1, -1], np.float32) / upem for c in pen.contours]
        return Glyph(ch, contours, gs[name].width / upem)


_faces: dict[str, Face] = {}


def face(rel: str) -> Face:
    if rel not in _faces:
        _faces[rel] = Face(rel)
    return _faces[rel]
