"""11 棋盤 — match cut from the card table: same camera, same two hands.
The felt tightens into the board, the cards dissolve, the pieces rise.
"""
import numpy as np

from . import segment
from ..fx import ease
from .s10_cards import table_scene


@segment("chess")
def render(ctx):
    morph = float(ctx.params.get("morph_seconds", 1.8))
    m = ease(min(1.0, ctx.t / morph))
    # continue the card-table clock so the camera drift doesn't jump at the cut
    cards = ctx.tl.segment("cards")
    t_prev = cards.shot_frames[-1][1] / ctx.fps
    return np.clip(table_scene(ctx, m, t_prev + ctx.t), 0, 1)
