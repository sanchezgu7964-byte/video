"""33 最後一句 — cut to black, and stay there."""
import numpy as np

from . import segment


@segment("last_line")
def render(ctx):
    return np.zeros((ctx.H, ctx.W), np.float32)
