"""意象鏡頭：剪輯層（edit.toml 的 insert = "名稱"）與刻版（plates.py、plates2.py）之間的接口。

每個意象都是一張刻出來的版（黑白木刻），只有「頭」是字草世界裡的一段（vignettes.py）。
舊的名字保留成別名：crowd 的 surge／formation／empty 分別是街頭的橫幅、被收編的方陣、散場的街。
"""
import numpy as np

from .plates import Plates

ALIAS = {("crowd", "surge"): "banner", ("crowd", "formation"): "formation", ("crowd", "empty"): "square"}
TOU_DUR = 13.0


class Inserts:
    def __init__(self, seed, tl=None):
        self.plates = Plates(seed)
        self.tl = tl
        self._tou = None

    def is_plate(self, name):
        return name != "tou"

    def render(self, name, p, t, frame, view=(1.0, 960, 540), **kw):
        p = float(np.clip(p, 0, 1))
        if name == "tou":
            if self._tou is None:
                from .vignettes import LeafletGrass
                self._tou = LeafletGrass(self.tl)
            tau = p * TOU_DUR
            return self._tou.render(int(round(tau * self._tou.fps)), burn=float(kw.get("burn", 0.0)))
        if name == "crowd":
            name = ALIAS[("crowd", kw.pop("variant", "surge"))]
        if "view" in kw:
            view = tuple(kw.pop("view"))
        return self.plates.render(name, p, t, view=view, **kw)
