"""剪輯層：把連續的場景切成鏡頭。

鏡頭種類
- scene：照常渲染場景，但用虛擬攝影機取景（view = [z, cx, cy]，可給 to = 結束時的取景做推拉搖移）
- black / white：純黑、純白（mute = true 時連聲音也切掉）
- flash：閃回一張先前的畫面（recall 清單中的第 src 張），可疊在目前的場景上（overlay）

明確寫在 config/edit.toml 的鏡頭之外，[[burst]] 會依場景資料自動生成一串鏡頭
（例如每一次踐踏切一個近景、作證時每一拍切一次）。burst 的區間內，明確鏡頭會被取代。
"""
import tomllib
from dataclasses import dataclass, field

import numpy as np

from .config import ROOT
from .gfx import View, ease
from .noise import rand

WIDE = (1.0, 960.0, 540.0)


@dataclass
class Shot:
    t0: float
    kind: str = "scene"
    view: tuple = WIDE
    to: tuple | None = None
    src: int = 0
    alpha: float = 1.0
    overlay: bool = False
    mute: bool = False
    tick: bool = False
    label: str = ""
    t1: float = 0.0
    follow: str | None = None
    insert: str | None = None
    p: tuple = (0.0, 1.0)
    kw: dict = field(default_factory=dict)

    def insert_p(self, t):
        u = (t - self.t0) / max(self.t1 - self.t0, 1e-6)
        return self.p[0] + (self.p[1] - self.p[0]) * u

    def view_at(self, t):
        if self.to is None:
            return clamp(*self.view)
        k = ease((t - self.t0) / max(self.t1 - self.t0, 1e-6))
        a, b = np.array(self.view), np.array(self.to)
        # 推拉用對數內插，速度才均勻
        z = float(np.exp(np.log(a[0]) + (np.log(b[0]) - np.log(a[0])) * k))
        return clamp(z, *(a[1:] + (b[1:] - a[1:]) * k))


def clamp(z, cx, cy, W=1920, H=1080):
    """取景不超出全景畫面。"""
    hw, hh = W / 2 / z, H / 2 / z
    return View(z, float(np.clip(cx, hw, W - hw)), float(np.clip(cy, hh, H - hh)))


class Edit:
    def __init__(self, tl, field_scene=None, end_scene=None):
        self.tl = tl
        raw = tomllib.loads((ROOT / "config/edit.toml").read_text())
        self.recall = [float(x) for x in raw["recall"]["times"]]
        shots = [self._shot(d) for d in raw.get("shot", [])]
        self._field = field_scene
        self._end = end_scene
        for b in raw.get("burst", []):
            gen = getattr(self, "_burst_" + b["type"])
            lo, hi = b["start"], b["end"]
            new = [s_ for s_ in gen(b) if s_.t0 < hi - 1e-6]     # 鏡頭串不越過自己的區間
            shots = [s for s in shots if not (lo <= s.t0 < hi)] + new
        # 剪接點對齊到格；同一格上有兩個鏡頭時，後寫的那個算數
        fps = tl.fps
        for s_ in shots:
            s_.t0 = round(s_.t0 * fps) / fps
        shots.sort(key=lambda s: s.t0)
        dedup = {}
        for s_ in shots:
            dedup[s_.t0] = s_
        shots = list(dedup.values())
        for i, s in enumerate(shots):
            s.t1 = shots[i + 1].t0 if i + 1 < len(shots) else tl.duration
        self.shots = [s for s in shots if s.t1 - s.t0 > 1e-4]
        self.starts = np.array([s.t0 for s in self.shots])

    # ── 讀取 ──
    def _shot(self, d):
        known = {"at", "kind", "view", "to", "src", "alpha", "overlay", "mute", "tick", "label", "follow", "insert", "p"}
        p = d.get("p", (0.0, 1.0))
        if isinstance(p, (int, float)):
            p = (p, p)
        return Shot(t0=float(d["at"]), kind=d.get("kind", "insert" if "insert" in d else "scene"),
                    view=tuple(d.get("view", WIDE)), to=tuple(d["to"]) if "to" in d else None,
                    src=int(d.get("src", 0)), alpha=float(d.get("alpha", 1.0)), overlay=bool(d.get("overlay", False)),
                    mute=bool(d.get("mute", False)), tick=bool(d.get("tick", "insert" in d)),
                    label=d.get("label", ""), follow=d.get("follow"), insert=d.get("insert"), p=tuple(p),
                    kw={k: v for k, v in d.items() if k not in known})

    def at(self, t):
        i = int(np.searchsorted(self.starts, t + 1e-6, side="right")) - 1
        return self.shots[max(i, 0)]

    @property
    def field(self):
        if self._field is None:
            from .scene_field import FieldScene
            self._field = FieldScene(self.tl)
        return self._field

    @property
    def end(self):
        if self._end is None:
            from .scene_end import EndScene
            self._end = EndScene(self.tl)
        return self._end

    # ── 自動生成的鏡頭串 ──
    def _burst_alternate(self, b):
        """在兩個（或多個）取景之間交替切，鏡頭越來越短。"""
        views = [tuple(v) for v in b["views"]]
        t, i, out = b["start"], 0, []
        d0, d1 = b["from"], b["to"]
        while t < b["end"] - 1e-3:
            k = (t - b["start"]) / (b["end"] - b["start"])
            out.append(Shot(t0=t, view=views[i % len(views)], tick=b.get("tick", True),
                            label=b.get("labels", [""] * len(views))[i % len(views)]))
            t += d0 + (d1 - d0) * k
            i += 1
        return out

    def _burst_names(self, b):
        """地底名字一閃：每個名字切一個極近的特寫。"""
        out = []
        for n in sorted(self.field.names, key=lambda n: n["t"])[:b.get("count", 3)]:
            t = n["t"] + 0.08
            out.append(Shot(t0=t, view=(4.0, n["x"] + 40, 720 + n["y"] - 10), tick=True, label="地底一閃而沉的名字"))
            out.append(Shot(t0=t + b.get("hold", 0.4), view=tuple(b["back"]), label="田野"))
        return out

    def _burst_stomps(self, b):
        """踐踏與刪刈：每一下切一個近景；加密之後，踐踏落地那一格閃白。"""
        f = self.field
        out = [Shot(t0=b["start"], view=WIDE, label="田野")]
        rng = rand(self.tl.seed, 50, 1)
        evs = [e for e in f.events if b["start"] <= e["t"] < b["end"]]
        n_st = n_bl = 0
        for e in evs:
            dense = e["t"] >= self.tl.c("stomp_halfbeat")
            if e["kind"] == "stomp":
                n_st += 1
            else:
                n_bl += 1
            # 每隔三下：踐踏切到行進的靴子，刪刈切到看客（不再每一下都切走，讓田野自己被看清楚）
            if e["kind"] == "stomp" and n_st % 4 == 2:
                d = 0.12 if dense else 0.3
                out.append(Shot(t0=e["t"] - d, kind="insert", insert="boots", p=(0.4, 0.5), tick=True,
                                kw={"step": None}, label="靴子"))
                out.append(Shot(t0=e["t"] + (1 / 30 if dense else 0.05), view=(2.4, e["xc"], 700), label="被踩扁的字"))
                out.append(Shot(t0=e["t"] + (0.25 if dense else 0.4), view=WIDE, label="田野"))
                continue
            if e["kind"] == "blade" and n_bl % 4 == 2:
                d = 0.1 if dense else 0.25
                out.append(Shot(t0=e["t"] + 0.12, kind="insert", insert="onlookers", p=(0.5, 0.6 + 0.1 * dense),
                                tick=True, label="看客"))
                out.append(Shot(t0=e["t"] + 0.12 + d, view=WIDE, label="田野"))
                continue
            if e["kind"] == "stomp":
                z = 3.0 if dense else 2.2
                out.append(Shot(t0=e["t"] - 0.07, view=(z, e["xc"], 690), tick=True, label="印版壓下"))
                if dense:
                    out.append(Shot(t0=e["t"], kind="white", tick=False, label="（白）"))
                    out.append(Shot(t0=e["t"] + 1 / 30, view=(z, e["xc"], 700), label="被壓扁的字"))
                out.append(Shot(t0=e["t"] + (0.28 if dense else 0.4), view=WIDE if not dense else
                                (1.35, rng.uniform(700, 1220), 600), label="田野"))
            else:
                # 刀：近景盯著一株會被切到的字
                tall = [p for p in f.plants if p.h * p.cut_u > e["hc"] + 6 and p.row >= 1] or f.plants
                p = tall[rng.integers(len(tall))]
                z = 2.6 if dense else 1.9
                out.append(Shot(t0=e["t"] - 0.16, view=(z, p.x, 720 - e["hc"]), tick=True, label="刀刃掃過"))
                out.append(Shot(t0=e["t"] + (0.2 if dense else 0.35), view=WIDE, label="田野"))
        return out

    def _burst_grid(self, b):
        """自愛之後：口號網格與被收編的野草交叉剪接，跟著網格的拍子。"""
        f = self.field
        beats = self.tl.beats_between(b["start"], b["end"])
        absorbed = sorted([p for p in f.plants if p.absorb_t is not None], key=lambda p: p.absorb_t)
        out, j = [], 0
        rng = rand(self.tl.seed, 51, 1)
        for i, t in enumerate(beats):
            s, hz = f.camera(t)
            if i % 3 == 2:
                out.append(Shot(t0=t, kind="insert", insert="crowd", kw={"variant": "formation", "beat": "sync"},
                                tick=True, label="方陣：一模一樣的旗"))
                continue
            if i % 2 == 0:
                cx, cy = rng.uniform(250, 1670), rng.choice([rng.uniform(120, 380), rng.uniform(760, 960)])
                out.append(Shot(t0=t, view=(3.0, cx, cy), tick=True, label="口號網格"))
            else:
                p = absorbed[min(j, len(absorbed) - 1)]
                j += 3
                x = 960 + s * (p.x - 960)
                out.append(Shot(t0=t, view=(2.6, x, hz - 50 * s), tick=True, label="被排進格子的野草"))
        return out

    def _burst_witness(self, b):
        """作證：分界線每一拍重切一次，鏡頭也每一拍切一次；輪廓出現時在兩側之間切。"""
        f = self.field
        pair_t = {round(p["t"], 3): p for p in f.pairs}
        from .clump import BASE
        out = [Shot(t0=b["start"], view=WIDE, label="分界線剖開畫面")]
        views = [WIDE, (2.2, BASE[0], BASE[1] - 50), (1.4, 960, 560)]
        labels = ["兩側", "站在線上的一叢", "銘文與兩側"]
        for i, t in enumerate(f.w_beats):
            pr = pair_t.get(round(float(t), 3))
            if pr is not None:
                ang = f.w_angle(t + 0.2)
                dvec = np.array([np.cos(ang), np.sin(ang)])
                nvec = np.array([-np.sin(ang), np.cos(ang)])
                ca = np.array(BASE) + pr["u"] * dvec + pr["v"] * nvec
                cb = np.array(BASE) + pr["u"] * dvec - pr["v"] * nvec
                half = self.tl.period / 2
                out.append(Shot(t0=t, view=(1.9, *ca), tick=True, label="一側的輪廓"))
                out.append(Shot(t0=t + half, view=(1.9, *cb), tick=True, label="另一側的輪廓"))
            elif i % 2 == 0:
                # 隔一拍才換一次取景：輪廓出現的那一拍之外，給畫面停留
                k = (i // 2) % len(views)
                out.append(Shot(t0=t, view=views[k], tick=True, label=labels[k]))
        return out

    def _burst_cycle(self, b):
        """火速：每一圈換一個取景；越轉越快時夾進先前所有畫面的閃回，最後兩秒只剩閃回。"""
        e = self.end
        from .clump import BASE
        views = [WIDE, (2.1, BASE[0], BASE[1] - 60), (1.6, 620, 560), (1.6, 1300, 560), (3.2, BASE[0], BASE[1] - 40)]
        out, r = [], 0
        n_recall = len(self.recall)
        tail = b["end"] - b["recall_tail"]
        for k, (tc, d) in enumerate(e.cycles):
            t = max(tc, b["start"])
            if t >= tail:
                break
            out.append(Shot(t0=t, view=views[k % len(views)], tick=True, label="生長、腐爛、重生"))
            if k >= 1:
                cal = min(d * 0.35, 0.45)
                out.append(Shot(t0=t + d * 0.62, kind="insert", insert="calendar", p=(0.0, 1.0), tick=True,
                                kw={"year": (k - 1) % 6}, label="撕下一頁日曆"))
                out.append(Shot(t0=t + d * 0.62 + cal, view=views[(k + 1) % len(views)], label="生長、腐爛、重生"))
            if d < 1.3 and r < n_recall:
                out.append(Shot(t0=t + min(d * 0.5, 0.2), kind="flash", src=r, tick=False, label="閃回"))
                out.append(Shot(t0=t + min(d * 0.5, 0.2) + 2 / 30, view=views[(k + 2) % len(views)], label="生長、腐爛、重生"))
                r += 1
        # 最後兩秒：一格一張，把前面全部的畫面過一遍
        t = tail
        fr = 1 / 30
        images = [("iron_house", 0.9, {}), ("petition", 0.8, {}), ("mouth", 1.0, {}), ("candle", 0.4, {}),
                  ("tou", 0.5, {}), ("grass_crack", 0.8, {}), ("roots_bones", 0.9, {}), ("leaflet", 0.9, {}),
                  ("boots", 0.5, {}), ("onlookers", 0.8, {}), ("crowd", 0.5, {"variant": "surge"}),
                  ("crowd", 0.5, {"variant": "formation"}), ("gate", 0.6, {}), ("crowd", 0.5, {"variant": "empty"}),
                  ("bars", 0.5, {})]
        m = 0
        while t < b["end"] - 1e-3:
            if r % 3 == 2 and m < len(images):
                name, pp, kw = images[m]
                out.append(Shot(t0=t, kind="insert", insert=name, p=(pp, pp), kw=kw, label="這一切"))
                m += 1
            else:
                out.append(Shot(t0=t, kind="flash", src=r % n_recall, label="這一切"))
            t += fr * (3 if r % 2 == 0 else 2)
            r += 1
        return out

    # ── 給分鏡文件用 ──
    def table(self):
        return [(s.t0, s.t1, s.kind, s.view, s.to, s.label) for s in self.shots]
