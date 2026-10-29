#!/usr/bin/env python3
"""依 config/timeline.toml 與 config/edit.toml 產生 STORYBOARD.md（時間線與逐鏡分鏡）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ycf import config  # noqa: E402
from ycf.edit import Edit  # noqa: E402

SECTIONS = [  # (標題, 起點 cue 或秒數, 一句說明)
    ("1 沉默", 0.0, "充實的黑：微粒、裂縫"),
    ("2 死與生", "leaf_in", "左邊熄滅、右邊殘影，交叉剪接越來越快"),
    ("3 泥與罪過", "mud_hang", "泥垂下、墜落、砸在地平線上"),
    ("4 字長成草", "mud_land", "喬木崩成筆畫，字從土裡長出；踐踏與刪刈的蒙太奇"),
    ("5 第一次坦然", "breath", "長鏡頭：殘字在明亮的安靜裡擺動"),
    ("6 愛與憎", "selflove", "暖光四秒；拉遠；口號網格與野草交叉剪接"),
    ("7 地火", "fire_rise", "光從底下透出；黑；白；熔岩；凍結"),
    ("8 第二次坦然", 144.0, "凍結的灰燼"),
    ("9 靜穆", "hush", "灰色平面，先前的畫面一閃而過；翻騰"),
    ("10 作證", "witness", "分界線每一拍重切，鏡頭每一拍一切"),
    ("11 火速", "cycle", "生長腐爛重生越轉越快，夾進所有先前的畫面；驟停"),
    ("12 去罷", "farewell", "全詩直排、被風吹散；落款；土色"),
]


INSERTS = {"iron_house": "鐵屋子", "mouth": "嘴", "candle": "燭", "grass_crack": "石縫的草", "roots_bones": "根與骨",
           "boots": "靴", "onlookers": "看客", "gate": "節烈牌坊", "calendar": "日曆", "bars": "牢窗",
           "lamp_window": "燈下寫字的人", "petition": "公車上書的呈文", "leaflet": "北京學界全體宣言", "tou": "「頭」"}


def size_name(z):
    if z < 1.25:
        return "全景"
    if z < 1.9:
        return "中景"
    if z < 2.7:
        return "近景"
    if z < 3.6:
        return "特寫"
    return "大特寫"


def tc(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    tl = config.load()
    edit = Edit(tl)
    fps = tl.fps
    starts = [tl.c(s) if isinstance(s, str) else s for _, s, _ in SECTIONS]
    ends = starts[1:] + [tl.duration]
    lines = tl.raw["line"]
    out = ["# 《野草》題辭　時間線與分鏡", "",
           f"全長 {tc(tl.duration)}（{tl.duration:.2f} 秒，{tl.nframes} 格）· {tl.W}×{tl.H} · {fps} fps · "
           f"{len(edit.shots)} 個鏡頭", "",
           "> 由 `make_storyboard.py` 依 `config/timeline.toml` 與 `config/edit.toml` 產生；改了剪輯之後重新執行即可。",
           "> 景別依虛擬攝影機的放大倍數：全景 <1.25×、中景 <1.9×、近景 <2.7×、特寫 <3.6×、大特寫 ≥3.6×。",
           "> 「意象」是刻版的插入鏡頭（ycf/plates.py、ycf/plates2.py），括號裡是它播到的進度；「頭」是字草世界裡的那一株。", "",
           "## 結構總表", "", "| 段落 | 開始 | 長度 | 鏡頭數 | 平均鏡頭長 | 內容 |", "|---|---|---|---|---|---|"]
    for (name, _, desc), a, b in zip(SECTIONS, starts, ends):
        n = sum(1 for s in edit.shots if a <= s.t0 < b)
        out.append(f"| {name} | {tc(a)} | {b - a:.2f} 秒 | {n} | {(b - a) / max(n, 1):.2f} 秒 | {desc} |")
    out.append("")
    for (name, _, desc), a, b in zip(SECTIONS, starts, ends):
        out += [f"## {name}", "", f"`{tc(a)} – {tc(b)}` · {desc}", ""]
        subs = [ln for ln in lines if a <= ln["at"] < b]
        if subs:
            out.append("字幕：" + "／".join(f"`{tc(ln['at'])}` {ln['text']}" for ln in subs))
            out.append("")
        out += ["| # | 時間 | 長度 | 格 | 種類 | 景別 | 取景中心 | 內容 |", "|---|---|---|---|---|---|---|---|"]
        for i, s in enumerate(e for e in edit.shots if a <= e.t0 < b):
            d = s.t1 - s.t0
            kind = {"scene": "", "black": "黑", "white": "白", "flash": "閃回", "insert": "意象"}[s.kind]
            if s.kind == "flash":
                kind += f"（{tc(edit.recall[s.src])}）"
            if s.kind == "insert":
                name = INSERTS.get(s.insert, s.insert)
                if s.insert == "crowd":
                    name = {"surge": "五四的橫幅", "formation": "被收編的方陣", "empty": "散場的街"}[s.kw.get("variant", "surge")]
                pp = f"{s.p[0]:.2f}" if s.p[0] == s.p[1] else f"{s.p[0]:.2f}→{s.p[1]:.2f}"
                kind += f"：{name}（{pp}）"
            if s.kind == "scene":
                z, cx, cy = s.view
                size = size_name(z)
                if s.to is not None:
                    size += "→" + size_name(s.to[0]) if size_name(s.to[0]) != size_name(z) else "（推移）"
                center = f"{z:.1f}× ({cx:.0f}, {cy:.0f})"
                if s.follow:
                    center += " 跟拍"
            else:
                size, center = "", ""
            extra = "；無聲" if s.mute else ""
            out.append(f"| {i + 1} | {tc(s.t0)} | {d:.3f} 秒 | {round(d * fps)} | {kind} | {size} | {center} | "
                       f"{s.label}{extra} |")
        out.append("")
    (ROOT / "STORYBOARD.md").write_text("\n".join(out))
    print(ROOT / "STORYBOARD.md", len(edit.shots), "shots")


if __name__ == "__main__":
    main()
