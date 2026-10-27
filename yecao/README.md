# 《野草》題辭 — 代碼生成短片

全片以 Python 逐影格渲染（pycairo 向量 + numpy 粒子／噪聲場），ffmpeg 合成影音。固定隨機種子，每次輸出一致。

## 目錄

| 路徑 | 內容 |
|---|---|
| `config/timeline.toml` | 時間軸、關鍵對位點、字幕全文、配色、年份／名字層、口號網格文字 |
| `analysis/beats.json` | 配樂分析結果（拍點、響度、低頻、onset） |
| `audio/eclosion.mp3` | 配樂 |
| `fonts/` | Noto Serif TC（Light／Regular）、Noto Sans TC Medium，由 `tools/fetch_fonts.sh` 產生 |
| `ycf/audio.py` | 拍點偵測（spectral flux + DP beat tracking）與混音（2:30 靜穆壓低、淡出、補無聲） |
| `ycf/fonts.py` | 由字形輪廓取出多邊形，字幕與字草共用 |
| `ycf/subtitles.py` | 一字一字浮現、銘文式刻出、落款、「手的痕跡」 |
| `ycf/scene_opening.py` | 第 1–3 段：微粒與裂縫、葉脈死與生、泥 |
| `ycf/scene_field.py` | 第 4–10 段：喬木崩成筆畫、字草物理、踐踏刪刈、網格、地火、灰燼、作證 |
| `ycf/scene_end.py` | 第 11–12 段：火速循環、去罷、土色 |
| `ycf/leaf.py`、`ycf/tree.py`、`ycf/clump.py` | 葉脈網路、L-system 喬木、「一叢野草」 |
| `render.py` | 渲染入口 |

## 準備

- Python 3、`numpy`、`scipy`、`Pillow`、`pycairo`、`fonttools`、`brotli`，以及 PATH 上的 `ffmpeg`。
- 配樂〈Eclosion〉（逆時針向、禮諳）有著作權，不放進 repo。請自行把音檔放到 `audio/eclosion.mp3`。
  `analysis/beats.json` 是由它算出的拍點與響度，已附上，所以沒有音檔也能渲染畫面（加 `--no-audio`）。
- 字體：執行 `tools/fetch_fonts.sh`。

## 使用

```bash
tools/fetch_fonts.sh                       # 第一次：取得字體
python3 render.py --analyze                # 換配樂或改 bpm_hint 後：重新偵測拍點
python3 render.py --still 66.5 123.9       # 單張檢查 → out/still_*.png
python3 render.py --range 61.5:71.5 --name s4   # 片段（含配樂）→ out/s4.mp4
python3 render.py --full                   # 全片 → out/yecao_tici.mp4
```

多核心並行：影格範圍切成數段，各段由場景起點重播模擬（只模擬、不繪製）到段首再開始渲染，結果與單線程一致。

## 配樂對位（分析結果）

66.26 BPM，拍距 0.905 s。關鍵點已校準至音樂實際轉折：

| 點 | 設定 | 說明 |
|---|---|---|
| crack | 22.70 | 22.6 秒音樂幾近無聲，22.7 秒重新進入 |
| mud_land | 42.00 | +14 dB 躍升 |
| breath | 81.00 | 強段退下 |
| dim | 121.00 | 2:01 下陷 |
| erupt | 123.32 | 2:03 爆點的拍 |
| restore | 157.81 | 2:38 拍點 |
| stop | 180.92 | 3:01 驟降前的最後一拍 |
| music_end | 205.53 | |
