#!/usr/bin/env python3
"""Write STORYBOARD.md: the timeline and shot-by-shot storyboard as text.

Timecodes, shot lengths, frame ranges and subtitle slots are read from
timeline.json (so the document follows every change to the edit); the visual
and sound descriptions live in STORY below. Run again after editing the timeline:

  python3 make_storyboard.py
"""
import os

from montage.timeline import DEFAULT_TIMELINE, ROOT, Timeline

ACTS = [
    (1, 6, "序：鐵與夜", "黑白、冷、工業。底片顆粒與暗角。只有火車的聲音。第 4 段第一次出現顏色。"),
    (7, 13, "列舉：世間萬物", "節拍格開始（90 BPM），鏡頭數遞增帶來加速；紙牌突然放慢；地下室驟然沉寂。"),
    (14, 18, "第一幕：肉身與時間", "從地下室的沉寂慢慢甦醒，仍偏暗、低飽和；鏡子處節奏回升，音樂下滑。"),
    (19, 23, "第二幕：血統與世界", "色彩回暖，再次加速，衝鋒是全片最快處。"),
    (24, 28, "第三幕：愛、夢與記憶", "突然放慢，最密的情感段落，以推軌變焦為頂點。"),
    (29, 33, "轉折與判詞", "回閃一切、驟停、遼闊的海，最後回到高架橋，切黑、靜默。"),
]

# per segment: picture, sound, and (optionally) per-shot descriptions and an edit note
STORY = {
    "viaduct": dict(
        picture="高處斜俯視一座連拱鐵路高架橋，自左下延伸到右上的霧裡。低斜的光把一整排拱洞的影子投在橋後的地面；"
                "下方一條暗色河流蜿蜒穿過。橋面上兩條鐵軌與護欄是全畫面最亮的細線。鏡頭極緩慢地向前推移。",
        sound="遠方的火車：車輪過軌縫的「da-dum」每拍一組，低沉的金屬嗡鳴與隆隆聲，從左側淡入。",
        edit="開場，從黑淡入。"),
    "rail_yard": dict(
        picture="夜間調車場的俯視：密集的平行鐵軌、一道斜向梯線分出的側線群、主線之間的道岔曲線。"
                "一列 11 節的火車（車頭與長方形車廂）緩緩駛過，在畫面中央的道岔處變軌到隔壁股道。"
                "鐵軌在場燈光暈下微微反光，一道反光隨鏡頭移動沿鋼軌滑行；燈桿頭是幾個帶光暈的亮點。",
        sound="火車變近、變大；變軌時輪緣發出細細的尖嘯；約 3.6 秒處一聲帶殘響的遠方汽笛。",
        edit="硬切。"),
    "smoke": dict(
        picture="夜空與一輪朦朧的月亮，底部是工廠的剪影（鋸齒屋頂、三根煙囪）。黑煙由 Perlin 捲曲雜訊推動的粒子組成，"
                "從煙囪升起、扭轉、擴散，越來越濃，逐漸遮住星點並飄過月亮。",
        sound="火車聲遠去並淡出。",
        edit="硬切。"),
    "last_judgment": dict(
        picture="荒涼的平原，一條鐵軌直直通向地平線，旁邊一排電線桿。太陽正好在鐵軌的消失點升起。"
                "畫面一開始是灰階，約在 15%–70% 之間漸變成彩色：天空由深藍經洋紅到橘色，鐵軌反射陽光。"
                "這是全片第一次出現顏色。",
        sound="一個長音在 6 秒內升高兩個八度，泛音逐漸打開；在切黑的那一格戛然而止。",
        edit="段尾硬切到黑。"),
    "blackout": dict(
        picture="全黑。黑暗中閃過 6 次，每次 3–4 格：2–3 個前面段落（高架橋、調車場、黑煙、最後審判）的畫面"
                "半透明地疊在一起，略微放大與偏移，一閃即逝。",
        sound="突然的寂靜之後，低頻 drone 淡入（D、A 的低音與輕微拍頻）。",
        edit="從第 4 段硬切；閃光由 seed 決定時間點。"),
    "sound": dict(
        picture="幾乎全黑。畫面中央一個極微弱、緩慢呼吸的光點，一圈圈波紋向外擴散、變淡，像聲音從深處傳來。",
        sound="drone 持續；每一道波紋出現時一聲很深、很輕的鐘聲，帶長殘響。",
        edit="硬切。"),
    "stars": dict(
        picture="三種星空。",
        shots=["密集星野，一條斜向的銀河帶，極慢推近。", "同心圓星軌（長曝光），弧線隨鏡頭變長。",
               "稀疏的星空，九顆亮星之間的連線逐條畫出，形成一個星座。"],
        sound="音樂開始：第一個木琴般的聲部（八分音符音型）；火車咔噠聲在底層變成心跳。",
        edit="從這段起，所有剪接點吸附到 90 BPM 的節拍格。"),
    "bread": dict(
        picture="暖色、深色背景，簡單幾何加上「膨脹」光影的麵包。",
        shots=["圓麵包俯視，方形割紋。", "法棍斜放，五道斜割口。", "一片吐司的切面，帶孔洞的麵包心。",
               "一盤八個小圓餐包。", "整條麵包被掰開：兩半分離，露出撕裂的麵包心，碎屑往下掉。"],
        sound="第二個聲部加入（十六分音符、錯位一拍），每小節一記低音。"),
    "libraries": dict(
        picture="東方與西方的圖書館交替閃動，鏡頭越來越短。西方：直立書脊與燙金線；東方：平放的線裝書堆與卷軸端面。",
        sound="第三個聲部加入（十六分音符的琶音）；每個剪接點一聲輕 tick。"),
    "cards": dict(
        picture="明顯放慢。綠色牌桌的世界，三層意義。",
        shots=["象徵：一張黑桃 A 置中，慢慢轉動、放大。",
               "占卜：十張牌依序發出，排成牌陣（太陽、星、月、命運之輪、塔，其餘背面朝上）。",
               "博弈：從上方斜看橢圓牌桌，中央五張公共牌，兩側各一隻手的剪影。"],
        sound="音型抽掉一半，只剩第一聲部的四分音符；一個持續的和弦墊底。"),
    "chess": dict(
        picture="匹配剪接：與紙牌最後一鏡完全相同的構圖。綠色橢圓牌桌收縮成方形，棋盤格浮現，"
                "紙牌淡出、棋子升起；兩側的手一直留在原位，最後左手微微前伸。",
        sound="同上，第二聲部輕輕回來。",
        edit="第一格與上一段最後一格相同（match cut）。"),
    "gallery": dict(
        picture="畫框與天窗交替，越剪越快。畫框裡的小畫重現前面的意象（拱橋、鐵軌、太陽、棋盤、人形）。",
        sound="三個聲部全部回來，音樂最滿；在段尾一刀切斷。"),
    "basement": dict(
        picture="驟然沉寂。幾乎全黑的石牆，只有一小塊帶鐵柵影子的暖光，在牆上極緩慢地移動；光裡有幾粒浮塵。",
        sound="音樂抽空：極輕的房間底噪，兩聲遠處的滴水。"),
    "body": dict(
        picture="只有腳與小腿的剪影，近景，緩慢行走；鏡頭跟著人走，地板接縫往後滑；地面有微弱的倒影。",
        sound="幾乎無聲：底噪與每一步落地的輕響（與畫面的步伐同步）。"),
    "nail": dict(
        picture="微距：黑暗中一片指甲的弧面，根部半月形的淡色弧線極緩慢地向兩側延伸；只有弧線附近是清晰的，其餘失焦。",
        sound="只有底噪。"),
    "shadow": dict(
        picture="側光照在牆上，一個人形站著；它的影子從腳下脫離，向右上漂開、變大、變淡，最後消失，人形仍站在原地。",
        sound="只有底噪。"),
    "mirrors": dict(
        picture="鏡中鏡的無限迴廊：遞迴縮小的鏡框，每一層都映出同一個人形，排成一列往深處退去。"
                "鏡面晃動，鏡子的數量逐鏡繁殖，縮放越來越快。冷色、低飽和。",
        sound="極簡音型重新進入，隨鏡子增加一個聲部一個聲部疊上去；火車心跳回到底層。"),
    "music_slide": dict(
        picture="畫面幾乎靜止：停在鏡子段的最後一格，極慢地下沉一點、變暗、褪色，近乎全黑。",
        sound="由聲音主導：所有音型連同一個長音一起向下滑 12 個半音並淡出。"),
    "frontier": dict(
        picture="全片第二次大片暖色，呼應第 4 段的日出：拂曉的開闊平原，太陽貼著地平線；"
                "一條迎光的細線斜斜劃過大地，是國界，上面有幾根小柱；遠處七匹馬的剪影沿地平線奔跑。",
        sound="溫暖的和聲（D 大調九和弦）緩緩展開。"),
    "bronze_saga": dict(
        picture="暗色背景的兩個靜物。",
        shots=["一個沉重的青銅圓柱砝碼，頂上有提環，高窗的光在側面留下一道亮線，表面有綠色銅鏽。",
               "一卷攤開的古老書卷，寫滿符文般的刻痕，兩端捲起，鏡頭緩慢橫移。"],
        sound="和聲轉到 B 小調。"),
    "algebra_fire": dict(
        picture="代數與火焰交替閃動，每一鏡都帶著另一方淡淡的殘影：偶數鏡是發光的公式（歐拉恆等式、圓方程、"
                "黃金比例……）、函數曲線與螺線；奇數鏡是程序化的火焰。",
        sound="和聲轉到 G 大七和弦。"),
    "charge": dict(
        picture="全片最快。紅色調（黑→血紅→熱橘），只有馬蹄、馬腿與塵土；背光的紅色塵霧，水平動態模糊，每鏡一個傾斜角。",
        sound="低沉的鼓點般脈衝，每個十六分音符一下，越接近結尾越重；和聲轉到 A。"),
    "days": dict(
        picture="黃昏的俯視縮時：上千個光點般的人影從四面八方流動，逐漸匯成一條彎曲的光之河，向右流去。",
        sound="和聲回到 D 大調，鼓點消失。"),
    "honeysuckle": dict(
        picture="突然放慢。忍冬花的極近特寫：捲曲的花瓣、細長的花蕊，柔光；背景其他花朵與葉子化為溫暖的散景。",
        sound="慢而透明的長音（A、E）。"),
    "love": dict(
        picture="溫暖的逆光中的手。",
        shots=["兩隻手從左右伸出，食指緩緩靠近，將觸未觸；指尖之間的光漸亮。",
               "只剩左手；右邊是空缺，原本另一隻手的位置只留下一點漸弱的餘光。"],
        sound="兩個長音；第二鏡只剩一個。"),
    "dream_treasure": dict(
        picture="地底般的構圖，呼應第 13 段地下室的石牆，但這次是溫暖的金色光：一堆發光的錢幣在凹處，"
                "幾道細裂縫透出光，金色的微塵緩緩上升。",
        sound="低而溫暖的長音（D、A、升 F）。"),
    "chance": dict(
        picture="深色背景上兩顆骰子在空中翻轉，畫出弧線；在 2.3 秒處停格，伴隨一下輕微的閃光。",
        sound="高處的長音，骰子的細碎聲響，停格時一切靜止。"),
    "memory": dict(
        picture="情緒頂點，推軌變焦：一個人形站在拱廊中央（拱廊呼應高架橋，地面是棋盤格）。鏡頭從 4 公尺退到 20 公尺，"
                "視角同時從 82° 收窄到約 18°；人形大小不變，背景的柱廊與地面被拉扯、壓縮。"
                "拉扯最快時加上放射狀模糊與色散。",
        sound="兩個長音隨推軌一上一下拉開各 5 個半音，濾波同時打開。"),
    "all_given": dict(
        picture="「這一切都給了你」：前面 28 段各取一格，依序以 2–3 格的速度回閃一遍。",
        sound="前面每一段聲音中最響的一小片段，依序快速重疊回閃。",
        edit="段尾驟停。"),
    "old_food": dict(
        picture="驟停，色彩褪回灰階。三個沉重的慢鏡頭。",
        shots=["虛偽：一張白色面具從中間裂開，兩半慢慢分離。",
               "失敗：九根直立的線一根接一根倒下，躺平在地上。",
               "屈辱：一塊厚板從上方壓下，下面的人形被壓低、彎下。"],
        sound="完全靜音，只剩一個極低的持續音（其他聲音在這段一律切斷）。"),
    "sea_sun": dict(
        picture="全片最遼闊的畫面：緩慢起伏的海面一直延伸到地平線，高懸的太陽在水面拉出一條閃爍的光帶。色彩飽滿卻安靜，顆粒較輕。",
        sound="慢而透明、近乎神聖的長音（D、A、D、E、A 依序進入），長殘響。"),
    "wasted_years": dict(
        picture="回到第 1 段的高架橋構圖，黑白：一列火車照常在橋上駛過；橋下田野中站著一個極小的人影，拖著長長的影子。",
        sound="只剩火車聲。"),
    "last_line": dict(
        picture="切黑，停留在黑畫面中（這一段沒有顆粒與灰塵）。",
        sound="火車聲在第 1 秒內淡出，之後約 2 秒完全靜默，直到結束。",
        edit="全片結束。"),
}


def tc(sec):
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:05.2f}"


def shot_lines(tl, seg):
    """Per-shot descriptions, generated from the segment code where the sequence lives there."""
    st = STORY.get(seg.name, {})
    n = len(seg.shot_frames)
    if seg.name == "libraries":
        from montage.segments.s09_libraries import SEQUENCE
        names = {"spines": "書架正面", "cells": "整面格櫃", "corridor": "無限延伸的書架走廊"}
        sides = {"west": "西方", "east": "東方"}
        return [f"{sides[sd]}・{names[k]}" for k, sd in SEQUENCE[:n]]
    if seg.name == "charge":
        from montage.segments.s22_charge import KINDS
        names = {"legs": "側面奔跑：數匹馬的腿與肚腹剪影，地面飛逝，塵土向後噴",
                 "strike": "一隻巨大的馬蹄重重落地，塵土爆開",
                 "cloud": "翻滾的塵雲，馬腿在其中閃現", "streaks": "速度線，馬腿一閃而過"}
        return [names[KINDS[i % len(KINDS)]] for i in range(n)]
    if seg.name == "gallery":
        out = []
        for i in range(n):
            v = i // 2
            if i % 2 == 0:
                out.append("單一大畫框近景" if v == 2 else "滿牆的畫框，鏡頭橫移")
            else:
                out.append(f"天窗：{[3, 4, 5, 6][v % 4]} 片窗格的斜光束與地上的光斑，浮塵")
        return out
    if seg.name == "mirrors":
        counts = seg.params.get("mirrors_per_shot", [1, 1, 2, 4, 6, 9, 16])
        return [f"{counts[min(i, len(counts) - 1)]} 面鏡子" for i in range(n)]
    if seg.name == "algebra_fire":
        return ["公式與曲線" if i % 2 == 0 else "火焰" for i in range(n)]
    if seg.name == "all_given":
        earlier = [s for s in tl.segments if s.index < seg.index]
        return [f"{earlier[i % len(earlier)].index + 1:02d} {earlier[i % len(earlier)].title}" for i in range(n)]
    return st.get("shots", [])


def build(tl):
    L = []
    w = L.append
    meta = tl.data["meta"]
    w(f"# 〈馬太福音二十五章三十節〉時間線與分鏡")
    w("")
    w(f"{meta.get('title', '')} · 全長 {tc(tl.duration)}（{tl.duration:.2f} 秒，{tl.n_frames} 格）· "
      f"{tl.W}×{tl.H} · {tl.fps} fps · {len(tl.segments)} 段")
    w("")
    w("> 這份文件由 `make_storyboard.py` 依 `timeline.json` 產生：時間碼、鏡頭長度、格數與字幕位置都是實際渲染值。"
      "改了時間軸之後重新執行即可更新。")
    w("")
    w(f"剪輯節拍：{tl.bpm:g} BPM，從「{tl.segment(tl.anchor_name).title}」起，標有吸附的段落其剪接點對齊節拍格"
      f"（1/16 = {round(tl.grid_seconds('1/16') * tl.fps)} 格，1/32 = {round(tl.grid_seconds('1/32') * tl.fps)} 格）。")
    w("")
    w("## 結構總表")
    w("")
    w("| # | 段落 | 開始 | 長度 | 鏡頭 | 畫面 | 吸附 |")
    w("|---|---|---|---|---|---|---|")
    for s in tl.segments:
        look = {"bw": "黑白", "color": "彩色"}.get(s.look, s.look)
        if s.name == "last_judgment":
            look = "黑白→彩色"
        if s.name == "music_slide":
            look = "褪色"
        w(f"| {s.index + 1:02d} | {s.title} | {tc(s.start_frame / tl.fps)} | {s.n_frames / tl.fps:.2f} 秒 | "
          f"{len(s.shot_frames)} | {look} | {s.spec.get('snap', '—')} |")
    w("")
    for a, b, name, note in ACTS:
        segs = tl.segments[a - 1:b]
        t0, t1 = segs[0].start_frame / tl.fps, segs[-1].end_frame / tl.fps
        w(f"## {name}（{a:02d}–{b:02d}，{tc(t0)}–{tc(t1)}）")
        w("")
        w(note)
        w("")
        for s in segs:
            st = STORY.get(s.name, {})
            t0 = s.start_frame / tl.fps
            t1 = s.end_frame / tl.fps
            w(f"### {s.index + 1:02d} {s.title}")
            w("")
            w(f"`{tc(t0)} – {tc(t1)}` · {s.n_frames / tl.fps:.2f} 秒 · {s.n_frames} 格（f{s.start_frame}–f{s.end_frame - 1}）"
              f" · {len(s.shot_frames)} 個鏡頭")
            w("")
            w(f"- **畫面**：{st.get('picture', '')}")
            shots = shot_lines(tl, s)
            if len(s.shot_frames) > 1 or shots:
                w("- **鏡頭**：")
                for i, (fa, fn) in enumerate(s.shot_frames):
                    desc = shots[i] if i < len(shots) else ""
                    w(f"  {i + 1}. `{tc((s.start_frame + fa) / tl.fps)}` {fn / tl.fps:.3f} 秒（{fn} 格）　{desc}")
            w(f"- **聲音**：{st.get('sound', '')}")
            if st.get("edit"):
                w(f"- **剪接**：{st['edit']}")
            subs = s.spec.get("subtitles", [])
            if subs:
                w("- **字幕**：" + "；".join(
                    f"{x['text']} `{tc(t0 + x.get('start', 0))}–{tc(t0 + min(x.get('end', 99), s.n_frames / tl.fps))}`"
                    for x in subs))
            w("")
    return "\n".join(L)


if __name__ == "__main__":
    tl = Timeline(DEFAULT_TIMELINE)
    path = os.path.join(ROOT, "STORYBOARD.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build(tl))
    print(path)
