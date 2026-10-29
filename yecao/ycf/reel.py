"""試片：依 config/reel.toml 把刻版鏡頭與字草世界的一段剪在一起，並為它合成聲音。

聲音的原則：具體的生活與公共空間——呼吸、幻燈機、講堂裡的「萬歲」與掌聲、街上的領喊與齊喊、
布被風吹、腳步、紙、靴落下。每一種聲音有自己的距離（濾波、殘響）、停頓與遮蔽；
鏡頭一切，聲音的空間也跟著切。
"""
import tomllib

import numpy as np
from scipy.signal import oaconvolve

from . import config, voice
from .config import ROOT
from .gfx import ease, ramp
from .noise import rand
from .plates import Plates
from .sound import SR, Bus, brown, crackle, curve, env_ad, filt, knock, pink, reverb_ir, swish, thud, tone, white


class Reel:
    def __init__(self, path=ROOT / "config/reel.toml"):
        self.tl = config.load()
        self.fps = self.tl.fps
        self.shots = tomllib.loads(path.read_text())["shot"]
        t = 0.0
        for s in self.shots:
            s["t0"] = t
            t += s["dur"]
            s["t1"] = t
        self.duration = t
        self.nframes = int(round(t * self.fps))
        self._plates = None
        self._vig = {}

    def shot_at(self, t):
        for s in self.shots:
            if s["t0"] <= t < s["t1"]:
                return s
        return self.shots[-1]

    def plates(self):
        if self._plates is None:
            self._plates = Plates(self.tl.seed)
        return self._plates

    def frame(self, f, plain=False):
        t = f / self.fps
        s = self.shot_at(t)
        u = (t - s["t0"]) / s["dur"]
        if s.get("kind") == "black":
            return np.zeros((self.tl.H, self.tl.W, 3), np.float32)
        if "vignette" in s:
            name = s["vignette"]
            if name not in self._vig:
                from .vignettes import LeafletGrass
                self._vig[name] = LeafletGrass(self.tl)
            a, b = s["tau"]
            tau = a + (b - a) * u
            return self._vig[name].render(int(round(tau * self.fps)))
        p0, p1 = s["p"]
        v0 = s["view"]
        v1 = s.get("to", v0)
        k = ease(u) if "to" in s else 0.0
        view = tuple(float(a + (b - a) * k) for a, b in zip(v0, v1))
        return self.plates().render(s["plate"], p0 + (p1 - p0) * u, t, view=view, plain=plain)

    # ── 聲音 ──────────────────────────────────────────────
    def sound(self):
        rng = rand(self.tl.seed, 700, 1)
        dur = self.duration
        dry, wet, far = Bus(dur), Bus(dur), Bus(dur)
        S = self.shots

        def seg(i0, i1):
            return S[i0]["t0"], S[i1]["t1"]

        def gate(x, t0, t1, fade=0.02):
            """把一段聲音限制在 [t0, t1)：切鏡就切空間。"""
            n = len(x)
            tt = np.arange(n) / SR
            g = np.clip(tt / fade, 0, 1) * np.clip((t1 - t0 - tt) / fade, 0, 1)
            return x * (g[:, None] if x.ndim == 2 else g)

        def add2(bus, st, t0, gain=1.0):
            bus.L[int(t0 * SR):int(t0 * SR) + len(st)] += st[: bus.n - int(t0 * SR), 0] * gain
            bus.R[int(t0 * SR):int(t0 * SR) + len(st)] += st[: bus.n - int(t0 * SR), 1] * gain

        # ── 鐵屋 ──
        a, b = seg(0, 2)
        n = int((b - a) * SR)
        tt = np.arange(n) / SR
        drone = filt(brown(rng, n), "lowpass", 70) * 2.0 + 0.05 * np.sin(2 * np.pi * 43 * tt) * (
            1 + 0.4 * np.sin(2 * np.pi * 0.11 * tt)) + 0.03 * np.sin(2 * np.pi * 61.5 * tt)
        drone *= np.clip(tt / 0.6, 0, 1)
        dry.add(gate(drone.astype(np.float32), a, b), a, 0.16)
        # 許多人的呼吸，慢、深、不齊
        wake_t = S[1]["t0"] + S[1]["dur"] * (0.30 - S[1]["p"][0]) / (S[1]["p"][1] - S[1]["p"][0])
        for k in range(16):
            per = rng.uniform(3.4, 5.2)
            ph = rng.uniform(0, per)
            pan = rng.uniform(-0.9, 0.9)
            g = rng.uniform(0.25, 0.6) * (1.8 if k == 0 else 1.0)
            tk = a - ph
            while tk < b:
                for (st, ln, lo, hi, amp) in ((0.0, 1.3, 500, 2200, 0.6), (1.5, 1.9, 280, 1300, 1.0)):
                    t0 = tk + st
                    if k == 0 and t0 > wake_t - 0.2:
                        continue          # 醒來的那個人，呼吸停住了
                    m = int(ln * SR)
                    e = np.sin(np.pi * np.linspace(0, 1, m)) ** 2
                    x = filt(white(rng, m), "bandpass", [lo, hi]) * e * amp * 0.05
                    if a <= t0 < b:
                        dry.add(gate(x, t0, b), t0, g, pan)
                        wet.add(gate(x, t0, b), t0, g * 0.25, pan)
                tk += per
        # 驚醒：一口吸進去的氣，然後耳鳴般的一條細音
        m = int(0.45 * SR)
        gasp = filt(white(rng, m), "bandpass", [700, 3000]) * env_ad(m, 0.05, 0.12) * 0.25
        dry.add(gasp, wake_t - 0.1, 1.0, 0.1)
        m = int((b - wake_t) * SR)
        ring = tone([3150], m / SR) * np.clip(np.arange(m) / SR / 1.5, 0, 1) * 0.012
        dry.add(gate(ring.astype(np.float32), wake_t, b), wake_t, 1.0, 0.0)
        # 抬頭時，鐵板一聲悶響的呻吟
        t_c = S[2]["t0"] + 0.9
        m = int(2.2 * SR)
        src = (np.sin(2 * np.pi * np.cumsum(70 + 25 * np.sin(np.linspace(0, 3, m))) / SR) > 0.6).astype(np.float32)
        groan = sum(filt(src, "bandpass", [f * 0.96, f * 1.04]) * g_ for f, g_ in ((640, 1.0), (1290, 0.5), (2210, 0.3)))
        groan *= np.sin(np.pi * np.linspace(0, 1, m)) ** 1.5
        dry.add(gate(groan, t_c, b), t_c, 0.22, -0.3)
        wet.add(gate(groan, t_c, b), t_c, 0.3, -0.3)

        # ── 看客：講堂 ──
        a, b = seg(4, 5)
        n = int((b - a) * SR)
        room = filt(pink(rng, n), "bandpass", [120, 900]) * 0.02
        dry.add(gate(room, a, b), a, 1.0)
        # 幻燈機換片：金屬的卡嗒，木框滑過
        for dt_, f_ in ((0.25, 950), (0.31, 380)):
            dry.add(knock(rng, f_, 0.12, 0.5) * 0.12, a + dt_, 1.0, 0.5)
            wet.add(knock(rng, f_, 0.12, 0.5) * 0.12, a + dt_, 0.5, 0.5)
        dry.add(swish(rng, 0.25, 1500, 3500, 0.6) * 0.08, a + 0.27, 1.0, 0.5)
        # 低低的人聲（還沒喊出來）
        murmur = np.zeros(int(3.2 * SR), np.float32)
        for k in range(26):
            ch = "外爭主權內除國賊"[k % 8]
            s_ = voice.syllable(rng, ch, rng.uniform(0.18, 0.3), rng.uniform(100, 150), rng.uniform(0.9, 1.1), 0.4)
            i = int(rng.uniform(0, 2.9) * SR)
            murmur[i:i + len(s_)] += s_[: len(murmur) - i] * rng.uniform(0.2, 0.5)
        murmur = filt(murmur, "lowpass", 1400) * 0.05
        dry.add(gate(murmur, a + 0.4, b), a + 0.4, 1.0, 0.2)
        wet.add(gate(murmur, a + 0.4, b), a + 0.4, 0.8, 0.2)
        # 「萬歲！」他們都拍掌歡呼起來
        t_b = a + 3.3
        cheer = voice.crowd(rng, ["ban", "zai"], 0.42, voices=18, f0=(115, 190), effort=1.3, jitter=0.04)
        cheer = np.stack([filt(cheer[:, 0], "lowpass", 3500), filt(cheer[:, 1], "lowpass", 3500)], 1)
        add2(dry, gate(cheer, t_b, b), t_b, 0.32)
        add2(wet, gate(cheer, t_b, b), t_b, 0.45)
        m = int((b - t_b - 0.3) * SR)
        clap = crackle(rng, m / SR, curve([(0, 0), (0.15, 520), (1.5, 420), (m / SR, 120)], m), 700, 6500, 1.0)
        dry.add(gate(clap, t_b + 0.3, b), t_b + 0.3, 1.1, 0.1)
        wet.add(gate(clap, t_b + 0.3, b), t_b + 0.3, 0.9, 0.1)

        # ── 街頭：一個人領喊，一群人跟著喊 ──
        a, b = seg(6, 7)
        n = int((b - a) * SR)
        tt = np.arange(n) / SR
        wind = filt(white(rng, n), "bandpass", [200, 1800]) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.23 * tt) ** 2) * 0.03
        dry.add(gate(wind, a, b), a, 1.0, -0.2)
        # 布被風吹：拍打的節奏跟著布上的波
        flap_env = np.clip(np.sin(2 * np.pi * 1.25 * tt) * 1.8 - 0.6, 0, 1) ** 2
        flap = filt(white(rng, n), "bandpass", [300, 2600]) * flap_env * 0.09
        dry.add(gate(flap, a, b), a, 1.0, 0.0)
        # 腳步：很多人，不齊
        for k in range(int((b - a) * 7)):
            t0 = a + rng.uniform(0, b - a)
            st = thud(rng, 90, 60, 0.12, 0.3, 0.8) * 0.12
            st[: int(0.08 * SR)] += crackle(rng, 0.08, 400, 1200, 5000, 0.05)
            dry.add(gate(st, t0, b), t0, rng.uniform(0.3, 0.8), rng.uniform(-0.8, 0.8))
        lead_a = voice.phrase(rng, "外爭主權", 0.36, 165, 1.0, 1.4)
        lead_b = voice.phrase(rng, "內除國賊", 0.36, 168, 1.0, 1.4)
        crowd_a = voice.crowd(rng, "外爭主權", 0.36, voices=22)
        crowd_b = voice.crowd(rng, "內除國賊", 0.36, voices=22)
        plan = [(a + 0.3, lead_a, None), (a + 1.9, None, crowd_a), (a + 3.6, lead_b, None), (a + 5.2, None, crowd_b)]
        for t0, ld, cr in plan:
            if ld is not None:
                x = filt(ld, "lowpass", 4500)
                dry.add(gate(x, t0, b), t0, 0.55, -0.35)
                wet.add(gate(x, t0, b), t0, 0.35, -0.35)
            if cr is not None:
                x = np.stack([filt(cr[:, 0], "lowpass", 3200), filt(cr[:, 1], "lowpass", 3200)], 1)
                add2(dry, gate(x, t0, b), t0, 0.62)
                add2(wet, gate(x, t0, b), t0, 0.8)

        # ── 傳單：喊聲退到遠處；紙；靴子走近，落下 ──
        a, b = seg(8, 11)
        n = int((b - a) * SR)
        tt = np.arange(n) / SR
        for t0, cr in ((a + 0.2, crowd_a), (a + 1.9, crowd_b), (S[11]["t0"] + 0.9, crowd_a)):
            x = np.stack([filt(cr[:, 0], "lowpass", 1100), filt(cr[:, 1], "lowpass", 1100)], 1)
            add2(far, gate(x, t0, b), t0, 0.22 if t0 < S[10]["t0"] else 0.07)
        rd = S[9]["t1"] - a
        rt = np.arange(int(rd * SR)) / SR
        rustle = crackle(rng, rd, 60 + 120 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * rt)), 2000, 9000, 0.25)
        dry.add(gate(rustle, a, S[9]["t1"]), a, 1.0, 0.2)
        # 帶鞋釘的靴子：一步一步走近
        steps = np.arange(a + 1.3, S[9]["t1"], 0.62)
        for k, t0 in enumerate(steps):
            g = 0.3 + 0.7 * (k + 1) / len(steps)
            dry.add(thud(rng, 75, 45, 0.25, 0.6, 0.6) * 0.3, t0, g, -0.3 + 0.15 * k)
            dry.add(knock(rng, 3400, 0.05, 0.3) * 0.12, t0 + 0.01, g, -0.3 + 0.15 * k)
        # 影子壓下來：一陣氣；落下：悶響、泥、然後什麼都沒有
        t_shadow = S[9]["t0"]
        dry.add(swish(rng, 0.6, 200, 900, 0.8) * 0.4, t_shadow, 1.0, -0.1)
        t_hit = S[10]["t0"]
        dry.add(thud(rng, 62, 30, 1.0, 1.2, 0.9), t_hit, 1.0, 0.0)
        wet.add(thud(rng, 62, 30, 1.0, 1.2, 0.9), t_hit, 0.3, 0.0)
        m = int(0.5 * SR)
        squelch = filt(white(rng, m), "lowpass", 900) * env_ad(m, 0.002, 0.12) * (
            1 + np.sin(2 * np.pi * 38 * np.arange(m) / SR)) * 0.5
        dry.add(squelch, t_hit + 0.02, 1.0, 0.0)
        # 抬起，走遠
        for k in range(3):
            t0 = S[11]["t0"] + 0.5 + k * 0.7
            dry.add(thud(rng, 75, 45, 0.25, 0.6, 0.6) * 0.3, t0, 0.5 / (k + 1), 0.3 + 0.2 * k)
            dry.add(knock(rng, 3400, 0.05, 0.3) * 0.12, t0 + 0.01, 0.5 / (k + 1), 0.3 + 0.2 * k)

        # ── 字草的世界 ──
        a, b = seg(12, 12)
        V = {"RISE": 1.6, "TOUCH": 6.0, "STOMP": 8.4}
        n = int((b - a) * SR)
        tt = np.arange(n) / SR
        wind = filt(white(rng, n), "bandpass", [150, 900]) * (0.4 + 0.6 * np.sin(2 * np.pi * 0.09 * tt + 1) ** 2) * 0.035
        wind *= np.clip(tt / 0.8, 0, 1) * np.clip((b - a - tt) / 1.2, 0, 1)
        dry.add(wind, a, 1.0, 0.0)
        wet.add(wind, a, 0.5, 0.0)
        # 冒芽：土裂開的細響；根往下：一陣一陣的細碎
        g_ = crackle(rng, 2.6, curve([(0, 5), (0.5, 60), (2.6, 8)], int(2.6 * SR)), 500, 4000, 0.35)
        dry.add(g_, a + V["RISE"], 1.0, 0.0)
        r_ = crackle(rng, 4.4, curve([(0, 3), (2, 18), (4.4, 2)], int(4.4 * SR)), 150, 1500, 0.5)
        dry.add(filt(r_, "lowpass", 1500), a + V["RISE"] + 0.5, 1.0, 0.0)
        # 碰到墨：低低的脈搏
        t_touch = a + V["TOUCH"]
        for k in range(int((b - t_touch) / 1.05)):
            t0 = t_touch + k * 1.05
            amp = 0.5 if t0 < a + V["STOMP"] else 0.28 * (1 - 0.1 * k)
            dry.add(thud(rng, 48, 38, 0.45, 1.0, 0.0) * amp, t0, 0.45, 0.0)
            dry.add(thud(rng, 46, 36, 0.35, 0.8, 0.0) * amp * 0.6, t0 + 0.24, 0.45, 0.0)
        # 靴又落下：一筆斷了
        t_s = a + V["STOMP"]
        dry.add(swish(rng, 0.45, 180, 800, 0.8) * 0.35, t_s - 0.4, 1.0, 0.0)
        dry.add(thud(rng, 58, 28, 1.3, 1.3, 1.0), t_s, 1.0, 0.0)
        wet.add(thud(rng, 58, 28, 1.3, 1.3, 1.0), t_s, 0.35, 0.0)
        dry.add(knock(rng, 2400, 0.08, 0.4) * 0.6, t_s + 0.05, 1.0, 0.35)
        dust = crackle(rng, 2.0, curve([(0, 200), (2.0, 0)], int(2.0 * SR)), 1500, 7000, 0.12)
        dry.add(dust, t_s + 0.05, 1.0, 0.0)

        # ── 混音：近的乾聲、殘響、遠處（更暗、更多殘響） ──
        ir = reverb_ir(rng, 1.6)
        ir_far = reverb_ir(rng, 3.2, damp=2200)

        def conv(st, k):
            return np.stack([oaconvolve(st[:, 0], k[:, 0])[: len(st)], oaconvolve(st[:, 1], k[:, 1])[: len(st)]], 1)

        mix = dry.stereo() + conv(wet.stereo(), ir) * 0.8 + conv(far.stereo(), ir_far) * 1.2
        # 黑：真的沒有聲音（連殘響也切掉）
        t = np.arange(len(mix)) / SR
        g = np.ones(len(mix), np.float32)
        for s in S:
            if s.get("kind") == "black" and s["dur"] >= 0.45:
                g *= 1 - ((t >= s["t0"]) & (t < s["t1"]))
        mix *= g[:, None]
        mix = mix[: int(dur * SR)]
        mix = np.stack([filt(mix[:, 0], "highpass", 25), filt(mix[:, 1], "highpass", 25)], 1)
        mix *= 10 ** (-1.0 / 20) / max(np.abs(mix).max(), 1e-6)
        return mix
