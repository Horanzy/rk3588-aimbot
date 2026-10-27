"""arena/laws/_wip_burst.py — 后坐力阶梯工况 + 突发平滑候选机制 (实验, 非现役律)。

为什么需要这个工况: 现有四组评测 (eval/integrate/fps_eval/selftest) 都没有
"连续射击" 这一类输入。游戏的每发后坐力在屏幕上就是**准星系下的一个阶梯**:
每一发把画面整体推下去 K 像素 (瞄准点在屏幕上的 y 阶跃 +K), 射击间隔 T。
采集帧率高于射击频率时, 律看得到每个阶梯, 于是把每一发当成一个独立的阶跃
去收敛。实测轨迹 (K=12px, T=100ms): 每发后先 10–20ms 的 pull 脉冲, 然后
前馈的均值继续推 → 反向过冲 (+12 → 0 → −6 → +12), 准星以射击频率来回摆。
这不是环路不稳, 是"每发一个阶跃 + 前馈均值"的重复瞬态。

本文件提供: 阶梯目标 + 指令/误差形态指标 + 候选机制并排测量。候选机制
全部只在开火窗内生效 (门 = 扳机, 固件三种输出模式都拿得到):
  base       现役 ff_pi_acc
  lpf        τ 一阶低通作用在期望速度上 (量化之前; 律账本记滤波后的 counts)
  kp         开火期 kp 缩放 (环路带宽降, 不加相位)
  ki         开火期 ki 缩放
  ff         开火期前馈缩放 (阶梯上过冲的来源是前馈均值; 0 = 只留 PI)
  igate      开火期 I 项距离门全开
  组合

指标 (开火窗内, 跳过前两发):
  pre_med/pre_p95  每发**击发前一刻**的真实 |e| (子弹离开时的瞄准误差)
  rmse             开火窗 |e| 均方根
  e_min            竖直误差的最小值 (负 = 过冲/回推的幅度)
  e_pp             竖直误差的峰峰值 (准星在目标上下摆动的总幅度)
  v_p95            指令竖直速度 p95 (pull 脉冲的峰值)
  v_rip            指令速度减去 T 窗滑动均值后的 rms / 均值 (高频纹波占比)
  jerk             指令竖直速度的帧间差 rms (px/ms²)

用法: python3 -m arena.laws._wip_burst
"""
from __future__ import annotations
import math
import random

from arena.core import Arena, ArenaConfig
from arena.scenarios import Target
from arena.laws.ff_pi_acc import FFPiAccLaw
from arena.laws.base import register


class RecoilTarget(Target):
    """每 period 一发: y 阶跃 +kick (画面被后坐力推下去)。x 可叠加匀速 strafe。"""

    def __init__(self, x=0.0, y=0.0, vx=0.0, t0=500.0, period=100.0, kick=12.0,
                 n=20):
        super().__init__(x, y, vx, 0.0)
        self.t0 = t0
        self.period = period
        self.kick = kick
        self.n = n
        self._next = t0
        self.shot_times = []

    def advance(self, h, t):
        self.x += self.vx * h
        while len(self.shot_times) < self.n and t >= self._next - 1e-9:
            self.y += self.kick
            self.shot_times.append(self._next)
            self._next += self.period
        if len(self.shot_times) >= self.n:
            self._next = float("inf")


class _SmoothLaw(FFPiAccLaw):
    """ff_pi_acc 的逐位副本 + 开火窗内的候选整形 + 指令轨迹。

    副本只为在"限幅之后、量化之前"插一个钩子 —— 整形必须作用在期望速度
    (px/ms) 上, 且律自己的 counts 账本记录的必须是真发出去的那份 (Smith 预测
    的在飞扣除与自身活动门都读它)。重复一份步进体是实验文件的代价。
    """

    def __init__(self, gate=(500.0, 3000.0), lpf_tau=0.0, kp_scale=1.0,
                 ki_scale=1.0, ff_scale=1.0, igate_open=False,
                 pfil_tau=0.0, wext_scale=1.0, box_hx=0.0, box_hy=0.0,
                 box_c=0.3, **kw):
        super().__init__(**kw)
        self.gate = gate
        self.lpf_tau = lpf_tau
        self.kp_scale = kp_scale
        self.ki_scale = ki_scale
        self.ff_scale = ff_scale
        self.igate_open = igate_open
        self.pfil_tau = pfil_tau
        self.wext_scale = wext_scale
        self.box_hx = box_hx
        self.box_hy = box_hy
        self.box_c = box_c

    def reset(self, cfg):
        super().reset(cfg)
        self._sx = self._sy = 0.0
        self._s_on = False
        self._px = self._py = 0.0
        self._p_on = False
        self.vtrace = []          # (t, vx, vy, ex, ey)
        self._last_v = (0.0, 0.0, 0.0, 0.0)

    def _fire(self, t):
        return self.gate[0] <= t <= self.gate[1]

    def step(self, t, obs):
        cfg = self.cfg
        if obs is not None and obs.new:
            self._update_filter(obs)

        fire = self._fire(t)
        cx = cy = 0
        age = t - self.t_pub
        if self.filt and age < self.STALE:
            w_state = self._w_update(cfg, cfg.h)
            Lc = cfg.L * self.l_comp
            cp = self.ch.at(self.t_pub - Lc)
            cn = self.ch.cum()
            b = max(self._last_beta, 1e-9)
            eps_x = self.ax_e * self._last_T * (self._last_alpha / b - 0.5)
            eps_y = self.ay_e * self._last_T * (self._last_alpha / b - 0.5)
            vff_x = self.fvx + eps_x
            vff_y = self.fvy + eps_y
            w_win = age + Lc
            wext = self.wext_scale if fire else 1.0
            ex = (self.fx + vff_x * w_win * wext
                  + 0.5 * self.ax_e * w_win * w_win
                  - cfg.s * (cn[0] - cp[0]))
            ey = (self.fy + vff_y * w_win * wext
                  + 0.5 * self.ay_e * w_win * w_win
                  - cfg.s * (cn[1] - cp[1]))
            # ---- 实验钩子 B: 软死区 (瞄准框) + 中心回拉 ----
            # e' = e - (1-c)*h*tanh(e/h): 箱内增益 c (回拉趋势), 箱外增益 → 1
            # (大误差/拉枪完全不受影响)。h=0 时退化为 e'=e (关闭)。
            if fire and self.box_hy > 0:
                h = self.box_hy
                ey = ey - (1.0 - self.box_c) * h * math.tanh(ey / h)
            if fire and self.box_hx > 0:
                h = self.box_hx
                ex = ex - (1.0 - self.box_c) * h * math.tanh(ex / h)

            r = math.hypot(ex, ey)
            gate = self.i_gate / (self.i_gate + r) if self.i_gate > 0 else 1.0
            if fire and self.igate_open:
                gate = 1.0

            kp = self.kp * (self.kp_scale if fire else 1.0)
            ki = self.ki * (self.ki_scale if fire else 1.0)
            i_lim = self.i_frac * self.max_v / max(ki, 1e-9)
            vx_u = kp * ex + ki * self.int_x
            vy_u = kp * ey + ki * self.int_y

            if ex * ex + ey * ey > cfg.fov_radius * cfg.fov_radius:
                self.int_x = self.int_y = 0.0
            else:
                wx = (vx_u > self.max_v and ex > 0) or (vx_u < -self.max_v and ex < 0)
                wy = (vy_u > self.max_v and ey > 0) or (vy_u < -self.max_v and ey < 0)
                if not wx:
                    self.int_x = max(-i_lim, min(i_lim, self.int_x + ex * cfg.h * gate))
                if not wy:
                    self.int_y = max(-i_lim, min(i_lim, self.int_y + ey * cfg.h * gate))

            # ---- 实验钩子 A: 只滤 P+I (前馈保持准时) ----
            if fire:
                if not self._p_on:
                    self._px, self._py = vx_u, vy_u
                    self._p_on = True
                if self.pfil_tau > 0:
                    a = 1.0 - math.exp(-cfg.h / self.pfil_tau)
                    self._px += a * (vx_u - self._px)
                    self._py += a * (vy_u - self._py)
                    vx_u, vy_u = self._px, self._py
            else:
                self._p_on = False
                self._px, self._py = vx_u, vy_u

            frame_dt = cfg.frame_dt if cfg.frame_dt > 0 else self.DT0
            gap_scale = 1.0 - max(0.0, min(1.0, (age - frame_dt) / max(1.0, cfg.L)))
            ff_gate = gate + (1.0 - gate) * (1.0 - w_state)
            ff_eff = self.ff_gain * ff_gate * gap_scale
            if fire:
                ff_eff *= self.ff_scale
            vx_u += ff_eff * vff_x
            vy_u += ff_eff * vff_y

            vx = max(-self.max_v, min(self.max_v, vx_u))
            vy = max(-self.max_v, min(self.max_v, vy_u))

            # ---- 实验钩子: 开火窗内的整形 (作用在期望速度 px/ms 上) ----
            if fire:
                if not self._s_on:
                    self._sx, self._sy = vx, vy      # 从当前指令起步, 无阶跃
                    self._s_on = True
                if self.lpf_tau > 0:
                    a = 1.0 - math.exp(-cfg.h / self.lpf_tau)
                    self._sx += a * (vx - self._sx)
                    self._sy += a * (vy - self._sy)
                    vx, vy = self._sx, self._sy
                else:
                    self._sx, self._sy = vx, vy
            else:
                self._s_on = False
                self._sx, self._sy = vx, vy
            # ------------------------------------------------------------
            self._last_v = (vx, vy, ex, ey)

            s = max(0.05, min(20.0, cfg.s))
            self.rem_x += vx * cfg.h / s
            self.rem_y += vy * cfg.h / s
            cx, cy, self.rem_x, self.rem_y = self._counts(
                self.rem_x, self.rem_y, s, cfg.count_limit)
        else:
            self.rem_x = self.rem_y = 0.0
            self.int_x = self.int_y = 0.0
            self._last_v = (0.0, 0.0, 0.0, 0.0)

        self.ch.add(t, cx, cy)
        self.vtrace.append((t,) + self._last_v)
        return cx, cy


@register("_wip_plain")
class _WipPlain(_SmoothLaw):
    """逐位副本对照 (所有整形关闭) — 用来确认副本与现役律逐位一致。"""


@register("_wip_box_pct")
class _WipBoxPct(FFPiAccLaw):
    """现役律 + 常开瞄准框 (smooth=25%, bbox 100×220): 量常开对标准四组的代价。"""

    def __init__(self, **kw):
        kw.setdefault("smooth", 25)
        kw.setdefault("box_w", 100.0)
        kw.setdefault("box_h", 220.0)
        super().__init__(**kw)


def run_recoil(law_factory, period=100.0, kick=12.0, n=20, fps=120,
               L_belief=50.0, L_true=50.0, s_belief=1.0, s_true=1.0,
               vx=0.0, noise=0.5, seed=1, max_v=1.5, drop_p=0.0):
    t0 = 500.0
    dur = t0 + period * n + 600.0
    cfg = ArenaConfig(noise_std=noise, fps=fps, duration=dur,
                      s_true=s_true, L_true=L_true, drop_p=drop_p)
    rng = random.Random(seed)
    tgt = RecoilTarget(vx=vx, t0=t0, period=period, kick=kick, n=n)
    ar = Arena(cfg, tgt, rng, cross0=(0.0, 0.0))
    res = ar.run(law_factory(), s_belief, L_belief, max_v)
    return res, tgt.shot_times, t0 + period * n, ar


def burst_metrics(res, shots, t_end, h, trace, box_hy=0.0, box_hx=0.0):
    t, ex, ey = res["t"], res["ex"], res["ey"]
    idx = [i for i, tt in enumerate(t) if tt <= t_end]
    # 跳过前两发 (起步瞬态), 只量稳态突发段
    t_ss = shots[min(5, len(shots) - 1)] if len(shots) > 6 else shots[0]
    win = [i for i, tt in enumerate(t) if t_ss <= tt <= t_end]
    e = [math.hypot(ex[i], ey[i]) for i in win]
    rmse = math.sqrt(sum(v * v for v in e) / len(e)) if e else float("inf")
    pre = []
    for ts in shots[2:]:
        cand = [i for i, tt in enumerate(t) if ts - 1e-9 > tt]
        if cand:
            i = cand[-1]
            pre.append(math.hypot(ex[i], ey[i]))
    ps = sorted(pre)
    pre_med = ps[len(ps) // 2] if ps else float("inf")
    pre_p95 = ps[min(len(ps) - 1, int(0.95 * len(ps)))] if ps else float("inf")
    eyw = [ey[i] for i in win]
    e_min = min(eyw) if eyw else 0.0
    e_pp = (max(eyw) - min(eyw)) if eyw else 0.0
    in_box = 0.0
    pre_box = 0.0
    if box_hy > 0:
        in_box = sum(1 for v in eyw if abs(v) <= box_hy) / max(1, len(eyw))
        pb = []
        for ts in shots[2:]:
            cand = [i for i, tt in enumerate(t) if ts - 1e-9 > tt]
            if cand:
                i = cand[-1]
                pb.append(max(0.0, abs(ey[i]) - box_hy))
        pre_box = sum(pb) / len(pb) if pb else 0.0
    # 指令速度 (律自己的期望速度, 不是量化后的 counts)
    tv = [x for x in trace if t_ss <= x[0] <= t_end]
    v = [x[2] for x in tv]
    vpos = sorted(x for x in v if x > 0)
    v_p95 = vpos[min(len(vpos) - 1, int(0.95 * len(vpos)))] if vpos else 0.0
    vmean = sum(v) / len(v) if v else 0.0
    # 纹波: 减去 T 窗滑动均值
    nw = max(1, int(round(100.0 / h)))       # 名义 T=100ms 的窗, 口径统一
    res_rip = []
    for i in range(len(v)):
        a = max(0, i - nw // 2); b = min(len(v), i + nw // 2 + 1)
        m = sum(v[a:b]) / (b - a)
        res_rip.append(v[i] - m)
    v_rip = (math.sqrt(sum(x * x for x in res_rip) / len(res_rip)) / vmean
             if vmean > 1e-9 else float("inf"))
    dv = [abs(v[i] - v[i - 1]) / h for i in range(1, len(v))]
    jerk = math.sqrt(sum(x * x for x in dv) / len(dv)) if dv else 0.0
    return {
        "pre_med": pre_med, "pre_p95": pre_p95, "rmse": rmse,
        "e_min": e_min, "e_pp": e_pp, "v_p95": v_p95,
        "v_rip": v_rip, "jerk": jerk, "diverged": res["diverged"],
        "in_box": in_box, "pre_box": pre_box,
    }


def _variants():
    g = (500.0, 3000.0)
    return [
        ("base", lambda: _SmoothLaw(gate=g)),
        ("ff0", lambda: _SmoothLaw(gate=g, ff_scale=0.0)),
        ("ff.5", lambda: _SmoothLaw(gate=g, ff_scale=0.5)),
        ("kp.50", lambda: _SmoothLaw(gate=g, kp_scale=0.50)),
        ("kp.25", lambda: _SmoothLaw(gate=g, kp_scale=0.25)),
        ("ki.33", lambda: _SmoothLaw(gate=g, ki_scale=0.33)),
        ("lpf10", lambda: _SmoothLaw(gate=g, lpf_tau=10.0)),
        ("lpf20", lambda: _SmoothLaw(gate=g, lpf_tau=20.0)),
        ("lpf40", lambda: _SmoothLaw(gate=g, lpf_tau=40.0)),
        ("ff0+lpf20", lambda: _SmoothLaw(gate=g, ff_scale=0.0, lpf_tau=20.0)),
        ("ff0+kp.5", lambda: _SmoothLaw(gate=g, ff_scale=0.0, kp_scale=0.50)),
        ("ff0+igate", lambda: _SmoothLaw(gate=g, ff_scale=0.0, igate_open=True)),
        ("box25", lambda: _SmoothLaw(gate=g, box_hy=27.5, box_hx=6.25, box_c=0.5)),
        ("pfil10", lambda: _SmoothLaw(gate=g, pfil_tau=10.0)),
        ("pfil20", lambda: _SmoothLaw(gate=g, pfil_tau=20.0)),
        ("pfil40", lambda: _SmoothLaw(gate=g, pfil_tau=40.0)),
        ("wext.75", lambda: _SmoothLaw(gate=g, wext_scale=0.75)),
        ("wext.50", lambda: _SmoothLaw(gate=g, wext_scale=0.50)),
        ("wext.25", lambda: _SmoothLaw(gate=g, wext_scale=0.25)),
        ("wext.5+lpf10", lambda: _SmoothLaw(gate=g, wext_scale=0.5, lpf_tau=10.0)),
        ("wext.5+pfil10", lambda: _SmoothLaw(gate=g, wext_scale=0.5, pfil_tau=10.0)),
        ("wext.75+pfil10", lambda: _SmoothLaw(gate=g, wext_scale=0.75, pfil_tau=10.0)),
        ("wext.75+pfil20", lambda: _SmoothLaw(gate=g, wext_scale=0.75, pfil_tau=20.0)),
        ("wext.75+lpf20", lambda: _SmoothLaw(gate=g, wext_scale=0.75, lpf_tau=20.0)),
        ("wext.5+lpf20", lambda: _SmoothLaw(gate=g, wext_scale=0.5, lpf_tau=20.0)),
    ]


def _table(period, kick, vx=0.0):
    print(f"\n########## 后坐阶梯 T={period:.1f}ms K={kick}px strafe_vx={vx} "
          f"120fps L=50 ##########")
    print(f"{'variant':12s} {'pre_med':>8s} {'pre_p95':>8s} {'rmse':>7s} "
          f"{'e_min':>7s} {'e_pp':>7s} {'v_p95':>7s} {'v_rip':>6s} {'jerk':>8s}")
    for name, fac in _variants():
        res, shots, t_end, ar, law = _run_with_law(fac, period, kick, vx)
        m = burst_metrics(res, shots, t_end, 2.0, law.vtrace)
        flag = " DIV" if m["diverged"] else ""
        print(f"{name:12s} {m['pre_med']:8.2f} {m['pre_p95']:8.2f} "
              f"{m['rmse']:7.2f} {m['e_min']:7.2f} {m['e_pp']:7.2f} "
              f"{m['v_p95']:7.3f} {m['v_rip']:6.2f} {m['jerk']:8.3f}{flag}")


def _run_with_law(law_factory, period, kick, vx):
    law = law_factory()
    t0 = 500.0
    n = 20
    dur = t0 + period * n + 600.0
    cfg = ArenaConfig(noise_std=0.5, fps=120, duration=dur)
    rng = random.Random(1)
    tgt = RecoilTarget(vx=vx, t0=t0, period=period, kick=kick, n=n)
    ar = Arena(cfg, tgt, rng, cross0=(0.0, 0.0))
    res = ar.run(law, 1.0, 50.0, 1.5)
    return res, tgt.shot_times, t0 + period * n, ar, law


def law_trace(law, fac, period, kick, vx):
    return law.vtrace


def _final_tables():
    """现役律 (ff_pi_acc) 的瞄准框/超前: 文档数字的复现路径。

    误差指标 (pre/rmse/e_min/e_pp/in_box) 由现役律直接给出; jerk 由上面的副本
    (逐位一致, 只是带指令轨迹) 的 box25 变体给出 (同一几何: hy=27.5/hx=6.25/c=0.5)。
    """
    cases = [("900rpm K=8 ", 66.7, 8.0), ("600rpm K=12", 100.0, 12.0),
             ("400rpm K=20", 150.0, 20.0)]
    print("\n########## 现役律: 瞄准框 smooth=25 / 超前 lead=5 ##########")
    print(f"{'case':12s} {'cfg':14s} {'pre':>6s} {'rmse':>6s} {'e_min':>6s} "
          f"{'e_pp':>6s} {'in_box':>6s}")
    for name, T, K in cases:
        for lab, kw in [("off", {}), ("box25", dict(smooth=25)),
                        ("box25+lead5", dict(smooth=25, lead=5))]:
            cfg = ArenaConfig(noise_std=0.5, fps=120, duration=500 + T * 20 + 600,
                              s_true=1.0, L_true=50.0)
            rng = random.Random(1)
            tgt = RecoilTarget(t0=500, period=T, kick=K, n=20)
            ar = Arena(cfg, tgt, rng, cross0=(0.0, 0.0))
            law = FFPiAccLaw(box_w=100.0, box_h=220.0, aim_y_frac=0.35, **kw)
            res = ar.run(law, 1.0, 50.0, 1.5)
            m = burst_metrics(res, tgt.shot_times, 500 + T * 20, 2.0, [],
                              box_hy=27.5, box_hx=6.25)
            print(f"{name:12s} {lab:14s} {m['pre_med']:6.2f} {m['rmse']:6.2f} "
                  f"{m['e_min']:6.1f} {m['e_pp']:6.1f} {m['in_box']*100:5.0f}%")


def main():
    _table(150.0, 20.0)    # 慢射速高后座 (400rpm) — 主工况
    _table(100.0, 12.0)    # 600rpm 中等后座
    _table(66.7, 12.0)     # 900rpm
    _table(150.0, 20.0, vx=0.25)   # 主工况 + 目标横移
    _final_tables()


if __name__ == "__main__":
    main()
