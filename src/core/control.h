// ============================================================================
//  control.h — ff_pi_acc 控制律的常数区 (结构参数, 编译期; 律内无手调增益 — 人手输入
//    只有 core/state.h 的速度刻度与标定出来的延迟):
//    收敛带宽 wn 由标定延迟 L 自动导出 (wn=(90°−PM)π/180/L), 阻尼比/前馈增益/
//    CUSUM 与 â 通道参数均给出物理出处; 跟踪滤波器增益、触发键位与接管保持窗
//    同在这里。各常量的推导与调整指引见 AGENTS.md "Tuning" 与
//    arena/laws/ff_pi_acc.py。
// ============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// ========================= 跟踪滤波器 (Smith 预测器, dt 归一) =========================
const float PRED_ALPHA0   = 0.50f;                   // 位置增益 @120fps; 实际 α=ALPHA0·dt/DT0 (帧率无关)
const float PRED_ALPHA_MAX = 0.90f;                  // α 饱和上限 (AGENTS 伪码 α=min(.9,·)): 帧率高于参考时
                                                     //   增益不再随 dt 归一比线性上涨
const float PRED_BETA0    = 0.03f;                   // 速度增益 @120fps; 实际 β=BETA0·dt/DT0 (0.03 = 失配带边缘余量)
const float PRED_BETA_MAX  = 0.60f;                  // β 饱和上限 (AGENTS 伪码 β=min(.6,·)); 重建旗标保持
                                                     //   时长的分母以同一上限封顶 — 两处必须同值
const float PRED_L_COMP   = 1.10f;                   // Smith 过补偿系数 (>1 帮欠补偿侧 L真>L̂, 危险方向)
const float PRED_DT0      = 1000.0f / 120.0f;        // 增益归一参考帧周期
const float TRACK_JUMP_GATE = 100.0f;                // 创新超此值 → 重置滤波器
const float TARGET_STALE_MS = 200.0f;                // 目标超时 → 暂停自瞄

// ========================= 控制律: 极点配置 PI + type-2 速度前馈 (ff_pi_acc) =========================
//  带宽由标定延迟 L 导出 (免手调); 前馈补跟踪速度; 方向矛盾 CUSUM 告警时该轴
//  v̂ 归零重拉 (变向/急停复用阶跃响应)。详见 arena/laws/ff_pi_acc.py 与 AGENTS.md。
const float FF_PM_DEG   = 50.0f;                     // 相位裕度: wn=(90°−PM)π/180/L (失配带 L20–80 全过的最快设计点)
const float FF_ZETA     = 1.0f;                      // 收敛阻尼比 (临界阻尼, 无过冲)
const float FF_GAIN_VAL = 1.0f;                      // 速度前馈增益: =1 是匀速目标零拖尾的精确开环指令
const float FF_I_GATE   = 8.0f;                      // 收敛区门控 / I 距离衰减尺度 (px)
const float FF_I_FRAC   = 1.0f;                      // 积分限幅 = I_FRAC×max_v/Ki
// CUSUM 参数 (均为 σ 倍数, 无量纲 — Page 序贯变化检测): K 漂移 0.5σ,
// C 单帧增量上限 3σ (拒后坐力式单帧踢脚), H 告警 9σ (同号持续 ~2 帧触发)
const float CUSUM_K = 0.5f;
const float CUSUM_C = 3.0f;
const float CUSUM_H = 9.0f;
// â 加速度偏差补偿通道 (创新均值反演, 修 α-β 对匀加速目标的结构滞后; 依据详见
//   arena/laws/ff_pi_acc.py — 无加速时 â≡0, 指令流与纯 PI+FF 逐位一致):
const float ACC_SIG_CLIP_K = 2.0f;   // 清洗创新 Huber 截断 (σ̂_r 倍数, M-estimator 标准断点)
const float ACC_TAU_L      = 4.0f;   // ȳ EMA 记忆 = ACC_TAU_L·L̂ (须 >> CUSUM 告警延迟, << 加速段时长)
const float ACC_RB_HOLD_N  = 1.5f;   // 重建旗标最短保持 = N·(T/β) (α-β 速度估计自身时间常数)
const float ACC_OW_ACTIV_K = 2.0f;   // 自身活动门限 θ_a = max_v/(K·L̂) (K 个延迟窗走完速度帽算剧烈)
const float ACC_SNR        = 10.0f;  // â 显著性地板倍数 (盖过失配带相关 dither 噪声驱动的均值游走)

// ========================= 开火期瞄准框 (软死区) =========================
// 后坐力在屏幕上是每发一个阶跃 (瞄准点被推下 K px, 间隔 T)。点瞄准把每一发当
//   独立阶跃去收敛, 指令是一串 pull 脉冲 (每发压一下再等下一发 — 非人力输出)。
//   开火期 (扳机按下) 把瞄准目标从一点换成检测框内的一个"躯干窗": 误差按
//      e' = e - (1-c)·h·tanh(e/h)
//   整形 — 箱内增益 c (中心回拉趋势, 永不摆烂), 箱外增益 → 1 (大误差/拉枪/失配
//   完全不受影响: 锁的远场原样)。框的尺寸就是操作员的调节量 (热参/CLI smooth,
//   竖直死区高度占检测框高度的百分比, 横向宽 = 它的一半):
//      hy = smooth·bbox_h/200,   hx = smooth·bbox_w/400
//   smooth=0 → h=0 → e'=e (关闭)。箱内中心增益 c 固定为 BOX_CENTER_GAIN —
//   它是"锁"的语义 (箱内半增益回拉), 不是手感旋钮。
//   实测 (arena/laws/ff_pi_acc.py + arena/laws/_wip_burst.py 的后坐阶梯工况,
//   120fps L=50, bbox 100×220px, smooth=25 → hy=27.5px):
//     400rpm/K20: rmse 9.13→7.45px, e_pp 25→22px, jerk 0.029→0.017
//     600rpm/K12: rmse 3.15→3.15px, jerk 0.022→0.013
//     900rpm/K8 : rmse 2.21→2.43px, jerk 0.017→0.010 (低后座档的容差代价)
//   失配/丢帧: L30 的 rmse 8.41→4.17 (最大改善); L70/丢帧各 +8%/+3% (击发前
//   误差同步变小), s±30%/60fps/横移不升;
//   开火中 80px 阶跃: 首次到达 172→188ms、settle 302→432ms (末端 (1-c)h 以 c
//   增益缓收 — 框的语义), 过冲 3.2→4.12px (仍在箱内)。常开会拖累标准四组
//   (OVERALL 123.62→149.95), 故只门控在开火期。
const float BOX_CENTER_GAIN = 0.5f;  // 箱内中心回拉增益 (箱外恒 1: 大误差/拉枪原样;
                                     //   实测膝点 — 0.3 更平滑但开火中拉枪的 settle
                                     //   多 100ms, 0.15 会让误差骑到箱沿上)
inline float smooth_axis(float e, float h, float c) {
    if (h <= 0.0f || c >= 1.0f) return e;
    return e - (1.0f - c) * h * std::tanh(e / h);
}
// 框的半宽/半高 (px): smooth 是竖直死区高度占检测框高的百分比, 横向宽取它的一半
//   (同一百分比口径), 中心 = 瞄准点。smooth=0 → 0 (恒等)。
inline float smooth_half_h(int pct, float bbox_h) {
    return (float)pct * bbox_h / 200.0f;
}
inline float smooth_half_w(int pct, float bbox_w) {
    return (float)pct * bbox_w / 400.0f;
}

// ========================= 追踪超前 (bullet lead) =========================
// 子弹有飞行时间: 目标横移时瞄准点应沿运动方向前移, 否则弹着点在目标身后 (准星在
//   身上却打不中)。前移量 = lead% × 检测框尺寸, 方向取自目标的**绝对**屏速 ——
//   TargetState.vx/vy 是估计器扣掉自身在途位移后的目标速度, 不是准星相对运动
//   (子弹不会因为准星向右移动就向右拐弯)。饱和速度 0.25 px/ms = 本库行为模型的
//   横移带 0.12–0.4 px/ms 的中点: 典型横移拿满 lead%, 慢目标按比例给, 静止目标 0
//   (零附近连续, 没有符号抖振)。两轴都生效; 偏移在 law_tick 里**最后一步**夹框:
//   先正常算瞄准点的绝对位置, 越出 YOLO 检测框就钉在框边 —— 永远不越框。
//   落地符号: 误差输入取 e_ref = e_target + lead (律对 e_ref 归零 → 准星在目标
//   **前方** lead px, 与运动同向; 取减号会变成拖后)。
const float LEAD_V_SAT = 0.25f;   // px/ms (横移带中点)
inline float lead_offset_px(int pct, float bbox_dim, float v_abs) {
    if (pct <= 0 || bbox_dim <= 0.0f) return 0.0f;
    return (float)pct * 0.01f * bbox_dim
           * std::clamp(v_abs / LEAD_V_SAT, -1.0f, 1.0f);
}
// 逐轴超前 (最后一步): 先按公式正常算, 再把瞄准点的**绝对位置**夹进 YOLO 框
//   [0, bbox_dim], 越界就钉在框边 —— 返回相对瞄准点的偏移。aim_frac 是瞄准点在
//   框内的比例 (横向 0.5, 纵向 y_off%)。永远不把瞄准点推出检测框。
inline float lead_clamped_axis_px(int pct, float bbox_dim, float v_abs,
                                  float aim_frac) {
    const float aim = aim_frac * bbox_dim;
    const float pos = std::clamp(aim + lead_offset_px(pct, bbox_dim, v_abs),
                                 0.0f, bbox_dim);
    return pos - aim;
}

// 单拍指令的硬上限 (counts, hid 落点): 120 counts/拍 @1kHz = 120 px/ms (hid 基线
//   1 px/count), 而出厂速度帽 -x 2667 px/s 只有 2.67 px/ms —— 它是帽的 45 倍, 故这不是
//   控制带而是失控保护: 只有积分器飞掉才撞得到, 撞到也只是让单份报告不成一次瞬移。
//   取 120 = 便于读的整百值 (选择规则); 它不参与"追不追得上"的约束 (那是 -x 的活)。
const int CMD_COUNT_CLAMP = 120;

// ========================= 触发键位 (HID 按钮位掩码) =========================
const uint16_t LEFT_KEY  = 0x01;
const uint16_t RIGHT_KEY = 0x02;
const uint16_t SIDE_KEY  = 0x10;                     // 侧键2: 抑制自瞄
const uint16_t SIDE_KEY2 = 0x08;                     // 侧键1
const uint16_t BOTH_SIDE_KEYS = SIDE_KEY | SIDE_KEY2; // 双侧键: 标定触发

const int KEEP_ALIVE_MS = 200;                       // 松开触发键后保持自瞄的时间

// 控制拍 (周期 TICK_MS): 把自瞄指令 (counts) 与标定激励序列合成进 HID 报文的
//  位移字节; 纯透传 (-a n / 热参 aim=0) 时不注入。cam_fps 用于丢帧期前馈衰减的
//  时间尺度, rpt 为 HID_REPORT_LEN 字节报文缓冲 (只改写位移字节)。
void control_apply(int cam_fps, uint8_t* rpt, int16_t real_x, int16_t real_y);

// pad 输出模式的律入口 (与 control_apply 同一份律数学, 逐句一致; 指标去向不同:
//  hid 量化成 counts 写报文位移字节, pad 交付期望速度 px/ms 由合并层换算成摇杆
//  偏转)。btns 为触发键位字 (fire→LEFT_KEY, ads→RIGHT_KEY, 由 pad 侧 RT/LT 门控
//  生成 — 侧键抑制与双侧键标定是 hid 键位语义, pad 键位字恒无那些位, 分支自然
//  惰性)。速度帽逐轴取 min(热参 x, 该轴有效满偏屏速 A_eff/1000): 注入打到该轴
//  满偏即该轴行程上限 (hid 两轴同为热参 x, 算式与单帽逐位相同)。返回注入门
//  (接管开 × 触发保持窗内)。
bool control_apply_pad(int cam_fps, uint16_t btns, float& out_vx, float& out_vy);
