// ============================================================================
//  pad_rp.h — PS5 遥控游玩输出后端 (-M rp): 以 libchiaki 纯控制会话
//    (CHIAKI_AUDIO_VIDEO_DISABLED) 经局域网 UDP 直连 PS5 遥控游玩服务,
//    音视频全丢, 只把合并后的逻辑手柄态发往主机。
//
//  分层与其它手柄后端 (pad_xinput / pad_p5g) 同构:
//    - 不占有 UDC (/dev/raw-gadget), 物理通道是局域网;
//    - 输入读取、合并层、摇杆账本、标定计划 (CAL_MODE_PAD) 与延迟槽 (RP_L_EST)
//      独立于本模块, 本模块只消费 pad_publish_snapshot() 并映射到 ChiakiControllerState。
// ============================================================================

#pragma once

#include <cstdint>

#include <chiaki/controller.h>

#include "io/pad_input.h"
#include "io/rp_account.h"

// 断线重连节拍 (ms): 与 pad_input 重扫、pad_p5g 加密狗重扫同一条 1s 节拍
constexpr int RP_RECONNECT_MS = 1000;

// 停机发送全零松键后的等待时长 (ms): 上游反馈发送线程的尾部重发首拍在 20 ms,
//   加 10 ms 调度余量保证首拍发出后再关闭会话 (与 ps5-input-bridge 一致)
constexpr int RP_RELEASE_WAIT_MS = 30;

// 允许丢包率上限: 局域网会话参数 (与 ps5-input-bridge 一致)
constexpr double RP_PACKET_LOSS_MAX = 0.05;

// 逻辑手柄态 → ChiakiControllerState (纯函数, 无 I/O):
//   - 摇杆直通 (PadLogical 与 ChiakiControllerState 同为上/左为负; 仅 XInput 取反)
//   - 扳机 uint8 0..255 直通
//   - PADBTN_BACK 与 PADBTN_TOUCH 同落 CHIAKI_CONTROLLER_BUTTON_TOUCHPAD
//     (PS5 游戏中触摸板是地图/记分板, 系统 Share 菜单对自瞄是干扰)
//   - 运动传感器置零、触控点置未按下 (-1)
void pad_rp_map(const PadLogical& st, ChiakiControllerState& out);

// 启动/停止 PS5 遥控游玩输出后端 (会话线程 + 1 kHz 泵线程)
bool pad_rp_start(const RpAccount& acc);
void pad_rp_stop();
