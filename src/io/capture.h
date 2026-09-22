// ============================================================================
//  capture.h — 采集/推理编排: 取帧 (io/hdmi_in) → RGA 裁剪/换格式 (io/rga_pp) →
//    NPU 一次推理 tick (io/npu_axcl) → 检测平移回窗口域 → NMS → FOV 目标筛选 →
//    estimator_step 发布 g_target; 加上标定采样 (g_calib_collect 期整条推理链跳过)
//    与三源截图采集、预览绘制。
//
//  --- 坐标域 (整条链的公共单位) ---
//  规范窗口 = 采集帧的中心 1:1 裁剪, 边长 W = max(CAP_SIZE, 模型输入边长): 控制律、
//  FOV 门与标定都活在这个域里, 截图也存它 —— 域只有一个, 否则 spd 的基线、FOV 半径与
//  标定的块宽各有一套像素含义。模型输入更小
//  (比如 320) 时看到的是这个窗口**再**中心 1:1 裁一次 (同一个 crop_center, 源不同),
//  于是模型的框平移 (W − 模型边长)/2 就回到窗口域 —— 这个平移是"模型看到的那块画面"
//  与"律看到的那块画面"之间唯一的换算, 放在一处。
//  取帧源是 2560×1440: W 由模型边长与 CAP_SIZE 的较大者给出, 故模型边长不能超过源边长
//  (裁剪不缩放 —— RGA 一次 improcess 的 srect 与 drect 同尺寸即无缩放, 见 io/rga_pp.h)。
//
//  --- BGR 窗口按需产出 (稳态每帧只有一次裁剪) ---
//  640 BGR 窗口的消费者只有三个: 一次截图落盘、一轮标定 (相关域从它出来)、预览打开。
//  稳态 (无截图事件、未标定、无预览) 每帧就只有一次 RGA: 源 → W RGB (模型输入)。
//  截图事件在**检测之后**才判定 (det 源要看本帧有没有目标), 所以 BGR 那一次裁剪也在
//  那时才做 —— 落盘的那一帧是同一个采集缓冲, 早裁晚裁是同一块画面。
//
//  --- 标定期 ---
//  g_calib_collect (标定状态机 io/calib_run.h 开) 为真时整条推理链跳过: 标定既不需要
//  检测也不需要注入, 省下的算力正好付相关的账, 而发布的目标自然过期 (g_target 走
//  TARGET_STALE_MS 超时路径), 跑完第一次检测经既有 TRACK_JUMP_GATE 重锁 —— 没有新分支。
//  采样用 core/calib.h 的采样器 (640 BGR 窗口 → 灰度 → INTER_AREA 降到 320 半分辨率
//  相关域 → 3×3 块一维投影相位相关), 逐帧位移连同激励槽位经 cal_note_sample 进历史;
//  g_calib_request 到达时跑该模式的拟合 (cal_fit) 与诊断, 成功才回写本输出模式的延迟
//  VAR (CAL_VAR_HID/PAD/P5G, 由调用方按 -M 选好)。
//  **断流会让本轮作废**: 相关域的"上一帧"与采样历史都跨了那次断流 (帧间隔是秒级), 重建
//  之后两者一起清掉, 本轮按"无样本"如实失败 (状态机摇头, 什么都不写); 若请求恰好是在
//  断流期间到达的, 它在这里被一次消费即清并按失败回执 —— 采样侧看不到画面时, 任何回写
//  都只能是编造的值。
//
//  --- 失锁 / 断流 (io/hdmi_in 判据与重建) ---
//  接收器失去 TMDS 锁时驱动会自己停流并等源重新协商模式, 重锁**不会**把流接回去 —— 症状
//  只有 poll 超时。本层把它当"流断了"处理: 拆掉会话按节拍重建 (首次打开与重建同一条路),
//  期间照常发布 (目标自然过期), 恢复后第一次检测经 TRACK_JUMP_GATE 重锁。取源失败**不结束
//  进程** (信号一侧是外部条件, 进程活着才能等到画面回来); 推理链打不开是配置事实, 与原来
//  一样立刻结束。断流的账 (失锁/停流判定次数、重建成败、fence 超时与它烧掉的时间) 进
//  `[AI FPS]` 行的增量段与退出时的 `收帧统计:` 行。
//
//  --- 预览 ---
//  预览画的是 640 BGR 窗口 (律看到的那块画面, 也是数据集存的那块) 加检测框与瞄准点。
//  **整幅 (2560×1440) 每帧再转一次格式被否决**, 理由三条: 它每帧多一次全帧 RGA
//  (源 11.06MB 里被读走的正是被否决的那部分)、多一次 11MB 的 CPU 拷贝, 而这笔开销恰好
//  加在控制环同一根线程上; 1440p 每帧 blit 也远超 X11 的预算 (120fps 下 3.3 亿像素/s);
//  而 FOV 圈与自瞄真正作用的那块画面本来就在窗口里 —— 调试视图要看的东西一个都没少。
//  预览需要板上有显示: ssh 会话没有 DISPLAY, 板上跑着 gdm 的本地图形会话才有 (main 在
//  预览关闭时清掉 DISPLAY, 免得没有窗口系统时 OpenCV 报错)。
//
//  --- 日志格式 (浏览器控制面板按行解析, 前缀不可改) ---
//    `模型: <架构> <边长>x<边长>[ <类数>类]`  — 架构是单个 token (无空格)
//    `[AI FPS] <检测帧率> fps …`              — 检测帧率 = 走完整条链的帧/s (面板画的就是它);
//                                               行尾是取帧层的增量账 (等 fence / 失锁 / 重建)
//    `采集统计: fire=… det=… auto=…`          — 退出时一行
//    `收帧统计: 交付 … / 取代 … / …`           — 退出时一行 (交付语义的账 + 断流的账)
//    `[HDMI] …`                               — 设备/格式/时序/低延迟/失锁/重建 (io/hdmi_in)
//    `[SAVE] <源>  (fire=… det=… auto=… drop=…)` — 每张截图一行
//    `[标定] …`                               — 标定引擎自己的口径 (io/calib_run.cpp)
// ============================================================================

#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "core/detect.h"
#include "io/calib_run.h"      // CalMode (激励计划/注入单位) 与 CAL_VAR_* (回写 VAR 名)
#include "io/npu_axcl.h"
#include "io/rga_pp.h"

// 采集源帧率 (Hz): 取帧层按接收器**锁定时序**写入 (io/hdmi_in.h 的 frame_hz) —— 帧率是
//   信号的属性, 不是一个可给的选择 (故无 -f): 采集率由信号决定, 唯一可测的是检测率
//   ([AI FPS] 行)。控制拍 (律的帧长尺度) 与标定采样窗深度读它。
float src_fps();

// 接收器打开之前的初值 = 部署信号实测帧率 (S1: 2560×1440@120 → 120.000fps, 相邻帧间隔
//   p50 8.333ms)。它只在"采集线程还没打开接收器"的那一段有效。
constexpr float SRC_FPS_DEFAULT = 120.0f;

// NMS 的 IoU 门: 0.45 = YOLO 官方导出工具链的默认值 (与 core/detect_test 的手算候选同
//   值) —— 类别各自做 NMS (core/detect.h 的 nms), 故这个门只决定"同一类里哪些框算同一
//   个目标"; 自瞄最终只取 FOV 圈内最近的一个, 0.45 与 0.5 的差别落在极近的重叠框上。
constexpr float NMS_IOU_THR = 0.45f;

// 一帧推理的产物 (窗口域坐标)
struct PipelineTick {
    std::vector<Detection> dets;    // 窗口域 (W) 坐标, 已 NMS (未过 FOV 门)
    NpuTick npu;                    // pack/H2D/exec/D2H/解码 逐段耗时
    double  rga_us   = 0;           // 本帧 RGA 合计 (一次或两次 improcess)
    double  total_us = 0;           // 本帧 RGA + NPU 的合计
    bool    ok = false;             // 本帧走完 (RGA 与 NPU 都成功)
};

// 一帧的编排: RGA 阶段 + NPU 会话。类与实例都可多份 (探针与固件各持一份), 但一次运行
//   只有一条采集线程驱动它 —— 缓冲一次分配逐帧复用, 不做每帧的分配 (抖动进控制环)。
class CapturePipeline {
public:
    CapturePipeline() = default;
    ~CapturePipeline();
    CapturePipeline(const CapturePipeline&) = delete;
    CapturePipeline& operator=(const CapturePipeline&) = delete;

    // 打开模型并建立会话, 再按模型输入边长定规范窗口 W = max(CAP_SIZE, 边长)。
    //   num_classes: 未折叠 DFL 头需要它才能把 attrs = 4·reg_max + 类数 切开 (标准导出
    //   的 reg_max 不是可观测的物理量, 故不猜 —— 网格头的类数由属性数自解, 给 0 即可)。
    //   source_side: 源帧的**短边** (裁剪不缩放, 窗口与模型输入都不得超过它; 源几何只有
    //   打开接收器之后才知道, 故由调用方交进来)。失败写 err 并返回 false。
    bool open(const std::string& model_path, int num_classes, int source_side,
              std::string* err = nullptr);
    void close();
    bool ready() const { return ready_; }

    int window_side() const { return win_side_; }      // W (律/FOV/标定的坐标域边长)
    int model_side()  const { return model_side_; }    // 模型输入边长 (由运行时自述)
    int num_classes() const { return num_classes_; }
    const NpuSession& npu() const { return npu_; }

    // 一次推理: 源 dmabuf → W RGB → [二级裁剪到模型输入] → NPU tick → 解码 → NMS,
    //   检测坐标已是窗口域 (模型域加 (W − 模型边长)/2)。conf/want_cls 由调用方逐帧给
    //   (热参数)。返回 false 时 err 说明是哪一段 (该帧不产生检测)。
    bool infer(const RgaSrc& src, float conf_thr, int want_cls, PipelineTick* out,
               std::string* err = nullptr);

    // 640 BGR 窗口 (截图/标定/预览的唯一取像口): 一次 RGA 到 (CAP_SIZE, BGR888)。
    //   返回的缓冲在下一个采集缓冲 (源不同) 被写之前有效 —— 调用方按帧用完即弃。
    const RgaDst* bgr_window(const RgaSrc& src, std::string* err = nullptr);

private:
    RgaPp     rga_;
    NpuSession npu_;
    int  win_side_ = 0, model_side_ = 0, num_classes_ = 0;
    bool ready_ = false;
};

// AI 采集/推理线程 (main 孵化): 取源 (io/hdmi_in, 缺省按驱动名解析; 失锁/断流时按节拍
//   重建, 见该文件头) 与模型, 逐帧跑上面那条链并把目标发布给控制拍 (estimator_step),
//   同时承担标定采样/拟合/回写与三源截图、预览。**取源失败不结束进程** (源是外部条件,
//   进程活着才能等到画面回来); 模型/推理链打不开则按配置错误结束整个进程 (取不到画面
//   就没有目标, 而"正在运行"是一个比"报错退出"更糟的状态)。
void ai_thread(std::string model_path, int target_cls, int num_classes,
               std::string cam_dev, bool preview,
               float init_l, std::string persist_path, const char* cal_var,
               CalMode cal_mode,
               std::string out_dir, int fire_ms, double auto_s,
               int cooldown_ms, int jpeg_quality);
