// ============================================================================
//  capture.cpp — capture.h 的实现: 逐帧编排 (取帧 → RGA → NPU → 解码 → 平移回窗口域
//    → NMS → FOV 选目标 → estimator_step), 标定采样/拟合/回写, 三源截图与预览。
//    坐标域、BGR 窗口按需产出、标定期跳过推理、预览口径的理由都在 capture.h 头部。
//    复现路径 (板端, root): sudo ./bin/aimbot -m engine/yolo11s.axmodel -c -1 -n 80 …
// ============================================================================

#include "io/capture.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>

#include "core/calib.h"
#include "core/estimator.h"
#include "core/state.h"
#include "io/hdmi_in.h"

namespace {

std::atomic<float> g_src_fps{SRC_FPS_DEFAULT};

// 逐段计时用 (单调钟, µs): RGA 是同步 ioctl, NPU 那三段在 io/npu_axcl.cpp 里各计各的
double now_us() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
}
void set_err(std::string* err, const std::string& m) { if (err) *err = m; }

// 有没有可用的窗口系统: X11 (DISPLAY) 或 Wayland (WAYLAND_DISPLAY) 之一非空即可。
//   本库的 OpenCV 用 Qt 后端, **没有平台插件时它是 qFatal 直接杀进程** (实测 RC=134),
//   而不是抛异常 —— 所以这条判据必须在进入 namedWindow 之前给出。部署形态正是 ssh 会话
//   (root, 没有 DISPLAY), 那里预览就该降级成"无预览 + 说清原因", 而不是让整条链陪葬。
bool has_display() {
    const char* d = std::getenv("DISPLAY");
    const char* w = std::getenv("WAYLAND_DISPLAY");
    return (d && *d) || (w && *w);
}

// 分位点 (窗口内的帧龄; 每 60s 清空重来, 故直接排序)
double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)(p / 100.0 * (double)(v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}

// 解码口径的短名: `模型:` 行的架构 token 必须是**单个** token (浏览器控制面板的
//   MODEL_RE 用 \S+ 取它, 紧接着要求 <宽>x<高>), 故这里各支都给一个无空格的短名。
const char* arch_token(HeadKind k) {
    switch (k) {
        case HeadKind::kYoloV5:  return "YOLOv5";
        case HeadKind::kYoloV8:  return "YOLOv8/11";
        case HeadKind::kEnd2End: return "端到端";
        case HeadKind::kYoloDfl: return "DFL";
    }
    return "?";
}

}  // namespace

float src_fps() { return g_src_fps.load(); }

CapturePipeline::~CapturePipeline() { close(); }

void CapturePipeline::close() {
    npu_.close();
    ready_ = false;
}

bool CapturePipeline::open(const std::string& model_path, int num_classes, int source_side,
                           std::string* err) {
    close();
    num_classes_ = num_classes;
    if (!npu_.open(model_path, err)) return false;

    model_side_ = npu_.input_side();
    if (model_side_ <= 0) {
        set_err(err, "模型输入不是方形 (边长未知): 网格几何与裁剪平移都定不下来");
        npu_.close();
        return false;
    }
    win_side_ = std::max(CAP_SIZE, model_side_);
    // 裁剪不缩放 (io/rga_pp.h): 窗口与模型输入都取自源帧的中心 1:1 矩形, 超过源的短边
    //   就没有那块画面。源几何在打开接收器之前不可知, 故由调用方把它交进来。
    if (win_side_ > source_side) {
        char b[256];
        snprintf(b, sizeof(b),
                 "窗口/模型输入 %d 超过源短边 %d (裁剪不缩放: 中心 1:1 矩形取不到那块画面)",
                 win_side_, source_side);
        set_err(err, b);
        npu_.close();
        return false;
    }

    // ---- 解码口径与类数 (先按几何判, 判不了再看属性数 —— 判据在 core/detect.cpp) ----
    size_t first = (size_t)-1;
    for (size_t k = 0; k < npu_.outputs().size(); ++k)
        if (npu_.outputs()[k].kept) { first = k; break; }
    if (first == (size_t)-1) {
        set_err(err, "该模型没有可解的检测输出 (布局解不出来的输出只进日志, 见 io/npu_axcl.h)");
        npu_.close();
        return false;
    }
    const NpuTensor& t0 = npu_.outputs()[first];
    const HeadKind kind = head_kind(t0.layout, num_classes_, model_side_);
    // 网格头的类数由属性数自解 (attrs − 4/5), 于是"没给类数"也能把 fallback 口径走对;
    //   未折叠 DFL 头没有这条出口 —— attrs = 4·reg_max + 类数, 而 reg_max 不是可观测量,
    //   故类数必须由调用方给 (猜一个分布长度等于猜框)。
    int classes = num_classes_;
    if (kind != HeadKind::kYoloDfl && kind != HeadKind::kEnd2End)
        classes = t0.layout.attrs - (kind == HeadKind::kYoloV5 ? 5 : 4);
    if (kind == HeadKind::kYoloDfl && classes < 1) {
        bool any_grid = false;
        for (const auto& t : npu_.outputs())
            if (t.kept && head_kind(t.layout, 0, model_side_) != HeadKind::kYoloDfl)
                any_grid = true;
        if (!any_grid) {
            set_err(err, "未折叠 DFL 头需要类数才能切开 attrs = 4·reg_max + 类数 (-n <类数>, "
                         "公开 YOLO11 模型是 80)");
            npu_.close();
            return false;
        }
        fprintf(stderr, "[NPU] ⚠ DFL 输出未解码: 未给类数 (-n); 该轮只用网格头\n");
    }
    if (classes > 0) num_classes_ = classes;

    std::cout << "模型: " << arch_token(kind) << " " << model_side_ << "x" << model_side_;
    if (classes > 0) std::cout << " " << classes << "类";
    std::cout << " (" << (npu_soc_name()[0] ? npu_soc_name() : "NPU") << ", "
              << model_path << ", 输出 " << npu_.outputs().size() << " 保留 " << npu_.kept_outputs()
              << ", 输入 U8 NHWC RGB)\n";
    ready_ = true;
    return true;
}

bool CapturePipeline::infer(const RgaSrc& src, float conf_thr, int want_cls, PipelineTick* out,
                            std::string* err) {
    if (!ready_) { set_err(err, "会话未就绪"); return false; }
    const double t0 = now_us();
    // 1) 源 → 窗口 W (RGB888): 规范帧, 也是模型输入更大时的输入本身
    const RgaDst* w = rga_.crop_center(src, win_side_, V4L2_PIX_FMT_RGB24, err);
    if (!w) return false;
    const uint8_t* in = (const uint8_t*)w->map;
    // 2) 模型输入更小 → 窗口再中心裁一次 (同一 crop_center, 源换成上一级的落点)
    if (model_side_ < win_side_) {
        const RgaSrc ws{w->fd, w->side, w->side, w->stride_px, w->fourcc};
        const RgaDst* m = rga_.crop_center(ws, model_side_, V4L2_PIX_FMT_RGB24, err);
        if (!m) return false;
        in = (const uint8_t*)m->map;
    }
    const double t1 = now_us();
    std::vector<Detection> dets =
        npu_.run(in, num_classes_, conf_thr, want_cls, out ? &out->npu : nullptr);
    const double t2 = now_us();
    if (out && !out->npu.ok) { set_err(err, "推理 tick 失败 (H2D/execute/D2H)"); return false; }
    // 3) 模型域 → 窗口域: 模型看到的是窗口的中心 1:1 子矩形, 平移 (W − 模型边长)/2
    const int off = (win_side_ - model_side_) / 2;
    for (auto& d : dets) { d.cx += (float)off; d.cy += (float)off; }
    if (out) {
        out->dets = nms(dets, NMS_IOU_THR);
        out->rga_us = t1 - t0;
        out->total_us = t2 - t0;
        out->ok = true;
    }
    return true;
}

const RgaDst* CapturePipeline::bgr_window(const RgaSrc& src, std::string* err) {
    // 中心 1:1 裁剪 (W == CAP_SIZE 时与窗口同源同矩形; W 更大时它是窗口的中心 640 子矩形
    //   —— 两个矩形都居中且无缩放, 故像素与"窗口再裁一次"逐位相同)
    return rga_.crop_center(src, CAP_SIZE, V4L2_PIX_FMT_BGR24, err);
}

// ============================== AI 采集线程 ==============================

void ai_thread(std::string model_path, int target_cls, int num_classes,
               std::string cam_dev, bool preview,
               float init_l, std::string persist_path, const char* cal_var,
               CalMode cal_mode,
               std::string out_dir, int fire_ms, double auto_s,
               int cooldown_ms, int jpeg_quality) {
    const bool collecting_enabled = !out_dir.empty();

    // ---- 接收器: 首次打开与失锁后的重建走同一条路 (判定与顺序见 io/hdmi_in.h) ----
    // 打不开**不结束进程**: 源没上电/还没锁/刚在换模式都会走到这里, 而进程活着才能等到
    //   画面回来 (接收器会失锁并让驱动自己停流, 那时唯一的恢复动作是重建, 不是退出)。
    HdmiIn in;
    CapturePipeline pipe;
    std::string err;
    if (!in.open(cam_dev, HDMI_BUF_DEFAULT, HdmiDelivery::Newest, &err))
        std::cerr << "[HDMI] 打开接收器失败: " << err << " (按 HDMI_REARM_MIN_MS 的节拍重试)\n";
    auto   t_src_try = std::chrono::steady_clock::now() - std::chrono::hours(1);
    int    src_attempts = 0;
    int    src_side = 0;                 // 上一次建推理链时的源短边 (模式变化要重建)
    double src_hz = src_fps();

    // 取源 + 按源几何建/重建推理链。失败即返回 false, 由调用方等下一个节拍 —— 信号一侧
    //   永不结束进程; 唯一的例外是**从未建起来过**的推理链 (模型打不开/窗口比源还大),
    //   那是配置事实, 与原来一样立刻结束。
    auto ensure_source = [&]() -> bool {
        if (!in.opened() && !in.rearm(&err)) {
            std::cerr << "[HDMI] 取源失败 (第 " << ++src_attempts << " 次): " << err
                      << " (按 " << HDMI_REARM_MIN_MS << "ms 节拍重试)\n";
            return false;
        }
        src_attempts = 0;
        t_src_try = std::chrono::steady_clock::now() - std::chrono::hours(1);  // 恢复即清节拍
        const HdmiFormat& f = in.format();
        if (in.frame_hz() > 0) g_src_fps.store((float)in.frame_hz());
        src_hz = src_fps();     // 模式变了帧率也变 (控制拍的帧长尺度每拍现读 src_fps)
        if (!pipe.ready() || std::min(f.width, f.height) != src_side) {
            // 源模式变了就重建推理链: 规范窗口是源的**中心 1:1 裁剪**, 新源短边装不下它
            //   时只能等源换回来 (进程照常活着, "按节拍再试一次"本身就是那条恢复路径)。
            if (!pipe.open(model_path, num_classes, std::min(f.width, f.height), &err)) {
                if (src_side == 0) {          // 从未建起来过: 配置错误, 立刻结束 (输入本身的问题)
                    std::cerr << "AI: " << err << "\n";
                    global_running = false;
                } else {
                    std::cerr << "AI: 源 " << f.width << "x" << f.height << " 下建推理链失败: "
                              << err << "\n  (等源换回可裁剪的模式; 进程不退出)\n";
                    in.close();               // 连源一起收回: 下一个节拍整条路重来
                }
                return false;
            }
            src_side = std::min(f.width, f.height);
            std::cout << "✅ 采集线程已启动 (" << f.width << "x" << f.height << " @ " << src_hz
                      << "Hz " << HdmiIn::fourcc_name(f.fourcc) << ", 窗口 " << pipe.window_side()
                      << ", 模型输入 " << pipe.model_side() << ", " << cam_dev << ")\n";
            std::cout.flush();
        }
        return true;
    };
    if (!ensure_source() && !global_running) return;   // 推理链建不起来: 与原来一样结束这一支

    // 预览要有窗口系统 (见 has_display): 没有就把预览关掉并说清原因, 让整条链照常跑 ——
    //   调试视图是附加物, 不该因为启动它的会话没有 DISPLAY 而带走采集/推理/控制。
    if (preview && !has_display()) {
        std::cerr << "⚠ 预览不可用: 没有 DISPLAY/WAYLAND_DISPLAY (预览需要板上本地图形会话; "
                     "ssh 会话没有显示, 板上跑着 gdm 的本地会话才有)\n  继续无预览运行\n";
        preview = false;
    }
    if (preview) {
        try {
            cv::namedWindow("Aimbot", cv::WINDOW_AUTOSIZE);
        } catch (const cv::Exception& e) {
            std::cerr << "⚠ 预览不可用 (窗口系统拒绝建窗): " << e.what()
                      << "\n  继续无预览运行\n";
            preview = false;
        }
    }

    EstimatorState est;

    // 标定采样状态: l_est 是运行态里唯一的标定量 (回写只有延迟), 拟合在 g_calib_request
    //   时一次跑完 (计划/窗口/拟合/诊断/回写都在 io/calib_run.h)
    float l_est = init_l;
    std::deque<CalibSample> hist;
    bool was_collecting = false;
    CalibSampler sampler;
    cv::Mat prev_dom;                       // 上一帧的相关域图 (CV_32F 320×320)
    long collect_frames = 0, collect_samples = 0;
    double collect_ms = 0.0;                // 采样本身的累计耗时 (每帧成本实测)
    auto cal_t0 = std::chrono::steady_clock::now();
    // 采样窗深度按打开时的帧率取 (帧数口径); 源模式变了帧率也变, 但只可能变小 (120→60),
    //   于是深度是超集 —— 模式变化本来就会清掉历史 (见循环里的重建分支), 不必重算。
    const size_t hist_max =
        (size_t)std::max(60, cal_hist_frames(cal_mode, (int)std::lround(src_hz)));

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> jitter(0.5, 1.5);
    auto roll_auto = [&]() {
        return std::chrono::milliseconds((long)(auto_s * 1000.0 * jitter(rng)));
    };
    auto t_start = std::chrono::steady_clock::now();
    auto next_auto = t_start + roll_auto();
    auto last_save = t_start - std::chrono::hours(1);
    auto last_fire = t_start - std::chrono::hours(1);
    long n_fire = 0, n_det = 0, n_auto = 0;

    // 检测率与逐段成本 (每 60s 一行: 检测率 = 走完整条链的帧/s —— 面板画的就是它)
    long   fps_cnt = 0;
    double sum_rga = 0, sum_npu = 0, sum_pack = 0, sum_h2d = 0, sum_exec = 0, sum_d2h = 0,
           sum_dec = 0, sum_wait = 0, sum_fence = 0;
    std::vector<double> ages;               // 本窗口的帧龄 (帧时间戳 → 检测就绪, ms)
    auto fps_t0 = std::chrono::steady_clock::now();

    // 取帧层自己的账: 每 60s 报的是**本窗口的增量** (累计值在"收帧统计:"里), 而"断流一次
    //   到底花了多少"只有这一处的四个量能回答 —— 等 fence 的次数与其中等到超时的次数,
    //   以及失锁/停在原地的判定次数与重建的成败次数 (口径与出处见 io/hdmi_in.h)。
    struct SrcCounters {
        uint64_t fence_waits = 0, fence_timeouts = 0, lock_lost = 0, stalled = 0, rearm_ok = 0,
                 rearm_fail = 0;
        double   fence_us_sum = 0.0;
    };
    auto snap_src = [&]() {
        SrcCounters c;
        c.fence_waits = in.fence_waits();
        c.fence_timeouts = in.fence_timeouts();
        c.fence_us_sum = in.fence_wait_us_sum();
        c.lock_lost = in.lock_lost();
        c.stalled = in.stalled();
        c.rearm_ok = in.rearm_ok();
        c.rearm_fail = in.rearm_fail();
        return c;
    };
    SrcCounters src_prev = snap_src();

    while (global_running) {
        const auto t_loop = std::chrono::steady_clock::now();

        // ---- 取源: 失锁/断流之后在这里重建, 按 HDMI_REARM_MIN_MS 的节拍重试 (见 io/hdmi_in.h) ----
        if (!in.opened()) {
            // 断流期间的标定请求: 采样侧拿不到画面, 这一轮量不出任何东西 —— 一次消费即清并按
            //   失败回执 (状态机随即摇头收尾), 绝不回写编造的值。
            if (g_calib_request.exchange(false)) {
                std::cout << "[标定] 无法测量: 采集断流 (重建中, 本轮无画面) — 未回写\n";
                std::cout.flush();
                g_calib_done.store(2);
            }
            if (!hdmi_rearm_due(t_src_try, t_loop)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            t_src_try = std::chrono::steady_clock::now();
            if (!ensure_source()) {
                if (!global_running) break;     // 从未建起来过的推理链: 配置错误, 与原来一样结束
                continue;
            }
            // 重建之后: 相关域的前一帧属于上一个会话 (跨了一次断流), 采样历史同理 ——
            //   这一轮标定因此作废 (按"无样本"如实失败), 由操作者重来。
            prev_dom.release();
            was_collecting = false;
        }

        HdmiFrame f;
        const auto t_wait0 = std::chrono::steady_clock::now();
        HdmiFail fail = HdmiFail::Ok;
        if (!in.wait_frame(&f, &err, &fail)) {
            if (fail == HdmiFail::Interrupted) {
                // 退出信号 (Ctrl+C/停机): 与流无关, 不重建也不当丢帧
                if (!global_running) break;
                continue;
            }
            // 取帧失败只有两种去向 (判据见 io/hdmi_in.h): 单次 fence 超时是丢了一帧, 下一帧
            //   照取 (驱动的下一次 DQBUF 会自行把上一根 fence signal 掉); 其余一律拆掉重建 ——
            //   重建是唯一能让画面回来的动作, 而"退出"会让整条链陪着一次失锁一起死。
            std::cerr << "AI: 取帧失败: " << err << "\n";
            if (hdmi_fail_needs_rearm(fail)) {
                std::cerr << "[HDMI] 按上面的原因拆掉会话, 重建采集 (已交付 " << in.delivered()
                          << " 帧; 期间发布的目标自然过期, 恢复后第一次检测经既有跳变门重锁)\n";
                in.close();     // 拆掉会话: 下一轮从取源开始 (含节拍)
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        const double wait_ms = elapsed_ms(now, t_wait0);
        const float conf_thr = g_conf_thr.load(), y_off_pct = g_y_off_pct.load(),
                    fov_r = g_fov_radius.load(), max_v = g_max_v.load();
        const RgaSrc src{f.fd, f.width, f.height, f.stride_px, f.fourcc};

        // 标定期整条推理链跳过 (标定不需要检测也不需要注入, 省下的算力付相关的账)
        const bool cal_collecting = g_calib_collect.load();
        if (cal_collecting && !was_collecting) {
            hist.clear(); prev_dom.release();
            collect_frames = collect_samples = 0; collect_ms = 0.0;
            cal_t0 = now;
        }
        was_collecting = cal_collecting;

        std::vector<Detection> filtered;
        bool found = false;
        float best_dx = 0, best_dy = 0;
        // 本帧的 640 BGR 窗口 — 三个消费者 (截图落盘 / 标定相关域 / 预览) 复用同一帧的
        //   同一次裁剪; 标定期它已经裁出来了, 因此那一轮不会为预览多付一次 RGA
        const RgaDst* bgr = nullptr;
        if (!cal_collecting) {
            PipelineTick tick;
            if (!pipe.infer(src, conf_thr, target_cls, &tick, &err)) {
                std::cerr << "AI: 推理失败: " << err << "\n";
                in.release(f.index);
                continue;
            }
            filtered = tick.dets;
            // FOV 门兼积分器边界: 圈内最近的一个 (瞄准点按 -y 的偏移落在部位上)
            float best_dist = 1e9f;
            const float cx0 = pipe.window_side() / 2.0f, cy0 = pipe.window_side() / 2.0f;
            for (const auto& d : filtered) {
                const float ty = d.cy + d.h * (0.5f - y_off_pct / 100.0f);
                const float dx = d.cx - cx0, dy = ty - cy0;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist < best_dist && dist < fov_r) {
                    best_dist = dist; best_dx = dx; best_dy = dy; found = true;
                }
            }
            // 帧龄 = 取一帧 + RGA + 推理的总时长 (帧时间戳 → 检测就绪), 是这条链的端到端
            //   延迟; 逐段成本同期累计, 每 60s 一行。取帧那一段含等 dma-fence (载荷写完
            //   的凭据, 见 io/hdmi_in.h) 与排空/归还, 故一并报出, 三者之和就是帧周期。
            ++fps_cnt;
            sum_rga += tick.rga_us; sum_npu += tick.npu.total_us();
            sum_pack += tick.npu.pack_us; sum_h2d += tick.npu.h2d_us;
            sum_exec += tick.npu.exec_us; sum_d2h += tick.npu.d2h_us;
            sum_dec += tick.npu.decode_us;
            sum_wait += 1000.0 * (double)wait_ms;
            sum_fence += in.last_fence_wait_us();
            ages.push_back(elapsed_ms(std::chrono::steady_clock::now(),
                                      std::chrono::steady_clock::time_point(
                                          std::chrono::nanoseconds(f.ts_ns))));
            const double win_s = elapsed_ms(now, fps_t0) / 1000.0;
            if (win_s >= 60.0) {
                const double n = std::max(1.0, (double)fps_cnt) * 1000.0;   // 各段累计是 µs
                // 取帧层的增量账: 等 fence 的次数与其中等到超时的次数 (超时那几次烧掉的
                //   就是上限 —— 账见 io/hdmi_in.h 的 HDMI_FENCE_WAIT_MS), 以及失锁/停流
                //   的判定次数与重建成败。稳态下这四项全 0, 有值就是这一分钟里发生过什么。
                const SrcCounters sc = snap_src();
                const uint64_t d_waits = sc.fence_waits - src_prev.fence_waits;
                const uint64_t d_to = sc.fence_timeouts - src_prev.fence_timeouts;
                const double d_us = sc.fence_us_sum - src_prev.fence_us_sum;
                printf("[AI FPS] %d fps (源 %.2fHz, 窗口 %.1fs) | 取帧 %.2fms (其中等 fence "
                       "%.2f) | RGA %.2fms | NPU %.2fms (pack %.2f H2D %.2f exec %.2f D2H "
                       "%.2f 解码 %.2f) | 合计 %.2fms/帧 | 帧龄(帧时间戳→检测就绪) p50 %.1f "
                       "p99 %.1f max %.1fms | 等 fence %llu 次 (成功 %llu 次 均值 %.2fms) | "
                       "超时 %llu 次 (白等 %llu×%dms) | 失锁 %llu 次 停流 %llu 次 | 重建 "
                       "%llu 成功 %llu 失败\n",
                       (int)((double)fps_cnt / win_s), src_hz, win_s, sum_wait / n,
                       sum_fence / n, sum_rga / n, sum_npu / n, sum_pack / n, sum_h2d / n,
                       sum_exec / n, sum_d2h / n, sum_dec / n,
                       (sum_wait + sum_rga + sum_npu) / n, pct(ages, 50), pct(ages, 99),
                       ages.empty() ? 0.0 : *std::max_element(ages.begin(), ages.end()),
                       (unsigned long long)d_waits,
                       (unsigned long long)(d_waits - d_to),
                       d_waits > d_to ? d_us / 1000.0 / (double)(d_waits - d_to) : 0.0,
                       (unsigned long long)d_to, (unsigned long long)d_to,
                       HDMI_FENCE_WAIT_MS,
                       (unsigned long long)(sc.lock_lost - src_prev.lock_lost),
                       (unsigned long long)(sc.stalled - src_prev.stalled),
                       (unsigned long long)(sc.rearm_ok - src_prev.rearm_ok),
                       (unsigned long long)(sc.rearm_fail - src_prev.rearm_fail));
                fflush(stdout);
                fps_cnt = 0; sum_rga = sum_npu = sum_pack = sum_h2d = sum_exec = sum_d2h
                    = sum_dec = sum_wait = sum_fence = 0;
                ages.clear(); fps_t0 = now;
                src_prev = sc;
            }
        }

        // 自身运动换算的账本来源与逐轴比例在 estimator_step 内部按输出模式取
        //   (一次快照, 见 core/estimator.cu); 标定期 found=false, 发布的目标自然过期。
        //   返回的实测 dt 是标定读数的量化底 (边沿 ±dt/2 与尾迹的 dt_first), 故采样
        //   必须在它之后。
        const float dt = estimator_step(est, now, found, best_dx, best_dy, l_est, max_v);

        if (cal_collecting) {
            bgr = pipe.bgr_window(src, &err);
            if (!bgr) {
                std::cerr << "AI: 标定取像失败: " << err << "\n";
                in.release(f.index);
                continue;
            }
            // 640 BGR 窗口 → 灰度 → INTER_AREA 降到半分辨率 320 相关域 (采样几何与两方案
            //   对照见 core/calib.h 头部): 先裁后压的决定随之落地 —— 窗口本身就是那块
            //   居中 1:1 矩形, 压缩只买采样率, 不改变标定场。
            const auto t_smp0 = std::chrono::steady_clock::now();
            cv::Mat win(CAP_SIZE, CAP_SIZE, CV_8UC3, (void*)bgr->map,
                        (size_t)bgr->stride_px * 3);
            cv::Mat gray, dm;
            cv::cvtColor(win, gray, cv::COLOR_BGR2GRAY);
            cv::resize(gray, gray, cv::Size(CALIB_SAMPLE_PX, CALIB_SAMPLE_PX), 0, 0,
                       cv::INTER_AREA);
            gray.convertTo(dm, CV_32F);
            ++collect_frames;
            if (!prev_dom.empty() && prev_dom.size() == dm.size()) {
                const CalibSampler::Frame fr = sampler.measure(prev_dom, dm);
                if (fr.ok[0] || fr.ok[1]) {
                    CalibSample smp;
                    smp.t = now; smp.dt_ms = dt;
                    smp.sx = fr.shift[0]; smp.sy = fr.shift[1];
                    smp.sx_all = fr.shift_all[0]; smp.sy_all = fr.shift_all[1];
                    for (int a = 0; a < 2; ++a) {
                        smp.ok[a] = fr.ok[a]; smp.resp[a] = fr.resp[a];
                        smp.spread[a] = fr.spread[a]; smp.n_static[a] = fr.n_static[a];
                    }
                    smp.slot = cal_note_sample(smp);   // 在线累计 + 打槽位
                    hist.push_back(smp);
                    while (hist.size() > hist_max) hist.pop_front();
                    ++collect_samples;
                }
            }
            prev_dom = dm;
            collect_ms += elapsed_ms(std::chrono::steady_clock::now(), t_smp0);
        }

        if (g_calib_request.exchange(false)) {
            // 拟合 (停顿 + 三读数; 账本不参与 —— 测量完全来自屏幕位移), 诊断与回写
            //   都在 io/calib_run.cpp 里 (hid 与手柄两种通道同一份口径)
            const CalResult cr = cal_fit(cal_mode, hist, g_cal_win.snapshot());
            const int done = cal_done_code(cr);
            cal_print_diag(cal_mode, cr, hist.size());
            // 标定期采样率实测 (要求不假设 120fps: 读数分辨率 = ±dt/2, 段跨 = T/dt 帧 —
            //   实测值连同每帧采样耗时一起进日志)
            if (collect_frames > 0) {
                const double wall = elapsed_ms(std::chrono::steady_clock::now(), cal_t0);
                const double ms = wall / (double)collect_frames;
                printf("[标定] 采样: 有效样本 %ld / 采集帧 %ld, 均值 %.2fms ≈ %.0ffps "
                       "(采集率 %d fps, 无样本帧 %.1f%%) | 采样耗时 %.2fms/帧 (几何 640→%d, "
                       "9 块×2 轴一维投影)\n",
                       collect_samples, collect_frames, ms, ms > 0 ? 1000.0 / ms : 0.0,
                       (int)std::lround(src_hz),
                       100.0 * (1.0 - (double)collect_samples /
                                          std::max(1.0, (double)collect_frames)),
                       collect_ms / std::max(1.0, (double)collect_frames), CALIB_SAMPLE_PX);
                fflush(stdout);
            }
            if (done == 1) {
                if (!persist_path.empty()) {
                    // 回写 VAR 名由输出模式在 main 里选好 (三套输出各一格延迟)
                    if (cal_writeback(cal_var, cr, persist_path))
                        std::cout << "[标定] 已回写 " << persist_path << " (" << cal_var << ")\n";
                    else std::cerr << "[标定] 回写失败\n";
                }
                l_est = cr.l_est;      // 只标延迟: 唯一进运行态的标定量
            } else
                std::cout << "[标定] 失败, 未回写 (无编造的值)\n";
            g_calib_done = done;
        }

        // ---- 三源截图 (det 源要看本帧有没有目标, 故判定在检测之后; BGR 窗口也在这里
        //      才裁 —— 只有真要落盘的那一帧付这次 RGA, 稳态每帧仍只有一次裁剪) ----
        std::string save_mode;
        if (collecting_enabled) {
            const bool cd_ok = elapsed_ms(now, last_save) >= cooldown_ms;
            if (g_left_down.load()) {
                if (g_cap_fire.load() && elapsed_ms(now, last_fire) >= fire_ms) {
                    save_mode = "fire"; last_fire = now;
                }
                if (g_cap_auto.load() && now >= next_auto) next_auto = now + roll_auto();
            } else {
                if (g_cap_det.load() && found && cd_ok) save_mode = "det";
                if (save_mode.empty() && g_cap_auto.load() && now >= next_auto && cd_ok)
                    save_mode = "auto";
            }
        }
        if (bgr == nullptr && (preview || !save_mode.empty()))
            bgr = pipe.bgr_window(src, &err);
        if (!save_mode.empty()) {
            if (bgr) {
                // 落盘的就是 640 BGR 窗口: 与已采的数据集同通道同几何 (存信道的读写两侧
                //   都不换格式, 数据集才可比)
                cv::Mat img(CAP_SIZE, CAP_SIZE, CV_8UC3, (void*)bgr->map,
                            (size_t)bgr->stride_px * 3);
                enqueue_save(make_filepath(out_dir + "/" + save_mode), img);
                last_save = now; next_auto = now + roll_auto();
                if (save_mode == "fire") ++n_fire;
                else if (save_mode == "det") ++n_det;
                else ++n_auto;
                std::cout << "[SAVE] " << save_mode << "  (fire=" << n_fire << " det=" << n_det
                          << " auto=" << n_auto << " drop=" << g_dropped.load() << ")\n";
            } else
                std::cerr << "AI: 截图取像失败: " << err << "\n";
        }

        if (preview) {
            if (bgr) {
                // 调试视图画在**副本**上: dma_heap 目的缓冲的 cache 纪律归 io/rga_pp
                //   (RGA 写 → 失效化 → CPU 读), CPU 往那块映射里写不在那条纪律里
                cv::Mat pv;
                cv::Mat(CAP_SIZE, CAP_SIZE, CV_8UC3, (void*)bgr->map,
                        (size_t)bgr->stride_px * 3).copyTo(pv);
                const float off_bgr = (float)(pipe.window_side() - CAP_SIZE) / 2.0f;
                for (const auto& d : filtered) {
                    cv::Rect box((int)(d.cx - d.w * .5f - off_bgr),
                                 (int)(d.cy - d.h * .5f - off_bgr), (int)d.w, (int)d.h);
                    box &= cv::Rect(0, 0, CAP_SIZE, CAP_SIZE);
                    cv::rectangle(pv, box, cv::Scalar(0, 255, 0), 2);
                    const float ty = d.cy + d.h * (0.5f - y_off_pct / 100.0f) - off_bgr;
                    cv::drawMarker(pv, cv::Point((int)(d.cx - off_bgr), (int)ty),
                                   cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 2);
                }
                if (collecting_enabled) {
                    char st[160];
                    snprintf(st, sizeof(st), "fire:%ld det:%ld auto:%ld drop:%ld%s  %.0ffps",
                             n_fire, n_det, n_auto, g_dropped.load(),
                             g_left_down.load() ? " [FIRE]" : "", src_hz);
                    cv::putText(pv, st, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                g_left_down.load() ? cv::Scalar(0, 0, 255)
                                                   : cv::Scalar(0, 255, 0), 2);
                }
                cv::imshow("Aimbot", pv);
                if (cv::waitKey(1) == 27) break;
            }
        }
        in.release(f.index);
    }

    if (preview) cv::destroyAllWindows();
    if (collecting_enabled)
        std::cout << "采集统计: fire=" << n_fire << " det=" << n_det << " auto=" << n_auto
                  << " dropped=" << g_dropped.load() << "\n";
    // 取帧层自己的账: 交付/取代/驱动丢/ERROR 是**交付语义**的账 (稳态三项全 0), 后四项是
    //   断流的账 —— 失锁与停流的判定次数、重建的成败、以及 fence 超时的次数与它烧掉的时间
    //   (超时白等的时长 = 次数 × 上限, 实测 poll 恰好走到上限才返回; 口径见 io/hdmi_in.h)。
    const HdmiFormat& f2 = in.format();
    const uint64_t n_to = in.fence_timeouts();
    std::cout << "收帧统计: 交付 " << in.delivered() << " / 取代 " << in.superseded()
              << " / 驱动丢 " << in.lost() << " / ERROR " << in.invalid()
              << " / 等 fence " << in.fence_waits() << " 次 (成功均值 "
              << (in.fence_waits() > n_to
                      ? in.fence_wait_us_sum() / (double)(in.fence_waits() - n_to) / 1000.0
                      : 0.0)
              << "ms, 最长 " << in.fence_wait_us_max() / 1000.0 << "ms) / fence 超时 " << n_to
              << " 次 (白等 " << n_to * (unsigned long long)HDMI_FENCE_WAIT_MS << "ms)"
              << " / 失锁 " << in.lock_lost() << " 次 停流 " << in.stalled()
              << " / 重建 " << in.rearm_ok() << " 成功 " << in.rearm_fail() << " 失败"
              << (in.rearm_ok() ? (" (上次恢复耗时 " + std::to_string((long)in.last_rearm_ms()) + "ms)")
                                : std::string())
              << " (" << f2.width << "x" << f2.height << " " << HdmiIn::fourcc_name(f2.fourcc)
              << ")\n";
    in.close();
    std::cout << "AI 线程已退出\n";
}
