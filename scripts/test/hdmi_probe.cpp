// ============================================================================
//  hdmi_probe.cpp — 采集层的板端验收探针: 解析设备 → 打印锁定时序与格式 →
//    连采 N 帧 (缺省 600) 并报实测帧率 / 交付延迟 / RGA 逐几何耗时 / 丢帧,
//    再落几张 PNG 供人眼确认裁剪区域与颜色通道。
//
//  编译 (scripts/compile.sh 里的目标, 与三个单测用同一份模块对象):
//    sudo ./build/hdmi_probe [帧数] [设备] [--fifo] [--slow <ms>] [--no-fence] [--dump <目录>]
//    * 设备缺省 = 按驱动名解析 (见 io/hdmi_in.h), 也可显式给 /dev/videoN
//    * --fifo: 走队列顺序交付 (对照路径), 用来量"最新帧语义买到了什么"
//    * --slow <ms>: 每帧故意睡这么久, 模拟消费者跟不上信号
//    * --no-fence: **只给测量用**, 不等 dma-fence 就交帧 —— 复现"交付时载荷还没写完"
//      (自检那行会读到几百字节变化; 生产路径禁止关, 理由见 io/hdmi_in.h)
//    * --dump <目录>: PNG 落盘位置 (缺省 /tmp)
//
//  返回值: 0 = 本次运行的验收项全过, 1 = 至少一项打了 ❌ (载荷稳定性自检、源几何对账、
//   三条逐字节对照、落图复核、取帧/重建/RGA 的失败)。账只收判决项 —— 帧率与耗时那几行是
//   测量, 不是判据; 而 poll 被信号打断 (Interrupted) 是退出路径, 不算失败。
//
//  失锁/断流: 探针按与生产路径同一条判据取帧 (io/hdmi_in.h 的 HdmiFail), 且**与生产路径
//   一样重建而不是退出** —— 判为流断了就 rearm() 并接着数帧 (只是丢帧这样的失败仍然报出
//   来)。这样一次探针长跑本身就是"失锁→重锁→画面回来"的验收: 给接收器写别的 EDID 即可
//   现场制造失锁 (复现命令见 io/hdmi_in.cpp 头部), 结果段里的失锁/停流/重建成败与 fence
//   超时是同一批账。
//
//  每段数字的定义与它为何这样定义 (不含糊其辞是这条链路的要求):
//    * 实测帧率 = 相邻两帧 buf.timestamp 之差 (驱动填的 CLOCK_MONOTONIC), 不是
//      "墙钟/帧数" —— 后者把探针自己的等待也算进去, 测不出信号的真实节拍。
//    * 交付延迟 = wait_frame 返回时刻 − 该帧的时间戳, 它由两段组成: 等 dma-fence
//      (低延迟模式下该帧的读出还在进行, 这一段是模式固有的) + userspace 侧开销
//      (poll 唤醒、排空、等 fence 之间的记账)。探针把两段分开报 —— 合起来才是"拿到手时
//      这帧有多旧"。
//    * 载荷稳定性自检 = 交付瞬间取该帧矩形首 4KB、6ms 后再取一次, 报变化字节。等过
//      fence 的路径必须是 0 (否则下游拿到的是半张画面); --no-fence 时必然非 0。
//    * 丢弃 (最新帧语义) = 被更新的帧取代、当场归还的那些帧。它随消费者变慢而增大,
//      这是设计行为不是故障 (旧帧没有价值), 队列深度只用来保证驱动不停。
//    * 驱动丢帧 = 交付帧的 sequence 缺口 (驱动在队列空时丢帧且不重发旧帧)。消费者跟得上
//      时它应为 0 (--slow 会把它顶起来, 那是对照)。
//    * RGA 耗时 = improcess() 的整段同步调用 (下发 + 等作业完成 + 目的缓冲失效化),
//      前 3 帧为预热 (首次调用含 RGA 上下文/页表建立), 不计入。
//    * [对照] 段 = 与 mmap 采集缓冲上写的纯 CPU 参照逐字节比对 (源是打包 RGB 时纯拷贝
//      与换通道都是无算术的置换, 故"字节数为 0"就是这条路径的正确性判据), 外加
//      裁剪 PNG 与原始帧 PNG 同区域的逐像素比对, 供人眼复核裁剪区域与颜色通道。
// ============================================================================

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unistd.h>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "io/hdmi_in.h"
#include "io/rga_pp.h"

namespace {

volatile sig_atomic_t g_stop = 0;
void on_sigint(int) { g_stop = 1; }

int64_t now_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

struct Stats {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    void reserve(size_t n) { v.reserve(n); }
    double pct(double q, double* mean = nullptr) const {
        if (v.empty()) { if (mean) *mean = 0.0; return 0.0; }
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        if (mean) {
            double t = 0;
            for (double x : s) t += x;
            *mean = t / (double)s.size();
        }
        size_t i = (size_t)(q * (double)s.size());
        if (i >= s.size()) i = s.size() - 1;
        return s[i];
    }
    void line(const char* what, const char* unit, double* mean_out = nullptr) const {
        double mean = 0;
        std::printf("  %-28s p50 %8.1f  p90 %8.1f  p99 %8.1f  max %9.1f %s (n=%zu)",
                    what, pct(0.50, &mean), pct(0.90), pct(0.99), pct(0.999), unit, v.size());
        if (mean_out) *mean_out = mean;
        std::printf("\n");
    }
};

// 源是打包 RGB (BGR3) 时的 CPU 参照实现: 按同一个矩形拷出来并做纯通道交换 ——
//   BGR3 (内存 B,G,R) → RGB888 (内存 R,G,B) 是一次置换, 没有算术, 所以逐字节相等
//   是这条路径的正确性判据 (有 CSC 的平面 YUV 源不在本对照的实现范围内)。
//   源已经是 RGB 的那一路 (二级裁剪的源 = 640 窗口缓冲) 用纯拷贝, 不换通道 ——
//   RGA 的 RGB→RGB 也是纯拷贝, 这是同一句话的另一半。
void cpu_crop(const uint8_t* src, int stride_bytes, int x0, int y0, int side, bool swap_rb,
              std::vector<uint8_t>* out) {
    out->resize((size_t)side * (size_t)side * 3);
    for (int y = 0; y < side; ++y) {
        const uint8_t* s = src + (size_t)(y0 + y) * (size_t)stride_bytes + (size_t)x0 * 3;
        uint8_t* d = out->data() + (size_t)y * (size_t)side * 3;
        if (!swap_rb) {
            std::memcpy(d, s, (size_t)side * 3);
            continue;
        }
        for (int x = 0; x < side; ++x) {
            d[x * 3 + 0] = s[x * 3 + 2];  // R
            d[x * 3 + 1] = s[x * 3 + 1];  // G
            d[x * 3 + 2] = s[x * 3 + 0];  // B
        }
    }
}

// 逐字节比对, 报出差异字节数; *diff_at 记首个差异的下标
long byte_diff(const uint8_t* a, const uint8_t* b, size_t n, size_t* diff_at) {
    long diff = 0;
    size_t first = 0;
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) { if (!diff) first = i; ++diff; }
    *diff_at = first;
    return diff;
}

bool png_write(const std::string& path, const cv::Mat& bgr) {
    if (bgr.empty()) return false;
    if (!cv::imwrite(path, bgr)) {
        std::fprintf(stderr, "⚠ 写 %s 失败\n", path.c_str());
        return false;
    }
    std::printf("  PNG: %s (%dx%d)\n", path.c_str(), bgr.cols, bgr.rows);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int frames = 600;
    std::string dev;
    std::string dumpdir = "/tmp";
    bool fifo = false;
    int slow_ms = 0;
    bool no_fence = false;
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--fifo") {
            fifo = true;
        } else if (a == "--no-fence") {
            no_fence = true;
        } else if (a == "--slow" && i + 1 < argc) {
            slow_ms = std::atoi(argv[++i]);
        } else if (a == "--dump" && i + 1 < argc) {
            dumpdir = argv[++i];
        } else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "未知参数 %s\n", a.c_str());
            return 2;
        } else if (positional == 0) {
            frames = std::atoi(a.c_str());
            ++positional;
        } else {
            dev = a;
        }
    }
    if (frames < 1) frames = 600;
    std::signal(SIGINT, on_sigint);
    // 本次运行的验收账: 每一个 ❌ 记一笔, 末尾由它决定返回值 —— 探针是验收工具, 打印出来的
    //   结论必须能被脚本判读, 只靠人眼看输出就等于没有判据。测量类输出 (帧率/延迟/耗时)
    //   不是判据, 故不记账。
    int nfail = 0;

    std::printf("== HDMI IN 探针: %d 帧, 交付=%s, 慢消费者=%dms, 设备=%s%s ==\n", frames,
                fifo ? "队列顺序(对照)" : "最新帧", slow_ms, dev.empty() ? "(按驱动名解析)" : dev.c_str(),
                no_fence ? ", 不等 fence(测量用: 载荷可能没写完)" : "");

    std::string err;
    HdmiIn cap;
    if (!cap.open(dev, HDMI_BUF_DEFAULT, fifo ? HdmiDelivery::QueueOrder : HdmiDelivery::Newest, &err)) {
        std::fprintf(stderr, "❌ 打开失败: %s\n", err.c_str());
        return 1;
    }
    cap.set_fence_wait(!no_fence);
    const HdmiFormat& f = cap.format();
    // 源几何与缓冲尺寸对账 (RGA 侧要求 = wstride*hstride*3, wstride 是像素)
    const RgaSrc src_geom{cap.buffers().empty() ? -1 : cap.buffers()[0].fd, f.width, f.height,
                          f.stride_px, f.fourcc};
    const int src_fmt = rga_format_of(src_geom.fourcc);
    const size_t src_need = (size_t)src_geom.stride_px * (size_t)src_geom.height * 3;
    std::printf("[RGA] 源 %s %dx%d wstride %dpx -> 缓冲要求 %zuB, 驱动 sizeimage %zuB %s\n",
                rga_format_name(src_fmt), src_geom.width, src_geom.height, src_geom.stride_px, src_need,
                f.plane_bytes, src_need == f.plane_bytes ? "(一致)" : "(❌ 不一致)");
    if (src_need != f.plane_bytes) ++nfail;
    const RgaRect crop = RgaPp::center_crop(f.width, f.height, CAP_SIZE);
    if (crop.side == 0) {
        std::fprintf(stderr, "❌ 源 %dx%d 装不下边长 %d 的 1:1 中心裁剪\n", f.width, f.height, (int)CAP_SIZE);
        return 1;
    }
    std::printf("[RGA] 规范窗口 = 源中心 (%d,%d) 起 %dx%d (1:1, 不缩放)\n", crop.x, crop.y, crop.side,
                crop.side);

    RgaPp pp;
    Stats st_interval, st_dispatch, st_fence, st_rga640rgb, st_rga640bgr, st_rga320;
    st_interval.reserve(frames);
    st_dispatch.reserve(frames);
    st_fence.reserve(frames);
    st_rga640rgb.reserve(frames);
    st_rga640bgr.reserve(frames);
    st_rga320.reserve(frames);

    int64_t first_ts = 0, last_ts = 0, prev_ts = 0;
    int n = 0;
    long diff_rgb = -1, diff_bgr = -1, diff_320 = -1;
    size_t diff_at_rgb = 0, diff_at_bgr = 0, diff_at_320 = 0;

    while (!g_stop && n < frames) {
        HdmiFrame fr;
        HdmiFail fail = HdmiFail::Ok;
        if (!cap.wait_frame(&fr, &err, &fail)) {
            std::fprintf(stderr, "❌ 取帧失败: %s\n", err.c_str());
            // Interrupted 是 poll 被信号打断的退出路径, 与流无关 (见 io/hdmi_in.h 的 HdmiFail),
            //   不计进验收账 —— 否则 Ctrl+C 结束的一次探针会被判成失败。
            if (fail != HdmiFail::Interrupted) ++nfail;
            // 与生产路径同一条恢复动作: 流断了就重建, 不是丢掉整条链 (判定见 io/hdmi_in.h)
            if (hdmi_fail_needs_rearm(fail) && !g_stop) {
                std::fprintf(stderr, "   → 重建 (失锁/断流) …\n");
                if (!cap.rearm(&err)) {
                    std::fprintf(stderr, "   ❌ 重建失败: %s\n", err.c_str());
                    ++nfail;
                    break;
                }
                continue;
            }
            break;
        }
        ++n;
        const int64_t t_now = now_ns();
        st_dispatch.add((double)(t_now - fr.ts_ns) / 1000.0);
        st_fence.add(cap.last_fence_wait_us());
        if (prev_ts != 0) st_interval.add((double)(fr.ts_ns - prev_ts) / 1e6);
        prev_ts = fr.ts_ns;
        if (first_ts == 0) first_ts = fr.ts_ns;
        last_ts = fr.ts_ns;

        // RGA 三条几何: 640 窗口 (RGB = 模型路径 / BGR = 截图路径) 与 320 (自 640 窗口
        //   再中心裁一次 = 小输入模型看到的窗口)。源用**本帧**的 dmabuf fd (每块缓冲
        //   各有自己的 fd), 不是固定的某一个 buffer。
        const RgaSrc fsrc{fr.fd, fr.width, fr.height, fr.stride_px, fr.fourcc};
        int64_t a = now_ns();
        const RgaDst* win_rgb = pp.crop_center(fsrc, CAP_SIZE, V4L2_PIX_FMT_RGB24, &err);
        int64_t b = now_ns();
        const RgaDst* win_bgr = pp.crop_center(fsrc, CAP_SIZE, V4L2_PIX_FMT_BGR24, &err);
        int64_t c = now_ns();
        if (!win_rgb || !win_bgr) {
            std::fprintf(stderr, "❌ RGA 失败: %s\n", err.c_str());
            ++nfail;
            break;
        }
        const RgaSrc mid{win_rgb->fd, win_rgb->side, win_rgb->side, win_rgb->stride_px,
                         V4L2_PIX_FMT_RGB24};
        const RgaDst* small = pp.crop_center(mid, win_rgb->side / 2, V4L2_PIX_FMT_RGB24, &err);
        int64_t d = now_ns();
        if (!small) {
            std::fprintf(stderr, "❌ RGA (二级裁剪) 失败: %s\n", err.c_str());
            ++nfail;
            break;
        }
        if (n >= 3) {  // 前 3 帧预热 (首次调用含 RGA 上下文/页表建立)
            st_rga640rgb.add((double)(b - a) / 1000.0);
            st_rga640bgr.add((double)(c - b) / 1000.0);
            st_rga320.add((double)(d - c) / 1000.0);
        }

        cap.release(fr.index);
        if (slow_ms > 0) usleep((useconds_t)slow_ms * 1000);
    }
    // 阶段 1 结束时的计数快照: 阶段 2 的对照/PNG 是百毫秒级的重活, 本身就会把队列压垮,
    //   那部分丢帧不是"消费者跟不上"的量, 不计进这里。
    const uint64_t lost_end = cap.lost(), sup_end = cap.superseded(), inv_end = cap.invalid();

    // ---- 阶段 2: 逐字节对照与落图 (另取一帧, 全程**不归还**它) ----
    // 不归还这一点是必须的: 归还之后驱动会立刻把新帧写进同一块缓冲, 而"CPU 参照 vs RGA
    //   结果"的比对要跨几次裁剪与几毫秒的 memcmp, 中间被写一次就变成了"两帧之差" ——
    //   实测就是这么来的 (1.2% 的差异, 与二级裁剪无关)。
    HdmiFrame fr;
    if (cap.wait_frame(&fr, &err)) {
        const int pitch = fr.stride_px * 3;
        // 载荷稳定性自检: 交付瞬间取矩形首 4KB, 6ms 后再取 —— 变化字节 = 交付时还没写完的
        //   那部分。低延迟模式的完成通知 (与 buf.timestamp 同一时刻) 早于载荷写完, 等 fence
        //   就是为这件事; 关掉 fence 等待 (--no-fence) 时这里必定非 0 —— 那正是这条结论的
        //   复现路径, 也是"半张画面进了下游"的量化形式。
        {
            const uint8_t* p = (const uint8_t*)fr.map + (size_t)crop.y * pitch + (size_t)crop.x * 3;
            std::vector<uint8_t> snap(p, p + 4096);
            usleep(6000);
            long d = 0;
            for (size_t i = 0; i < snap.size(); ++i)
                if (snap[i] != p[i]) ++d;
            std::printf("[自检] 交付后 6ms 内该帧首 4KB 变化 %ld 字节%s (等 fence 之后应为 0)\n", d,
                        d ? " ❌" : " ✓");
            // --no-fence 正是这条结论的复现路径 (交付时载荷还在写, 非 0 是预期), 故只在
            //   等 fence 的那条路径上把非 0 记成失败。
            if (d && !no_fence) ++nfail;
        }
        const RgaSrc fsrc{fr.fd, fr.width, fr.height, fr.stride_px, fr.fourcc};
        const RgaDst* win_rgb = pp.crop_center(fsrc, CAP_SIZE, V4L2_PIX_FMT_RGB24, &err);
        const RgaDst* win_bgr = pp.crop_center(fsrc, CAP_SIZE, V4L2_PIX_FMT_BGR24, &err);
        const RgaDst* small = nullptr;
        if (win_rgb && win_bgr) {
            const RgaSrc mid{win_rgb->fd, win_rgb->side, win_rgb->side, win_rgb->stride_px,
                             V4L2_PIX_FMT_RGB24};
            small = pp.crop_center(mid, win_rgb->side / 2, V4L2_PIX_FMT_RGB24, &err);
        }
        if (!win_rgb || !win_bgr || !small) {
            std::fprintf(stderr, "❌ 阶段 2 RGA 失败: %s\n", err.c_str());
            ++nfail;
        } else {
            const uint8_t* base = (const uint8_t*)fr.map;
            std::printf("[对照] 源 '%s' %dx%d 行距 %dB, 取中心矩形 (%d,%d) %dx%d\n",
                        HdmiIn::fourcc_name(fr.fourcc).c_str(), fr.width, fr.height, pitch, crop.x,
                        crop.y, crop.side, crop.side);
            if (fr.fourcc == V4L2_PIX_FMT_BGR24 || fr.fourcc == V4L2_PIX_FMT_RGB24) {
                const bool src_is_bgr = (fr.fourcc == V4L2_PIX_FMT_BGR24);
                std::vector<uint8_t> ref;
                // 640 RGB888: 源是 BGR3 时参照要换通道
                cpu_crop(base, pitch, crop.x, crop.y, crop.side, src_is_bgr, &ref);
                diff_rgb = byte_diff(ref.data(), (const uint8_t*)win_rgb->map, (size_t)win_rgb->bytes,
                                     &diff_at_rgb);
                // 640 BGR888: 源是 BGR3 时是纯拷贝
                cpu_crop(base, pitch, crop.x, crop.y, crop.side, !src_is_bgr, &ref);
                diff_bgr = byte_diff(ref.data(), (const uint8_t*)win_bgr->map, (size_t)win_bgr->bytes,
                                     &diff_at_bgr);
                std::printf("[对照] 640 RGB888 (源 BGR3 → 换通道) 差异字节 %ld / %zu%s\n", diff_rgb,
                            win_rgb->bytes, diff_rgb ? " ❌" : " ✓");
                if (diff_rgb) ++nfail;
                std::printf("[对照] 640 BGR888 (源 BGR3 → 纯拷贝) 差异字节 %ld / %zu%s\n", diff_bgr,
                            win_bgr->bytes, diff_bgr ? " ❌" : " ✓");
                if (diff_bgr) ++nfail;
                // 二级裁剪: 源是 640 RGB 窗口 (已经是 RGB), 故参照也是纯拷贝
                const RgaRect c2 = RgaPp::center_crop(win_rgb->side, win_rgb->side, small->side);
                std::vector<uint8_t> ref2;
                cpu_crop((const uint8_t*)win_rgb->map, win_rgb->side * 3, c2.x, c2.y, c2.side, false,
                         &ref2);
                diff_320 = byte_diff(ref2.data(), (const uint8_t*)small->map, (size_t)small->bytes,
                                     &diff_at_320);
                std::printf("[对照] 320 RGB888 (自 640 窗口再中心裁 (%d,%d) %dx%d, 纯拷贝) 差异字节 "
                            "%ld / %zu%s\n",
                            c2.x, c2.y, c2.side, c2.side, diff_320, small->bytes,
                            diff_320 ? " ❌" : " ✓");
                if (diff_320) ++nfail;
                // 通道落点: 挑矩形内 |R−B| 最大的像素做样本 —— 源 BGR3 的 R (源字节 2) 应落在
                //   目的 RGB888 的第 0 字节。挑彩色像素是因为灰像素上"换没换通道"看不出来。
                int bx = crop.x, by = crop.y;
                long best = -1;
                for (int y = 0; y < crop.side; ++y)
                    for (int x = 0; x < crop.side; ++x) {
                        const uint8_t* p = base + (size_t)(crop.y + y) * pitch + (size_t)(crop.x + x) * 3;
                        const long d = std::abs((int)p[0] - (int)p[2]);
                        if (d > best) { best = d; bx = crop.x + x; by = crop.y + y; }
                    }
                {
                    const uint8_t* s = base + (size_t)by * pitch + (size_t)bx * 3;
                    const uint8_t* o = (const uint8_t*)win_rgb->map +
                                       (size_t)(by - crop.y) * win_rgb->stride_px * 3 +
                                       (size_t)(bx - crop.x) * 3;
                    std::printf("[对照] 通道落点: 源 BGR3 字节 [B,G,R] -> 目的 RGB888 字节 [R,G,B]; "
                                "矩形内 |R−B| 最大的样本 (源坐标 %d,%d) 源(%02x,%02x,%02x) -> 目的(%02x,%02x,%02x)\n",
                                bx, by, s[0], s[1], s[2], o[0], o[1], o[2]);
                }
            } else {
                std::printf("[对照] 源是平面 YUV, 纯拷贝/通道交换的参照实现不适用 (需要一次 CSC), 跳过\n");
            }

            // PNG: 原帧 (标出裁剪框) / 两个 640 窗口 / 320 窗口 —— 给人眼确认裁剪区域与颜色
            cv::Mat raw_bgr;
            if (fr.fourcc == V4L2_PIX_FMT_BGR24) {
                raw_bgr = cv::Mat(fr.height, fr.width, CV_8UC3, (void*)fr.map, pitch).clone();
            } else if (fr.fourcc == V4L2_PIX_FMT_RGB24) {
                cv::Mat raw_rgb(fr.height, fr.width, CV_8UC3, (void*)fr.map, pitch);
                cv::cvtColor(raw_rgb, raw_bgr, cv::COLOR_RGB2BGR);
            }
            if (!raw_bgr.empty()) {
                cv::Mat marked = raw_bgr.clone();
                cv::rectangle(marked, cv::Rect(crop.x, crop.y, crop.side, crop.side),
                              cv::Scalar(0, 255, 0), 3);
                png_write(dumpdir + "/hdmi_raw.png", raw_bgr);
                png_write(dumpdir + "/hdmi_raw_crop_marked.png", marked);
            } else {
                std::printf("  PNG: 源是平面 YUV, 原帧 PNG 未落 (见 io/hdmi_in.h 的格式说明)\n");
            }
            cv::Mat rgb_bgr;
            cv::cvtColor(cv::Mat(win_rgb->side, win_rgb->side, CV_8UC3, win_rgb->map), rgb_bgr,
                         cv::COLOR_RGB2BGR);
            png_write(dumpdir + "/hdmi_crop640_rgb.png", rgb_bgr);
            png_write(dumpdir + "/hdmi_crop640_bgr.png",
                      cv::Mat(win_bgr->side, win_bgr->side, CV_8UC3, win_bgr->map));
            cv::Mat small_bgr;
            cv::cvtColor(cv::Mat(small->side, small->side, CV_8UC3, small->map), small_bgr,
                         cv::COLOR_RGB2BGR);
            png_write(dumpdir + "/hdmi_crop320_rgb.png", small_bgr);

            // 落图复核: 把写出去的两张 PNG 读回来, 比对裁剪区逐像素 —— 这是"人眼能确认
            //   裁剪区域与颜色通道"的程序化形式 (读回的是 PNG 解码结果, 与 cv::imwrite 的
            //   BGR 约定同一条路)
            {
                const cv::Mat a = cv::imread(dumpdir + "/hdmi_crop640_rgb.png");
                const cv::Mat b = cv::imread(dumpdir + "/hdmi_raw.png");
                if (!a.empty() && !b.empty() && b.cols >= crop.x + crop.side &&
                    b.rows >= crop.y + crop.side) {
                    long d = 0;
                    for (int y = 0; y < a.rows; ++y)
                        for (int x = 0; x < a.cols; ++x)
                            if (a.at<cv::Vec3b>(y, x) != b.at<cv::Vec3b>(crop.y + y, crop.x + x)) ++d;
                    std::printf("[落图] 裁剪 PNG vs 原始帧 PNG 同区域逐像素差异 %ld / %d%s\n", d,
                                a.rows * a.cols, d ? " ❌" : " ✓");
                    if (d) ++nfail;
                }
            }
        }
    }

    const double span_s = (last_ts > first_ts) ? (double)(last_ts - first_ts) / 1e9 : 0.0;
    std::printf("== 结果: 交付 %d 帧 (请求 %d), 时间跨度 %.3fs ==\n", n, frames, span_s);
    std::printf("  实测帧率 %8.3f fps (由帧时间戳导出; 相邻帧间隔 p50 %.3fms p99 %.3fms)\n",
                span_s > 0 ? (double)(n - 1) / span_s : 0.0, st_interval.pct(0.50), st_interval.pct(0.99));
    std::printf("  交付延迟 (取帧返回 − 帧时间戳; 含低延迟模式固有的等 fence 那段):\n");
    st_dispatch.line("dispatch 合计", "us");
    st_fence.line("  其中等 dma-fence", "us");
    if (!st_rga640rgb.v.empty()) {
        std::printf("  RGA 每帧耗时 (improcess 同步 + 目的缓冲失效化, 前 3 帧预热不计):\n");
        st_rga640rgb.line("640 窗口 RGB888 (模型)", "us");
        st_rga640bgr.line("640 窗口 BGR888 (截图)", "us");
        st_rga320.line("320 窗口 RGB888 (二级裁剪)", "us");
    }
    std::printf("  丢弃 (最新帧语义) %llu | 驱动丢帧 (sequence 缺口) %llu | 驱动 ERROR 归还 %llu\n",
                (unsigned long long)sup_end, (unsigned long long)lost_end,
                (unsigned long long)inv_end);
    // 断流的账 (口径见 io/hdmi_in.h): fence 超时那几次烧掉的就是上限, 与成功的等待分开报
    const uint64_t f_waits = cap.fence_waits(), f_to = cap.fence_timeouts();
    std::printf("  等 fence %llu 次 (成功 %llu 次, 均值 %.2fms, 最长 %.2fms) | fence 超时 %llu 次 "
                "(白等 %.0fms = %llu×%dms)\n",
                (unsigned long long)f_waits, (unsigned long long)(f_waits - f_to),
                f_waits > f_to ? cap.fence_wait_us_sum() / (double)(f_waits - f_to) / 1000.0 : 0.0,
                cap.fence_wait_us_max() / 1000.0, (unsigned long long)f_to,
                (double)f_to * (double)HDMI_FENCE_WAIT_MS, (unsigned long long)f_to,
                HDMI_FENCE_WAIT_MS);
    std::printf("  失锁 %llu 次 | 停流 %llu 次 | 重建 %llu 成功 %llu 失败%s\n",
                (unsigned long long)cap.lock_lost(), (unsigned long long)cap.stalled(),
                (unsigned long long)cap.rearm_ok(), (unsigned long long)cap.rearm_fail(),
                cap.rearm_ok() ? (" (末次恢复耗时 " + std::to_string((long)cap.last_rearm_ms()) +
                                  "ms, 当前 " + cap.timing_text() + ")").c_str()
                               : "");
    if (diff_rgb >= 0) {
        std::printf("  逐字节对照: 640 RGB %ld 差异 | 640 BGR %ld 差异 | 320 RGB %ld 差异 (期望全 0)",
                    diff_rgb, diff_bgr, diff_320);
        if (diff_rgb) std::printf(" 首个差异 @%zu", diff_at_rgb);
        if (diff_bgr) std::printf(" 首个差异 @%zu", diff_at_bgr);
        if (diff_320) std::printf(" 首个差异 @%zu", diff_at_320);
        std::printf("\n");
    }
    if (g_stop) std::printf("  (收到 SIGINT, 已停流并释放)\n");
    return nfail ? 1 : 0;   // 返回值 = 本次运行的验收结论, 见文件头的"返回值"一行
}
