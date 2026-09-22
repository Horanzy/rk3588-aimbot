// ============================================================================
//  pipeline_probe — build/pipeline_probe 板端验收探针 (不是单测; 需要 root、AXCL 卡
//    与活动的 HDMI 信号):
//    用**生产路径本身** (io/capture.h 的 CapturePipeline 与 io/hdmi_in 的取帧层)
//    跑两个问题:
//      (1) 活信号: 连采 N 帧走完整条链 (取帧 → RGA 窗口 → [二级裁剪] → NPU → 解码
//          → NMS), 报实测检测帧率与逐段耗时 (RGA / pack / H2D / execute / D2H / 解码)
//          与帧龄百分位 —— 桌面画面上的候选数应当为 0, 这一段的用途是"链走得通、
//          每段多贵";
//      (2) 真图检测: 取一帧**真实的采集缓冲** (驱动给的 dmabuf, 几何/行距都来自驱动),
//          把一张真图按 1:1 贴进它的中心 640 窗口 (该缓冲此刻在 userspace 手里, 驱动不
//          再持有), 再对这一帧跑同一条链, 打印窗口域的框与置信度并落盘窗口 PNG 与
//          标注 PNG。于是除"这一帧的像素内容"之外没有任何一处是合成的: 取帧、RGA、
//          NPU、解码、平移、NMS 都是生产代码。
//    落盘 (--out 前缀, 默认 pipeline_probe): <前缀>_window.png (窗口原样),
//      <前缀>_boxes.png (画上框与类别/置信度)。
//
//  用法: sudo LD_LIBRARY_PATH=/usr/lib/axcl ./build/pipeline_probe <model.axmodel> \
//      [--image <jpg>] [--classes <n>] [--cam <节点>] [--frames <n>] [--conf <t>]
//      [--out <前缀>]
//    未给 --image 时只跑 (1) 段。
//
//  进程收尾用 _Exit: axcl 的库在静态析构里 abort (实测 RC=134), stdout 已在开头设为
//    无缓冲, 故管道里也不丢行。
// ============================================================================

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "io/capture.h"
#include "io/hdmi_in.h"

static double now_ms() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

struct Stat { double avg = 0, p50 = 0, p99 = 0, mx = 0; };
static Stat summarize(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    s.avg = sum / (double)v.size();
    s.p50 = v[v.size() * 50 / 100]; s.p99 = v[v.size() * 99 / 100]; s.mx = v.back();
    return s;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string model, image, cam, out = "pipeline_probe";
    int classes = 0, frames = 300;
    float conf = 0.5f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* n) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "✗ %s 需要取值\n", n); exit(2); }
            return argv[++i];
        };
        if (a == "--image") image = val("--image");
        else if (a == "--classes") classes = std::stoi(val("--classes"));
        else if (a == "--cam") cam = val("--cam");
        else if (a == "--frames") frames = std::stoi(val("--frames"));
        else if (a == "--conf") conf = std::stof(val("--conf"));
        else if (a == "--out") out = val("--out");
        else if (!a.empty() && a[0] != '-') model = a;
        else { fprintf(stderr, "✗ 未知参数 %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) {
        fprintf(stderr, "用法: pipeline_probe <model.axmodel> [--image <jpg>] [--classes <n>] "
                        "[--cam <节点>] [--frames <n>] [--conf <t>] [--out <前缀>]\n");
        return 2;
    }

    // ---- 接收器 (生产取帧层: 裸 V4L2 + 等 dma-fence + 最新帧语义) ----
    HdmiIn in;
    std::string err;
    std::string dev = HdmiIn::resolve_device(cam, &err);
    if (dev.empty() || !in.open(dev, HDMI_BUF_DEFAULT, HdmiDelivery::Newest, &err)) {
        fprintf(stderr, "✗ 打开接收器失败: %s\n", err.c_str());
        return 1;
    }
    const HdmiFormat& fmt = in.format();
    if (in.frame_hz() > 0) fprintf(stdout, "[探针] 采集源 %s, %.3fHz\n",
                                   in.timing_text().c_str(), in.frame_hz());

    // ---- 生产逐帧链 ----
    CapturePipeline pipe;
    if (!pipe.open(model, classes, std::min(fmt.width, fmt.height), &err)) {
        fprintf(stderr, "✗ 建立会话失败: %s\n", err.c_str());
        return 1;
    }
    fprintf(stdout, "[探针] 规范窗口 %d, 模型输入 %d, 窗口域平移 %+d (模型域 → 窗口域)\n",
            pipe.window_side(), pipe.model_side(), (pipe.window_side() - pipe.model_side()) / 2);

    auto run_one = [&](const RgaSrc& src, std::vector<Detection>* dets, PipelineTick* tick) {
        PipelineTick t;
        if (!pipe.infer(src, conf, -1, &t, &err)) {
            fprintf(stderr, "✗ 推理失败: %s\n", err.c_str());
            return false;
        }
        if (tick) *tick = t;
        if (dets) *dets = t.dets;
        return true;
    };

    // ---- (1) 活信号: 检测帧率 + 逐段耗时 + 帧龄 ----
    std::vector<double> rga, npu, age, pack, h2d, exec, d2h, dec;
    int n = 0, nfail = 0, ndet_total = 0;
    const double t0 = now_ms();
    for (int i = 0; i < frames && global_running; ++i) {
        HdmiFrame f;
        if (!in.wait_frame(&f, &err)) { fprintf(stderr, "✗ 取帧失败: %s\n", err.c_str());
                                        if (++nfail > 10) break; continue; }
        const double a0 = now_ms();
        const RgaSrc src{f.fd, f.width, f.height, f.stride_px, f.fourcc};
        PipelineTick t;
        if (run_one(src, nullptr, &t)) {
            ++n;
            rga.push_back(t.rga_us); npu.push_back(t.npu.total_us());
            pack.push_back(t.npu.pack_us); h2d.push_back(t.npu.h2d_us);
            exec.push_back(t.npu.exec_us); d2h.push_back(t.npu.d2h_us);
            dec.push_back(t.npu.decode_us);
            // 帧龄 = 帧时间戳 → 检测就绪 (时间戳与 CLOCK_MONOTONIC 同源, 取帧层已校验)
            age.push_back(now_ms() - (double)f.ts_ns / 1e6);
            ndet_total += (int)t.dets.size();
        }
        in.release(f.index);
    }
    const double wall = now_ms() - t0;
    const Stat sr = summarize(rga), sn = summarize(npu), sa = summarize(age),
               sp = summarize(pack), sh = summarize(h2d), se = summarize(exec),
               sd = summarize(d2h), sk = summarize(dec);
    fprintf(stdout,
            "\n[1] 活信号 (%d 帧走完整条链 / 取帧失败 %d, 墙钟 %.0fms)\n"
            "    检测帧率 %.1f fps\n"
            "    RGA        avg %.2f p50 %.2f p99 %.2f max %.2f µs\n"
            "    NPU 合计   avg %.2f p50 %.2f p99 %.2f max %.2f µs\n"
            "      pack %.2f | H2D %.2f | exec %.2f | D2H %.2f | 解码 %.2f µs (avg)\n"
            "    帧龄       avg %.2f p50 %.2f p99 %.2f max %.2f ms (帧时间戳 → 检测就绪)\n"
            "    检测候选合计 %d (窗口 %d 上的桌面画面)\n",
            n, nfail, wall, wall > 0 ? 1000.0 * (double)n / wall : 0.0,
            sr.avg, sr.p50, sr.p99, sr.mx, sn.avg, sn.p50, sn.p99, sn.mx,
            sp.avg, sh.avg, se.avg, sd.avg, sk.avg, sa.avg, sa.p50, sa.p99, sa.mx,
            ndet_total, pipe.window_side());

    // ---- (2) 真图: 贴进一帧真实采集缓冲的中心窗口, 再走同一条链 ----
    int rc = 0;
    if (!image.empty()) {
        cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
        if (img.empty()) { fprintf(stderr, "✗ 读不到图: %s\n", image.c_str()); rc = 1; }
        else {
            HdmiFrame f;
            if (!in.wait_frame(&f, &err)) { fprintf(stderr, "✗ 取帧失败: %s\n", err.c_str());
                                            rc = 1; }
            else {
                const int side = pipe.window_side();
                cv::Mat big = cv::Mat::zeros(f.height, f.width, CV_8UC3);   // 背景: 全黑
                cv::Mat small;
                cv::resize(img, small, cv::Size(side, side), 0, 0, cv::INTER_AREA);
                // 1:1 贴进中心 (行距按驱动的像素行距; 采集帧的内存布局是驱动给的)
                for (int y = 0; y < side; ++y)
                    memcpy((uint8_t*)big.data + (size_t)(y + (f.height - side) / 2) * f.width * 3
                               + (size_t)(f.width - side) / 2 * 3,
                           small.data + (size_t)y * small.step, (size_t)side * 3);
                cv::Mat dst(f.height, f.width, CV_8UC3, (void*)f.map,
                            (size_t)f.stride_px * 3);
                big.copyTo(dst);      // dmabuf 是 vb2-dc 一致内存: CPU 写立刻对设备可见
                const RgaSrc src{f.fd, f.width, f.height, f.stride_px, f.fourcc};
                std::vector<Detection> dets;
                if (run_one(src, &dets, nullptr)) {
                    fprintf(stdout, "\n[2] 真图检测 (图 %d×%d 缩到 %d×%d, 贴进真实采集缓冲的"
                                    "中心窗口)\n", img.cols, img.rows, side, side);
                    fprintf(stdout, "    框 %zu 个 (窗口域坐标; 窗口 = 图中的同一块画面):\n",
                            dets.size());
                    for (const auto& d : dets)
                        fprintf(stdout, "      cls %2d conf %.3f  中心 (%.1f, %.1f) 尺寸 "
                                        "%.1f×%.1f  → 左上 (%.0f, %.0f)\n",
                                d.class_id, d.conf, d.cx, d.cy, d.w, d.h,
                                d.cx - d.w / 2, d.cy - d.h / 2);
                    const RgaDst* bgr = pipe.bgr_window(src, &err);
                    if (!bgr) fprintf(stderr, "✗ 取窗口失败: %s\n", err.c_str());
                    else {
                        cv::Mat win(side, side, CV_8UC3, (void*)bgr->map,
                                    (size_t)bgr->stride_px * 3);
                        cv::Mat pv; win.copyTo(pv);    // 画在副本上 (dma_heap 缓冲不写)
                        cv::imwrite(out + "_window.png", win);
                        for (const auto& d : dets) {
                            cv::Rect box((int)(d.cx - d.w / 2), (int)(d.cy - d.h / 2),
                                         (int)d.w, (int)d.h);
                            box &= cv::Rect(0, 0, side, side);
                            cv::rectangle(pv, box, cv::Scalar(0, 255, 0), 2);
                            char lab[64];
                            snprintf(lab, sizeof(lab), "%d %.2f", d.class_id, d.conf);
                            cv::putText(pv, lab, cv::Point(box.x, std::max(12, box.y - 4)),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                        }
                        cv::imwrite(out + "_boxes.png", pv);
                        fprintf(stdout, "    落盘: %s_window.png / %s_boxes.png\n",
                                out.c_str(), out.c_str());
                    }
                } else rc = 1;
                in.release(f.index);
            }
        }
    }
    if (!image.empty() && rc == 0) fprintf(stdout, "\n[探针] 通过 (活信号链 + 真图端到端)\n");
    in.close();
    fflush(stdout);
    std::_Exit(rc);
}
