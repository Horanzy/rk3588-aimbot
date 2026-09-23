// ============================================================================
//  model_probe — build/model_probe 板端验收探针 (不是单测; 需要 root 与 AXCL 卡):
//    给定一个 .axmodel, 把运行时的 **IO 契约** 逐张量打出来 (名字/形状/元素类型/
//    字节数/在解码契约里的角色: 输入是否 U8 NHWC RGB, 哪些输出被保留、哪些被跳过及
//    原因), 可选喂一张真图跑完整条链 (letterbox → RGB → H2D → 推理 → D2H → 解码 →
//    NMS) 并把**实际喂进去的输入张量**与**每个保留输出的原始字节**落盘 (供别处的
//    参照实现逐字节比对), 再报每个保留输出的**数值实测** (码长与评分通道量程 —— 见下),
//    最后按 N 次迭代报各段耗时的百分位 (保留输出多于一个时 D2H 另给逐张量的表) 与等效
//    带宽。
//
//  --xfer-sweep: 只做传输尺寸扫描 (H2D/D2H 各尺寸的百分位 + 最小二乘拟合与残差),
//    用来定"每次传输的固定开销 + 每字节速率"这条延迟预算的地基; 给了模型则把该模型
//    自己的输入/输出字节数也作为扫描点 (真实尺寸上的点比外推可信)。
//
//  图形口径与 AXERA 参考实现 (axcl-samples/examples/base/common.hpp 的
//    get_input_data_letterbox + detection.hpp 的 get_out_bbox) 逐像素同一几何:
//    等比缩放 (int(scale·边) 截断) + 居中补零, 框按 (p − 补零偏移) × 原边长/缩放边长
//    换回原图并夹到 [0, 边长−1]。**通道序是这对比对里唯一的差别**: 参考实现的
//    ax_yolo11 调 get_input_data_letterbox 时不传 bgr2rgb, 于是喂 BGR; 本探针喂
//    **RGB** (转换后的模型自带 RGB 契约, 见 io/npu_axcl.h), 故对同一次解码结果,
//    置信度会差几个点、框会差 1–3px (同一张量下的两套通道序本就该有这点差);
//    `--swap-rb` 喂 BGR, 用来把这条差别消掉做同口径对比。落盘的输入张量就是喂进去的
//    那些字节, 故与别处的比对是逐字节的, 不靠"口径应当一致"。
//
//  类数 (--classes): 三支网格/提案口径的类数由属性数导出, 不需要它; **未折叠 DFL 头
//    需要真类数**才能把 attrs = 4·reg_max + 类数 切开, 不给则该输出解不出 (打印提示)。
//
//  用法: sudo LD_LIBRARY_PATH=/usr/lib/axcl ./build/model_probe <model.axmodel> [选项]
//    --image <jpg>  跑一张真图并打印解码+NMS 后的框 (模型输入像素与原图坐标两套)
//    --dump <前缀>  落盘 --image 喂进去的输入张量 (<前缀>.in.u8) 与每个保留输出
//                   (<前缀>.out<N>.f32), 都是无头部的原始字节
//    --swap-rb      输入喂 BGR (参考实现的通道序), 默认喂 RGB (转换契约)
//    --iters <n>    计时迭代次数 (默认 200)
//    --classes <n>  DFL 头的类数 (默认 0 = 由属性数导出; 公开 YOLO11 模型给 80)
//    --conf <t>     解码置信度阈值 (默认 0.5, 与固件 -t 的缺省同值)
//    --nms <t>      NMS 的 IoU 阈值 (默认 0.45)
//    --cls <id>     只解码该类 (默认 -1 = 不筛)
//    --xfer-sweep   只做传输尺寸扫描 (可与模型一起给, 用其真实尺寸当扫描点)
//
//  进程收尾: axcl 的库在静态析构里 abort (未 join 的线程 → terminate, 实测 RC=134),
//    故本探针用 _Exit 收尾跳过静态析构; stdout 已在程序开头设为无缓冲, 写出的行不受
//    影响 (管道里也一样)。
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <axcl.h>
#include <axcl_rt.h>
#include <axcl_rt_memory.h>
#include <axcl_rt_type.h>

#include "core/detect.h"
#include "io/npu_axcl.h"

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

struct Stat { double avg = 0, p50 = 0, p90 = 0, p99 = 0, mx = 0; };
static Stat summarize(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    s.avg = sum / (double)v.size();
    s.p50 = v[v.size() * 50 / 100];
    s.p90 = v[v.size() * 90 / 100];
    s.p99 = v[v.size() * 99 / 100];
    s.mx = v.back();
    return s;
}
// 一段耗时的百分位行; bytes != 0 时一并给等效带宽 (字节/µs = MB/s)
static void row(const char* tag, const std::vector<double>& v, size_t bytes = 0) {
    const Stat s = summarize(v);
    printf("  %-8s %8.1f %8.1f %8.1f %8.1f %8.1f", tag, s.avg, s.p50, s.p90, s.p99, s.mx);
    if (bytes) printf("   %7.1f MB/s", (double)bytes / s.avg);
    printf("\n");
}
static void print_dims(const NpuTensor& t) {
    for (int i = 0; i < t.dims.nbDims; ++i)
        printf("%s%lld", i ? "," : "", (long long)t.dims.d[i]);
}

// 参考实现的 letterbox 几何: 等比缩放 (截断取整) + 居中补零
struct Letterbox { int resize_w = 0, resize_h = 0, pad_w = 0, pad_h = 0; double scale = 0; };
static std::vector<uint8_t> letterbox_feed(const cv::Mat& bgr, int S, bool swap_rb, Letterbox& geo) {
    const double sc = std::min((double)S / bgr.rows, (double)S / bgr.cols);
    geo.scale = sc;
    geo.resize_w = (int)(sc * bgr.cols);
    geo.resize_h = (int)(sc * bgr.rows);
    cv::Mat r;
    cv::resize(bgr, r, cv::Size(geo.resize_w, geo.resize_h));
    geo.pad_w = (S - geo.resize_w) / 2;
    geo.pad_h = (S - geo.resize_h) / 2;
    cv::Mat canvas(S, S, CV_8UC3, cv::Scalar(0, 0, 0));
    r.copyTo(canvas(cv::Rect(geo.pad_w, geo.pad_h, geo.resize_w, geo.resize_h)));
    cv::Mat fed;
    // 契约 (io/npu_axcl.h) 是 RGB; --swap-rb 不换通道, 即按参考实现的 BGR 口径喂
    if (swap_rb) fed = canvas;
    else cv::cvtColor(canvas, fed, cv::COLOR_BGR2RGB);
    std::vector<uint8_t> v((size_t)S * S * 3);
    memcpy(v.data(), fed.data, v.size());
    return v;
}

// ---- 保留输出的数值实测: 最小非零相邻差 (码长) + 评分通道 ----
//   设备侧的输出张量按一个量程量化, 于是它的值落在一个码长网格上 —— 两个数说明这件事:
//   不同非零值之间的最小相邻差 (码长), 与落在它整数倍上的比例 (整张一个量程时 100%,
//   浮点或多量程的张量则比例低、差值极小)。评分通道 (obj/类分数) 的量程是 0..1, 坐标
//   通道是几百像素, 两者共用一张张量的量程时, 后者的码长就是前者的分辨率 —— 实测:
//   apex/codwz/R6 100% 落在网格上, 且类分数通道的最大值恰好等于一个码长 (即置信度只剩
//   一位); R6 更极端, obj/类通道整条落到码 0 (实测 0 候选, 而同帧同输入喂原 ONNX 有
//   7765 条 >0.5)。这是转换产物的性质, 本探针不改数值, 只把它摆出来。
static void report_output_stats(const NpuSession& s, int classes, bool placeholder_input) {
    printf("\n输出张量实测 (%s):\n", placeholder_input
           ? "本次输入是全 0 占位, 故评分通道恒为 0; 步长是产物的性质, 与输入无关"
           : "本次输入是真图, 步长是产物的性质, 通道量程随输入变");
    for (size_t i = 0; i < s.outputs().size(); ++i) {
        const NpuTensor& t = s.outputs()[i];
        if (!t.kept) continue;
        const float* p = (const float*)s.output_host(i);
        const size_t n = t.bytes / sizeof(float);
        std::vector<float> u;
        for (size_t k = 0; k < n; ++k) { const float a = std::fabs(p[k]); if (a > 0) u.push_back(a); }
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        double step = 0, on_grid = 0;
        if (u.size() >= 2) {
            step = (double)u[1] - (double)u[0];
            for (size_t k = 1; k < u.size(); ++k)
                step = std::min(step, (double)u[k] - (double)u[k - 1]);
            size_t hit = 0;
            for (float v : u) {
                const double q = (double)v / step;
                if (std::fabs(q - (double)std::llround(q)) < 1e-3) ++hit;
            }
            on_grid = 100.0 * (double)hit / (double)u.size();
        }
        printf("  OUT[%zu] 不同非零值 %zu 个, 最小非零相邻差 %.6f (落在它整数倍上 %.1f%% —— "
               "100%% 即整张按该步长量化, 浮点张量则比例低且差值极小)\n",
               i, u.size(), step, on_grid);

        // 评分通道: 按该输出的解码口径取 obj/类分数所在通道 (属性在前/在后共用 outVal),
        //   报最大与 >0 的条数 —— 一张评分通道整条为 0 的张量意味着置信度信息在量化时
        //   就没了 (坐标通道仍完好), 解码只能给 0 候选。
        const HeadKind k = head_kind(t.layout, classes, s.input_side());
        int c_first = -1, c_last = -1;
        const char* what = "评分";
        if (k == HeadKind::kEnd2End) { c_first = 4; c_last = 5; what = "conf/类"; }
        else if (k == HeadKind::kYoloV5) { c_first = 4; c_last = t.layout.attrs - 1; what = "obj+类"; }
        else if (k == HeadKind::kYoloV8) { c_first = 4; c_last = t.layout.attrs - 1; what = "类分数"; }
        else if (k == HeadKind::kYoloDfl) {
            const int nb = t.layout.attrs - classes;
            what = "类 logits";
            if (classes > 0 && nb > 0 && nb % 4 == 0) { c_first = nb; c_last = t.layout.attrs - 1; }
        }
        if (c_first < 0) { printf("        评分通道: 该输出的口径未定 (DFL 头见 --classes), 跳过\n"); continue; }
        float mx = -1e30f;
        size_t nz = 0;
        for (int a = c_first; a <= c_last; ++a)
            for (int j = 0; j < t.layout.num; ++j) {
                const float v = outVal(p, t.layout, a, j);
                if (v > mx) mx = v;
                if (v > 0) ++nz;
            }
        printf("        通道 %d..%d (%s) 最大 %.6f, >0 的 %zu 条\n", c_first, c_last, what, mx, nz);
    }
}

// ---- 传输尺寸扫描: 逐尺寸的百分位 + t = a + B/rate 的最小二乘拟合与残差 ----
static void xfer_sweep(const NpuSession* model, int iters) {
    std::vector<size_t> sizes = {16u << 10, 64u << 10, 128u << 10, 256u << 10, 512u << 10,
                                 1u << 20, 2u << 20, 4u << 20, 8u << 20};
    if (model) {   // 真实尺寸: 该模型的输入与每个保留输出
        for (const auto& t : model->inputs()) sizes.push_back(t.bytes);
        for (const auto& t : model->outputs()) if (t.kept) sizes.push_back(t.bytes);
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    const size_t maxb = sizes.back();

    void* dev = nullptr;
    if (axclrtMalloc(&dev, maxb, AXCL_MEM_MALLOC_HUGE_FIRST)) {
        printf("设备缓冲分配失败 (%zu B)\n", maxb);
        return;
    }
    std::vector<uint8_t> host(maxb, 0x5a);

    printf("\n=== 传输尺寸扫描 (每尺寸 %d 次; 单位 µs; 含该模型自身尺寸) ===\n", iters);
    printf("  %10s %10s %9s %9s %9s | %9s %9s %9s | %9s %9s\n", "尺寸", "字节", "H2D p50",
           "H2D p90", "H2D p99", "D2H p50", "D2H p90", "D2H p99", "H2D MB/s", "D2H MB/s");
    std::vector<double> B, th, td;
    for (size_t b : sizes) {
        for (int i = 0; i < 5; ++i) {   // 预热
            axclrtMemcpy(dev, host.data(), b, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(host.data(), dev, b, AXCL_MEMCPY_DEVICE_TO_HOST);
        }
        std::vector<double> vh, vd;
        vh.reserve(iters); vd.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const double t0 = now_us();
            axclrtMemcpy(dev, host.data(), b, AXCL_MEMCPY_HOST_TO_DEVICE);
            const double t1 = now_us();
            axclrtMemcpy(host.data(), dev, b, AXCL_MEMCPY_DEVICE_TO_HOST);
            const double t2 = now_us();
            vh.push_back(t1 - t0); vd.push_back(t2 - t1);
        }
        const Stat sh = summarize(vh), sd = summarize(vd);
        printf("  %10zu %10zu %9.1f %9.1f %9.1f | %9.1f %9.1f %9.1f | %9.1f %9.1f\n",
               b, b, sh.p50, sh.p90, sh.p99, sd.p50, sd.p90, sd.p99,
               (double)b / sh.avg, (double)b / sd.avg);
        B.push_back((double)b); th.push_back(sh.avg); td.push_back(sd.avg);
    }

    // 两遍最小二乘: 全部点 (≥64KB) 与带宽支配段 (≥256KB); 残差逐点给出
    auto fit = [&](const std::vector<double>& T, const char* tag, bool bandwidth_only) {
        std::vector<double> bb, tt;
        for (size_t i = 0; i < B.size(); ++i)
            if (!bandwidth_only || B[i] >= 262144) { bb.push_back(B[i]); tt.push_back(T[i]); }
        const size_t n = bb.size();
        if (n < 2) return;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (size_t i = 0; i < n; ++i) { sx += bb[i]; sy += tt[i]; sxx += bb[i] * bb[i]; sxy += bb[i] * tt[i]; }
        const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);   // µs / byte
        const double icept = (sy - slope * sx) / n;                       // µs / 次调用
        double worst = 0; char worst_at[32] = "-";
        for (size_t i = 0; i < n; ++i) {
            const double pred = icept + slope * bb[i];
            const double e = (tt[i] - pred) / pred * 100.0;
            if (fabs(e) > worst) { worst = fabs(e); snprintf(worst_at, sizeof worst_at, "%.0f B", bb[i]); }
        }
        printf("  拟合 %-16s: 每次调用 %.1f µs + %.3f µs/MB (%.1f MB/s); 残差最大 %.1f%% @ %s\n",
               tag, icept, slope * 1048576.0, 1.0 / slope, worst, worst_at);
    };
    printf(" H2D:\n   ");
    fit(th, "全部点 (≥64KB)", false);
    printf("   ");
    fit(th, "带宽段 (≥256KB)", true);
    printf(" D2H:\n   ");
    fit(td, "全部点 (≥64KB)", false);
    printf("   ");
    fit(td, "带宽段 (≥256KB)", true);
    printf(" 注: 每次调用都有百微秒级固定项, 小尺寸项离线性最远 —— 预算用'带宽段'的斜率\n"
           "     与每个尺寸自己的实测值, 不拿一条直线外推到小尺寸。\n");
    axclrtFree(dev);
}

static int probe_main(int argc, char** argv) {
    std::string model_path, image_path, dump_prefix;
    int iters = 200, want_cls = -1, classes = 0;
    float conf = 0.5f, nms_thr = 0.45f;
    bool sweep = false, swap_rb = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--image") image_path = next();
        else if (a == "--dump") dump_prefix = next();
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--classes") classes = atoi(next().c_str());
        else if (a == "--conf") conf = (float)atof(next().c_str());
        else if (a == "--nms") nms_thr = (float)atof(next().c_str());
        else if (a == "--cls") want_cls = atoi(next().c_str());
        else if (a == "--xfer-sweep") sweep = true;
        else if (a == "--swap-rb") swap_rb = true;
        else if (a == "-h" || a == "--help") {
            printf("用法: %s <model.axmodel> [--image <jpg>] [--dump <前缀>] [--iters N]\n"
                   "          [--classes n] [--conf t] [--nms t] [--cls id] [--swap-rb]\n"
                   "       %s --xfer-sweep [--iters N] [<model.axmodel>]\n", argv[0], argv[0]);
            return 0;
        } else model_path = a;
    }

    std::string err;
    if (!npu_init(&err)) { printf("❌ %s\n", err.c_str()); return 1; }
    int vmaj = 0, vmin = 0, vpat = 0;
    axclrtGetVersion(&vmaj, &vmin, &vpat);
    printf("=== model_probe ===\n卡: %s   AXCL 运行时: %d.%d.%d\n", npu_soc_name(), vmaj, vmin, vpat);

    if (sweep) {
        NpuSession s;
        if (!model_path.empty() && !s.open(model_path, &err)) {
            printf("❌ %s\n", err.c_str()); return 1;
        }
        xfer_sweep(model_path.empty() ? nullptr : &s, iters);
        return 0;
    }
    if (model_path.empty()) { printf("❌ 缺模型路径 (见 -h)\n"); return 1; }

    NpuSession s;
    if (!s.open(model_path, &err)) { printf("❌ %s\n", err.c_str()); return 1; }

    // ---------------- IO 契约 ----------------
    size_t skipped = 0;
    for (const auto& t : s.outputs()) if (!t.kept) ++skipped;
    printf("\n模型: %s\nIO 契约: 输入 %zu 个 / 输出 %zu 个 (保留 %zu / 跳过 %zu)\n",
           model_path.c_str(), s.inputs().size(), s.outputs().size(), s.kept_outputs(), skipped);
    bool has_dfl = false;
    for (const auto& t : s.inputs()) {
        printf("  IN [0] %-28s %-4s dims ", t.name.c_str(), npu_dtype_str(t.raw_dtype));
        print_dims(t);
        printf("  %8zu B   ", t.bytes);
        if (s.input_side() > 0 && t.bytes == (size_t)s.input_side() * s.input_side() * 3)
            printf("NHWC RGB 3 通道 ✓ (运行时 layout 字段=%s 与 dims 不符, 判据是 dims)\n",
                   npu_layout_str(t.raw_layout));
        else
            printf("⚠ 与 U8/NHWC/3 通道契约不符 (方形边长 %d, layout 字段=%s)\n",
                   s.input_side(), npu_layout_str(t.raw_layout));
    }
    for (size_t i = 0; i < s.outputs().size(); ++i) {
        const NpuTensor& t = s.outputs()[i];
        printf("  OUT[%zu] %-28s %-4s dims ", i, t.name.c_str(), npu_dtype_str(t.raw_dtype));
        print_dims(t);
        printf("  %8zu B   ", t.bytes);
        if (!t.kept) { printf("%s\n", t.note.c_str()); continue; }
        const HeadKind k = head_kind(t.layout, classes, s.input_side());
        if (k == HeadKind::kYoloDfl) has_dfl = true;
        printf("保留: 布局 attrs=%d num=%d (%s)", t.layout.attrs, t.layout.num,
               t.layout.attrs_first ? "属性在前" : "属性在后");
        if (t.layout.dfl)
            printf(", DFL 网格 %d×%d 步长 %d", t.layout.gh, t.layout.gw, t.layout.stride);
        printf(" | 口径 %s | ", head_kind_name(k));
        if (k == HeadKind::kYoloDfl) {
            const int nb = t.layout.attrs - classes;
            if (classes > 0 && nb > 0 && nb % 4 == 0)
                printf("每边分布长度 %d (由 attrs − 类数 %d 再 /4)", nb / 4, classes);
            else
                printf("⚠ 需要 --classes <类数> 才能切分 (attrs − 类数)/4");
        } else {
            printf("类数 = 属性数 − %d = %d", k == HeadKind::kYoloV5 ? 5 : 4,
                   t.layout.attrs - (k == HeadKind::kYoloV5 ? 5 : 4));
        }
        printf("\n");
    }
    printf("每 tick 传输量: H2D %zu B / D2H %zu B (只算保留的输出)\n", s.in_bytes(), s.out_bytes());
    if (has_dfl && classes <= 0)
        printf("⚠ 该模型有未折叠 DFL 头: 解码需要真类数, 用 --classes <n> 给出\n");

    // ---------------- 输入张量 (真图或占位) ----------------
    const int S = s.input_side();
    if (S <= 0) { printf("❌ 输入非方形, 本探针的 letterbox 口径不适用\n"); return 1; }
    cv::Mat img;
    std::vector<uint8_t> in((size_t)S * S * 3, 0);
    Letterbox geo;
    if (!image_path.empty()) {
        img = cv::imread(image_path);
        if (img.empty()) { printf("❌ 读图失败: %s\n", image_path.c_str()); return 1; }
        in = letterbox_feed(img, S, swap_rb, geo);
        printf("\n输入: %s (%d×%d) → letterbox %d×%d (等比 %.4f, resize %d×%d, 补零 上下 %d)"
               " → %s %zu B\n", image_path.c_str(), img.cols, img.rows, S, S, geo.scale,
               geo.resize_w, geo.resize_h, geo.pad_h, swap_rb ? "BGR (--swap-rb)" : "RGB",
               in.size());
    } else {
        printf("\n输入: 占位 (全 0; 未给 --image —— NPU 的执行时间与数据无关)\n");
    }

    if (!dump_prefix.empty() && image_path.empty())
        printf("⚠ --dump 需要 --image (要落盘的是实际喂进去的张量)\n");

    // ---------------- 一次真跑 + 解码 ----------------
    NpuTick tick;
    auto dets = s.run(in.data(), classes, conf, want_cls, &tick);
    std::vector<Detection> final_dets = nms(dets, nms_thr);
    printf("\n解码: 候选 %zu 条 → NMS (IoU %.2f) 后 %zu 条\n", dets.size(), nms_thr,
           final_dets.size());
    std::sort(final_dets.begin(), final_dets.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    // 诊断输出: 低阈值或占位输入下候选可以上万条, 只列前若干条 (其余的用计数说明)
    const size_t kListMax = 50;
    const size_t shown = std::min(final_dets.size(), kListMax);
    if (image_path.empty()) {
        printf("  %-4s %-6s %8s %8s %8s %8s %8s\n", "#", "类", "conf", "x0", "y0", "x1", "y1");
        for (size_t i = 0; i < shown; ++i) {
            const Detection& d = final_dets[i];
            printf("  %-4zu %-6d %7.1f%% %8.1f %8.1f %8.1f %8.1f\n", i + 1, d.class_id,
                   d.conf * 100, d.cx - d.w * .5, d.cy - d.h * .5, d.cx + d.w * .5, d.cy + d.h * .5);
        }
    } else {
        printf("  %-4s %-6s %8s %8s %8s %8s %8s | %-8s %-8s %-8s %-8s\n", "#", "类", "conf",
               "x0", "y0", "x1", "y1", "原图x0", "原图y0", "原图x1", "原图y1");
        // 参考实现的换算式: (p − 补零偏移) × 原边长/缩放边长, 夹到 [0, 边长−1]
        const double rx = geo.resize_w ? (double)img.cols / geo.resize_w : 0;
        const double ry = geo.resize_h ? (double)img.rows / geo.resize_h : 0;
        auto mx = [&](double v) {
            return std::max(std::min((v - geo.pad_w) * rx, (double)img.cols - 1), 0.0); };
        auto my = [&](double v) {
            return std::max(std::min((v - geo.pad_h) * ry, (double)img.rows - 1), 0.0); };
        for (size_t i = 0; i < shown; ++i) {
            const Detection& d = final_dets[i];
            const double x0 = d.cx - d.w * .5, y0 = d.cy - d.h * .5;
            const double x1 = d.cx + d.w * .5, y1 = d.cy + d.h * .5;
            printf("  %-4zu %-6d %7.1f%% %8.1f %8.1f %8.1f %8.1f | %8.0f %8.0f %8.0f %8.0f\n",
                   i + 1, d.class_id, d.conf * 100, x0, y0, x1, y1, mx(x0), my(y0), mx(x1), my(y1));
        }
    }
    if (final_dets.size() > shown)
        printf("  ⋯ 余 %zu 条未列 (诊断输出只列前 %zu 条)\n", final_dets.size() - shown, kListMax);

    // 输出张量本身的数值实测 (量化步长与评分通道量程 —— 见 report_output_stats 注释)
    report_output_stats(s, classes, image_path.empty());

    // ---------------- 落盘 (逐字节可比) ----------------
    if (!dump_prefix.empty() && !image_path.empty()) {
        printf("落盘 (无头部原始字节; .in.u8 的通道序见上面的输入行):\n");
        auto wr = [&](const std::string& p, const void* d, size_t n) -> bool {
            FILE* f = fopen(p.c_str(), "wb");
            if (!f) { printf("  ❌ 打开失败 %s\n", p.c_str()); return false; }
            const bool ok = fwrite(d, 1, n, f) == n;
            fclose(f);
            printf("  %s %s (%zu B)\n", ok ? "✅" : "❌", p.c_str(), n);
            return ok;
        };
        wr(dump_prefix + ".in.u8", in.data(), in.size());
        for (size_t i = 0; i < s.outputs().size(); ++i) {
            const NpuTensor& t = s.outputs()[i];
            if (!t.kept) continue;
            char path[512];
            snprintf(path, sizeof path, "%s.out%zu.f32", dump_prefix.c_str(), i);
            wr(path, s.output_host(i), t.bytes);
        }
    }

    // ---------------- 计时 ----------------
    const int warm = 20;
    for (int i = 0; i < warm; ++i) s.run(in.data(), classes, conf, want_cls, nullptr);
    std::vector<double> vh2d, vexec, vd2h, vdec, vtot;
    vh2d.reserve(iters); vexec.reserve(iters);
    vd2h.reserve(iters); vdec.reserve(iters); vtot.reserve(iters);
    // 逐保留输出的 D2H (每张量一张表; 下标 = 保留顺序, 与 IO 契约段的行序一致)
    std::vector<std::vector<double>> veach(s.kept_outputs());
    for (auto& v : veach) v.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        NpuTick t;
        s.run(in.data(), classes, conf, want_cls, &t);
        vh2d.push_back(t.h2d_us); vexec.push_back(t.exec_us);
        vd2h.push_back(t.d2h_us); vdec.push_back(t.decode_us); vtot.push_back(t.total_us());
        for (int k = 0; k < t.d2h_n && k < (int)veach.size(); ++k) veach[k].push_back(t.d2h_each_us[k]);
    }
    printf("\n计时 (%d 次, 预热 %d 次; 单位 µs):\n", iters, warm);
    printf("  %-8s %8s %8s %8s %8s %8s\n", "段", "avg", "p50", "p90", "p99", "max");
    row("H2D", vh2d, s.in_bytes());
    row("execute", vexec);
    row("D2H", vd2h, s.out_bytes());
    row("decode", vdec);
    row("合计", vtot);
    if (veach.size() > 1) {
        printf("  逐输出 D2H (保留顺序, 与上面的 OUT 行序一致):\n");
        std::vector<size_t> kbytes;
        for (const auto& t : s.outputs()) if (t.kept) kbytes.push_back(t.bytes);
        for (size_t k = 0; k < veach.size(); ++k) {
            char tag[24];
            snprintf(tag, sizeof tag, "D2H[%zu]", k);
            row(tag, veach[k], k < kbytes.size() ? kbytes[k] : 0);
        }
    }
    printf("  等效检测率 (合计 avg): %.1f fps  (execute 是输入已在显存时的 NPU 纯执行时间)\n",
           1e6 / summarize(vtot).avg);
    return 0;
}

// axcl 的库在静态析构里 abort (未被 join 的线程 → terminate, 实测 RC=134): 收尾用 _Exit
//   跳过静态析构。stdout 在进 probe_main 前就设成无缓冲, 写出的行照旧 (管道里也一样)。
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int rc = probe_main(argc, argv);
    fflush(stdout);
    fflush(stderr);
    std::_Exit(rc);
}
