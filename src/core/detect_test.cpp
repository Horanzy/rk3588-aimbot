// ============================================================================
//  detect_test — build/detect_test 单测 (scripts/compile.sh 构建并执行一次):
//    [1] 布局解析: 四种真实导出形状 (端到端 [1,300,6] / YOLOv8-11 转置
//        [1,8,2100] / YOLOv5 [1,10647,7] / 未折叠 DFL [1,80,80,144]) 与解析器
//        拒收的形状 (rank 5、DFL 非方网格、通道数过小)
//    [2] 锚点数由模型几何给出: (S/8)²+(S/16)²+(S/32)² 复现实测三档 (320→2100,
//        416→3549, 640→8400), 3×3549 = 10647; 不能整除步长时返回 0 (几何判不了)
//    [3] 口径判定: 四个真实布局各落到哪一支 (与配置的类数无关 —— 几何优先);
//        几何缺席时退回"属性数 vs 类数"口径, 类数给对时结论一致; 端到端靠
//        "属性在后且恰 6 个"这一物理结构判定
//    [4] 逐支解码: 手算候选的 cx/cy/w/h/conf/类 与实解逐字段比对 (端到端取属性
//        在后; v8-11 取转置; v5 取 objectness×类分数), 以及类/阈值筛选
//    [5] DFL 头: 手算分布 (单点、均布、两点) 的期望值 × 步长 与解出的框逐字段比对;
//        类分数过 sigmoid; 步长未知时整块跳过 (不猜分布长度)
//    [6] NMS: 同类重叠被抑制、异类同框保留、部分重叠按 IoU 阈值取舍、同分排序
//  全部断言通过输出 ALL PASS 并返回 0; 任一断言失败返回非零 (compile.sh 的 set -e
//  终止编译)。
// ============================================================================

#include <cmath>
#include <iostream>
#include <vector>

#include "core/detect.h"

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { std::cout << "  ok  " << msg << "\n"; } \
    else { std::cerr << "  FAIL " << msg << "\n"; ++g_fail; } \
} while (0)

static bool near_(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol * (1.f + std::fabs(b));
}
static Dims dims3(int64_t a, int64_t b) { Dims d; d.nbDims = 3; d.d[0] = 1; d.d[1] = a; d.d[2] = b; return d; }
static Dims dims4(int64_t a, int64_t b, int64_t c) {
    Dims d; d.nbDims = 4; d.d[0] = 1; d.d[1] = a; d.d[2] = b; d.d[3] = c; return d;
}

int main() {
    std::cout << "=== detect_test (检测输出解析) ===\n";

    // ---------------- [1] 布局解析 ----------------
    {
        std::cout << "[1] 布局解析\n";
        OutputLayout l;
        CHECK(parseOutputLayout(dims3(300, 6), l, 640) && l.attrs == 6 && l.num == 300
              && !l.attrs_first && !l.dfl,
              "端到端 [1,300,6]: 属性在后、候选 300、非 DFL");
        CHECK(parseOutputLayout(dims3(8, 2100), l, 320) && l.attrs == 8 && l.num == 2100
              && l.attrs_first && !l.dfl,
              "v8-11 转置 [1,8,2100]: 属性在前、候选 2100");
        CHECK(parseOutputLayout(dims3(10647, 7), l, 416) && l.attrs == 7 && l.num == 10647
              && !l.attrs_first,
              "YOLOv5 [1,10647,7]: 属性在后、候选 10647");
        CHECK(parseOutputLayout(dims3(6, 8400), l, 640) && l.attrs == 6 && l.num == 8400
              && l.attrs_first,
              "单类双类网格头 [1,6,8400]: 属性在前");
        CHECK(parseOutputLayout(dims4(80, 80, 144), l, 640) && l.dfl && l.gw == 80
              && l.gh == 80 && l.num == 6400 && l.attrs == 144 && l.stride == 8,
              "DFL [1,80,80,144] 输入 640: 网格 80×80、步长 8、属性 144");
        CHECK(parseOutputLayout(dims4(20, 20, 144), l, 640) && l.stride == 32,
              "DFL 第三层 [1,20,20,144]: 步长 32 (由输入边长与网格数导出)");
        Dims d5{}; d5.nbDims = 5; d5.d[0] = 1; d5.d[1] = 80; d5.d[2] = 80; d5.d[3] = 1; d5.d[4] = 144;
        CHECK(!parseOutputLayout(d5, l, 640), "rank 5 不是可解布局 (跳过)");
        CHECK(!parseOutputLayout(dims4(80, 40, 144), l, 640),
              "非方网格 [1,80,40,144] 判为 NCHW 原始头而非 DFL (跳过)");
        CHECK(!parseOutputLayout(dims4(80, 80, 4), l, 640), "通道数 <5 的 rank4 不是检测头 (跳过)");
        CHECK(!parseOutputLayout(dims4(80, 80, 144), l, 641),
              "输入边长不能被网格整除时不算 DFL 网格 (步长必须整数)");
        Dims d1{}; d1.nbDims = 1; d1.d[0] = 8400;
        CHECK(!parseOutputLayout(d1, l, 640), "rank 1 不是可解布局 (跳过)");
    }

    // ---------------- [2] 锚点数由几何给出 ----------------
    {
        std::cout << "[2] 锚点数 (三个特征层步长 8/16/32)\n";
        CHECK(anchor_count(320) == 2100, "320 → 40²+20²+10² = 2100");
        CHECK(anchor_count(256) == 1344, "256 → 32²+16²+8² = 1344 (实测 Z v11s_256 的候选数)");
        CHECK(anchor_count(416) == 3549, "416 → 52²+26²+13² = 3549");
        CHECK(anchor_count(640) == 8400, "640 → 80²+40²+20² = 8400");
        CHECK(3 * anchor_count(416) == 10647, "YOLOv5 的三锚 = 3×3549 = 10647 (实测候选数)");
        CHECK(anchor_count(0) == 0 && anchor_count(-8) == 0, "边长未知/非正 → 0 (几何判不了)");
        CHECK(anchor_count(300) == 0, "边长不能被步长整除 → 0");
    }

    // ---------------- [3] 口径判定 ----------------
    {
        std::cout << "[3] 口径判定 (几何优先, 类数不参与)\n";
        OutputLayout l;
        parseOutputLayout(dims3(8, 2100), l, 320);
        CHECK(head_kind(l, 4, 320) == HeadKind::kYoloV8, "[1,8,2100] @320 → v8-11 (候选数 = G)");
        parseOutputLayout(dims3(11, 2100), l, 320);
        CHECK(head_kind(l, 7, 320) == HeadKind::kYoloV8, "[1,11,2100] @320 → v8-11");
        parseOutputLayout(dims3(10, 2100), l, 320);
        CHECK(head_kind(l, 6, 320) == HeadKind::kYoloV8, "[1,10,2100] @320 → v8-11");
        parseOutputLayout(dims3(13, 2100), l, 320);
        CHECK(head_kind(l, 9, 320) == HeadKind::kYoloV8, "[1,13,2100] @320 → v8-11");
        parseOutputLayout(dims3(10647, 7), l, 416);
        CHECK(head_kind(l, 2, 416) == HeadKind::kYoloV5, "[1,10647,7] @416 → v5 (候选数 = 3G)");
        parseOutputLayout(dims3(5, 3549), l, 416);
        CHECK(head_kind(l, 1, 416) == HeadKind::kYoloV8, "[1,5,3549] @416 → v8-11 (单类)");
        parseOutputLayout(dims3(6, 8400), l, 640);
        CHECK(head_kind(l, 2, 640) == HeadKind::kYoloV8, "[1,6,8400] @640 → v8-11 (实测 CS)");
        parseOutputLayout(dims3(13, 1344), l, 256);
        CHECK(head_kind(l, 9, 256) == HeadKind::kYoloV8, "[1,13,1344] @256 → v8-11 (实测 Z 256)");
        parseOutputLayout(dims3(300, 6), l, 640);
        CHECK(head_kind(l, 4, 640) == HeadKind::kEnd2End,
              "[1,300,6] @640 → 端到端 (候选数与 G、3G 都不等, 属性在后且恰 6 个)");
        parseOutputLayout(dims4(80, 80, 144), l, 640);
        CHECK(head_kind(l, 80, 640) == HeadKind::kYoloDfl, "[1,80,80,144] → DFL 头");

        // 几何缺席 (输入边长未知) 时的退路口径: 类数给对则结论一致
        OutputLayout l2;
        parseOutputLayout(dims3(10647, 7), l2, 0);
        CHECK(head_kind(l2, 2, 0) == HeadKind::kYoloV5,
              "几何缺席 + 类数正确 → 退路口径仍判 v5 (属性数 = 5 + 类数)");
        OutputLayout l3;
        parseOutputLayout(dims3(8, 2100), l3, 0);
        CHECK(head_kind(l3, 4, 0) == HeadKind::kYoloV8,
              "几何缺席 + 类数正确 → 退路口径仍判 v8-11 (属性数 = 4 + 类数)");
        OutputLayout l4;
        parseOutputLayout(dims3(300, 6), l4, 0);
        CHECK(head_kind(l4, 80, 0) == HeadKind::kEnd2End,
              "几何缺席时端到端仍由'属性在后且恰 6 个'判出 (与类数无关)");
    }

    // ---------------- [4] 逐支解码 ----------------
    {
        std::cout << "[4] 逐支解码 (手算候选)\n";
        OutputLayout l;

        // 端到端: 第 0 行 (x1,y1,x2,y2,conf,cls) = (10,20,50,80,0.9,3)
        std::vector<float> e(300 * 6, 0.f);
        e[0] = 10; e[1] = 20; e[2] = 50; e[3] = 80; e[4] = 0.9f; e[5] = 3;
        e[6 * 1 + 4] = 0.95f; e[6 * 1 + 5] = 3;          // 第 1 行: 置信度更高, 同一个框
        e[6 * 2 + 4] = 0.8f;  e[6 * 2 + 5] = 5;          // 第 2 行: 另一类
        parseOutputLayout(dims3(300, 6), l, 640);
        auto d = decode_outputs({e.data()}, {l}, 80, 0.5f, -1, 640);
        CHECK(d.size() == 3, "端到端: 阈值 0.5 下 3 个候选全部留下");
        CHECK(near_(d[0].cx, 30) && near_(d[0].cy, 50) && near_(d[0].w, 40)
              && near_(d[0].h, 60) && near_(d[0].conf, 0.9f) && d[0].class_id == 3,
              "端到端第 0 行: cx/cy/w/h = 30/50/40/60, conf 0.9, 类 3");
        auto d3 = decode_outputs({e.data()}, {l}, 80, 0.5f, 3, 640);
        CHECK(d3.size() == 2 && d3[0].conf == 0.9f && d3[1].conf == 0.95f,
              "端到端按类筛 (want_cls=3) 只留该类, 且按缓冲顺序 (未排序)");
        auto dh = decode_outputs({e.data()}, {l}, 80, 0.92f, -1, 640);
        CHECK(dh.size() == 1 && dh[0].conf == 0.95f, "端到端阈值 0.92 只留 0.95 那条");

        // v8-11 转置 [1,8,2100]: 候选 100 的 (cx,cy,w,h) 在 0..3, 类分数在 4..7
        std::vector<float> v8(8 * 2100, 0.f);
        auto set_v8 = [&](int i, float cx, float cy, float w, float h,
                          std::initializer_list<float> cls) {
            v8[0 * 2100 + i] = cx; v8[1 * 2100 + i] = cy;
            v8[2 * 2100 + i] = w;  v8[3 * 2100 + i] = h;
            int c = 0; for (float s : cls) v8[(4 + c++) * 2100 + i] = s;
        };
        set_v8(100, 100, 200, 50, 40, {0.1f, 0.2f, 0.8f, 0.3f});
        set_v8(7, 11, 12, 13, 14, {0.1f, 0.2f, 0.3f, 0.6f});
        parseOutputLayout(dims3(8, 2100), l, 320);
        auto dv8 = decode_outputs({v8.data()}, {l}, 4, 0.5f, -1, 320);
        CHECK(dv8.size() == 2, "v8-11: 阈值 0.5 下留下 2 个候选 (类分数最大者)");
        // 逐候选按缓冲顺序出 (候选 7 在候选 100 之前)
        CHECK(dv8[0].class_id == 3 && near_(dv8[0].conf, 0.6f) && near_(dv8[0].cx, 11)
              && near_(dv8[0].w, 13),
              "v8-11 候选 7: 框 11/12/13/14, 类 3 / conf 0.6 (逐候选取最大类分数)");
        CHECK(dv8[1].class_id == 2 && near_(dv8[1].conf, 0.8f) && near_(dv8[1].cx, 100)
              && near_(dv8[1].cy, 200) && near_(dv8[1].w, 50) && near_(dv8[1].h, 40),
              "v8-11 候选 100: 框 100/200/50/40, conf = 类分数最大 0.8, 类 2");
        auto dv8f = decode_outputs({v8.data()}, {l}, 4, 0.5f, 3, 320);
        CHECK(dv8f.size() == 1 && near_(dv8f[0].cx, 11), "v8-11 按类筛只留 want_cls 那一条");

        // YOLOv5 [1,10647,7]: 每行 (cx,cy,w,h,obj,c0,c1), conf = obj × 类分数
        std::vector<float> v5(10647 * 7, 0.f);
        auto set_v5 = [&](int i, float cx, float cy, float w, float h, float obj,
                          float c0, float c1) {
            float* p = v5.data() + (size_t)i * 7;
            p[0] = cx; p[1] = cy; p[2] = w; p[3] = h; p[4] = obj; p[5] = c0; p[6] = c1;
        };
        set_v5(0, 30, 40, 10, 12, 0.9f, 0.5f, 0.8f);      // conf = 0.72, 类 1
        set_v5(1, 1, 2, 3, 4, 0.9f, 0.9f, 0.1f);          // conf = 0.81, 类 0
        set_v5(2, 5, 6, 7, 8, 0.4f, 0.9f, 0.9f);          // conf = 0.36 (低于阈值)
        parseOutputLayout(dims3(10647, 7), l, 416);
        auto dv5 = decode_outputs({v5.data()}, {l}, 2, 0.5f, -1, 416);
        CHECK(dv5.size() == 2, "v5: objectness × 类分数 过阈值的 2 条");
        CHECK(dv5[0].class_id == 1 && near_(dv5[0].conf, 0.72f) && near_(dv5[0].cx, 30),
              "v5 第 0 行: conf = 0.9×0.8 = 0.72, 类 1");
        CHECK(dv5[1].class_id == 0 && near_(dv5[1].conf, 0.81f) && near_(dv5[1].h, 4),
              "v5 第 1 行: conf = 0.9×0.9 = 0.81, 类 0");
        auto dv5f = decode_outputs({v5.data()}, {l}, 2, 0.5f, 0, 416);
        CHECK(dv5f.size() == 1 && near_(dv5f[0].conf, 0.81f), "v5 按类筛");

        // 多输出: 两个不同口径的缓冲混排时, 逐缓冲各按自己的布局解
        OutputLayout la, lb;
        parseOutputLayout(dims3(300, 6), la, 640);
        parseOutputLayout(dims3(6, 8400), lb, 640);        // 属性在前 (每个属性一整行)
        std::vector<float> ee(300 * 6, 0.f);
        ee[4] = 0.7f; ee[5] = 2;                          // 端到端第 0 行: conf 0.7 / 类 2
        std::vector<float> gg(8400 * 6, 0.f);              // 类 1 的分数留 0 → 最大者是类 0
        gg[4 * 8400 + 3] = 0.9f;                           // 属性在前: 第 3 候选的 conf
        auto dm = decode_outputs({ee.data(), gg.data()}, {la, lb}, 3, 0.5f, -1, 640);
        CHECK(dm.size() == 2 && dm[0].class_id == 2 && dm[0].conf == 0.7f
              && dm[1].class_id == 0 && near_(dm[1].conf, 0.9f),
              "多输出: 逐缓冲按各自布局解 (端到端一行类 2 + 网格一条类 0)");
    }

    // ---------------- [5] DFL 头 ----------------
    {
        std::cout << "[5] DFL 头 (分布期望 × 步长)\n";
        const int gh = 80, attrs = 144, nc = 80, reg = 16;
        std::vector<float> o((size_t)gh * gh * attrs, 0.f);
        auto anchor = [&](int row, int col) {
            return o.data() + ((size_t)row * gh + col) * attrs;
        };
        // 每个格的类分数默认给 logit −4 (sigmoid 0.01799), 故只有下面指定的格能过阈值 ——
        //   全 0 的格会以 sigmoid(0)=0.5 恰好压线, 那不是本用例要测的东西。
        for (int a = 0; a < gh * gh; ++a)
            for (int c = 0; c < nc; ++c) o[(size_t)a * attrs + 4 * reg + c] = -4.f;

        float* p = anchor(1, 2);                          // 格心 (20, 12), 步长 8
        // 每条边一条完整分布 (未列出的桶给 −1000, 即 exp 下溢成 0):
        //   左: 单点 bin 1 → 期望 1;  上: 16 桶等权 → 7.5;  右: 两点 2/5 等权 → 3.5;  下: 单点 0 → 0
        auto fill = [&](float* d, float v) { for (int j = 0; j < reg; ++j) d[j] = v; };
        fill(p + 0 * reg, -1000.f); p[0 * reg + 1] = 0.f;
        fill(p + 1 * reg, 0.f);
        fill(p + 2 * reg, -1000.f); p[2 * reg + 2] = 0.f; p[2 * reg + 5] = 0.f;
        fill(p + 3 * reg, -1000.f); p[3 * reg + 0] = 0.f;
        p[4 * reg + 0] = 0.f;                             // 类 0: logit 0 → 0.5
        p[4 * reg + 7] = 2.f;                             // 类 7: logit 2 → 0.8808
        OutputLayout l;
        parseOutputLayout(dims4(gh, gh, attrs), l, 640);
        // 另一格 (0,0): 四条边都单点 0 (期望 0) + 上面的默认负类分数 → 低阈值下按类 0 解出

        auto d = decode_outputs({o.data()}, {l}, nc, 0.5f, -1, 640);
        CHECK(d.size() == 1, "DFL: 阈值 0.5 下只留那条 (其余格的类分数 0.01799)");
        // ltrb = (1, 7.5, 3.5, 0) × 步长 8 = (8, 60, 28, 0), 格心 (20, 12) →
        //   x1 = 12, y1 = −48, x2 = 48, y2 = 12 → cx 30, cy −18, w 36, h 60
        CHECK(near_(d[0].cx, 30) && near_(d[0].cy, -18) && near_(d[0].w, 36) && near_(d[0].h, 60),
              "DFL 框: 单点/等权/两点分布的期望 1/7.5/3.5/0 × 步长 8 → cx 30, cy −18, w 36, h 60");
        CHECK(near_(d[0].conf, 0.8808f, 1e-3f) && d[0].class_id == 7,
              "DFL 类分数: logit 2 → sigmoid 0.8808, 取最大者 (类 7)");
        auto d7 = decode_outputs({o.data()}, {l}, nc, 0.5f, 0, 640);
        CHECK(d7.empty(), "DFL 按类筛: 留下的那条最大类是 7, want_cls=0 时不留");

        // 低阈值: 只把 (0,0) 格抬起来 —— 其余格的类分数压到 logit −20 (sigmoid 2e-9)
        for (int a = 0; a < gh * gh; ++a)
            for (int c = 0; c < nc; ++c) o[(size_t)a * attrs + 4 * reg + c] = -20.f;
        for (int k = 0; k < 4; ++k) {
            fill(anchor(0, 0) + k * reg, -1000.f);
            anchor(0, 0)[k * reg + 0] = 0.f;             // 四条边都单点 bin 0 → 距离 0
        }
        for (int c = 0; c < nc; ++c) anchor(0, 0)[4 * reg + c] = 0.f;   // 类 0: logit 0 → 0.5
        auto d0 = decode_outputs({o.data()}, {l}, nc, 0.001f, 0, 640);
        CHECK(d0.size() == 1 && d0[0].class_id == 0 && near_(d0[0].conf, 0.5f)
              && near_(d0[0].cx, 4) && near_(d0[0].cy, 4) && near_(d0[0].w, 0)
              && near_(d0[0].h, 0),
              "DFL 第 (0,0) 格: 四条边都单点 bin 0 → 距离 0, 框退化为格心 (4,4); "
              "类分数 logit 0 → conf 0.5");
        CHECK(decode_outputs({o.data()}, {l}, nc, 0.6f, -1, 640).empty(),
              "DFL 阈值语义与其他支一致: conf 0.5 在阈值 0.6 下不留 (不严格大于)");
        OutputLayout lu;
        parseOutputLayout(dims4(gh, gh, attrs), lu, 0);    // 输入边长未知 → 步长 0
        CHECK(lu.stride == 0 && decode_outputs({o.data()}, {lu}, nc, 0.1f, -1, 0).empty(),
              "DFL 步长未知时不猜分布长度/步长, 整块跳过");
        CHECK(decode_outputs({o.data()}, {l}, 141, 0.1f, -1, 640).empty(),
              "类数配置与属性数不自洽 (144−141 不是 4 的倍数) 时整块跳过");
    }

    // ---------------- [6] NMS ----------------
    {
        std::cout << "[6] NMS\n";
        Detection a{5, 5, 10, 10, 0.9f, 0};   // 中心 (5,5) 10×10
        Detection b{5, 5, 10, 10, 0.8f, 0};   // 同框同类, 低分
        Detection c{7.5f, 5, 10, 10, 0.7f, 0};// 同类偏移 2.5px: IoU = 75/125 = 0.6
        Detection e{105, 5, 10, 10, 0.6f, 0}; // 同类不重叠
        Detection f{5, 5, 10, 10, 0.95f, 1};  // 异类同框
        auto r = nms({a, b, c, e, f}, 0.45f);
        CHECK(r.size() == 3, "同类同框与 IoU 0.6 的被抑制, 异类同框与不重叠的留下");
        CHECK(r[0].class_id == 0 && r[0].conf == 0.9f,
              "按类分组 (类号升序) 组内按 conf 降序: 首条是类 0 的最高分");
        bool kept_hi = false, kept_far = false;
        for (auto& x : r) { if (x.conf == 0.9f) kept_hi = true; if (x.conf == 0.6f) kept_far = true; }
        CHECK(kept_hi && kept_far, "同类里最高分与不相交的那条保留");
        auto r2 = nms({a, c}, 0.7f);
        CHECK(r2.size() == 2, "阈值放到 0.7 (IoU 0.6 < 0.7) 时两条都留");
        auto r3 = nms({}, 0.45f);
        CHECK(r3.empty(), "空输入 → 空输出");
    }

    std::cout << (g_fail ? "FAILED\n" : "ALL PASS\n");
    return g_fail ? 1 : 0;
}
