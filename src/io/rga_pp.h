// ============================================================================
//  rga_pp.h — RGA 裁剪/格式阶段: 采集帧 (V4L2 dmabuf) 的**中心 1:1 裁剪** →
//    dma_heap 目的缓冲 (RGA 与 CPU 共用), 每次裁剪一次 improcess()。
//
//  几何: 本库的规范帧 = 边长 CAP_SIZE (= 640, core/state.h) 的中心 1:1 裁剪 ——
//    控制律、FOV 门与标定都活在这个窗口里 (上一平台是 nvvidconv 的同边长中心裁剪),
//    截图采集存的就是它。模型输入更小 (比如 320 边长) 时看到的是这个窗口**再中心
//    1:1 裁一次** (与上一平台"第二次中心裁剪"的关系一模一样): 同一个 crop_center
//    作用于不同的源, 源是采集帧时得到窗口, 源是窗口缓冲时得到模型输入。
//    **不缩放**: 分辨率一变, 每 count 的屏幕位移就变, 于是速度倍率 spd 与标定出的
//    延迟所在的物理量都跟着变; 保持 1:1 则"屏幕像素"在整条链路里是同一个单位。
//    RGA 一次 improcess 里 srect 与 drect 同尺寸即无缩放, 这里按不变量断言。
//
//  缓冲尺寸规则 (实测, 违反即被驱动拒绝, 与性能无关):
//    * planar YUV (NV12/NV16/NV24) 的 wstride 单位是**像素**, 且 UV 平面沿用 Y 的
//      stride: 缓冲总大小 = wstride × hstride × (3/2 / 2 / 3)。按 UV stride =
//      wstride/2 紧排会被判尺寸不足 ("Only get buffer N byte ... but current image
//      required M byte")。
//    * RGB888/BGR888 的 wstride 单位也是**像素**, 总大小 = wstride × hstride × 3;
//      传 V4L2 的 bytesperline (如 7680) 会被当成 7680 像素而要 3 倍大小。
//    两条合起来就是**一条规则**: 把源几何按 V4L2 的口径给出 (bytesperline 换算成
//    像素行距, sizeimage 是整帧字节), RGA 要求的缓冲大小恰好等于驱动给的 sizeimage
//    —— 2560×1440 的四个格式都逐字节对得上, 故本层不需要任何补齐。
//
//  为什么目的缓冲走 /dev/dma_heap/system: 它给的是能交给 RGA 的 dmabuf fd, 也让
//    输出的 CPU 映射 (mmap 同一块) 与 RGA 写的是同一份物理内存 —— 截图/对照实现
//    不需要再拷一次。**分配失败是硬错误, 不退化** (退到 userspace 缓冲会让 RGA 拿到
//    虚拟地址, 每次作业多一次内核侧的引脚与映射, 延迟与正确性都不可控)。
//
//  缓存一致性 (实测踩过): /dev/dma_heap/system 的 CPU 映射是**带 cache** 的 (这也是
//    为什么有 system-uncached 这个兄弟堆), 而 RGA 用 DMA 写 —— 写完之后 CPU 读到的是
//    上一次留在 cache 里的同一段内存, 差异不是随机的而是**上一次的内容**: 实测同一块
//    目的缓冲在同源两次裁剪之间 CPU 读差异 0 (两次都读同一份陈旧 cache), 而
//    DMA_BUF_IOCTL_SYNC(START|READ) 失效化之后与源的通道交换结果**逐像素 0 差异**
//    (同步前后差 200960 字节)。所以 crop_center 在 improcess 之后立刻失效化一次目的
//    缓冲, 保证调用方紧接着读 CPU 映射拿到的是本次结果; 每块缓冲在其下一次被写之前
//    补一次 END|READ 收尾 (CPU 侧从不写这些缓冲, 故只有读向的同步)。
//    源侧不需要: V4L2 的 mmap 是 vb2-dc 的一致内存, 同一句检查下同步前后 CPU 读 0 差异。
//
//  显式 wrapbuffer_fd_t(...) 而不是可变参数的 wrapbuffer_fd(): 后者按参数个数分派
//    到不同的 wstride/hstride 组合, 参数顺序错了照样编译过 —— 裁切几何上不该有这种
//    静默的自由度。
//
//  ---- 板端实测 (命令: sudo ./build/hdmi_probe 600; 源 2560×1440 BGR3 活动信号) ----
//  每帧一次 improcess 的同步耗时 (含目的缓冲失效化那一步; 前 3 帧预热不计):
//    | 目的                    | p50   | p90   | p99   | max   | 每帧占比 (120fps 8.33ms) |
//    |-------------------------|-------|-------|-------|-------|--------------------------|
//    | 640 窗口 RGB888 (模型)  | 560.6 | 596.5 | 680.5 | 690.1 | 6.7% (p99 8.2%)          |
//    | 640 窗口 BGR888 (截图)  | 559.1 | 595.9 | 669.7 | 709.0 | 6.7% (p99 8.0%)          |
//    | 320 窗口 RGB888 (二级)  | 279.7 | 318.2 | 339.2 | 402.2 | 3.4%                     |
//    (单位 µs。同一几何在不同运行间 p50 漂移 ±15% (640 窗口落在 428–561µs), 与失效化的
//     cache 维护量及同机其它负载有关; 生产路径每帧只有**一次**裁剪, 即上表第一行。)
//  成本随"读多少源像素"走而不随输出走 (源 11059200B 里取 640×640), 三处几何同源同码:
//  二级裁剪读的是 640 窗口那块小缓冲, 所以它比一级便宜一半。
//
//  逐字节正确性 (同一命令的 [对照] 段, 与 mmap 采集缓冲上写的纯 CPU 参照逐字节比对):
//    640 RGB888 (源 BGR3 → 换通道) 0 / 1228800 差异
//    640 BGR888 (源 BGR3 → 纯拷贝) 0 / 1228800 差异
//    320 RGB888 (自 640 窗口再中心裁 (160,160) 320×320, 纯拷贝) 0 / 307200 差异
//  即: 中心矩形**(960,400) 起 640×640** 就是源帧的中心, 无缩放, 无偏移; 二级裁剪的源
//  已是 RGB, 故它是纯拷贝 (RGA 的 RGB→RGB 不做通道交换, BGR→RGB 才做)。落图三处又可
//  人眼复核: 裁剪 PNG 与原始帧 PNG 的同一区域**逐像素 0 差异 (0/409600)**。
//  通道落点: 矩形内 |R−B| 最大的样本 (源坐标 1227,400) 源字节 B,G,R = 12,9a,db →
//  目的 RGB888 字节 = db,9a,12 —— 源的第一通道 (R) 落在目的的第 0 字节 (R); 该像素在
//  落下的 PNG 里是 RGB(219,154,18) 的琥珀色, 与原始帧同一位置的像素逐位相同。
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <linux/videodev2.h>

#include "core/state.h"  // CAP_SIZE (规范窗口边长)

// dma_heap 设备: 全是 /dev/dma_heap/system (通用 CMA/system 堆, 板上实测可选; 它的
//   兄弟 system-uncached 只改 cache 属性, 对 RGA 无影响, 不引入)
constexpr const char* RGA_DMA_HEAP = "/dev/dma_heap/system";

// RGA 目的缓冲的几何上限外的约束: 目的边长必须 ≤ 源边长 (裁剪不缩放), 且源/目的
//   行距都是像素单位。这里只声明"本库会用到的目的格式", 用 V4L2 的 fourcc 表达
//   (源与目的同一套词汇, 映射函数只有一个)。
inline bool rga_dst_format_ok(uint32_t fourcc) {
    return fourcc == V4L2_PIX_FMT_RGB24 || fourcc == V4L2_PIX_FMT_BGR24;
}

// V4L2 fourcc → RGA 格式 (RK_FORMAT_*); 未收录返回 -1。收录范围 = 驱动能给的四源
//   格式 + 本库要用的两目格式 (见文件头"缓冲尺寸规则")。
int rga_format_of(uint32_t fourcc);
const char* rga_format_name(int rga_fmt);

// 源几何: 逐帧来自 io/hdmi_in 的 HdmiFrame (fd + 像素行距 + 四字符码)
struct RgaSrc {
    int      fd        = -1;
    int      width     = 0;
    int      height    = 0;
    int      stride_px = 0;  // 像素行距 (= RGA 的 wstride)
    uint32_t fourcc    = 0;
};

// 目的缓冲: 一块 dma_heap/system 的 dmabuf, RGA 写它, CPU 经 map 读它。
struct RgaDst {
    int      fd        = -1;
    void*    map       = nullptr;
    int      side      = 0;
    int      stride_px = 0;   // == side (行距不补齐: 目的就是 RGA 的输出布局)
    size_t   bytes     = 0;   // == stride_px × side × 3
    uint32_t fourcc    = 0;
};

// 一次裁剪的矩形 (源域)
struct RgaRect {
    int x = 0, y = 0, side = 0;
};

class RgaPp {
public:
    RgaPp() = default;
    ~RgaPp();
    RgaPp(const RgaPp&) = delete;
    RgaPp& operator=(const RgaPp&) = delete;

    // 中心 1:1 裁剪矩形: 源中心取 side×side; 源任一边小于 side 时无解 (返回 side=0)
    static RgaRect center_crop(int src_w, int src_h, int side);

    // 把 src 的中心 side×side 裁到 (side, dst_fourcc) 的目的缓冲 (无缩放), 返回它的
    //   只读视图 (缓冲按 (side, fourcc) 建一次, 之后逐帧复用同一块内存), 失败返回
    //   nullptr 并写 err (RGA 自己的错误文字一并带上)。
    const RgaDst* crop_center(const RgaSrc& src, int side, uint32_t dst_fourcc,
                              std::string* err = nullptr);

    // 已建立的目的缓冲 (crop_center 之前/之后都能取)
    const RgaDst* buffer(int side, uint32_t dst_fourcc) const;
    size_t buffers() const { return bufs_.size(); }

private:
    struct Entry {
        int side = 0;
        uint32_t fourcc = 0;
        RgaDst dst;
        bool read_pending = false;  // 本次写之后 CPU 侧已失效化, 等下一次写前收尾
    };
    Entry* find(int side, uint32_t fourcc);
    const Entry* find(int side, uint32_t fourcc) const;
    bool prepare(int side, uint32_t fourcc, std::string* err);
    int heap_fd_ = -1;      // /dev/dma_heap/system (首次 prepare 时打开)
    // 缓冲表按 (边长, 格式) 建一次后逐帧复用。存 unique_ptr 而不是 Entry 本体: 调用方
    //   拿着 crop_center 返回的指针用 (二级裁剪的源就是上一级的落点), 容器扩容不能把
    //   那些指针搬走 —— 地址稳定是这条 API 的一部分。
    std::vector<std::unique_ptr<Entry>> bufs_;
};
