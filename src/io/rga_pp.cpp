// ============================================================================
//  rga_pp.cpp — rga_pp.h 的实现: 一次 improcess() 的中心 1:1 裁剪 (无缩放) 到
//    dma_heap 目的缓冲; 缓冲尺寸规则与几何理由全在 io/rga_pp.h 的头部。
//    复现路径 (板端, root): ./build/hdmi_probe 600
// ============================================================================

#include "io/rga_pp.h"

#include <cstddef>  // 必须在 im2d.h 之前: librga 的头用了 NULL 却没自己包含它
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <linux/dma-buf.h>
#include <linux/dma-heap.h>

#include "im2d.h"
#include "rga.h"

namespace {

void set_err(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

// 目的缓冲的 CPU 映射带 cache, RGA 是用 DMA 写的 —— 见 rga_pp.h 的"缓存一致性"段:
//   设备写完之后 CPU 这边必须先失效化, 否则读到的是上一次留在 cache 里的同一段内存。
//   (V4L2 的 mmap 是 vb2-dc 的一致内存, 同一句检查下同步前后 CPU 读 0 差异, 故源侧不需要。)
bool dmabuf_invalidate(int fd) {
    dma_buf_sync s;
    std::memset(&s, 0, sizeof(s));
    s.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) == 0;
}

bool dmabuf_read_done(int fd) {
    dma_buf_sync s;
    std::memset(&s, 0, sizeof(s));
    s.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) == 0;
}

}  // namespace

int rga_format_of(uint32_t fourcc) {
    switch (fourcc) {
        case V4L2_PIX_FMT_BGR24: return RK_FORMAT_BGR_888;
        case V4L2_PIX_FMT_RGB24: return RK_FORMAT_RGB_888;
        case V4L2_PIX_FMT_NV24:  return RK_FORMAT_YCbCr_444_SP;
        case V4L2_PIX_FMT_NV16:  return RK_FORMAT_YCbCr_422_SP;
        case V4L2_PIX_FMT_NV12:  return RK_FORMAT_YCbCr_420_SP;
        default: return -1;
    }
}

const char* rga_format_name(int rga_fmt) {
    switch (rga_fmt) {
        case RK_FORMAT_BGR_888:      return "RK_FORMAT_BGR_888";
        case RK_FORMAT_RGB_888:      return "RK_FORMAT_RGB_888";
        case RK_FORMAT_YCbCr_444_SP: return "RK_FORMAT_YCbCr_444_SP";
        case RK_FORMAT_YCbCr_422_SP: return "RK_FORMAT_YCbCr_422_SP";
        case RK_FORMAT_YCbCr_420_SP: return "RK_FORMAT_YCbCr_420_SP";
        default: return "(未知)";
    }
}

RgaPp::~RgaPp() {
    for (std::unique_ptr<Entry>& e : bufs_) {
        if (e->dst.map) munmap(e->dst.map, e->dst.bytes);
        if (e->dst.fd >= 0) ::close(e->dst.fd);
    }
    bufs_.clear();
    if (heap_fd_ >= 0) ::close(heap_fd_);
}

RgaPp::Entry* RgaPp::find(int side, uint32_t fourcc) {
    for (std::unique_ptr<Entry>& e : bufs_)
        if (e->side == side && e->fourcc == fourcc) return e.get();
    return nullptr;
}

const RgaPp::Entry* RgaPp::find(int side, uint32_t fourcc) const {
    for (const std::unique_ptr<Entry>& e : bufs_)
        if (e->side == side && e->fourcc == fourcc) return e.get();
    return nullptr;
}

const RgaDst* RgaPp::buffer(int side, uint32_t dst_fourcc) const {
    const Entry* e = find(side, dst_fourcc);
    return e ? &e->dst : nullptr;
}

RgaRect RgaPp::center_crop(int src_w, int src_h, int side) {
    RgaRect r;
    if (side <= 0 || src_w < side || src_h < side) return r;  // side = 0 = 无解
    r.side = side;
    r.x = (src_w - side) / 2;
    r.y = (src_h - side) / 2;
    return r;
}

bool RgaPp::prepare(int side, uint32_t fourcc, std::string* err) {
    if (find(side, fourcc)) return true;  // 已建: 逐帧复用同一块内存
    if (side <= 0) {
        set_err(err, "目的边长 " + std::to_string(side) + " 非法");
        return false;
    }
    if (!rga_dst_format_ok(fourcc)) {
        set_err(err, "目的格式 fourcc 0x" + std::to_string(fourcc) + " 不是本层收录的 RGB888/BGR888");
        return false;
    }
    if (heap_fd_ < 0) {
        heap_fd_ = ::open(RGA_DMA_HEAP, O_RDWR | O_CLOEXEC);
        if (heap_fd_ < 0) {
            set_err(err, std::string("打开 ") + RGA_DMA_HEAP + " 失败: " + std::strerror(errno) +
                             " (以 root 运行; 目的缓冲必须是能交给 RGA 的 dmabuf)");
            return false;
        }
    }
    Entry e;
    e.side = side;
    e.fourcc = fourcc;
    e.dst.fd = -1;
    e.dst.side = side;
    e.dst.stride_px = side;
    e.dst.bytes = (size_t)side * (size_t)side * 3;  // RGB888/BGR888 恒为 3B/px
    e.dst.fourcc = fourcc;
    dma_heap_allocation_data d;
    std::memset(&d, 0, sizeof(d));
    d.len = e.dst.bytes;
    d.fd_flags = O_RDWR | O_CLOEXEC;
    d.heap_flags = 0;
    if (ioctl(heap_fd_, DMA_HEAP_IOCTL_ALLOC, &d) < 0) {
        // 分配失败不退化: 退到 userspace 缓冲会让 RGA 每次作业多一次内核侧引脚/映射
        set_err(err, "DMA_HEAP_IOCTL_ALLOC(" + std::to_string(e.dst.bytes) + "B, 边长 " +
                         std::to_string(side) + ") 失败: " + std::strerror(errno));
        return false;
    }
    e.dst.fd = d.fd;
    e.dst.map = mmap(nullptr, e.dst.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, e.dst.fd, 0);
    if (e.dst.map == MAP_FAILED) {
        e.dst.map = nullptr;
        ::close(e.dst.fd);
        e.dst.fd = -1;
        set_err(err, std::string("mmap dma_heap 缓冲失败: ") + std::strerror(errno));
        return false;
    }
    bufs_.push_back(std::unique_ptr<Entry>(new Entry(e)));
    return true;
}

const RgaDst* RgaPp::crop_center(const RgaSrc& src, int side, uint32_t dst_fourcc, std::string* err) {
    if (!prepare(side, dst_fourcc, err)) return nullptr;
    Entry* e = find(side, dst_fourcc);
    if (!e) {
        set_err(err, "目的缓冲丢失");
        return nullptr;
    }
    const int sfmt = rga_format_of(src.fourcc);
    if (sfmt < 0) {
        set_err(err, "源格式 fourcc 0x" + std::to_string(src.fourcc) + " 不在 RGA 映射表内");
        return nullptr;
    }
    const RgaRect r = center_crop(src.width, src.height, side);
    if (r.side == 0) {
        set_err(err, "源 " + std::to_string(src.width) + "x" + std::to_string(src.height) +
                         " 装不下边长 " + std::to_string(side) + " 的 1:1 中心裁剪; 不缩放是设计约束");
        return nullptr;
    }
    // 上一帧交出去之后 CPU 读过这块缓冲: 按 dma-buf 的约定收尾 (CPU 侧从不写这些缓冲,
    //   故只有读向的 END), 再让 RGA 写
    if (e->read_pending) {
        dmabuf_read_done(e->dst.fd);
        e->read_pending = false;
    }
    // 显式 wrapbuffer_fd_t(fd, w, h, wstride, hstride, fmt): wstride/hstride 是**像素**
    //   (源侧 wstride = 驱动的 bytesperline 换算值, hstride 在单平面布局里就是高度)
    rga_buffer_t s = wrapbuffer_fd_t(src.fd, src.width, src.height, src.stride_px, src.height, sfmt);
    rga_buffer_t d = wrapbuffer_fd_t(e->dst.fd, e->dst.side, e->dst.side, e->dst.stride_px,
                                     e->dst.side, rga_format_of(e->dst.fourcc));
    im_rect srect;
    srect.x = r.x;
    srect.y = r.y;
    srect.width = side;
    srect.height = side;
    im_rect drect;
    drect.x = 0;
    drect.y = 0;
    drect.width = side;   // 与 srect 同尺寸 = 不缩放
    drect.height = side;
    const im_rect prect = {0, 0, 0, 0};  // 无 pattern
    // 7 参形式 = 同步调用 (usage = IM_SYNC): 返回即作业完成, 这正是逐帧延迟里要算的那段
    const IM_STATUS st = improcess(s, d, (rga_buffer_t){0}, srect, drect, prect, IM_SYNC);
    if (st != IM_STATUS_SUCCESS) {
        set_err(err, std::string("improcess 失败 (源 ") + rga_format_name(sfmt) + " " +
                         std::to_string(src.width) + "x" +
                         std::to_string(src.height) + " wstride " + std::to_string(src.stride_px) + "px, "
                         "中心 " + std::to_string(r.x) + "," + std::to_string(r.y) + " " + std::to_string(side) +
                         " -> " + rga_format_name(rga_format_of(e->dst.fourcc)) + " " + std::to_string(side) +
                         "x" + std::to_string(side) + "): " + imStrError(st));
        return nullptr;
    }
    // RGA 写完立刻让 CPU 侧失效化: 缓存一致性是这条缓冲契约的一部分, 不能外包给调用方
    //   (忘一次就是"截到的是上一帧", 而且只在画面动的地方看得出来)
    if (!dmabuf_invalidate(e->dst.fd)) {
        set_err(err, "目的缓冲 DMA_BUF_IOCTL_SYNC 失效化失败");
        return nullptr;
    }
    e->read_pending = true;
    return &e->dst;
}
