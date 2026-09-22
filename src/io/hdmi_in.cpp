// ============================================================================
//  hdmi_in.cpp — hdmi_in.h 的实现: 裸 V4L2 MULTIPLANAR 取帧 (mmap + EXPBUF),
//    最新帧语义的交付, timing 锁定门, low_latency 前置位, 失锁/断流的判定与重建。
//    设计理由与实测口径全在 io/hdmi_in.h 的头部; 本文件只放"怎么做"。
//
//  复现路径 (板端, root):
//    * 取帧/交付/裁剪对照: ./build/hdmi_probe 600
//    * 失锁与重建: 给接收器写一份**别的** EDID 会让它落下 HPD, 源随之重新协商 ——
//        sudo v4l2-ctl -d /dev/video0 --set-edid=pad=0,file=/usr/lib/firmware/edid/1920x1080.bin,format=raw
//      源切到新模式的瞬间接收器失锁、驱动自己停流; 恢复部署 EDID 的做法见
//      /usr/local/sbin/hdmirx-set-edid.sh (它开机时把 /usr/local/share/hdmirx/1440p120.edid
//      写进接收器, 是本机 EDID 的权威来源):
//        sudo v4l2-ctl -d /dev/video0 --set-edid=pad=0,file=/usr/local/share/hdmirx/1440p120.edid,format=raw
//      aimbot 侧看 [HDMI] 的重建行与 [AI FPS] 行的增量段 (等 fence/失锁/重建), 探针侧看
//      build/hdmi_probe 的结果段 (同一批判据与账)。
// ============================================================================

#include "io/hdmi_in.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

namespace {

// low_latency 模块参数节点 (0664 root:root; 驱动只在 STREAMON 时读)
const char* kLowLatencyPath = "/sys/module/rockchip_hdmirx/parameters/low_latency";

void set_err(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

std::string errno_text() { return std::string(std::strerror(errno)); }

int64_t now_us() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000LL + t.tv_nsec / 1000;
}

// low_latency 是内核的 bool 模块参数: sysfs 读回是 "Y"/"N" (param_get_bool 的打印形式),
//   写入接受 "1"/"0"/"y"/"n"/"Y"/"N"。判"已置位"要认这两种形式 —— 把内核的正确回读
//   ("Y") 当成失败是错的。
bool low_latency_on(const std::string& s) { return s == "Y" || s == "y" || s == "1"; }

// ioctl 包装: EINTR 重试 (信号随时可能到, 不该把它变成设备错误)
int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r < 0 && errno == EINTR);
    return r;
}

// 读一个小文本文件的首行并去掉结尾空白
bool read_line(const std::string& path, std::string* out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    char buf[4096] = {0};
    const bool ok = std::fgets(buf, sizeof(buf), f) != nullptr;
    std::fclose(f);
    if (!ok) return false;
    size_t n = std::strlen(buf);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ')) buf[--n] = 0;
    *out = buf;
    return true;
}

}  // namespace

std::string HdmiIn::fourcc_name(uint32_t f) {
    char s[5] = {static_cast<char>(f & 0xff), static_cast<char>((f >> 8) & 0xff),
                 static_cast<char>((f >> 16) & 0xff), static_cast<char>((f >> 24) & 0xff), 0};
    return std::string(s);
}

std::string HdmiIn::timing_text() const {
    char b[256];
    const uint32_t tw = timing_.width + timing_.hfrontporch + timing_.hsync + timing_.hbackporch;
    const uint32_t th = timing_.height + timing_.vfrontporch + timing_.vsync + timing_.vbackporch;
    std::snprintf(b, sizeof(b), "%ux%u @ %.2fHz (总 %ux%u, 像素时钟 %llu Hz)", timing_.width,
                  timing_.height, frame_hz(), tw, th, (unsigned long long)timing_.pixelclock);
    return std::string(b);
}

double HdmiIn::frame_hz() const {
    // 帧率由**总**尺寸导出 (含消隐): 有效区尺寸 × 像素时钟 得到的是像素率, 不是帧率
    const uint32_t tw = timing_.width + timing_.hfrontporch + timing_.hsync + timing_.hbackporch;
    const uint32_t th = timing_.height + timing_.vfrontporch + timing_.vsync + timing_.vbackporch;
    if (timing_.pixelclock == 0 || tw == 0 || th == 0) return 0.0;
    return (double)timing_.pixelclock / ((double)tw * (double)th);
}

// ============================== 设备解析 ==============================
std::string HdmiIn::resolve_device(const std::string& spec, std::string* err) {
    // 显式给节点就原样用: 调用方(或调试者)知道自己在指哪块
    if (!spec.empty()) return spec;

    const char* kDir = "/sys/class/video4linux";
    DIR* d = opendir(kDir);
    if (!d) {
        set_err(err, std::string("打不开 ") + kDir + ": " + errno_text());
        return std::string();
    }
    std::vector<std::string> hits;
    while (dirent* e = readdir(d)) {
        if (std::strncmp(e->d_name, "video", 5) != 0) continue;
        std::string name;
        if (!read_line(std::string(kDir) + "/" + e->d_name + "/name", &name)) continue;
        if (name.find(HDMIRX_NAME_SUBSTR) != std::string::npos) hits.push_back("/dev/" + std::string(e->d_name));
    }
    closedir(d);
    std::sort(hits.begin(), hits.end());  // readdir 顺序不定, 排序保证确定性
    if (hits.empty()) {
        set_err(err, std::string("没有名字含 \"") + HDMIRX_NAME_SUBSTR + "\" 的采集节点 (扫 " + kDir +
                       "/video*/name); 接收到信号时它应显示为 stream_hdmirx");
        return std::string();
    }
    if (hits.size() > 1) {
        std::string m = "多个节点都匹配 \"" + std::string(HDMIRX_NAME_SUBSTR) + "\":";
        for (const auto& h : hits) m += " " + h;
        set_err(err, m);
        return std::string();
    }
    return hits[0];
}

// ============================== 低延迟前置位 ==============================
bool HdmiIn::set_low_latency(std::string* err) {
    std::string cur;
    if (!read_line(kLowLatencyPath, &cur)) {
        set_err(err, std::string("读不到 ") + kLowLatencyPath + " (rockchip_hdmirx 未加载?): " + errno_text());
        return false;
    }
    if (!low_latency_on(cur)) {
        FILE* f = std::fopen(kLowLatencyPath, "w");
        if (!f) {
            set_err(err, std::string("写 ") + kLowLatencyPath + " 失败 (" + errno_text() +
                           ") — 低延迟采集是硬要求, 且驱动只在 STREAMON 时读它一次; 以 root 运行");
            return false;
        }
        const int n = std::fputs("1\n", f);
        std::fclose(f);
        if (n < 0) {
            set_err(err, std::string("写 ") + kLowLatencyPath + " 失败");
            return false;
        }
    }
    // 回读校验: 写完只看进程自己的返回值不足以说明驱动拿到的是 1
    std::string after;
    if (!read_line(kLowLatencyPath, &after) || !low_latency_on(after)) {
        set_err(err, std::string(kLowLatencyPath) + " 回读 = \"" + after + "\" (期望 Y/1)");
        return false;
    }
    low_latency_ = 1;
    if (!low_latency_announced_) {   // 重建会重开一次会话, 这句话不必每次重复
        low_latency_announced_ = true;
        std::printf("[HDMI] 低延迟: low_latency=%s (%s, 驱动在 STREAMON 时读取; 帧完成中断取 line_flag "
                    "且跳过 6 帧预热)\n",
                    after.c_str(), kLowLatencyPath);
    }
    return true;
}

// ============================== 等 timing 锁定 ==============================
bool HdmiIn::query_lock(std::string* err) {
    if (fd_ < 0) {
        set_err(err, "设备未打开");
        return false;
    }
    v4l2_dv_timings t;
    std::memset(&t, 0, sizeof(t));
    if (xioctl(fd_, VIDIOC_QUERY_DV_TIMINGS, &t) < 0) {
        set_err(err, errno_text());
        return false;
    }
    timing_ = t.bt;
    return true;
}

bool HdmiIn::wait_timing_lock(std::string* err) {
    std::string last;
    for (int waited = 0; waited <= HDMI_LOCK_WAIT_MS; waited += HDMI_LOCK_POLL_MS) {
        if (query_lock(&last)) {
            std::printf("[HDMI] 时序锁定: %s\n", timing_text().c_str());
            return true;
        }
        if (waited < HDMI_LOCK_WAIT_MS) usleep(HDMI_LOCK_POLL_MS * 1000);
    }
    set_err(err, "VIDIOC_QUERY_DV_TIMINGS 在 " + std::to_string(HDMI_LOCK_WAIT_MS) +
                     "ms 内未锁定 (驱动最后报: " + last +
                     "); 未接源时是 \"Link has been severed\", 接了但 TMDS 未锁是 \"No locks available\"");
    return false;
}

// ============================== 缓冲建立 ==============================
bool HdmiIn::queue_buffer(int index) {
    v4l2_plane planes[VIDEO_MAX_PLANES];
    std::memset(planes, 0, sizeof(planes));
    v4l2_buffer b;
    std::memset(&b, 0, sizeof(b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = index;
    b.length = VIDEO_MAX_PLANES;
    b.m.planes = planes;
    return xioctl(fd_, VIDIOC_QBUF, &b) == 0;
}

// ============================== 打开 / 重建 ==============================
bool HdmiIn::open(const std::string& spec, int num_buffers, HdmiDelivery delivery, std::string* err) {
    close();
    delivery_ = delivery;
    num_buffers_ = num_buffers;
    spec_ = spec;
    if (num_buffers < HDMI_BUF_MIN || num_buffers > HDMI_BUF_MAX) {
        set_err(err, "缓冲块数 " + std::to_string(num_buffers) + " 超出 [" + std::to_string(HDMI_BUF_MIN) +
                         ", " + std::to_string(HDMI_BUF_MAX) + "]");
        return false;
    }
    dev_path_ = resolve_device(spec, err);
    if (dev_path_.empty()) return false;
    return arm(err);
}

// 一次采集会话的建立。设备解析不在这里 (open 解析一次, rearm 复用): 节点号在一次启动内
//   是稳定的, 跨重启才不稳定 (见 io/hdmi_in.h)。
bool HdmiIn::arm(std::string* err) {
    if (dev_path_.empty()) {
        set_err(err, "没有设备路径 (先 open 一次)");
        return false;
    }
    if (fd_ < 0) {
        // O_NONBLOCK: DQBUF 用非阻塞 + poll 等待 —— 排空队列那一步必须有"现在没有更多了"
        //   这个明确的返回 (EAGAIN), 阻塞式 DQBUF 表达不了它
        fd_ = ::open(dev_path_.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd_ < 0) {
            set_err(err, "打开 " + dev_path_ + " 失败: " + errno_text());
            return false;
        }

        v4l2_capability cap;
        std::memset(&cap, 0, sizeof(cap));
        if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
            set_err(err, "VIDIOC_QUERYCAP 失败: " + errno_text());
            close();
            return false;
        }
        driver_ = reinterpret_cast<const char*>(cap.driver);
        // 名字解析的复核: 节点自述的驱动名也要对得上 (两处一致才认)
        if (driver_.find(HDMIRX_NAME_SUBSTR) == std::string::npos) {
            set_err(err, dev_path_ + " 的驱动是 \"" + driver_ + "\", 不含 \"" + HDMIRX_NAME_SUBSTR + "\"");
            close();
            return false;
        }
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) || !(cap.capabilities & V4L2_CAP_STREAMING)) {
            set_err(err, dev_path_ + " 不具备 MULTIPLANAR 采集 + 流能力 (capabilities 0x" +
                           std::to_string(cap.capabilities) + ")");
            close();
            return false;
        }
        std::printf("[HDMI] 设备 %s (驱动 %s, %s)\n", dev_path_.c_str(), driver_.c_str(),
                    reinterpret_cast<const char*>(cap.bus_info));
    }

    // ---- 起流前先确已锁定, 锁定之后才读格式 ----
    // 顺序是硬的: 缓冲尺寸由**锁定后的**格式导出, 而模式变化会同时改分辨率与帧率
    // (见 io/hdmi_in.h 的"失锁/断流与重建"); 首次打开与重建共用这一条。
    if (!wait_timing_lock(err)) {
        close();
        return false;
    }

    // ---- 格式: 只读, 不设置 (格式由信号决定, S_FMT 对任何别的格式都是 EINVAL) ----
    v4l2_format f;
    std::memset(&f, 0, sizeof(f));
    f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(fd_, VIDIOC_G_FMT, &f) < 0) {
        set_err(err, "VIDIOC_G_FMT 失败: " + errno_text());
        close();
        return false;
    }
    fmt_.fourcc = f.fmt.pix_mp.pixelformat;
    fmt_.width = (int)f.fmt.pix_mp.width;
    fmt_.height = (int)f.fmt.pix_mp.height;
    fmt_.bpl_bytes = (int)f.fmt.pix_mp.plane_fmt[0].bytesperline;
    fmt_.plane_bytes = (size_t)f.fmt.pix_mp.plane_fmt[0].sizeimage;

    // RGA 的 wstride 是**像素**单位, 故把驱动的字节行距换算成像素: RGB 系 ÷3, 平面 YUV
    //   的行距本身就是像素 (UV 平面沿用 Y 的 stride, 见文件头)。RGB 系下 bpl 必须是 3 的
    //   整数倍 —— 驱动的 bpl = ALIGN(width*3, 64) 在 width*3 不是 64 倍数时只按 64 取整,
    //   此时 bpl 可能不被 3 整除, 那是个无法用像素行距表达的布局, 明确报错而不是凑一个数
    //   (实机 2560/1920/1280/640 宽下 bpl 都能被 3 整除)。
    switch (fmt_.fourcc) {
        case V4L2_PIX_FMT_BGR24:
            if (fmt_.bpl_bytes % 3 != 0) {
                set_err(err, "BGR3 的 bytesperline " + std::to_string(fmt_.bpl_bytes) +
                                 " 不是 3 的整数倍, 无法表达成像素行距");
                close();
                return false;
            }
            fmt_.stride_px = fmt_.bpl_bytes / 3;
            break;
        case V4L2_PIX_FMT_NV24:
        case V4L2_PIX_FMT_NV16:
        case V4L2_PIX_FMT_NV12:
            fmt_.stride_px = fmt_.bpl_bytes;
            break;
        default:
            set_err(err, "不支持的采集格式 '" + fourcc_name(fmt_.fourcc) + "' (驱动的格式表是 BGR3/NV24/NV16/NV12)");
            close();
            return false;
    }
    if (fmt_.width <= 0 || fmt_.height <= 0 || fmt_.bpl_bytes <= 0 || fmt_.plane_bytes == 0) {
        set_err(err, "G_FMT 给出无效几何");
        close();
        return false;
    }
    std::printf("[HDMI] 格式 '%s' %dx%d 行距 %dB (%dpx) sizeimage %zuB (驱动按 ALIGN(w×bpp/8,64) 给, 不重算)\n",
                fourcc_name(fmt_.fourcc).c_str(), fmt_.width, fmt_.height, fmt_.bpl_bytes,
                fmt_.stride_px, fmt_.plane_bytes);

    // ---- 低延迟: 必须在 STREAMON 之前置位 (驱动只在 STREAMON 读一次) ----
    if (!set_low_latency(err)) {
        close();
        return false;
    }

    // ---- 缓冲: REQBUFS(MMAP) → 逐块 QUERYBUF + mmap + EXPBUF ----
    v4l2_requestbuffers req;
    std::memset(&req, 0, sizeof(req));
    req.count = (uint32_t)num_buffers_;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        set_err(err, "VIDIOC_REQBUFS 失败: " + errno_text());
        close();
        return false;
    }
    if ((int)req.count < HDMI_BUF_MIN) {
        set_err(err, "驱动只给了 " + std::to_string(req.count) + " 块缓冲 (< " + std::to_string(HDMI_BUF_MIN) + ")");
        close();
        return false;
    }
    bufs_.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_plane planes[VIDEO_MAX_PLANES];
        std::memset(planes, 0, sizeof(planes));
        v4l2_buffer b;
        std::memset(&b, 0, sizeof(b));
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = i;
        b.length = VIDEO_MAX_PLANES;
        b.m.planes = planes;
        if (xioctl(fd_, VIDIOC_QUERYBUF, &b) < 0) {
            set_err(err, "VIDIOC_QUERYBUF(" + std::to_string(i) + ") 失败: " + errno_text());
            close();
            return false;
        }
        HdmiBuffer& buf = bufs_[i];
        buf.index = (int)i;
        buf.length = (size_t)planes[0].length;
        // 四种格式在 V4L2 里都是单平面 (平面 YUV 的 UV 与 Y 同处一个缓冲), 故只映射
        //   plane[0]; 平面长度仍读出来与 G_FMT 的 sizeimage 对账, 免得布局变了还照搬
        if (planes[0].length != fmt_.plane_bytes) {
            std::printf("[HDMI] 注意: 块 %u 的 plane[0].length=%u 与 G_FMT 的 sizeimage %zu 不一致\n", i,
                        b.m.planes[0].length, fmt_.plane_bytes);
        }
        buf.map = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, planes[0].m.mem_offset);
        if (buf.map == MAP_FAILED) {
            buf.map = nullptr;
            set_err(err, "mmap(块 " + std::to_string(i) + ") 失败: " + errno_text());
            close();
            return false;
        }
        v4l2_exportbuffer exp;
        std::memset(&exp, 0, sizeof(exp));
        exp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        exp.index = i;
        exp.plane = 0;
        exp.flags = O_RDWR | O_CLOEXEC;
        if (xioctl(fd_, VIDIOC_EXPBUF, &exp) < 0) {
            set_err(err, "VIDIOC_EXPBUF(块 " + std::to_string(i) + ") 失败: " + errno_text() +
                             " — 没有 dmabuf 就没有零拷贝通路");
            close();
            return false;
        }
        buf.fd = exp.fd;
    }
    std::printf("[HDMI] %zu 块缓冲 (MMAP %zuB/块 + EXPBUF dmabuf fd 全部导出)\n", bufs_.size(), bufs_[0].length);

    for (size_t i = 0; i < bufs_.size(); ++i) {
        if (!queue_buffer(bufs_[i].index)) {
            set_err(err, "QBUF(" + std::to_string(i) + ") 失败: " + errno_text());
            close();
            return false;
        }
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        set_err(err, "VIDIOC_STREAMON 失败: " + errno_text() +
                         " (未锁定 timing 时驱动报 \"timing is invalid\")");
        close();
        return false;
    }
    streaming_ = true;
    return true;
}

// 拆掉一次采集会话, 但**留着 fd**: 停流 → 释放缓冲 (munmap → 关 dmabuf → REQBUFS(0))。
//   流已经死了的时候每一步都可能失败 (驱动自己停过流, 队列也可能已在 error 态), 一律容忍:
//   这一步的目标是把设备恢复成"没被打开过的样子", 而真正的恢复路径是随后的重新 arm。
//   万一 REQBUFS(0) 也没成功, 设备关闭时内核照样把缓冲收回 (fd 才是唯一的所有者)。
void HdmiIn::unstream() {
    if (fd_ >= 0 && streaming_) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    streaming_ = false;
    hold_index_ = -1;
    // 先解映射再释放 (内核要求释放时缓冲没有被映射), dmabuf fd 是内核给的 dup, 一并关掉
    const bool had_bufs = !bufs_.empty();
    for (HdmiBuffer& b : bufs_) {
        if (b.map) munmap(b.map, b.length);
        if (b.fd >= 0) ::close(b.fd);
        b.map = nullptr;
        b.fd = -1;
    }
    bufs_.clear();
    if (had_bufs && fd_ >= 0) {
        v4l2_requestbuffers req;
        std::memset(&req, 0, sizeof(req));
        req.count = 0;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        req.memory = V4L2_MEMORY_MMAP;
        xioctl(fd_, VIDIOC_REQBUFS, &req);
    }
    seq_valid_ = false;
    ts_checked_ = false;
    fence_fail_streak_ = 0;
}

// ============================== 关流 ==============================
void HdmiIn::close() {
    unstream();
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

// ============================== 重建 ==============================
// 失锁/断流之后的恢复路径 (判据与顺序见 io/hdmi_in.h 的"失锁/断流与重建")。
//   失败时设备处于关闭状态, 再调一次就是一次完整的新尝试 —— 调用方按 HDMI_REARM_MIN_MS
//   的节拍重试, 于是"源还没上电/还在换模式"与"设备真坏了"在行为上一致: 继续等。
bool HdmiIn::rearm(std::string* err) {
    const auto t0 = std::chrono::steady_clock::now();
    const HdmiFormat before = fmt_;
    const double hz_before = frame_hz();

    if (dev_path_.empty()) {   // open 从未成功过: 把设备解析补上 (rearm 可独立使用)
        dev_path_ = resolve_device(spec_, err);
        if (dev_path_.empty()) {
            ++rearm_fail_;
            return false;
        }
    }

    if (fd_ >= 0) {
        std::printf("[HDMI] 重建: 拆掉当前会话 (STREAMOFF 容错 → munmap → 关 dmabuf → REQBUFS(0)) "
                    "并重新打开 %s\n",
                    dev_path_.c_str());
        std::fflush(stdout);
        unstream();
        ::close(fd_);
        fd_ = -1;
    }
    if (!arm(err)) {          // arm 失败时内部已 close()
        ++rearm_fail_;
        return false;
    }
    ++rearm_ok_;
    last_rearm_ms_ = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
    std::printf("[HDMI] 重建成功: 恢复耗时 %.0fms — %s, %zu 块缓冲已入队并起流 "
                "(断流前 %dx%d @ %.2fHz)\n",
                last_rearm_ms_, timing_text().c_str(), bufs_.size(), before.width, before.height,
                hz_before);
    std::fflush(stdout);
    return true;
}

HdmiIn::~HdmiIn() { close(); }

// ============================== 取帧 ==============================
// 单次非阻塞取帧: 0 = 取到, 1 = 现在没有, -1 = 错误(写 err)。
//   ts_type / err_flag 是驱动在缓冲 flags 里给的两件事, 不塞进对外的 HdmiFrame。
int HdmiIn::dequeue(HdmiFrame* f, uint32_t* ts_type, bool* err_flag, std::string* err) {
    v4l2_plane planes[VIDEO_MAX_PLANES];
    std::memset(planes, 0, sizeof(planes));
    v4l2_buffer b;
    std::memset(&b, 0, sizeof(b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    b.memory = V4L2_MEMORY_MMAP;
    b.length = VIDEO_MAX_PLANES;
    b.m.planes = planes;
    if (xioctl(fd_, VIDIOC_DQBUF, &b) < 0) {
        if (errno == EAGAIN) return 1;
        set_err(err, "VIDIOC_DQBUF 失败: " + errno_text());
        return -1;
    }
    if (b.index >= bufs_.size()) {
        set_err(err, "DQBUF 给出越界下标 " + std::to_string(b.index));
        return -1;
    }
    const HdmiBuffer& buf = bufs_[b.index];
    f->index = (int)b.index;
    f->fd = buf.fd;
    f->map = buf.map;
    f->fourcc = fmt_.fourcc;
    f->width = fmt_.width;
    f->height = fmt_.height;
    f->stride_px = fmt_.stride_px;
    f->plane_bytes = fmt_.plane_bytes;
    f->ts_ns = (int64_t)b.timestamp.tv_sec * 1000000000LL + (int64_t)b.timestamp.tv_usec * 1000LL;
    f->sequence = b.sequence;
    // low_latency 时驱动把本缓冲的 dma-fence 挂在 timecode.userbits 上 (4 字节小端 fd,
    //   0xffffffff = 无)。fence 才是"这一帧已经写完"的凭据 —— 何时可以读缓冲见 io/hdmi_in.h。
    int fence;
    std::memcpy(&fence, b.timecode.userbits, sizeof(fence));
    f->fence_fd = (fence == -1) ? -1 : fence;
    *ts_type = b.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK;
    *err_flag = (b.flags & V4L2_BUF_FLAG_ERROR) != 0;
    return 0;
}

bool HdmiIn::wait_frame(HdmiFrame* out, std::string* err, HdmiFail* fail) {
    set_fail(fail, HdmiFail::Ok);
    if (out == nullptr) {
        set_err(err, "wait_frame: out 为空");
        set_fail(fail, HdmiFail::Error);
        return false;
    }
    if (!streaming_) {
        set_err(err, "wait_frame: 未起流 (流断过之后要先 rearm)");
        set_fail(fail, HdmiFail::NoStream);
        return false;
    }
    // 最新帧语义: 手上那帧在被取代之后才归还 —— 它已经不是"最新"了
    if (delivery_ == HdmiDelivery::Newest && hold_index_ >= 0) {
        queue_buffer(hold_index_);
        ++superseded_;
        hold_index_ = -1;
    }

    for (int waited = 0, empty_rounds = 0;;) {
        pollfd p;
        p.fd = fd_;
        p.events = POLLIN;
        p.revents = 0;
        const int pr = poll(&p, 1, HDMI_POLL_SLICE_MS);
        if (pr < 0) {
            if (errno == EINTR) {   // 信号立刻可能到 (退出路径): 不重建, 交给调用方看 global_running
                set_err(err, "poll 被信号打断");
                set_fail(fail, HdmiFail::Interrupted);
                return false;
            }
            set_err(err, std::string("poll 失败: ") + errno_text());
            set_fail(fail, HdmiFail::Error);
            return false;
        }
        if (pr > 0 && (p.revents & (POLLERR | POLLHUP | POLLNVAL))) {
            char b[64];
            std::snprintf(b, sizeof(b), "0x%x", p.revents);
            set_err(err, std::string("poll 报设备异常 revents=") + b);
            set_fail(fail, HdmiFail::Error);
            return false;
        }
        if (pr > 0) {
            // 排空: 只要还有已完成的缓冲就接着取, 只留最新的一帧, 其余当场归还 ——
            //   消费者比信号慢时这样"交出来的帧"始终贴近当前, 而不是队列头那帧
            HdmiFrame newest{};
            bool have = false;
            for (;;) {
                HdmiFrame f{};
                uint32_t ts_type = 0;
                bool err_flag = false;
                const int r = dequeue(&f, &ts_type, &err_flag, err);
                if (r < 0) {
                    set_fail(fail, HdmiFail::Error);
                    return false;
                }
                if (r == 1) break;
                if (err_flag) {
                    // 驱动以 VB2_BUF_STATE_ERROR 归还的缓冲 (停流路径) 没有有效载荷
                    ++invalid_;
                    queue_buffer(f.index);
                    continue;
                }
                if (!ts_checked_) {
                    ts_checked_ = true;
                    if (ts_type != V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
                        set_err(err, "该队列的时间戳类型不是 MONOTONIC (flags=0x" + std::to_string(ts_type) +
                                         "), 交付延迟与帧率统计的口径不成立");
                        queue_buffer(f.index);
                        set_fail(fail, HdmiFail::Error);
                        return false;
                    }
                    if (!ts_announced_) {
                        ts_announced_ = true;
                        std::printf("[HDMI] 时间戳基准 = CLOCK_MONOTONIC (队列声明 "
                                    "V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC)\n");
                    }
                }
                if (have) {
                    // 被新帧取代: 当场归还, 它那份 fence 也不再需要 (fd 是内核给的 dup, 得关)
                    close_fence(newest.fence_fd);
                    queue_buffer(newest.index);
                    ++superseded_;
                }
                newest = f;
                have = true;
                if (delivery_ == HdmiDelivery::QueueOrder) break;  // 对照路径: 一次只取一帧
            }
            if (!have) {
                // poll 说有帧却一帧都取不到: 再来一轮 (队列状态与 poll 不一致, 有界重试)
                if (++empty_rounds > 3) {
                    set_err(err, "poll 连续返回可读但 DQBUF 取不到帧");
                    set_fail(fail, HdmiFail::Error);
                    return false;
                }
                continue;
            }

            // 序号缺口 = 驱动少交付的帧 (队列空时驱动按 drop the frame! 丢帧, 不重发旧帧)
            if (seq_valid_) {
                const uint32_t d = newest.sequence - last_seq_;
                if (d > 1) lost_ += (d - 1);
            }
            seq_valid_ = true;
            last_seq_ = newest.sequence;

            // 载荷写完的凭据 = fence (见 io/hdmi_in.h): 等不到就不交出去 —— 半张画面进了下游
            //   比这里报错糟得多 (标定会把"还在写的那几行"当成真实位移)。
            //   **但缓冲必须当场归还**: 驱动的丢帧路径 (line_flag 中断里找不到下一个缓冲就
            //   drop the frame!, 不重发旧帧) 会给该缓冲留下一个永不再 signal 的 fence, 于是
            //   慢消费者跑长一点就必然遇到它; 不归还是永久少一块 —— 4 块里漏 3 块之后驱动就
            //   再没有缓冲可写 (它自己要求 ≥2 块), 流从此一帧都没有 (长跑实测: 前 3 次超时
            //   都还活着, 第 4 次之后 poll 再也不返回)。归还的是一块过了时限的旧帧, 内容本来
            //   就要丢。
            if (!wait_fence(newest.fence_fd, err)) {
                newest.fence_fd = -1;      // wait_fence 两条路都已关掉这个内核给的 dup
                queue_buffer(newest.index);
                const std::string fence_err = err ? *err : std::string("帧的 dma-fence 没有 signal");
                ++fence_fail_streak_;
                // 载荷没写完这件事本身有三种去向 (判据与出处见 io/hdmi_in.h): 接收器已经不在
                //   锁上 = 断流的第一个症状 (不必等下一帧); 锁定但连续到线 = 流不产数据;
                //   否则就是丢了一帧 (驱动的下一次 DQBUF 会把上一根 fence signal 掉)。
                std::string lerr;
                if (!query_lock(&lerr)) {
                    ++lock_lost_;
                    set_err(err, fence_err + " — 且接收器未锁定 (" + lerr + "): 断流, 需要重建");
                    set_fail(fail, HdmiFail::LockLost);
                } else if (hdmi_fence_streak_is_loss(fence_fail_streak_)) {
                    ++stalled_;
                    set_err(err, fence_err + " — 连续 " + std::to_string(fence_fail_streak_) +
                                     " 帧如此而接收器仍锁定: 流不产数据, 需要重建");
                    set_fail(fail, HdmiFail::Stalled);
                } else {
                    set_fail(fail, HdmiFail::Fence);
                }
                return false;
            }
            newest.fence_fd = -1;   // 已关, 不留给调用方
            fence_fail_streak_ = 0;

            hold_index_ = newest.index;
            ++delivered_;
            *out = newest;
            return true;
        }

        // 本片内没有帧: 先问接收器还在不在锁上 —— "信号掉了"只有这一个快速可观测量
        //   (见 io/hdmi_in.h)。它失败就是断流, 察觉延迟上限 = 一片而不是整个预算。
        waited += HDMI_POLL_SLICE_MS;
        std::string lerr;
        if (!query_lock(&lerr)) {
            ++lock_lost_;
            set_err(err, "poll " + std::to_string(HDMI_POLL_SLICE_MS) + "ms 内无帧且接收器未锁定 (" +
                             lerr + "): 信号/流断了, 需要重建");
            set_fail(fail, HdmiFail::LockLost);
            return false;
        }
        if (waited >= HDMI_POLL_TIMEOUT_MS) {
            ++stalled_;
            set_err(err, "poll 在 " + std::to_string(HDMI_POLL_TIMEOUT_MS) +
                             "ms 内无帧而接收器仍锁定: 流在驱动侧停了 (或缓冲被消费者占死), "
                             "需要重建 —— 不是消费者落后 (消费者落后表现为序号缺口, 不表现为无帧)");
            set_fail(fail, HdmiFail::Stalled);
            return false;
        }
    }
}

// 等 fence: sync_file 的 poll 在 fence signal 时返回 POLLIN, 未 signal 时挂回调等唤醒 ——
//   比按 SYNC_IOC_FILE_INFO 忙轮询好 (后者实测要轮询五千次才等到 5ms)
void HdmiIn::close_fence(int fd) {
    if (fd >= 0) ::close(fd);
}

bool HdmiIn::wait_fence(int fd, std::string* err) {
    if (fd < 0) {
        last_fence_wait_us_ = 0.0;
        return true;
    }
    if (!fence_wait_) {  // 测量用 (见 io/hdmi_in.h): 不等, 但 fd 照样关
        last_fence_wait_us_ = 0.0;
        close_fence(fd);
        return true;
    }
    ++fence_waits_;
    const int64_t t0 = now_us();
    for (;;) {
        pollfd p;
        p.fd = fd;
        p.events = POLLIN;
        p.revents = 0;
        const int pr = poll(&p, 1, HDMI_FENCE_WAIT_MS);
        if (pr > 0 && (p.revents & POLLIN)) {
            last_fence_wait_us_ = (double)(now_us() - t0);
            // 账只记**成功的**等待: 超时那几次烧掉的恰好是上限 (poll 到点才返回, 实测
            //   40038–40081µs), 与成功的分布混在一起就再也看不出上限该多大 (见
            //   HDMI_FENCE_WAIT_MS 的选择规则)。
            fence_wait_us_sum_ += last_fence_wait_us_;
            if (last_fence_wait_us_ > fence_wait_us_max_) fence_wait_us_max_ = last_fence_wait_us_;
            close_fence(fd);
            return true;
        }
        if (pr > 0) {
            char b[64];
            std::snprintf(b, sizeof(b), "0x%x", p.revents);
            set_err(err, std::string("帧的 dma-fence 报了异常 revents=") + b + " (载荷状态不明)");
            close_fence(fd);
            return false;
        }
        if (pr < 0 && errno == EINTR) continue;  // 信号打断: 继续等 (上限就是这条 poll 的超时)
        ++fence_timeouts_;
        last_fence_wait_us_ = (double)(now_us() - t0);
        set_err(err, pr == 0 ? ("帧的 dma-fence 在 " + std::to_string(HDMI_FENCE_WAIT_MS) +
                                "ms 内没有 signal (载荷没写完; 低延迟模式下属异常)")
                             : (std::string("poll dma-fence 失败: ") + errno_text()));
        close_fence(fd);
        return false;
    }
}

void HdmiIn::release(int index) {
    if (index < 0) return;
    if (index != hold_index_) {
        // 不是手上那帧: 只可能已经归还过 (Newest 下由 wait_frame 代还), 不重复入队
        return;
    }
    queue_buffer(index);
    hold_index_ = -1;
}
