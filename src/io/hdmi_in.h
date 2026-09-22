// ============================================================================
//  hdmi_in.h — RK3588 板载 HDMI RX (rk_hdmirx) 的取帧层: 裸 V4L2, MULTIPLANAR,
//    零拷贝 (mmap + EXPBUF 两条通路), 最新帧语义。
//
//  为什么是裸 V4L2 而不是 GStreamer / cv::VideoCapture: 本平台的信号是
//  2560×1440 的 24 位 RGB, 120Hz 下 11059200B × 120 = 1.33 GB/s。走
//  v4l2src ! videoconvert ! appsink 或 cv::VideoCapture(..., CAP_V4L2) 的每一帧
//  都要"内核缓冲 → userspace 大拷贝 → 逐像素 CSC → cv::Mat"几次全帧搬运, 单核的
//  相当一部分就交给搬 11MB/帧了; 而直接采到 V4L2 缓冲上、把 EXPBUF 导出的 dmabuf
//  fd 交给 RGA, 每帧只被硬件搬一次 (见 io/rga_pp.h 的缓冲尺寸规则)。延迟面同样
//  倒向裸 V4L2: GStreamer 的 queue 深度就是最坏情况的额外排队 (而本驱动的行为是
//  "队列空就丢帧, 不重发旧帧")。
//
//  设备解析: /dev/videoN 的下标跨重启不稳定 (实测: 同一张板子两次启动 hdmirx 落在
//  不同号上), 故缺省按**节点名**解析 —— /sys/class/video4linux/videoN/name 里含
//  "hdmirx" 的那个 (驱动名 rk_hdmirx, 节点名 stream_hdmirx, 判据取两者共有的子串);
//  匹配到多个是错误并列出来 (与 io/hid_mouse.h 的多义判据同一口径), 显式给
//  /dev/videoN 则原样使用。解析结果由打开后的 VIDIOC_QUERYCAP 复核 (driver 字段
//  必须是 rk_hdmirx), 即"名字解析 + 驱动自述"两处一致才算选中。
//
//  **格式由信号决定, 不由我们选**: 驱动 g_out_fmts 只有 BGR24/NV24/NV16/NV12 四种,
//  且 hdmirx_s_fmt_vid_cap_mplane() 对 pixelformat != cur_fmt_fourcc 直接 -EINVAL
//  (实测: 锁定 2560×1440 信号下 S_FMT 任何别的格式都失败)。所以本层只读 G_FMT 并
//  适应, 不尝试设置。四个格式全是 V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE (节点没有
//  单平面采集能力), 平面数 1 (平面 YUV 的 UV 与 Y 同处一个缓冲, 首平面 length =
//  sizeimage = 整帧)。
//
//  **stride 与 sizeimage 一律取自驱动, 绝不按 width×bpp 重算**: 驱动的规则是
//  bpl = ALIGN(width × bpp/8, 64), sizeimage = bpl × height (平面 YUV 的 UV 平面
//  沿用 Y 的 stride, 故整帧 = bpl×height×(1/1.5/2/3))。实测 2560×1440 BGR3:
//  ALIGN(7680,64) = 7680, 7680×1440 = 11059200 ✓ (同一公式在 640×480 与
//  1920×1080 档位也对得上)。本层把 bytesperline 换算成**像素**行距 (RGB 系 ÷3,
//  平面 YUV 直接用), 这正是 RGA 要的 wstride 单位 (见 io/rga_pp.h)。
//
//  **STREAMON 之前必须等到 timing 锁定**: hdmirx_start_stream() 里
//  `if (!hdmirx_dev->get_timing) return -EINVAL;` — 未锁定就起流只有
//  "timing is invalid" 一条报错, 拿不到流。故 open() 在 QBUF 之前轮询
//  VIDIOC_QUERY_DV_TIMINGS 到成功为止, 上限 2000ms 每 200ms 一次 (与驱动自己在
//  hdmirx_wait_lock_and_get_timing 里给锁定+读格式的 10×200ms 重试预算同一个
//  数), 超时按失败返回并把驱动给的最后一条错误一起打出来 (未接源时
//  QUERY_DV_TIMINGS 报 ENOLINK "Link has been severed"; 已接但 TMDS 未锁报
//  ENOLCK "No locks available" — 两者都是"还没锁上", 都在重试窗口内)。
//
//  --- 失锁 / 断流与重建 (rearm) ---
//  接收器会**在源切模式 (主机/主机的 HDR 开关、游戏机开机或切输入) 时失去 TMDS 锁**,
//  驱动随即自己停流 (`stream start stopping` / `stream stopping finished`), 源重发模式后
//  接收器重新锁上 (`signal lock ok` + `hdmirx_format_change: New format: …`)。**重锁不会
//  把流接回去**: 驱动停流之后 vb2 队列不再产帧, 而它自己不会替调用方重新 STREAMON。
//  于是"信号回来了但画面永远没有"是这条链路的常态故障, 症状只有 poll 超时 —— 一个把
//  poll 超时当成"再等等"的消费者会一直等下去, 而把它当成"设备坏了"的消费者会把整条
//  链带走 (后者正是本层要否掉的行为)。
//
//  判据分两层, 因为两类"没有帧"的物理原因不同、恢复动作也不同:
//    (1) **接收器锁定是唯一的快速可观测量**。poll 分片 (HDMI_POLL_SLICE_MS) 内没有帧时查
//        VIDIOC_QUERY_DV_TIMINGS: 它失败就是"信号/流断了" (驱动在失去 TMDS 锁时
//        hdmirx_g_dv_timings 直接报 not locked), 立刻按断流处理 —— 察觉延迟上限 = 一片,
//        而不是整个 poll 预算。同一个查询也用在 fence 超时上: 载荷没写完 + 未锁定 = 那
//        一帧是断流的第一个症状, 不必等下一帧。
//    (2) **整个 poll 预算 (HDMI_POLL_TIMEOUT_MS) 内没有帧而接收器仍在锁上**, 或者连续
//        HDMI_FENCE_FAIL_MAX 帧的 dma-fence 都没 signal, 则是"流在驱动侧停了/队列不再
//        交付"(消费侧或驱动侧), 同样按断流处理。这一支存在的意义是**不把"信号还在、
//        驱动还在产帧"的假象当作健康** —— 恢复路径对两者是同一条。
//  两类都返回 HdmiFail::{LockLost,Stalled} 并进入 rearm(); 单次 fence 超时返回
//  HdmiFail::Fence (丢了一帧, 驱动的下一次 DQBUF 会自行把上一根 fence signal 掉, 见
//  HDMI_FENCE_FAIL_MAX), 被信号打断返回 HdmiFail::Interrupted (退出路径, 不重建)。
//
//  **重建的顺序与理由**: STREAMOFF (容忍错误 —— 流已经死了, 这一步本来就可能失败) →
//  munmap → 关掉 EXPBUF 导出的 dmabuf fd → REQBUFS(0) → 关设备 → 重新打开 → 有界等锁
//  → 读 G_FMT → 重建缓冲/入队 → STREAMON。最后两步的先后是硬的: 缓冲尺寸由**锁定后的**
//  格式导出 (模式变化同时改分辨率与帧率, 1440p 与 1080p 的 11.06MB/8.29MB 缓冲不是同一
//  种东西), 所以"先锁后读"是分配正确的缓冲的前提 —— 本层的 arm() 把 wait_timing_lock
//  放在 G_FMT 之前, 这条顺序对首次打开与重建是同一条。
//  重建**不重新解析设备节点**: 节点号在一次启动内是稳定的 (跨重启才不稳定, 见上), 首次
//  打开解析出来的路径被记住并复用; 记住的路径打不开就按失败报出来 (那才是真的没了)。
//
//  重建的失败是**可重试的**而不是终局: rearm() 失败时设备处于关闭状态, 再调一次就是
//  完整的一次新尝试 (调用方按 HDMI_REARM_MIN_MS 的节拍重试)。这条让"源还没上电/还在换
//  模式"与"设备坏了"在行为上一致 —— 都是继续等, 而不是带走进程。
//
//  **最新帧语义** (HdmiDelivery::Newest, 生产路径): 消费者比信号慢时, 按队列顺序
//  逐帧处理会越落越远 —— 交出去的永远是队列头那帧, 它的年龄随落后程度累积, 而
//  控制环读的是交叉位置, 读的是几分钟前(字面意义)的画面。故 wait_frame 一拿到
//  DQBUF 就继续非阻塞 DQBUF 排空队列, 只留最新的一帧, 其余当场 QBUF 归还, 归还的
//  帧计入 superseded(); 手上还没 release 的旧帧在下一次 wait_frame 开头一并归还
//  (它就是被新帧取代的那一帧)。这样"交出来的帧的年龄"与消费者的速度无关, 只由
//  一帧的传输时间与此刻队列里恰好有几帧决定 (实测见 hdmi_in.cpp 头部)。队列深度
//  因此只买"驱动不停"的余量 — 驱动在 line_flag 中断里找不到下一个 buffer 就
//  `drop the frame!` 丢掉该帧且不重发, 所以宁可多留几块也不要让队列空。
//  HdmiDelivery::QueueOrder 是同一份代码的对照路径 (严格按队列顺序取, 只由
//  release 归还): 它存在的意义就是让"最新帧语义买到了什么"可以被量出来, 不是在
//  生产里选择的一条路。
//
//  时间戳: 该队列声明 V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC (驱动 queue_setup 里
//  timestamp_flags), 所以 buf.timestamp 与 CLOCK_MONOTONIC 同源, 可以直接和
//  clock_gettime(CLOCK_MONOTONIC) 相减 —— 交付延迟与帧率统计都靠这一点, 故首次交付时
//  校验一次时间戳类型, 不是 MONOTONIC 就按硬错误返回 (度量口径错了比测不出来更糟)。
//
//  **低延迟开关与帧完成凭据 (dma-fence) 是同一件事的两半**: rockchip_hdmirx 的
//  low_latency 由**本层在 STREAMON 前写入** (/sys/module/rockchip_hdmirx/parameters/
//  low_latency), 不交给部署脚本 —— 驱动只在 STREAMON 时读它一次
//  (hdmirx_start_streaming: low_latency && fps>=59 → delay_line=10, 帧完成中断从
//  dma_idle 提前到 line_flag, 并跳过 6 帧预热过滤), 所以"先写参数再起流"这个顺序的
//  保证点只能是调用 STREAMON 的那段代码: 交给脚本等于把一条硬要求变成启动纪律, 而上一轮
//  遗留的 0 与"故意关掉"在文件里长得一模一样。写入幂等 (已是置位就不写), 写完回读校验,
//  不是置位就按硬错误返回 (本层运行环境已经需要 root: /dev/rga 与 raw_gadget 都是
//  root-only)。该参数是内核 bool, sysfs 回读是 "Y"/"N" (实测写入 "1" 后回读 "Y"),
//  所以判"已置位"认这两种写法。
//
//  低延迟的代价是**完成通知早于载荷写完**, 而凭据是 fence: 开着 low_latency 时
//  VIDIOC_DQBUF 返回的那一刻帧并没有写完 —— 不等 fence 就交帧, 探针的自检 (交付瞬间取
//  该帧矩形首 4KB, 6ms 后再取) 读到 **198 字节变化**, 即那 4KB 里有 198 字节是这 6ms 内
//  才被写进去的; 照原样读缓冲拿到的是"半张画面", 而按帧内位移做的任何测量 (标定的相位
//  相关就是) 会把这种变化当成真实位移。驱动为此在每次 DQBUF 时把该缓冲的 dma-fence 挂在
//  buf.timecode.userbits 上 (4 字节小端 fd, 0xffffffff = 无): DQBUF 返回时它恒为 active,
//  等它 signal 之后同一段 4KB 在 6ms 内 0 字节变化 (探针自检同一句话, 两个分支分别 0 与
//  198)。所以 wait_frame 在把帧交出去之前**等这个 fence** (poll + HDMI_FENCE_WAIT_MS
//  上限), 并**每次关掉这个 fd** —— 它是内核给的 dup, 四块缓冲循环出四个 fd, 不关就是每帧
//  漏一个。非低延迟模式 userbits 是 0xffffffff (约定无效), 完成点在 dma_idle, 载荷本来
//  完整, 无需等 (本层的测量开关绕开这一段, 见 set_fence_wait)。
//
//  由此, 时间戳的位置也确定了: buf.timestamp 填在**完成通知**处 (与该次 vb2 完成同一
//  时刻), 而载荷要到 fence signal 才完整 —— 于是 "取帧返回 − 时间戳" 这个交付延迟就是
//  "等 fence" + "userspace 侧开销" 两段之和, 探针把两者分开报 (实测 p50 6.9–7.2ms 与
//  0.8ms)。这个量最终进的是标定的 L (L 从画面测, 不从时间戳测), 口径不冲突; 但它决定了
//  任何跨帧的 age 计算必须知道这一点。
//
//  ---- 板端实测 (命令: sudo ./build/hdmi_probe 600; 2560×1440@120 BGR3 活动信号) ----
//  帧率 120.000fps (帧时间戳导出; 相邻帧间隔 p50 8.333 / p99 8.338ms), 交付 600 帧,
//  **0 丢弃 (最新帧语义) / 0 序号缺口 / 0 驱动 ERROR 归还** —— 消费者跟得上时这三项必须
//  全 0, 因为队列里恒有 ≥3 块空缓冲 (手上最多留 1 块)。
//  交付延迟 = 取帧返回 − 帧时间戳: p50 7841.0 / p90 7848.6 / p99 7855.4 / max 7919.5µs;
//  其中等 dma-fence: p50 6903.0 / p90 7319.0 / p99 7326.0 / max 7808.0µs
//  (三轮 600 帧下 fence p50 落在 6903–7180µs, 交付延迟 p50 7826–7841µs: fence 的等待长度
//  与"我们到达 DQBUF 的相位"有关, 而到达相位又被本帧之前的处理量决定; 两段之差
//  ≈0.8ms 是 userspace 侧开销 —— poll 唤醒、排空、记账)。
//  最新帧语义买到的东西 (每帧故意睡 30ms 模拟慢消费者, 300 帧):
//    最新帧: 交付帧龄 p50 16.2ms, 有界 (不再随运行时长增长), 且载荷完整 (自检 0 字节);
//    队列顺序 (对照): 交付帧龄 p50 91.2ms —— 落后 5.6 倍, 并且最终因占满队列导致某块的
//    fence 再也不 signal 而失败 (缓冲被占死时驱动走 drop the frame! 分支, 不重发旧帧)。
// ============================================================================

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <linux/videodev2.h>

// 设备解析判据: 驱动名 rk_hdmirx 与 /sys 节点名 stream_hdmirx 共有的子串。
constexpr const char* HDMIRX_NAME_SUBSTR = "hdmirx";

// 缓冲块数: 下限 2 = 驱动队列的 min_buffers_needed (HDMIRX_REQ_BUFS_MIN = 2, 源码),
//   缺省 4 = 实测 120fps 下 0 丢帧 (最新帧语义下消费者手上最多留 1 块, 故 2 块够
//   驱动不停, 4 块给"归还 与 下次取"之间的空隙留余量), 上限 8 = 每块缓冲都是一帧
//   物理连续的 DMA32 内存 (1440p RGB 下 11.06MB/块), 而队列深度只买余量不买延迟,
//   故不再往上开 (V4L2 自己的 VIDEO_MAX_FRAME=32 是协议上限, 不是设计边界)。
constexpr int HDMI_BUF_MIN     = 2;
constexpr int HDMI_BUF_DEFAULT = 4;
constexpr int HDMI_BUF_MAX     = 8;

// 等 timing 锁定的窗口: 与驱动 hdmirx_wait_lock_and_get_timing 给锁定+读格式的
//   10×200ms 重试预算同一个数 (2000ms / 200ms), 超时按失败返回而不是继续起流。
constexpr int HDMI_LOCK_WAIT_MS = 2000;
constexpr int HDMI_LOCK_POLL_MS = 200;

// 等 dma-fence 的上限: 低延迟模式给出的完成通知早于载荷写完, fence 才是写完的凭据
//   (实测 120fps 下 5.27ms, 即约 0.63 个帧周期), 而本层支持的信号最低到 60fps
//   (帧周期 16.7ms) —— 上限取 2 个 60fps 帧周期再取整 = 40ms。
//   选择规则 (改这个数要按它算): fence 在该帧载荷写完时 signal, 而这个时刻落在这一帧的
//   传输窗口内, 故**合法等待 ≤ 一个帧周期**; 上限取它的 2 倍 = 给最慢支持信号 (60fps 的
//   16.7ms) 与调度抖动留的整倍余量 —— 不能再小: 压到 20ms 就是 1.2 倍余量, 会把"晚一点
//   但真的会写完"的帧误判成丢帧, 而一次误判的代价是半张画面进下游 (比 40ms 的等待重得
//   多)。超时侧的成本是算出来的: 超时意味着这一帧的 fence 永不再 signal (驱动的丢帧
//   路径), 白等的就是上限本身 —— 40ms 的线程时间加一帧的检测 (120fps 下 4.8 个帧周期)。
//   这笔成本的实际发生率与成功等待的真实分布由 fence_waits / fence_timeouts /
//   fence_wait_us_sum 三个观测直接给出 ([AI FPS] 行的增量段与退出时的"收帧统计:"),
//   "上限该不该更小"由那两个数回答 —— 判据是量出来的, 不是估出来的。
constexpr int HDMI_FENCE_WAIT_MS = 40;

// 连等 fence 超时的判定线 = 队列块数 − 1 (消费者手上最多留 1 块, 见"最新帧语义"): 驱动的
//   丢帧路径只在"队列里没有空缓冲"时触发, 而它给该缓冲留下的 fence **永远不会再 signal**
//   (下一次 DQBUF 会把它顺手 signal 掉 —— 内核日志 hdmirx_dqbuf_get_done_fence: last fence
//   not signal, signal now!, 所以单次超时是驱动能自愈的丢帧)。HDMI_BUF_DEFAULT − 1 帧
//   = 缓冲整整转完一圈而**没有一帧载荷写完** —— 那不是丢一帧, 是流不再产数据 (重建才是
//   恢复路径)。按 40ms 上限算, 这条线对应 120ms 的窗口。
constexpr int HDMI_FENCE_FAIL_MAX = HDMI_BUF_DEFAULT - 1;

// wait_frame 的一次 poll 分片: 分片结束且无帧时查一次接收器锁定状态 —— "信号掉了"只有
//   这一个快速可观测量 (见文件头)。取 200ms = 驱动自己的锁定重试步长 (HDMI_LOCK_POLL_MS),
//   于是断流被察觉的延迟上限是一片 (200ms = 120fps 下 24 帧 / 60fps 下 12 帧), 而总的
//   无帧容忍仍是 HDMI_POLL_TIMEOUT_MS (分片不改变预算, 只把"还没有帧"按片检查一次锁定)。
//   健康信号下第一片就拿到帧, 故稳态里这个查询一次都不发生。
constexpr int HDMI_POLL_SLICE_MS = 200;

// wait_frame 的 poll 总预算: 120fps 信号每 8.33ms 出一帧, 2000ms = 240 帧的时间,
//   到这个量级还没帧而接收器仍在锁上, 就不是抖动而是流停了 (或缓冲被消费者占死),
//   需要重建而不是继续等。
constexpr int HDMI_POLL_TIMEOUT_MS = 2000;

// 两次重建尝试的**起点**之间的最小间隔: 一次尝试本身含"有界等锁" (HDMI_LOCK_WAIT_MS),
//   所以失败的尝试已经把下一次推后到 ≥2s; 这条 1s 只在尝试很快失败时起作用 (节点打不开),
//   免得按拍率空转。1s 是"源重发模式并重新锁定"这条现场时间量的下界 —— 复现它的命令在
//   io/hdmi_in.cpp 头部 (给接收器写别的 EDID → 源重新协商), 比它更密的尝试只是把同一件
//   事重复问一遍。
constexpr int HDMI_REARM_MIN_MS = 1000;

// 一块采集缓冲的两条通路: mmap (CPU 参照实现 / 截图) 与 EXPBUF 导出的 dmabuf fd
//   (RGA 直接消费, 不落 userspace)。两者是同一块内核缓冲, index 就是 V4L2 下标。
struct HdmiBuffer {
    int    index  = -1;
    void*  map    = nullptr;
    size_t length = 0;      // = plane_fmt[0].sizeimage
    int    fd     = -1;     // dmabuf fd
};

// 信号决定的采集几何 (G_FMT 原样; stride 不重算)
struct HdmiFormat {
    uint32_t fourcc      = 0;
    int      width       = 0;
    int      height      = 0;
    int      bpl_bytes   = 0;   // plane_fmt[0].bytesperline (驱动的字节行距)
    int      stride_px   = 0;   // 像素行距 (= RGA 的 wstride): RGB 系 bpl/3, 平面 YUV = bpl
    size_t   plane_bytes = 0;   // plane_fmt[0].sizeimage (整帧)
};

// 交付的一帧: fd / map / 几何指向同一块内核缓冲, release 之前一直有效。
struct HdmiFrame {
    int         index      = -1;
    int         fd         = -1;
    const void* map        = nullptr;
    uint32_t    fourcc     = 0;
    int         width      = 0;
    int         height     = 0;
    int         stride_px  = 0;
    size_t      plane_bytes = 0;
    int64_t     ts_ns      = 0;   // 帧完成时刻 (CLOCK_MONOTONIC, 驱动填)
    uint32_t    sequence   = 0;   // 驱动序号 (缺口 = 驱动丢帧)
    int         fence_fd   = -1;  // low_latency 时随缓冲给出的 dma-fence (-1 = 无)
};

// 交付语义 (见文件头"最新帧语义")
enum class HdmiDelivery {
    Newest,       // 生产路径: 只交最新一帧, 过期帧当场归还
    QueueOrder,   // 对照路径: 严格按队列顺序, 只由 release 归还
};

// 取帧失败的去向 (与设备无关的纯判据, 调用方据此决定"重建"还是"下一帧再来"):
enum class HdmiFail {
    Ok = 0,
    Fence,        // 帧到手但载荷没写完 (dma-fence 没 signal) — 丢了一帧
    LockLost,     // 接收器未锁定 (poll 分片无帧, 或 fence 超时后查到) — 流断了
    Stalled,      // 整个 poll 预算无帧而接收器仍锁定, 或连续 fence 超时到线 — 流不产数据
    Error,        // ioctl/poll/设备异常 (含时间戳口径不符) — 会话状态可疑
    Interrupted,  // poll 被信号打断 (退出路径, 与流无关)
    NoStream,     // 未起流: 调用方在重建失败之后继续取帧了
};

// 是否该拆掉会话重建 (见文件头): 只有"流断了"与"会话状态可疑"值得重建。
//   单次 fence 超时是丢帧 (驱动的下一次 DQBUF 会自行 signal 掉它), 连续到
//   HDMI_FENCE_FAIL_MAX 由 wait_frame 归为 Stalled; Interrupted/NoStream 既不重建也不
//   当作丢帧 (前者是退出信号, 后者是调用顺序)。
constexpr bool hdmi_fail_needs_rearm(HdmiFail f) {
    return f == HdmiFail::LockLost || f == HdmiFail::Stalled || f == HdmiFail::Error;
}
// 连续 fence 超时到线 (推导见 HDMI_FENCE_FAIL_MAX): 到线之后在 wait_frame 里就按 Stalled 报
constexpr bool hdmi_fence_streak_is_loss(int streak) { return streak >= HDMI_FENCE_FAIL_MAX; }

// 重建节拍: 两次尝试起点之间至少 HDMI_REARM_MIN_MS (出处见该常量)。
inline bool hdmi_rearm_due(std::chrono::steady_clock::time_point last,
                           std::chrono::steady_clock::time_point now) {
    return now - last >= std::chrono::milliseconds(HDMI_REARM_MIN_MS);
}

class HdmiIn {
public:
    HdmiIn() = default;
    ~HdmiIn();
    HdmiIn(const HdmiIn&) = delete;
    HdmiIn& operator=(const HdmiIn&) = delete;

    // 解析设备节点: spec 空 = 按 HDMIRX_NAME_SUBSTR 扫 /sys/class/video4linux/*/name;
    //   非空 = 原样使用 (显式 /dev/videoN)。失败返回空串并写 err。
    static std::string resolve_device(const std::string& spec, std::string* err);

    // 打开 → 解析/校驱动 → 等 timing 锁定 → 读 G_FMT → 置 low_latency →
    //   REQBUFS/QUERYBUF/mmap/EXPBUF → QBUF 全部入队 → STREAMON。任一步失败即 close() 并返回
    //   false; spec 与块数被记住, 之后的 rearm() 用同一份参数重来 (不重新解析节点, 见文件头)。
    bool open(const std::string& spec, int num_buffers = HDMI_BUF_DEFAULT,
              HdmiDelivery delivery = HdmiDelivery::Newest, std::string* err = nullptr);
    // 重建: 拆掉当前会话 (见文件头) → 重新打开 → 有界等锁 → 按锁定后的格式重建缓冲并起流。
    //   失败返回 false 并写 err, 此时设备处于**关闭**状态 —— 再调一次就是一次完整的新尝试
    //   (open() 从未成功过时也可以直接用它, 它会把设备解析补上)。成功返回 true 并打印
    //   重建耗时与前后格式。
    bool rearm(std::string* err = nullptr);
    void close();
    bool opened() const { return fd_ >= 0; }

    // 接收器锁定状态: VIDIOC_QUERY_DV_TIMINGS 成功 = true (顺带刷新 timing_ 与帧率);
    //   失败 = false 并把驱动给的最后一条错误写进 err (ENOLINK "Link has been severed" =
    //   源不在, ENOLCK "No locks available" = TMDS 未锁)。它是"信号掉了"唯一的快速可观测量。
    bool query_lock(std::string* err = nullptr);

    // 等一帧并交付 (Newest 下顺带把过期帧归还)。返回 false 时 err 说明原因, fail 给出
    //   **去向** (要重建 / 只是丢了一帧 / 与流无关), 见 HdmiFail 与 hdmi_fail_needs_rearm。
    bool wait_frame(HdmiFrame* out, std::string* err = nullptr, HdmiFail* fail = nullptr);
    // 归还一帧 (Newest 下 wait_frame 也会替调用方归还手上那帧)
    void release(int index);

    // 等不等 fence —— **只给测量用**: 关掉它, wait_frame 会在载荷没写完时就交帧 (实测
    //   交付后 6ms 内该帧仍有几千字节在变), 生产路径禁止关 (半张画面进下游比报错糟得多)。
    //   探针用它复现"低延迟模式的完成通知早于载荷写完"这条结论: 默认 0 字节变化,
    //   --no-fence 时非 0。
    void set_fence_wait(bool on) { fence_wait_ = on; }
    bool fence_wait_enabled() const { return fence_wait_; }

    // ---- 观测 (探针与日志) ----
    const std::string& dev_path() const { return dev_path_; }
    const std::string& driver()   const { return driver_; }
    const HdmiFormat&  format()   const { return fmt_; }
    const v4l2_bt_timings& timings() const { return timing_; }
    const std::vector<HdmiBuffer>& buffers() const { return bufs_; }
    int      low_latency() const { return low_latency_; }
    HdmiDelivery delivery() const { return delivery_; }
    uint64_t superseded() const { return superseded_; }   // 最新帧语义归还的旧帧
    uint64_t lost()       const { return lost_; }         // 驱动少交付的帧 (序号缺口)
    uint64_t delivered()  const { return delivered_; }
    uint64_t invalid()    const { return invalid_; }      // 驱动以 ERROR 归还的缓冲
    uint64_t fence_waits() const { return fence_waits_; } // 等过 fence 的帧数
    // 上一帧等 dma-fence 的实际耗时 (µs): 交付延迟里低延迟模式的固有成分, 探针据此把它
    //   与 userspace 侧开销分开报 (两者合起来才是"拿到手时这帧有多旧")
    double   last_fence_wait_us() const { return last_fence_wait_us_; }
    // fence 等待的账 (HDMI_FENCE_WAIT_MS 那条选择规则的实测落点): 等过多少次 / 其中等到
    //   超时多少次 / 等待总时长与最长一次 (µs)。**成功的等待不超过一个帧周期, 超时的那几次
    //   烧掉的恰好是上限** —— 把这两个量分开是"上限该多大"唯一诚实的判据。
    uint64_t fence_timeouts()   const { return fence_timeouts_; }
    double   fence_wait_us_sum() const { return fence_wait_us_sum_; }
    double   fence_wait_us_max() const { return fence_wait_us_max_; }
    uint64_t lock_lost()  const { return lock_lost_; }    // 判为"接收器未锁定"的次数
    uint64_t stalled()    const { return stalled_; }      // 判为"锁定但流不产数据"的次数
    uint64_t rearm_ok()   const { return rearm_ok_; }     // 重建成功次数
    uint64_t rearm_fail() const { return rearm_fail_; }   // 重建失败次数
    // 上一次成功重建的耗时 (ms; 0 = 还没重建过): "断流→画面回来"的实测恢复时间
    double   last_rearm_ms() const { return last_rearm_ms_; }

    // 锁定时序导出的帧率 (Hz) = 像素时钟 / (总宽 × 总高): 帧率是**信号的属性**,
    //   由接收器报的时序给出 (有效区尺寸乘像素时钟得到的是像素率, 不是帧率);
    //   未锁定/时序缺失时返回 0。控制拍与标定采样窗按它取本源的帧长尺度。
    double frame_hz() const;
    // 'BGR3' 这样的四字符码文字 (日志与探针用)
    static std::string fourcc_name(uint32_t f);
    // 锁定时序的一行文字: 有效尺寸 / 总尺寸 / 像素时钟 / 导出帧率
    std::string timing_text() const;

private:
    bool set_low_latency(std::string* err);
    bool wait_timing_lock(std::string* err);
    // 一次采集会话的建立: (fd 已打开时) 等锁 → 读格式 → low_latency → 缓冲/入队 → STREAMON。
    //   设备解析不在这里 —— open() 解析一次, rearm() 复用 (见文件头)。任一步失败即 close()。
    bool arm(std::string* err);
    // 拆掉会话但**留着 fd**: STREAMOFF (容错) → munmap → 关 dmabuf fd → REQBUFS(0)。
    //   流已经死了的时候每一步都可能失败, 一律容忍 (见文件头)。
    void unstream();
    bool queue_buffer(int index);
    // 单次非阻塞取帧: 0 = 取到, 1 = 现在没有, -1 = 错误 (写 err)。ts_type / err_flag
    //   是驱动在缓冲 flags 里给的两件事, 不塞进对外的 HdmiFrame。
    int dequeue(HdmiFrame* f, uint32_t* ts_type, bool* err_flag, std::string* err);
    // 等该帧的 dma-fence 到 signal (载荷写完的凭据), 等到/等不到都把这个 fd 关掉:
    //   它是内核给本次 DQBUF 的 dup, 不关就是每帧漏一个。fd < 0 = 本次没有 fence,
    //   直接算通过 (非低延迟模式的完成点就是 dma_idle, 载荷本来完整)。
    bool wait_fence(int fd, std::string* err);
    static void close_fence(int fd);
    static void set_fail(HdmiFail* f, HdmiFail v) { if (f) *f = v; }
    int  hold_index_ = -1;      // 已交付未归还的缓冲 (Newest 下下一次取帧时归还)
    int  fd_ = -1;
    bool streaming_ = false;
    bool ts_checked_ = false;   // 本次会话校验过时间戳类型
    bool ts_announced_ = false; // 时间戳基准那句话只报一次 (重建会重开会话)
    std::string spec_, dev_path_, driver_;
    int  num_buffers_ = HDMI_BUF_DEFAULT;
    HdmiFormat fmt_;
    v4l2_bt_timings timing_{};
    std::vector<HdmiBuffer> bufs_;
    HdmiDelivery delivery_ = HdmiDelivery::Newest;
    int low_latency_ = -1;
    uint64_t superseded_ = 0, lost_ = 0, delivered_ = 0, invalid_ = 0, fence_waits_ = 0;
    uint64_t fence_timeouts_ = 0, lock_lost_ = 0, stalled_ = 0, rearm_ok_ = 0, rearm_fail_ = 0;
    double last_fence_wait_us_ = 0.0;
    double fence_wait_us_sum_ = 0.0, fence_wait_us_max_ = 0.0, last_rearm_ms_ = 0.0;
    int  fence_fail_streak_ = 0;      // 连续 fence 超时 (成功的取帧清零; 线见 HDMI_FENCE_FAIL_MAX)
    bool fence_wait_ = true;
    bool low_latency_announced_ = false;
    bool seq_valid_ = false;
    uint32_t last_seq_ = 0;
};
