// ============================================================================
//  hdmi_test — build/hdmi_test 单测 (scripts/compile.sh 构建并执行一次): 取帧层的
//    **判据与推导关系**, 不需要信号也不碰设备 (真机的失锁→重锁→画面回来由
//    scripts/test/hdmi_probe 与 aimbot 的 [HDMI] 行验收, 见 io/hdmi_in.h)。
//    [1] 失败去向: 每个 HdmiFail 落到哪一支 —— 只有"流断了/会话状态可疑"才重建,
//        单次 fence 超时是丢帧, 被信号打断与调用顺序错误两者都不重建
//    [2] 连续 fence 超时的判定线 = 队列块数 − 1 (缓冲转完一圈没有一帧载荷写完),
//        线上一格仍是丢帧
//    [3] 重建节拍: 两次尝试起点间隔不足 HDMI_REARM_MIN_MS 就不该再试, 到点即试
//    [4] 常量的推导关系 (改一个必须连带改另一个): fence 上限覆盖一个 60fps 帧周期且
//        不超过它的 2.5 倍、poll 预算是整分片、分片 = 驱动自己的锁定重试步长、
//        重建节拍介于一次锁定步长与一次有界等锁之间、缓冲块数的上下界
//  全部断言通过输出 ALL PASS 并返回 0; 任一断言失败返回非零 (compile.sh 的 set -e
//  终止编译)。
// ============================================================================

#include <chrono>
#include <iostream>

#include "io/hdmi_in.h"

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { std::cout << "  ok  " << msg << "\n"; } \
    else { std::cerr << "  FAIL " << msg << "\n"; ++g_fail; } \
} while (0)

int main() {
    std::cout << "=== hdmi_test (取帧层的判据) ===\n";

    // ---------------- [1] 失败去向 ----------------
    {
        std::cout << "[1] 失败去向 (要重建 / 只是丢帧 / 与流无关)\n";
        CHECK(!hdmi_fail_needs_rearm(HdmiFail::Ok), "Ok 不重建 (成功路径)");
        CHECK(hdmi_fail_needs_rearm(HdmiFail::LockLost),
              "LockLost 重建 (poll 分片无帧且接收器未锁定 = 流断了)");
        CHECK(hdmi_fail_needs_rearm(HdmiFail::Stalled),
              "Stalled 重建 (锁定但整预算无帧 / 连续 fence 超时到线 = 流不产数据)");
        CHECK(hdmi_fail_needs_rearm(HdmiFail::Error),
              "Error 重建 (ioctl/poll 异常, 会话状态可疑)");
        CHECK(!hdmi_fail_needs_rearm(HdmiFail::Fence),
              "单次 Fence 不重建 (下一帧照取: 驱动的下一次 DQBUF 会自己 signal 掉那根 fence)");
        CHECK(!hdmi_fail_needs_rearm(HdmiFail::Interrupted),
              "Interrupted 不重建 (被信号打断, 与流无关)");
        CHECK(!hdmi_fail_needs_rearm(HdmiFail::NoStream),
              "NoStream 不重建 (调用顺序错误: 重建失败之后不该取帧)");
        CHECK((int)HdmiFail::Ok == 0, "Ok == 0 (默认值的语义就是'没有失败')");
    }

    // ---------------- [2] 连续 fence 超时的判定线 ----------------
    {
        std::cout << "[2] 连续 fence 超时的判定线\n";
        CHECK(HDMI_FENCE_FAIL_MAX == HDMI_BUF_DEFAULT - 1,
              "线 = 队列块数 − 1 = 3 (消费者手上最多留 1 块, 其余整圈都没有一帧载荷写完)");
        CHECK(!hdmi_fence_streak_is_loss(1),
              "第 1 次超时是丢帧 (驱动下一次 DQBUF 把上一根 fence signal 掉 = 自愈路径)");
        CHECK(!hdmi_fence_streak_is_loss(HDMI_FENCE_FAIL_MAX - 1), "线上一格仍是丢帧");
        CHECK(hdmi_fence_streak_is_loss(HDMI_FENCE_FAIL_MAX), "到线即判流不产数据");
        CHECK(hdmi_fence_streak_is_loss(HDMI_FENCE_FAIL_MAX + 1), "过线之后一直判 (不会回头)");
    }

    // ---------------- [3] 重建节拍 ----------------
    {
        std::cout << "[3] 重建节拍\n";
        const auto t0 = std::chrono::steady_clock::now();
        const auto at = [&](int ms) { return t0 + std::chrono::milliseconds(ms); };
        CHECK(!hdmi_rearm_due(at(0), at(0)), "同一时刻不重复尝试");
        CHECK(!hdmi_rearm_due(at(HDMI_REARM_MIN_MS - 1), at(0)),
              "差 1ms 还不到节拍, 不再试");
        CHECK(hdmi_rearm_due(at(HDMI_REARM_MIN_MS), at(0)), "刚好到节拍即试");
        CHECK(hdmi_rearm_due(at(HDMI_REARM_MIN_MS + 1), at(0)), "超过节拍即试");
        CHECK(hdmi_rearm_due(at(10 * HDMI_REARM_MIN_MS), at(0)), "久未尝试时立即试 (首次打开)");
    }

    // ---------------- [4] 常量的推导关系 ----------------
    {
        std::cout << "[4] 常量的推导关系\n";
        const double frame_60 = 1000.0 / 60.0;   // 本层支持的最慢信号的一帧周期 (ms)
        CHECK(HDMI_FENCE_WAIT_MS > frame_60,
              "fence 上限 > 一个 60fps 帧周期 (合法等待 ≤ 一帧: 载荷在该帧的传输窗口内写完)");
        CHECK((double)HDMI_FENCE_WAIT_MS <= 2.5 * frame_60,
              "fence 上限 ≤ 2.5 个 60fps 帧周期 (取 2 倍再按整十取整 = 40ms, 不虚耗)");
        CHECK(HDMI_POLL_SLICE_MS == HDMI_LOCK_POLL_MS,
              "poll 分片 = 驱动自己的锁定重试步长 (断流察觉延迟上限就是一片)");
        CHECK(HDMI_POLL_TIMEOUT_MS % HDMI_POLL_SLICE_MS == 0,
              "poll 总预算是整分片 (分片只改变检查粒度, 不改变预算)");
        CHECK(HDMI_POLL_TIMEOUT_MS / HDMI_POLL_SLICE_MS >= 4,
              "预算至少含 4 片 (分片不把无帧容忍削短)");
        CHECK(HDMI_REARM_MIN_MS >= HDMI_LOCK_POLL_MS,
              "重建节拍 ≥ 一个锁定步长 (比这更密的尝试只是把同一件事重复问一遍)");
        CHECK(HDMI_REARM_MIN_MS <= HDMI_LOCK_WAIT_MS,
              "重建节拍 ≤ 一次有界等锁 (失败的尝试本身已把下一次推后, 节拍只在快速失败时起作用)");
        CHECK(HDMI_BUF_MIN <= HDMI_BUF_DEFAULT && HDMI_BUF_DEFAULT <= HDMI_BUF_MAX,
              "缓冲块数上下界包住缺省值");
        CHECK(HDMI_BUF_MIN >= 2, "驱动自己的 min_buffers_needed = 2 是下限");
    }

    std::cout << (g_fail ? "FAILED\n" : "ALL PASS\n");
    return g_fail ? 1 : 0;
}
