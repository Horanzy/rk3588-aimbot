// ============================================================================
//  pad_rp.cpp — pad_rp.h 的实现
//
//  线程模型与停机序列:
//    1. 会话线程 (g_sess_thread): 循环维护 ChiakiSession 生命周期 (init → start →
//       join 阻塞整场 → fini); 断开后等待 RP_RECONNECT_MS (可被 stop 条件变量唤醒)
//       重连。
//    2. 泵线程 (g_pump_thread): 1 kHz timerfd 轮询 pad_publish_snapshot(&seq);
//       仅在 seq 变化且映射后的 ChiakiControllerState 与上次发送不同时调用
//       chiaki_session_set_controller_state (上游发送线程负责 200 ms 保活与
//       20/50/100/200 ms 尾部重发)。
//    3. 同一把 g_sess_mtx 保护会话指针生命周期 (泵的 set_controller_state ↔
//       会话线程的 init/fini), g_session_live 防止 stop 恰好落在重连间隙时操作
//       已销毁会话。
//    4. 停机序列: 置 g_running=false 唤醒重连等待 → 先 join 泵线程 (保证不再下发新
//       状态) → 发全零松键 → 等待 RP_RELEASE_WAIT_MS (30 ms, 等首拍 20 ms 尾部
//       重发) → chiaki_session_stop → join 会话线程。
// ============================================================================

#include "io/pad_rp.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

#include <sys/timerfd.h>
#include <unistd.h>

#include <chiaki/session.h>
#include <chiaki/sock.h>

#include "core/state.h"
#include "io/pad_output.h"

namespace {

std::once_flag g_chiaki_init_once;
ChiakiErrorCode g_chiaki_init_err = CHIAKI_ERR_UNKNOWN;

std::mutex g_sess_mtx;
ChiakiSession g_session{};
bool g_session_live = false;
std::atomic<bool> g_session_connected{false};

std::mutex g_stop_mtx;
std::condition_variable g_stop_cv;
std::atomic<bool> g_rp_running{false};

std::thread g_sess_thread;
std::thread g_pump_thread;

ChiakiLog g_chiaki_log{};
RpAccount g_acc{};

void rp_chiaki_log_cb(ChiakiLogLevel level, const char* msg, void* user) {
    (void)user;
    if (level & (CHIAKI_LOG_ERROR | CHIAKI_LOG_WARNING | CHIAKI_LOG_INFO)) {
        std::cout << "[chiaki] " << (msg ? msg : "") << "\n";
    }
}

void rp_event_cb(ChiakiEvent* event, void* user) {
    (void)user;
    if (!event) return;
    switch (event->type) {
    case CHIAKI_EVENT_CONNECTED:
        g_session_connected.store(true, std::memory_order_release);
        std::cout << "✅ [RP] 已连接 PS5 (" << g_acc.host << "); 纯控制会话激活\n";
        break;
    case CHIAKI_EVENT_LOGIN_PIN_REQUEST:
        g_session_connected.store(false, std::memory_order_release);
        std::cerr << "❌ [RP] PS5 要求 4 位登录 PIN — 无头模式不支持输入 PIN, 请在 PS5 设置中关闭该账号的登录密码\n";
        break;
    case CHIAKI_EVENT_QUIT:
        g_session_connected.store(false, std::memory_order_release);
        std::cout << "⚠ [RP] 会话退出: " << chiaki_quit_reason_string(event->quit.reason) << "\n";
        break;
    default:
        break;
    }
}

void rp_session_loop() {
    while (g_rp_running.load(std::memory_order_acquire)) {
        std::string iface_err;
        std::string iface = !g_acc.iface.empty() ? g_acc.iface : rp_find_wired_iface(&iface_err);
        if (iface.empty() || rp_is_wireless_iface(iface)) {
            std::cerr << "❌ [RP] 无可用有线网卡 ("
                      << (iface.empty() ? iface_err : iface + " 是无线网卡")
                      << "), 拒绝回落 WiFi\n";
            std::unique_lock<std::mutex> ulk(g_stop_mtx);
            g_stop_cv.wait_for(ulk, std::chrono::milliseconds(RP_RECONNECT_MS), [] {
                return !g_rp_running.load(std::memory_order_acquire);
            });
            continue;
        }
        chiaki_sock_set_bind_iface(iface.c_str());

        ChiakiConnectInfo info{};
        info.host = g_acc.host.c_str();
        info.ps5 = true;
        std::memcpy(info.psn_account_id, g_acc.account_id.data(), 8);
        std::memcpy(info.regist_key, g_acc.regist_key.data(), 16);
        std::memcpy(info.morning, g_acc.morning.data(), 16);
        // 360p@30 仅为占位预设; audio_video_disabled = CHIAKI_AUDIO_VIDEO_DISABLED 时
        //   session.c 会禁用音视频流请求
        chiaki_connect_video_profile_preset(&info.video_profile,
                                            CHIAKI_VIDEO_RESOLUTION_PRESET_360p,
                                            CHIAKI_VIDEO_FPS_PRESET_30);
        info.audio_video_disabled = CHIAKI_AUDIO_VIDEO_DISABLED;
        info.video_profile_auto_downgrade = true;
        info.enable_dualsense = false;
        info.packet_loss_max = RP_PACKET_LOSS_MAX;

        bool started = false;
        g_session_connected.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(g_sess_mtx);
            if (!g_rp_running.load(std::memory_order_acquire)) break;
            std::memset(&g_session, 0, sizeof(g_session));
            ChiakiErrorCode err = chiaki_session_init(&g_session, &info, &g_chiaki_log);
            if (err != CHIAKI_ERR_SUCCESS) {
                std::cerr << "❌ [RP] chiaki_session_init 失败: " << chiaki_error_string(err) << "\n";
            } else {
                chiaki_session_set_event_cb(&g_session, rp_event_cb, nullptr);
                err = chiaki_session_start(&g_session);
                if (err != CHIAKI_ERR_SUCCESS) {
                    std::cerr << "❌ [RP] chiaki_session_start 失败: " << chiaki_error_string(err) << "\n";
                    chiaki_session_fini(&g_session);
                    std::memset(&g_session, 0, sizeof(g_session));
                } else {
                    g_session_live = true;
                    started = true;
                }
            }
        }

        if (started) {
            chiaki_session_join(&g_session);
            g_session_connected.store(false, std::memory_order_release);
            std::lock_guard<std::mutex> lk(g_sess_mtx);
            g_session_live = false;
            chiaki_session_fini(&g_session);
            std::memset(&g_session, 0, sizeof(g_session));
        }

        if (!g_rp_running.load(std::memory_order_acquire)) break;
        std::unique_lock<std::mutex> ulk(g_stop_mtx);
        g_stop_cv.wait_for(ulk, std::chrono::milliseconds(RP_RECONNECT_MS), [] {
            return !g_rp_running.load(std::memory_order_acquire);
        });
    }
}

void rp_pump_loop() {
    int tfd = timerfd_create(CLOCK_MONOTONIC, 0);
    if (tfd >= 0) {
        struct itimerspec its{};
        its.it_value.tv_nsec = 1;
        its.it_interval.tv_nsec = 1'000'000'000 / DEFAULT_FREQ;
        timerfd_settime(tfd, 0, &its, nullptr);
    }

    uint64_t last_seq = 0;
    bool have_last = false;
    ChiakiControllerState last_sent{};
    chiaki_controller_state_set_idle(&last_sent);

    while (g_rp_running.load(std::memory_order_acquire)) {
        if (tfd >= 0) {
            uint64_t exp = 0;
            ssize_t r = read(tfd, &exp, sizeof(exp));
            if (r < 0 && errno == EINTR) continue;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1'000'000 / DEFAULT_FREQ));
        }
        if (!g_rp_running.load(std::memory_order_acquire)) break;

        uint64_t seq = 0;
        PadLogical st = pad_publish_snapshot(&seq);
        if (have_last && seq == last_seq) continue;
        last_seq = seq;

        ChiakiControllerState mapped{};
        pad_rp_map(st, mapped);
        if (have_last && chiaki_controller_state_equals(&mapped, &last_sent)) continue;

        std::lock_guard<std::mutex> lk(g_sess_mtx);
        if (g_session_live && g_session_connected.load(std::memory_order_acquire)) {
            ChiakiErrorCode err = chiaki_session_set_controller_state(&g_session, &mapped);
            if (err == CHIAKI_ERR_SUCCESS) {
                last_sent = mapped;
                have_last = true;
            } else {
                // 发送通道已失效 (如 Takion socket 关闭返回 EBADF): 立即停会话触发 1s 重连
                g_session_connected.store(false, std::memory_order_release);
                have_last = false;
                chiaki_session_stop(&g_session);
            }
        } else {
            have_last = false;
        }
    }

    if (tfd >= 0) close(tfd);
}

} // namespace

void pad_rp_map(const PadLogical& st, ChiakiControllerState& out) {
    chiaki_controller_state_set_idle(&out);

    static constexpr struct {
        uint16_t src;
        uint32_t dst;
    } kBtnMap[] = {
        { PADBTN_A,          CHIAKI_CONTROLLER_BUTTON_CROSS },
        { PADBTN_B,          CHIAKI_CONTROLLER_BUTTON_MOON },
        { PADBTN_X,          CHIAKI_CONTROLLER_BUTTON_BOX },
        { PADBTN_Y,          CHIAKI_CONTROLLER_BUTTON_PYRAMID },
        { PADBTN_LB,         CHIAKI_CONTROLLER_BUTTON_L1 },
        { PADBTN_RB,         CHIAKI_CONTROLLER_BUTTON_R1 },
        { PADBTN_BACK,       CHIAKI_CONTROLLER_BUTTON_TOUCHPAD },
        { PADBTN_START,      CHIAKI_CONTROLLER_BUTTON_OPTIONS },
        { PADBTN_GUIDE,      CHIAKI_CONTROLLER_BUTTON_PS },
        { PADBTN_L3,         CHIAKI_CONTROLLER_BUTTON_L3 },
        { PADBTN_R3,         CHIAKI_CONTROLLER_BUTTON_R3 },
        { PADBTN_DPAD_UP,    CHIAKI_CONTROLLER_BUTTON_DPAD_UP },
        { PADBTN_DPAD_DOWN,  CHIAKI_CONTROLLER_BUTTON_DPAD_DOWN },
        { PADBTN_DPAD_LEFT,  CHIAKI_CONTROLLER_BUTTON_DPAD_LEFT },
        { PADBTN_DPAD_RIGHT, CHIAKI_CONTROLLER_BUTTON_DPAD_RIGHT },
        { PADBTN_TOUCH,      CHIAKI_CONTROLLER_BUTTON_TOUCHPAD },
    };
    for (const auto& m : kBtnMap) {
        if (st.btns & m.src) out.buttons |= m.dst;
    }

    out.l2_state = st.lt;
    out.r2_state = st.rt;
    out.left_x   = st.lx;
    out.left_y   = st.ly;
    out.right_x  = st.rx;
    out.right_y  = st.ry;
}

bool pad_rp_start(const RpAccount& acc) {
    std::string iface_err;
    std::string iface = !acc.iface.empty() ? acc.iface : rp_find_wired_iface(&iface_err);
    if (iface.empty()) {
        std::cerr << "❌ [RP] 仅允许走有线以太网, 但 " << iface_err << "\n";
        return false;
    }
    if (!rp_valid_iface(iface) || rp_is_wireless_iface(iface)) {
        std::cerr << "❌ [RP] 网卡 \"" << iface << "\" 不是物理有线以太网卡 (禁止走 WiFi)\n";
        return false;
    }
    std::cout << "✅ [RP] 仅走有线网卡: " << iface << " (SO_BINDTODEVICE; 有线断开不回落 WiFi)\n";

    std::call_once(g_chiaki_init_once, [] {
        g_chiaki_init_err = chiaki_lib_init();
        chiaki_log_init(&g_chiaki_log, CHIAKI_LOG_ERROR | CHIAKI_LOG_WARNING | CHIAKI_LOG_INFO,
                        rp_chiaki_log_cb, nullptr);
    });
    if (g_chiaki_init_err != CHIAKI_ERR_SUCCESS) {
        std::cerr << "❌ [RP] chiaki_lib_init 失败: " << chiaki_error_string(g_chiaki_init_err) << "\n";
        return false;
    }

    g_acc = acc;
    g_rp_running.store(true, std::memory_order_release);
    g_sess_thread = std::thread(rp_session_loop);
    g_pump_thread = std::thread(rp_pump_loop);
    return true;
}

void pad_rp_stop() {
    g_rp_running.store(false, std::memory_order_release);
    g_stop_cv.notify_all();

    if (g_pump_thread.joinable()) g_pump_thread.join();

    bool need_release_wait = false;
    {
        std::lock_guard<std::mutex> lk(g_sess_mtx);
        if (g_session_live) {
            ChiakiControllerState idle{};
            chiaki_controller_state_set_idle(&idle);
            chiaki_session_set_controller_state(&g_session, &idle);
            need_release_wait = true;
        }
    }
    if (need_release_wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(RP_RELEASE_WAIT_MS));
        std::lock_guard<std::mutex> lk(g_sess_mtx);
        if (g_session_live) chiaki_session_stop(&g_session);
    }

    if (g_sess_thread.joinable()) g_sess_thread.join();
}
