// ============================================================================
//  rp_regist — PS5 遥控游玩 (-M rp) 配对与连通探针 (build/rp_regist)
//
//  两种用法:
//    1. 一次性配对 (生成 0600 凭据文件, 默认写入 run/rp-account):
//         ./build/rp_regist --host <PS5_IP> --account-id <B64> --pin <8位> \
//             [--out run/rp-account] [--name 客厅]
//       PS5 端: 设置 → 系统 → 遥控游玩 → 启用遥控游玩 → 关联设备 (8 位 PIN)。
//       Account-ID Base64 可用 third_party/chiaki-ng/scripts/psn-account-id.py 获取。
//
//    2. 连通探针 (验证凭据可连上 PS5 纯控制会话, 连上即退出 0):
//         ./build/rp_regist --probe [--account run/rp-account] [--timeout 15]
// ============================================================================

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

#include <chiaki/regist.h>
#include <chiaki/session.h>
#include <chiaki/sock.h>

#include "io/pad_rp.h"
#include "io/rp_account.h"

namespace {

constexpr int REGIST_TIMEOUT_S = 60;
constexpr int PROBE_TIMEOUT_S  = 15;

void usage(const char* prog) {
    std::cout
        << "用法:\n"
        << "  配对: " << prog << " --host <PS5_IP> --account-id <Base64> --pin <8位PIN> "
           "[--out run/rp-account] [--name <标签>] [--iface <有线网卡>]\n"
        << "  探针: " << prog << " --probe [--account run/rp-account] [--iface <有线网卡>] [--timeout <秒>]\n";
}

struct RegistWait {
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;
    ChiakiRegistEventType result = CHIAKI_REGIST_EVENT_TYPE_FINISHED_FAILED;
    ChiakiRegisteredHost host_out{};
    bool have_host = false;
};

void on_regist_event(ChiakiRegistEvent* ev, void* user) {
    auto* w = static_cast<RegistWait*>(user);
    std::lock_guard<std::mutex> lk(w->mtx);
    w->result = ev->type;
    if (ev->type == CHIAKI_REGIST_EVENT_TYPE_FINISHED_SUCCESS && ev->registered_host) {
        w->host_out = *ev->registered_host;
        w->have_host = true;
    }
    w->done = true;
    w->cv.notify_all();
}

struct ProbeWait {
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;
    int rc = 1;
    std::string msg;
};

void on_probe_event(ChiakiEvent* ev, void* user) {
    auto* w = static_cast<ProbeWait*>(user);
    std::lock_guard<std::mutex> lk(w->mtx);
    switch (ev->type) {
    case CHIAKI_EVENT_CONNECTED:
        w->rc = 0;
        w->msg = "connected";
        w->done = true;
        w->cv.notify_all();
        break;
    case CHIAKI_EVENT_LOGIN_PIN_REQUEST:
        w->rc = 2;
        w->msg = "PS5 要求 4 位登录 PIN (请在 PS5 设置中关闭该账号登录密码)";
        w->done = true;
        w->cv.notify_all();
        break;
    case CHIAKI_EVENT_QUIT:
        if (!w->done) {
            w->rc = 1;
            w->msg = chiaki_quit_reason_string(ev->quit.reason);
            w->done = true;
            w->cv.notify_all();
        }
        break;
    default:
        break;
    }
}

void log_cb(ChiakiLogLevel level, const char* msg, void* user) {
    (void)user;
    if (level == CHIAKI_LOG_ERROR) {
        std::cerr << "[chiaki] " << (msg ? msg : "") << "\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    bool probe = false;
    std::string host, b64_id, pin_s, out_path = "run/rp-account", name, iface_arg;
    int timeout_s = -1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--probe") probe = true;
        else if (a == "--host"       && i + 1 < argc) host = argv[++i];
        else if (a == "--account-id" && i + 1 < argc) b64_id = argv[++i];
        else if (a == "--pin"        && i + 1 < argc) pin_s = argv[++i];
        else if ((a == "--out" || a == "--account") && i + 1 < argc) out_path = argv[++i];
        else if (a == "--name"       && i + 1 < argc) name = argv[++i];
        else if (a == "--iface"      && i + 1 < argc) iface_arg = argv[++i];
        else if (a == "--timeout"    && i + 1 < argc) timeout_s = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::cerr << "❌ 未知参数: " << a << "\n"; usage(argv[0]); return 1; }
    }

    auto resolve_wired = [&](const std::string& preferred) -> std::string {
        std::string err;
        std::string w = !preferred.empty() ? preferred : rp_find_wired_iface(&err);
        if (w.empty()) {
            std::cerr << "❌ 仅允许走有线以太网, 但 " << err << "\n";
            return "";
        }
        if (!rp_valid_iface(w) || rp_is_wireless_iface(w)) {
            std::cerr << "❌ 网卡 \"" << w << "\" 不是物理有线以太网卡 (禁止走 WiFi)\n";
            return "";
        }
        return w;
    };

    ChiakiErrorCode init_err = chiaki_lib_init();
    if (init_err != CHIAKI_ERR_SUCCESS) {
        std::cerr << "❌ chiaki_lib_init 失败: " << chiaki_error_string(init_err) << "\n";
        return 1;
    }
    ChiakiLog log{};
    chiaki_log_init(&log, CHIAKI_LOG_ERROR | CHIAKI_LOG_WARNING, log_cb, nullptr);

    if (probe) {
        RpAccount acc{};
        std::string err;
        if (!rp_account_load(out_path, acc, &err)) {
            std::cerr << "❌ " << err << "\n";
            return 1;
        }
        std::string wired = resolve_wired(!iface_arg.empty() ? iface_arg : acc.iface);
        if (wired.empty()) return 1;
        chiaki_sock_set_bind_iface(wired.c_str());
        std::cout << "✅ 探测仅走有线网卡: " << wired << " (SO_BINDTODEVICE; 若报 EPERM 请用 sudo)\n";
        int wait_s = timeout_s > 0 ? timeout_s : PROBE_TIMEOUT_S;

        ChiakiConnectInfo info{};
        info.host = acc.host.c_str();
        info.ps5 = true;
        std::memcpy(info.psn_account_id, acc.account_id.data(), 8);
        std::memcpy(info.regist_key, acc.regist_key.data(), 16);
        std::memcpy(info.morning, acc.morning.data(), 16);
        chiaki_connect_video_profile_preset(&info.video_profile,
                                            CHIAKI_VIDEO_RESOLUTION_PRESET_360p,
                                            CHIAKI_VIDEO_FPS_PRESET_30);
        info.audio_video_disabled = CHIAKI_AUDIO_VIDEO_DISABLED;
        info.video_profile_auto_downgrade = true;
        info.enable_dualsense = false;
        info.packet_loss_max = RP_PACKET_LOSS_MAX;

        ChiakiSession sess{};
        ChiakiErrorCode e = chiaki_session_init(&sess, &info, &log);
        if (e != CHIAKI_ERR_SUCCESS) {
            std::cerr << "❌ chiaki_session_init 失败: " << chiaki_error_string(e) << "\n";
            return 1;
        }
        ProbeWait pw;
        chiaki_session_set_event_cb(&sess, on_probe_event, &pw);
        e = chiaki_session_start(&sess);
        if (e != CHIAKI_ERR_SUCCESS) {
            std::cerr << "❌ chiaki_session_start 失败: " << chiaki_error_string(e) << "\n";
            chiaki_session_fini(&sess);
            return 1;
        }

        {
            std::unique_lock<std::mutex> lk(pw.mtx);
            pw.cv.wait_for(lk, std::chrono::seconds(wait_s), [&] { return pw.done; });
        }
        chiaki_session_stop(&sess);
        chiaki_session_join(&sess);
        chiaki_session_fini(&sess);

        if (!pw.done) {
            std::cerr << "❌ 探测超时 (" << wait_s << "s): 检查 PS5 开机/待机网络与同网段连通性 (" << acc.host << ")\n";
            return 1;
        }
        if (pw.rc == 0) {
            std::cout << "✅ 探测成功: 已连通 PS5 (" << acc.host << ")\n";
            return 0;
        }
        std::cerr << "❌ 探测失败: " << pw.msg << "\n";
        return pw.rc;
    }

    if (!rp_valid_host(host) || b64_id.empty() || pin_s.size() != 8) {
        std::cerr << "❌ 配对需要 --host <PS5_IP> --account-id <Base64> --pin <8位数字>\n";
        usage(argv[0]);
        return 1;
    }
    uint8_t account_id[8] = {};
    if (!rp_base64_decode_8(b64_id, account_id)) {
        std::cerr << "❌ --account-id 必须是 8 字节 Base64\n";
        return 1;
    }
    uint32_t pin = 0;
    for (char c : pin_s) {
        if (c < '0' || c > '9') {
            std::cerr << "❌ --pin 必须是 8 位数字\n";
            return 1;
        }
        pin = pin * 10u + (uint32_t)(c - '0');
    }

    std::string wired = resolve_wired(iface_arg);
    if (wired.empty()) return 1;
    chiaki_sock_set_bind_iface(wired.c_str());
    std::cout << "✅ 配对仅走有线网卡: " << wired << " (SO_BINDTODEVICE; 若报 EPERM 请用 sudo)\n";

    ChiakiRegistInfo rinfo{};
    rinfo.target = CHIAKI_TARGET_PS5_1;
    rinfo.host = host.c_str();
    std::memcpy(rinfo.psn_account_id, account_id, 8);
    rinfo.pin = pin;

    RegistWait rw;
    ChiakiRegist regist{};
    ChiakiErrorCode re = chiaki_regist_start(&regist, &log, &rinfo, on_regist_event, &rw);
    if (re != CHIAKI_ERR_SUCCESS) {
        std::cerr << "❌ chiaki_regist_start 失败: " << chiaki_error_string(re) << "\n";
        return 1;
    }

    int wait_s = timeout_s > 0 ? timeout_s : REGIST_TIMEOUT_S;
    bool finished = false;
    {
        std::unique_lock<std::mutex> lk(rw.mtx);
        finished = rw.cv.wait_for(lk, std::chrono::seconds(wait_s), [&] { return rw.done; });
    }
    if (!finished) chiaki_regist_stop(&regist);
    chiaki_regist_fini(&regist);

    if (!finished) {
        std::cerr << "❌ 配对超时 (" << wait_s << "s): 确认 PS5 处于关联设备界面且 PIN 未过期\n";
        return 1;
    }
    if (rw.result != CHIAKI_REGIST_EVENT_TYPE_FINISHED_SUCCESS || !rw.have_host) {
        std::cerr << "❌ 配对失败: 检查 --host / --account-id / --pin\n";
        return 1;
    }

    RpAccount acc{};
    acc.host = host;
    std::memcpy(acc.account_id.data(), account_id, 8);
    std::memcpy(acc.regist_key.data(), rw.host_out.rp_regist_key, 16);
    std::memcpy(acc.morning.data(), rw.host_out.rp_key, 16);
    acc.name = name;
    acc.iface = iface_arg;

    std::string err;
    if (!rp_account_save(out_path, acc, &err)) {
        std::cerr << "❌ " << err << "\n";
        return 1;
    }
    std::cout << "✅ 配对成功: 凭据已写入 " << out_path << " (0600)\n";
    return 0;
}
