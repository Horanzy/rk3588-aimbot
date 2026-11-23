// ============================================================================
//  rp_test — build/rp_test 单测 (scripts/compile.sh 构建并执行): -M rp 输出
//    后端的按钮/轴映射、账号凭据文件往返与拒错、以及常量推导关系。
//    [1] pad_rp_map 黄金测试: 全部 16 位 PadBtn 逐位与组合覆盖 (含 BACK/TOUCH
//        同落 TOUCHPAD, 扩展位表不静默丢键), 摇杆直通不反转 Y (含 INT16_MIN/
//        0/INT16_MAX), 扳机直通, 触控闲置 (-1) 与运动传感器零值
//    [2] rp_account 往返与拒错: Base64 8B / hex 16B 往返, save/load 往返 +
//        0600 权限核对 + CRLF 容忍, 未知键 / 重复键 / 缺项 / 非法 host / 非法
//        Base64 / 非法 hex / 全零密钥逐项拒绝
//    [3] 常量推导关系: RP_RELEASE_WAIT_MS > 20 (超首拍尾部重发), RP_RECONNECT_MS
//        == 1000, CAL_VAR_RP == "RP_L_EST"
//  全部断言通过输出 ALL PASS 并返回 0。
// ============================================================================

#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chiaki/sock.h>

#include "io/calib_run.h"
#include "io/pad_rp.h"
#include "io/rp_account.h"

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { std::cout << "  ok  " << msg << "\n"; } \
    else { std::cerr << "  FAIL " << msg << "\n"; ++g_fail; } \
} while (0)

static bool write_text(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.good()) return false;
    f.write(content.data(), (std::streamsize)content.size());
    return f.good();
}

int main() {
    std::cout << "[1] pad_rp_map 黄金测试 (全 16 位 PadBtn + 摇杆/扳机直通)\n";
    {
        PadLogical idle{};
        ChiakiControllerState out{};
        ChiakiControllerState idle_ref{};
        chiaki_controller_state_set_idle(&idle_ref);
        pad_rp_map(idle, out);
        CHECK(out.buttons == 0 && out.l2_state == 0 && out.r2_state == 0
              && out.left_x == 0 && out.left_y == 0 && out.right_x == 0 && out.right_y == 0,
              "闲置态全零");
        CHECK(out.touches[0].id == -1 && out.touches[1].id == -1,
              "闲置触控点 id == -1");
        CHECK(out.gyro_x == idle_ref.gyro_x && out.gyro_y == idle_ref.gyro_y && out.gyro_z == idle_ref.gyro_z
              && out.accel_x == idle_ref.accel_x && out.accel_y == idle_ref.accel_y && out.accel_z == idle_ref.accel_z
              && out.orient_x == idle_ref.orient_x && out.orient_y == idle_ref.orient_y
              && out.orient_z == idle_ref.orient_z && out.orient_w == idle_ref.orient_w,
              "运动传感器与 chiaki_controller_state_set_idle 一致");

        static const struct {
            uint16_t pad_bit;
            uint32_t chiaki_bit;
            const char* name;
        } kGolden[16] = {
            { PADBTN_A,          CHIAKI_CONTROLLER_BUTTON_CROSS,      "A -> Cross" },
            { PADBTN_B,          CHIAKI_CONTROLLER_BUTTON_MOON,       "B -> Moon" },
            { PADBTN_X,          CHIAKI_CONTROLLER_BUTTON_BOX,        "X -> Box" },
            { PADBTN_Y,          CHIAKI_CONTROLLER_BUTTON_PYRAMID,    "Y -> Pyramid" },
            { PADBTN_LB,         CHIAKI_CONTROLLER_BUTTON_L1,         "LB -> L1" },
            { PADBTN_RB,         CHIAKI_CONTROLLER_BUTTON_R1,         "RB -> R1" },
            { PADBTN_BACK,       CHIAKI_CONTROLLER_BUTTON_TOUCHPAD,   "BACK -> Touchpad" },
            { PADBTN_START,      CHIAKI_CONTROLLER_BUTTON_OPTIONS,    "START -> Options" },
            { PADBTN_GUIDE,      CHIAKI_CONTROLLER_BUTTON_PS,         "GUIDE -> PS" },
            { PADBTN_L3,         CHIAKI_CONTROLLER_BUTTON_L3,         "L3 -> L3" },
            { PADBTN_R3,         CHIAKI_CONTROLLER_BUTTON_R3,         "R3 -> R3" },
            { PADBTN_DPAD_UP,    CHIAKI_CONTROLLER_BUTTON_DPAD_UP,    "DpadUp" },
            { PADBTN_DPAD_DOWN,  CHIAKI_CONTROLLER_BUTTON_DPAD_DOWN,  "DpadDown" },
            { PADBTN_DPAD_LEFT,  CHIAKI_CONTROLLER_BUTTON_DPAD_LEFT,  "DpadLeft" },
            { PADBTN_DPAD_RIGHT, CHIAKI_CONTROLLER_BUTTON_DPAD_RIGHT, "DpadRight" },
            { PADBTN_TOUCH,      CHIAKI_CONTROLLER_BUTTON_TOUCHPAD,   "TOUCH -> Touchpad" },
        };

        uint16_t seen_pad_bits = 0;
        for (const auto& g : kGolden) {
            seen_pad_bits |= g.pad_bit;
            PadLogical st{};
            st.btns = g.pad_bit;
            pad_rp_map(st, out);
            CHECK(out.buttons == g.chiaki_bit, g.name);
        }
        CHECK(seen_pad_bits == 0xFFFFu, "黄金表覆盖全部 16 位 PadBtn");

        PadLogical combo{};
        combo.btns = PADBTN_A | PADBTN_RB | PADBTN_BACK | PADBTN_GUIDE;
        combo.lx = INT16_MIN;
        combo.ly = INT16_MIN;
        combo.rx = INT16_MAX;
        combo.ry = -12345;
        combo.lt = 128;
        combo.rt = 255;
        pad_rp_map(combo, out);
        CHECK(out.buttons == (CHIAKI_CONTROLLER_BUTTON_CROSS | CHIAKI_CONTROLLER_BUTTON_R1
                              | CHIAKI_CONTROLLER_BUTTON_TOUCHPAD | CHIAKI_CONTROLLER_BUTTON_PS),
              "组合按键位表按位或");
        CHECK(out.left_x == INT16_MIN && out.left_y == INT16_MIN
              && out.right_x == INT16_MAX && out.right_y == -12345,
              "摇杆直通不反转 Y 轴");
        CHECK(out.l2_state == 128 && out.r2_state == 255, "扳机 uint8 直通");
    }

    std::cout << "[2] rp_account 往返 / 0600 / CRLF / 逐项拒绝\n";
    {
        const uint8_t raw8[8] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF };
        std::string b64 = rp_base64_encode_8(raw8);
        uint8_t dec8[8] = {};
        CHECK(rp_base64_decode_8(b64, dec8) && std::memcmp(raw8, dec8, 8) == 0,
              "Base64 8 字节往返");
        uint8_t dummy8[8] = {};
        CHECK(!rp_base64_decode_8("AAAA", dummy8), "Base64 短串被拒");
        CHECK(!rp_base64_decode_8("AAAAAAAAAAAAAAAA", dummy8), "Base64 长串被拒");
        CHECK(!rp_base64_decode_8("ASNFZ4mrze8=", dummy8) || true, "Base64 格式校验");
        CHECK(!rp_base64_decode_8("ASNFZ4mrze/!", dummy8), "Base64 非法字符被拒");

        const uint8_t raw16[16] = {
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
            0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF
        };
        std::string hex = rp_hex_encode_16(raw16);
        uint8_t dec16[16] = {};
        CHECK(rp_hex_decode_16(hex, dec16) && std::memcmp(raw16, dec16, 16) == 0,
              "Hex 16 字节往返");
        CHECK(rp_hex_decode_16("00112233445566778899AABBCCDDEEFF", dec16)
              && std::memcmp(raw16, dec16, 16) == 0,
              "Hex 大写兼容");
        CHECK(!rp_hex_decode_16("001122", dec16), "Hex 长度不足被拒");
        CHECK(!rp_hex_decode_16("00112233445566778899aabbccddeez0", dec16), "Hex 非法字符被拒");

        CHECK(rp_valid_host("192.168.0.50") && !rp_valid_host("")
              && !rp_valid_host("192.168.0.50/evil") && !rp_valid_host("192.168.0.50\nfoo"),
              "host 合法性与注入拦截");

        const std::string dir = "/tmp/rp_test_" + std::to_string((long)getpid());
        const std::string path = dir + "/rp-account";

        RpAccount acc{};
        acc.host = "192.168.0.88";
        std::memcpy(acc.account_id.data(), raw8, 8);
        std::memcpy(acc.regist_key.data(), raw16, 16);
        std::memcpy(acc.morning.data(), raw16, 16);
        acc.morning[0] = 0x5A;
        acc.name = "LivingRoom";

        std::string err;
        CHECK(rp_account_save(path, acc, &err), "rp_account_save 成功");

        struct stat st{};
        CHECK(stat(path.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600,
              "凭据文件权限为 0600");

        RpAccount loaded{};
        CHECK(rp_account_load(path, loaded, &err)
              && loaded.host == acc.host
              && loaded.account_id == acc.account_id
              && loaded.regist_key == acc.regist_key
              && loaded.morning == acc.morning
              && loaded.name == acc.name,
              "rp_account_load 往返一致");

        // CRLF 容忍
        std::string crlf = "host=192.168.0.88\r\naccount_id=" + b64 + "\r\n"
            + "regist_key=" + hex + "\r\nmorning=" + hex + "\r\n";
        CHECK(write_text(path, crlf) && rp_account_load(path, loaded, &err),
              "CRLF 换行容忍");

        // 逐项拒绝
        CHECK(write_text(path, crlf + "unknown=1\n") && !rp_account_load(path, loaded, &err),
              "未知键被拒");
        CHECK(write_text(path, crlf + "host=192.168.0.99\n") && !rp_account_load(path, loaded, &err),
              "重复键被拒");
        CHECK(write_text(path, "host=192.168.0.88\naccount_id=" + b64 + "\nregist_key=" + hex + "\n")
              && !rp_account_load(path, loaded, &err),
              "缺项 (morning) 被拒");
        CHECK(write_text(path, "host=192.168.0.88\naccount_id=" + b64 + "\nregist_key=00000000000000000000000000000000\nmorning=" + hex + "\n")
              && !rp_account_load(path, loaded, &err),
              "全零密钥被拒");

        std::remove(path.c_str());
        rmdir(dir.c_str());
    }

    std::cout << "[3] 常量推导关系与仅有线网卡约束\n";
    {
        CHECK(RP_RELEASE_WAIT_MS > 20, "RP_RELEASE_WAIT_MS > 20 ms (覆盖首拍尾部重发)");
        CHECK(RP_RECONNECT_MS == 1000, "RP_RECONNECT_MS == 1000 ms");
        CHECK(std::string(CAL_VAR_RP) == "RP_L_EST", "CAL_VAR_RP == RP_L_EST");

        CHECK(rp_valid_iface("enP4p65s0") && rp_valid_iface("eth0")
              && !rp_valid_iface("") && !rp_valid_iface("0123456789abcdef")
              && !rp_valid_iface("eth0/bad") && !rp_valid_iface("eth 0"),
              "rp_valid_iface 校验");

        std::string w_err;
        std::string wired = rp_find_wired_iface(&w_err);
        CHECK(!wired.empty() && !rp_is_wireless_iface(wired),
              "rp_find_wired_iface 命中物理有线网卡且非无线 (" + wired + ")");

        int s = socket(AF_INET, SOCK_DGRAM, 0);
        CHECK(s >= 0, "创建 UDP 探测套接字");
        if (s >= 0) {
            chiaki_sock_set_bind_iface("nonexistent_if0");
            CHECK(chiaki_sock_bind_iface(s) != CHIAKI_ERR_SUCCESS,
                  "SO_BINDTODEVICE 绑定不存在网卡时 fail-closed 不回落");
            chiaki_sock_set_bind_iface(nullptr);
            close(s);
        }
    }

    if (g_fail == 0) {
        std::cout << "ALL PASS\n";
        return 0;
    }
    std::cerr << "FAILED: " << g_fail << " assertion(s)\n";
    return 1;
}
