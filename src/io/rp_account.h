// ============================================================================
//  rp_account.h — PS5 遥控游玩 (-M rp) 的配对凭据文件读写与校验
//
//  文件格式 (默认 run/rp-account, 权限 0600, 不进版本库):
//    key=value 逐行文本 (# 注释与空行忽略, 容忍 CRLF):
//      host       — PS5 局域网 IP / 主机名 (非空, 无空白/控制符/斜杠, < 256 字节)
//      account_id — PSN Account-ID (8 字节, Base64 编码)
//      regist_key — 配对注册密钥 (16 字节, 32 位十六进制)
//      morning    — 配对会话密钥 (16 字节, 32 位十六进制)
//      name       — 可选人类可读标签
//      iface      — 可选绑定有线网卡名 (如 enP4p65s0; 留空 = 自动检测物理有线以太网)
//    未知键、重复键、缺项或格式错误一律拒绝, 不静默忽略。
// ============================================================================

#pragma once

#include <array>
#include <cstdint>
#include <string>

struct RpAccount {
    std::string host;
    std::array<uint8_t, 8>  account_id{};
    std::array<uint8_t, 16> regist_key{};
    std::array<uint8_t, 16> morning{};
    std::string name;
    std::string iface;
};

// 主机名/IP 校验: 非空, < 256 字节, 不含控制符/空白/斜杠 (防路径或头注入)
bool rp_valid_host(const std::string& host);

// 网卡名校验: 非空, < 16 字节 (IFNAMSIZ), 仅允许 [A-Za-z0-9._:-]
bool rp_valid_iface(const std::string& iface);

// 无线网卡判定: /sys/class/net/<iface>/wireless 或 phy80211 存在即视为无线
bool rp_is_wireless_iface(const std::string& iface);

// 自动查找物理有线以太网卡 (排除 lo、虚拟接口与无线接口):
//   优先返回 operstate == "up" 的有线网卡; 若全部插口未连线 (operstate != "up"),
//   仍返回首个物理有线网卡名 (供 SO_BINDTODEVICE 死绑, 断线时绝不回落到 WiFi)。
//   未找到任何物理有线网卡时写 err 并返回空串。
std::string rp_find_wired_iface(std::string* err = nullptr);

// 8 字节 Base64 编解码 (用于 PSN Account-ID; 解码必须恰好产出 8 字节)
bool rp_base64_decode_8(const std::string& b64, uint8_t out[8]);
std::string rp_base64_encode_8(const uint8_t in[8]);

// 16 字节十六进制编解码 (32 位 hex, 大小写不敏感; 解码必须恰好产出 16 字节)
bool rp_hex_decode_16(const std::string& hex, uint8_t out[16]);
std::string rp_hex_encode_16(const uint8_t in[16]);

// 读取并校验凭据文件; 失败时写入 err (若非空) 并返回 false
bool rp_account_load(const std::string& path, RpAccount& out, std::string* err = nullptr);

// 原子写入凭据文件 (0600 权限, 同目录临时文件 + fsync + rename);
//   写入前同样校验字段有效性, 失败时写入 err 并返回 false
bool rp_account_save(const std::string& path, const RpAccount& acc, std::string* err = nullptr);
