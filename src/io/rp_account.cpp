// ============================================================================
//  rp_account.cpp — rp_account.h 的实现
// ============================================================================

#include "io/rp_account.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

void set_err(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    size_t b = s.size();
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

bool any_nonzero(const uint8_t* p, size_t n) {
    uint8_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc |= p[i];
    return acc != 0;
}

int b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

int hex_val(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool ensure_parent_dir(const std::string& path, std::string* err) {
    size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return true;
    std::string dir = path.substr(0, slash);
    std::string cur;
    for (size_t i = 0; i < dir.size(); ++i) {
        cur.push_back(dir[i]);
        if (dir[i] == '/' || i + 1 == dir.size()) {
            if (cur == "/") continue;
            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                set_err(err, "创建目录失败: " + cur + " (" + std::strerror(errno) + ")");
                return false;
            }
        }
    }
    return true;
}

} // namespace

bool rp_valid_host(const std::string& host) {
    if (host.empty() || host.size() >= 256) return false;
    for (unsigned char c : host) {
        if (c <= 32 || c == 127 || c == '/' || c == '\\') return false;
    }
    return true;
}

bool rp_valid_iface(const std::string& iface) {
    if (iface.empty() || iface.size() >= 16) return false;
    for (unsigned char c : iface) {
        if (!(std::isalnum(c) || c == '.' || c == '_' || c == ':' || c == '-')) return false;
    }
    return true;
}

bool rp_is_wireless_iface(const std::string& iface) {
    if (!rp_valid_iface(iface)) return false;
    struct stat st{};
    const std::string base = "/sys/class/net/" + iface;
    return stat((base + "/wireless").c_str(), &st) == 0
        || stat((base + "/phy80211").c_str(), &st) == 0;
}

std::string rp_find_wired_iface(std::string* err) {
    DIR* dir = opendir("/sys/class/net");
    if (!dir) {
        set_err(err, "无法打开 /sys/class/net: " + std::string(std::strerror(errno)));
        return "";
    }
    std::vector<std::string> names;
    while (struct dirent* ent = readdir(dir)) {
        std::string n = ent->d_name;
        if (n == "." || n == ".." || n == "lo") continue;
        names.push_back(n);
    }
    closedir(dir);
    std::sort(names.begin(), names.end());

    std::string first_wired;
    for (const auto& n : names) {
        if (!rp_valid_iface(n)) continue;
        const std::string base = "/sys/class/net/" + n;
        struct stat st{};
        // 必须有 device 符号链接 (排除虚拟网桥/tun/veth)
        if (stat((base + "/device").c_str(), &st) != 0) continue;
        // 必须是 ARPHRD_ETHER (type == 1) 且非无线网卡
        std::ifstream tf(base + "/type");
        int iftype = 0;
        if (!(tf >> iftype) || iftype != 1) continue;
        if (rp_is_wireless_iface(n)) continue;

        if (first_wired.empty()) first_wired = n;
        std::ifstream of(base + "/operstate");
        std::string state;
        if (of >> state && state == "up") return n;
    }
    if (!first_wired.empty()) return first_wired;
    set_err(err, "未检测到物理有线以太网卡 (/sys/class/net)");
    return "";
}

bool rp_base64_decode_8(const std::string& b64_in, uint8_t out[8]) {
    std::string s = trim(b64_in);
    // 8 字节 Base64: 11 个有效字符 + 1 个 '=' (长度 12), 或无填充 11 字符
    if (s.size() != 12 && s.size() != 11) return false;
    if (s.size() == 12 && s[11] != '=') return false;
    uint8_t buf[9] = {};
    int bits = 0;
    uint32_t acc = 0;
    size_t pos = 0;
    for (size_t i = 0; i < 11; ++i) {
        int v = b64_val((unsigned char)s[i]);
        if (v < 0) return false;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            buf[pos++] = (uint8_t)((acc >> bits) & 0xFF);
        }
    }
    // 11 * 6 = 66 位 = 8 字节 + 2 个余位 (必须为 0)
    if (pos != 8 || (acc & ((1u << bits) - 1u)) != 0) return false;
    std::memcpy(out, buf, 8);
    return true;
}

std::string rp_base64_encode_8(const uint8_t in[8]) {
    static const char* kTbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(12);
    for (size_t i = 0; i < 8; i += 3) {
        uint32_t b0 = in[i];
        uint32_t b1 = (i + 1 < 8) ? in[i + 1] : 0;
        uint32_t b2 = (i + 2 < 8) ? in[i + 2] : 0;
        uint32_t tri = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kTbl[(tri >> 18) & 0x3F]);
        out.push_back(kTbl[(tri >> 12) & 0x3F]);
        out.push_back((i + 1 < 8) ? kTbl[(tri >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < 8) ? kTbl[tri & 0x3F] : '=');
    }
    return out;
}

bool rp_hex_decode_16(const std::string& hex_in, uint8_t out[16]) {
    std::string s = trim(hex_in);
    if (s.size() != 32) return false;
    for (size_t i = 0; i < 16; ++i) {
        int hi = hex_val((unsigned char)s[2 * i]);
        int lo = hex_val((unsigned char)s[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

std::string rp_hex_encode_16(const uint8_t in[16]) {
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(32);
    for (size_t i = 0; i < 16; ++i) {
        out.push_back(kHex[(in[i] >> 4) & 0xF]);
        out.push_back(kHex[in[i] & 0xF]);
    }
    return out;
}

bool rp_account_load(const std::string& path, RpAccount& out, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        set_err(err, "无法打开 RP 凭据文件: " + path + " (" + std::strerror(errno) + ")");
        return false;
    }

    RpAccount acc{};
    bool seen_host = false, seen_aid = false, seen_rkey = false, seen_morning = false;
    bool seen_name = false, seen_iface = false;
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::string s = trim(line);
        if (s.empty() || s[0] == '#') continue;

        size_t eq = s.find('=');
        if (eq == std::string::npos) {
            set_err(err, path + ":" + std::to_string(lineno) + ": 缺少 '='");
            return false;
        }
        std::string key = trim(s.substr(0, eq));
        std::string val = trim(s.substr(eq + 1));

        if (key == "host") {
            if (seen_host) { set_err(err, path + ": 重复键 host"); return false; }
            if (!rp_valid_host(val)) { set_err(err, path + ": 非法 host \"" + val + "\""); return false; }
            acc.host = val;
            seen_host = true;
        } else if (key == "account_id") {
            if (seen_aid) { set_err(err, path + ": 重复键 account_id"); return false; }
            if (!rp_base64_decode_8(val, acc.account_id.data())) {
                set_err(err, path + ": account_id 必须是 8 字节 Base64");
                return false;
            }
            seen_aid = true;
        } else if (key == "regist_key") {
            if (seen_rkey) { set_err(err, path + ": 重复键 regist_key"); return false; }
            if (!rp_hex_decode_16(val, acc.regist_key.data()) || !any_nonzero(acc.regist_key.data(), 16)) {
                set_err(err, path + ": regist_key 必须是非零 32 位十六进制");
                return false;
            }
            seen_rkey = true;
        } else if (key == "morning") {
            if (seen_morning) { set_err(err, path + ": 重复键 morning"); return false; }
            if (!rp_hex_decode_16(val, acc.morning.data()) || !any_nonzero(acc.morning.data(), 16)) {
                set_err(err, path + ": morning 必须是非零 32 位十六进制");
                return false;
            }
            seen_morning = true;
        } else if (key == "name") {
            if (seen_name) { set_err(err, path + ": 重复键 name"); return false; }
            for (unsigned char c : val) {
                if (c < 32 || c == 127) {
                    set_err(err, path + ": name 含非法控制字符");
                    return false;
                }
            }
            acc.name = val;
            seen_name = true;
        } else if (key == "iface") {
            if (seen_iface) { set_err(err, path + ": 重复键 iface"); return false; }
            if (!rp_valid_iface(val)) { set_err(err, path + ": 非法 iface \"" + val + "\""); return false; }
            acc.iface = val;
            seen_iface = true;
        } else {
            set_err(err, path + ":" + std::to_string(lineno) + ": 未知键 \"" + key + "\"");
            return false;
        }
    }

    if (!seen_host || !seen_aid || !seen_rkey || !seen_morning) {
        set_err(err, path + ": 缺少必需字段 (host / account_id / regist_key / morning)");
        return false;
    }

    out = std::move(acc);
    return true;
}

bool rp_account_save(const std::string& path, const RpAccount& acc, std::string* err) {
    if (!rp_valid_host(acc.host)) {
        set_err(err, "非法 host: " + acc.host);
        return false;
    }
    if (!any_nonzero(acc.regist_key.data(), 16) || !any_nonzero(acc.morning.data(), 16)) {
        set_err(err, "regist_key 与 morning 不可全零");
        return false;
    }
    if (!ensure_parent_dir(path, err)) return false;

    std::ostringstream oss;
    oss << "# PS5 Remote Play pairing credentials (0600)\n";
    oss << "host=" << acc.host << "\n";
    oss << "account_id=" << rp_base64_encode_8(acc.account_id.data()) << "\n";
    oss << "regist_key=" << rp_hex_encode_16(acc.regist_key.data()) << "\n";
    oss << "morning=" << rp_hex_encode_16(acc.morning.data()) << "\n";
    if (!acc.name.empty()) {
        oss << "name=" << acc.name << "\n";
    }
    if (!acc.iface.empty()) {
        if (!rp_valid_iface(acc.iface)) {
            set_err(err, "非法 iface: " + acc.iface);
            return false;
        }
        oss << "iface=" << acc.iface << "\n";
    }
    const std::string content = oss.str();

    std::string tmp = path + ".tmp." + std::to_string((long)getpid());
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        set_err(err, "无法创建临时文件 " + tmp + ": " + std::strerror(errno));
        return false;
    }
    // 明确兜底 0600 权限 (不受 umask 意外放宽影响, 只收紧不放宽)
    fchmod(fd, 0600);

    size_t written = 0;
    while (written < content.size()) {
        ssize_t n = write(fd, content.data() + written, content.size() - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            set_err(err, "写入失败 " + tmp + ": " + std::strerror(errno));
            close(fd);
            unlink(tmp.c_str());
            return false;
        }
        written += (size_t)n;
    }
    fsync(fd);
    close(fd);

    if (rename(tmp.c_str(), path.c_str()) != 0) {
        set_err(err, "重命名失败 " + tmp + " -> " + path + ": " + std::strerror(errno));
        unlink(tmp.c_str());
        return false;
    }
    return true;
}
