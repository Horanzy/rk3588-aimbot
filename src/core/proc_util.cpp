// ============================================================================
//  proc_util.cpp — proc_util.h 的实现: 实例锁的取用与两处 /proc 读取。
//
//  为什么是 flock 而不是"查 pidfile 里那个 pid 还在不在": 文件锁挂在**打开的文件
//    描述**上, 进程退出 (含被 kill -9、崩溃、内核 OOM 收走) 时由内核释放 —— 没有
//    残骸可清, 也不可能把"别人重启后复用了同一个 pid"读成"上一个实例还在跑"。
//    pidfile 两样都有: 崩溃留下一个假 pid, 而 pid 复用让这个假 pid 还能存在进程。
//  锁文件放 /run (tmpfs): 一次重启自然清空, 与"进程级独占"这个语义一致。
//
//  /proc/<pid>/cmdline 是 NUL 分隔且以 NUL 结尾的 argv; 只在诊断输出里出现, 故
//    读不到就退回空串, 由调用方只说 pid, 不猜。
// ============================================================================

#include "core/proc_util.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

const char* INSTANCE_LOCK_PATH = "/run/aimbot.lock";

// 接管的有界等待。SIGTERM 之后的 grace: 本库自己的停机路径是"信号唤醒阻塞的端点 ioctl
//   → 三线程 join → 释放 UDC", 实测亚秒级完成, 3s 是它一个量级以上的余量 —— 超过它就不
//   是"在停"而是卡在某个不返回的调用里, 只剩 SIGKILL。SIGKILL 之后 1s: 等它从 pid 表消失
//   (僵尸另说 —— 锁在进程死亡时已由内核释放)。
const int INSTANCE_TERM_GRACE_MS = 3000;
const int INSTANCE_LOCK_WAIT_MS  = 2000;

std::string proc_cmdline(int pid) {
    if (pid <= 0) return "";
    std::ifstream f("/proc/" + std::to_string(pid) + "/cmdline", std::ios::binary);
    if (!f) return "";
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char& c : s) if (c == '\0') c = ' ';
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

std::vector<std::pair<int, std::string>> proc_fd_holders(const char* node, int exclude_pid) {
    std::vector<std::pair<int, std::string>> out;
    struct stat want{};
    if (stat(node, &want) != 0) return out;              // 节点不存在: 没人持有它
    DIR* p = opendir("/proc");
    if (!p) return out;
    struct dirent* de;
    while ((de = readdir(p)) != nullptr) {
        if (de->d_name[0] < '0' || de->d_name[0] > '9') continue;   // /proc 下只关注 pid 目录
        const int pid = atoi(de->d_name);
        if (pid <= 0 || pid == exclude_pid) continue;
        const std::string fdd = std::string("/proc/") + de->d_name + "/fd";
        DIR* fd = opendir(fdd.c_str());
        if (!fd) continue;                               // 进程刚退出, 或不属于本用户 (读不到)
        bool hit = false;
        struct dirent* fe;
        while (!hit && (fe = readdir(fd)) != nullptr) {
            if (fe->d_name[0] == '.') continue;
            struct stat got{};
            const std::string fdp = fdd + "/" + fe->d_name;
            if (stat(fdp.c_str(), &got) != 0) continue;   // 该描述符已关闭
            // (dev, ino) 对字符设备与普通文件同样成立: 前者比设备号, 后者比 inode ——
            //   锁文件是普通文件, 用设备号是找不到持有者的。
            hit = got.st_dev == want.st_dev && got.st_ino == want.st_ino;
        }
        closedir(fd);
        if (hit) out.push_back({pid, proc_cmdline(pid)});
    }
    closedir(p);
    std::sort(out.begin(), out.end());                    // 输出与扫描顺序无关
    return out;
}


// 自己的可执行文件的位置 → 部署根: /proc/self/exe = <root>/bin/aimbot, 故根 = 它的上两级。
//   不依赖调用目录, 也不依赖 argv[0] (它可能是相对路径, 也可能已被替换)。
static std::string own_root() {
    char self[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) return "";
    self[n] = '\0';
    std::string p = self;
    for (int up = 0; up < 2; ++up) {
        const size_t s = p.rfind('/');
        if (s == std::string::npos) return "";
        p.resize(s);
    }
    return p;
}

std::string proc_exe_path(int pid) {
    if (pid <= 0) return "";
    char buf[PATH_MAX];
    const ssize_t n = readlink(("/proc/" + std::to_string(pid) + "/exe").c_str(), buf, sizeof(buf) - 1);
    if (n <= 0) return "";                                 // 已退出, 或不是本用户的进程
    buf[n] = '\0';
    return buf;
}

bool proc_is_ours(int pid, const std::string& root) {
    if (pid <= 0 || root.empty()) return false;
    const std::string exe = proc_exe_path(pid);
    if (exe.empty()) return false;                         // 读不到就不动它 (宁可不接管)
    return exe.size() > root.size() && exe.compare(0, root.size(), root) == 0
           && exe[root.size()] == '/';
}

static bool pid_gone(int pid) { return kill(pid, 0) != 0 && errno == ESRCH; }

static bool wait_pid_gone(int pid, int ms) {
    for (int t = 0; t < ms; t += 20) {
        if (pid_gone(pid)) return true;
        usleep(20000);
    }
    return pid_gone(pid);
}

bool proc_terminate_owned(int pid, int grace_ms) {
    if (pid <= 0 || pid_gone(pid)) return true;
    if (kill(pid, SIGTERM) != 0 && errno == ESRCH) return true;
    if (wait_pid_gone(pid, grace_ms)) return true;
    std::cerr << "[接管] pid " << pid << " 未在 " << grace_ms << "ms 内自己退出 → SIGKILL\n";
    kill(pid, SIGKILL);
    return wait_pid_gone(pid, 1000);
}

// 接管之后再取锁: 锁由内核在进程死亡时释放, 所以这里等的是"内核已放锁"这件事, 不是给
//   进程的宽限期 —— 20ms 一格的有界轮询足够覆盖释放(与可能的回收)的时延。
static bool flock_retry(int fd, int ms) {
    for (int t = 0; t < ms; t += 20) {
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) return true;
        if (errno != EWOULDBLOCK) return false;
        usleep(20000);
    }
    return false;
}

bool instance_lock_acquire() {
    static int fd = -1;                                   // 常开: 锁活到这个进程结束
    fd = open(INSTANCE_LOCK_PATH, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        std::cerr << "❌ 实例锁打不开 (" << INSTANCE_LOCK_PATH << "): " << strerror(errno) << "\n";
        return false;
    }
    // 两轮: 第一轮失败且持有者是本库自己的进程时接管它, 第二轮取锁即成功。第二轮的
    //   失败路径就是终局 (外人持有, 或接管之后锁仍取不到)。
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
            // 内容 = 自己的 pid, 给后来者点名用。锁本身与它无关 (内核在 fd 上记账),
            //   故写失败只是少一行信息, 不影响独占性 —— 报警告, 不失败。
            const std::string me = std::to_string((long)getpid()) + "\n";
            if (ftruncate(fd, 0) != 0 || pwrite(fd, me.data(), (ssize_t)me.size(), 0) < 0)
                std::cerr << "⚠ 实例锁内容写入失败 (" << strerror(errno)
                          << "): 后到的实例将读不到本进程的 pid\n";
            return true;
        }
        if (errno != EWOULDBLOCK) {
            std::cerr << "❌ 实例锁加锁失败: " << strerror(errno) << "\n";
            return false;
        }
        // 持有者以"谁打开着锁文件"为准 —— 锁文件的内容可能过期 (上一个实例被 kill -9 或
        //   崩溃时它不会自己清), 内容只在扫不到持有者时作为 pid 提示用。
        auto holders = proc_fd_holders(INSTANCE_LOCK_PATH, (int)getpid());
        int pid = holders.size() == 1 ? holders[0].first : 0;
        if (pid == 0) {
            char buf[32] = {0};
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            if (n > 0) pid = atoi(buf);
        }
        const std::string cl = pid > 0 ? proc_cmdline(pid) : "";
        // 本库自己的进程 (bin/aimbot 与 build/ 下的探针): 接管 —— 换一个实例就是在换
        //   这一次运行, 不该要求人去 kill。边界见 proc_util.h: 根之外的进程不碰。
        bool all_ours = !holders.empty();
        for (const auto& h : holders)
            if (!proc_is_ours(h.first, own_root())) all_ours = false;
        if (attempt == 0 && all_ours && proc_is_ours(pid, own_root())) {
            bool all_gone = true;
            for (const auto& h : holders) {
                std::cerr << "[接管] 上一个实例还在跑: pid " << h.first
                          << (h.second.empty() ? " (命令行读不到)" : "  " + h.second) << " → SIGTERM\n";
                if (!proc_terminate_owned(h.first, INSTANCE_TERM_GRACE_MS)) all_gone = false;
            }
            if (all_gone && flock_retry(fd, INSTANCE_LOCK_WAIT_MS)) {
                std::cerr << "[接管] 持有者已结束, 实例锁已收归本进程\n";
                continue;                                  // 回循环头再 flock 一次
            }
            std::cerr << "❌ 接管失败: pid " << pid << " 已结束但锁仍取不到\n"
                         "   UDC / 采集设备 / NPU 卡一次只归一个进程; 本进程已退出, 未触碰任何设备\n";
            return false;
        }
        std::cerr << "❌ 已有 aimbot 实例在运行";
        if (pid > 0) std::cerr << ": pid " << pid << (cl.empty() ? " (命令行读不到)" : "  " + cl);
        else         std::cerr << " (持有者的 pid 读不到, 锁内容为空)";
        std::cerr << "\n"
                     "   UDC / 采集设备 / NPU 卡一次只归一个进程; 本进程已退出, 未触碰任何设备\n";
        if (holders.empty())
            std::cerr << "   (打开锁文件的进程扫不到: 权限或文件系统受限)\n";
        else if (!all_ours)
            std::cerr << "   持有者里有部署根之外的进程, 按边界一个都不碰; 先停掉它们再启动\n";
        else
            std::cerr << "   持有者都是本库自己的进程, 正常情况下会被自动接管 —— 走到这里说明接管没成功\n";
        return false;
    }
    return false;
}
