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
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

const char* INSTANCE_LOCK_PATH = "/run/aimbot.lock";

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
            hit = S_ISCHR(got.st_mode) && got.st_rdev == want.st_rdev;
        }
        closedir(fd);
        if (hit) out.push_back({pid, proc_cmdline(pid)});
    }
    closedir(p);
    std::sort(out.begin(), out.end());                    // 输出与扫描顺序无关
    return out;
}

bool instance_lock_acquire() {
    static int fd = -1;                                   // 常开: 锁活到这个进程结束
    fd = open(INSTANCE_LOCK_PATH, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        std::cerr << "❌ 实例锁打不开 (" << INSTANCE_LOCK_PATH << "): " << strerror(errno) << "\n";
        return false;
    }
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
    char buf[32] = {0};
    const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
    const int pid = n > 0 ? atoi(buf) : 0;
    const std::string cl = proc_cmdline(pid);
    std::cerr << "❌ 已有 aimbot 实例在运行";
    if (pid > 0) std::cerr << ": pid " << pid << (cl.empty() ? " (命令行读不到)" : "  " + cl);
    else         std::cerr << " (持有者的 pid 读不到, 锁内容为空)";
    std::cerr << "\n"
                 "   UDC / 采集设备 / NPU 卡一次只归一个进程; 本进程已退出, 未触碰任何设备\n";
    if (pid > 0) std::cerr << "   先停掉它:  kill " << pid << "\n";
    return false;
}
