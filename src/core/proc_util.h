// ============================================================================
//  proc_util.h — 实例闸门与 /proc 诊断: 三样独占资源 (UDC 的 raw_gadget 会话、
//    采集设备、AXCL 卡) 一次只归一个进程, 故启动时先取一把文件锁。锁被持有时先
//    用 /proc 点出持有者(pid + 命令行), 再按一条边界决定动作: 可执行文件落在部署
//    根**之内**的进程是本库自己的东西 (bin/aimbot 与 build/ 下的探针), 直接接管
//    (SIGTERM → 有界等待 → SIGKILL) 后继续启动 —— 换一个实例就是在换这一次运行;
//    落在根**之外**的进程不碰, 只列出来并以非 0 退出 (我们造的可以结束, 别人造的不碰)。
// ============================================================================

#pragma once

#include <string>
#include <utility>
#include <vector>

// 实例锁文件 (见 proc_util.cpp 的取舍说明)。是系统路径: 部署目录可任意改名移动。
extern const char* INSTANCE_LOCK_PATH;

// /proc/<pid>/cmdline (NUL 分隔的 argv) → 空格分隔的一行; 进程已退出或无权限时回空串
//   (调用方只说 pid)。
std::string proc_cmdline(int pid);

// 持有 node 这个设备节点 (按节点身份比较, 不是路径字符串 — 软链/绑定挂载的另一条
//   路径也算) 的进程, 每项 = pid + 命令行, 按 pid 升序。exclude_pid < 0 = 不排除任何
//   进程; 调用方排除自己 (它自己持有那份描述符是本层的既定事实)。
std::vector<std::pair<int, std::string>> proc_fd_holders(const char* node, int exclude_pid);

// /proc/<pid>/exe 解析出的绝对路径 (进程已退出或读不到时回空串)
std::string proc_exe_path(int pid);

// pid 的可执行文件是否落在 root 之内 (root 自己先 realpath)。部署根之内的进程是本库
//   自己的东西, 可以被接管; 之外的任何进程都不碰 —— 这条边界就是"我们造的可以结束,
//   别人造的不碰"。
bool proc_is_ours(int pid, const std::string& root);

// 结束一个自己的进程: SIGTERM → 最多 grace_ms 等它自己退 → 仍活着则 SIGKILL →
//   再有界地等它从 pid 表消失。返回 true = 它已经不在了 (或本来就不在)。
bool proc_terminate_owned(int pid, int grace_ms);

// 取单实例锁: true = 拿到 (锁常开, 随进程一起消失)。锁被自己人持有时先接管它再取;
//   被外人持有、或锁本身不可用, 打出持有者与原因后返回 false。两种失败都已打印原因,
//   调用方直接退出。只碰锁文件 —— 调用点必须在打开 UDC / 采集设备 / NPU 之前。
bool instance_lock_acquire();
