// ============================================================================
//  proc_util.h — 实例闸门与 /proc 诊断: 三样独占资源 (UDC 的 raw_gadget 会话、
//    采集设备、AXCL 卡) 一次只归一个进程, 故启动时先取一把文件锁; 锁被持有时用
//    /proc 点出持有者(命令行), 供人手去停。同一读取也服务于"UDC 被谁占着"这类
//    报错 —— 那种情况下要杀的就是这里列出的进程。
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

// 取单实例锁: true = 拿到 (锁常开, 随进程一起消失), false = 已有实例 (打出持有者的
//   pid 与命令行) 或锁本身不可用 (打出原因)。两种失败都已打印原因, 调用方直接退出。
// 只碰锁文件 —— 调用点必须在打开 UDC / 采集设备 / NPU 之前。
bool instance_lock_acquire();
