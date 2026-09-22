#! /bin/bash
# ==============================================================================
#  平台准备脚本 — RK3588 板端执行一次, 幂等: 把固件跑起来需要的系统前提备齐。
#  启动脚本 (scripts/game/*.sh) 与面板 (webui) 在拉起 aimbot 之前都调它, 任一项
#  不成立就报错退出 —— 前提不齐时固件根本起不来, 报错比拉起一个必然失败的进程有用。
#
#  三件事, 全是"部署机提供、本库消费"的东西 (与本库放在哪里无关):
#
#   ① librga — RGA 裁切/格式转换的用户态库。发行版仓库里没有这个包, 上游的发布形式
#      是源码仓库 + 各架构的预编译 librga.so, 故本脚本"取上游 → 校验版本 → 装到
#      /usr/local → 回读校验", 使 scripts/compile.sh 一条 -lrga /
#      -I/usr/local/include/rga 就能编译链接。
#      **为什么必须 1.10.x**: 6.1 内核的 rga3 驱动拒绝旧的 1.3.2 (dmesg 报
#      "rga: ID[n]: invalid setting core by user" / "job assign failed", improcess 返回
#      RGA_BLIT fail: Invalid argument)。钉在 1.10.6_[3] —— 板端实测可用的那个上游提交
#      (2b32edc), 版本号从两处独立读出并比对: 头文件 im2d_version.h 的四个宏与
#      libs/Linux/gcc-aarch64/librga.so 里编进去的 "rga_api version" 串。任何一处不等于
#      钉住的版本就报错退出, 不装一个没核对过的库。
#
#   ② raw_gadget — USB 输出通道的内核模块 (mainline drivers/usb/gadget/legacy/
#      raw_gadget.c)。模块本身是部署机的事 (发行版包, 或按内核文档
#      Documentation/usb/raw_gadget.rst 出树编译), 本脚本只负责把它载入并保持载入、
#      放节点权限、把 UDC 腾空 —— 设备栈自己 (src/io/usbraw.cpp) 直接开 /dev/raw-gadget,
#      会话 open 即绑 UDC、close 即解绑。持久化写 /etc/modules-load.d: 少了它, 一次重启
#      就把 USB 输出通道整条带走, 症状是 aimbot 启动失败而不是"忘了 modprobe"。
#
#   ③ axcl — AX650N 加速卡的运行时 (库在 /usr/lib/axcl, 设备节点 /dev/axcl_host)。
#      它是板级 BSP 的一部分, 本脚本**只核对不安装**: 固件链接 -laxcl_rt/-laxcl_pkg 并以
#      root 打开 /dev/axcl_host, 缺任何一项都起不来。设备节点不在时试一次 modprobe
#      (BSP 自己的 modules-load.d 条目被清掉的情形), 仍不在就按失败报出。
#
#  设备树与内核配置的改动**不在本脚本范围内** (HDMI RX 的使能、rga3 编进内核、axcl 主机
#  驱动的镜像化都是镜像自身的既成事实, 见 README.md 的"板端前提"); 本脚本只处理用户态
#  库、模块载入与节点权限这类运行期可达的东西。
#
#  用法: sudo bash scripts/setup_platform.sh [--ref <librga 分支/提交>] [--url <地址>]
#    已是目标版本时 librga 那一段什么都不做 (只打印版本), 故可以反复执行。
# ==============================================================================
set -e

ROOT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
WORK="$ROOT/build/librga-src"                     # 上游检出位置 (在 .gitignore 覆盖的 build/ 下)
PREFIX_INC=/usr/local/include/rga
PREFIX_LIB=/usr/local/lib

# 钉住的上游版本与提交 (见文件头)
RGA_VER_EXPECT="1.10.6_[3]"
RGA_REF_DEFAULT="main"                            # 上游 main 在 2b32edc 时 = 1.10.6_[3]
RGA_URL_DEFAULT="https://github.com/airockchip/librga.git"
# 装机的架构只卖一份预编译库: aarch64 板子对 libs/Linux/gcc-aarch64
RGA_PREBUILT="libs/Linux/gcc-aarch64/librga.so"

REF="$RGA_REF_DEFAULT"
while [ $# -gt 0 ]; do
    case "$1" in
        --ref) REF="$2"; shift 2 ;;
        --url) RGA_URL="$2"; shift 2 ;;
        *) echo "用法: $0 [--ref <分支/提交>] [--url <仓库地址>]"; exit 2 ;;
    esac
done
RGA_URL="${RGA_URL:-$RGA_URL_DEFAULT}"

if [ "$(id -u)" != "0" ]; then
    echo "❌ 需要 root (要写 /usr/local、/etc/modules-load.d 与设备节点权限)"
    exit 1
fi
if [ "$(uname -m)" != "aarch64" ]; then
    echo "❌ 本脚本按 aarch64 的预编译库装 ($RGA_PREBUILT); 当前 $(uname -m)"
    exit 1
fi

# 从 im2d_version.h 拼出 "1.10.6_[3]" 形式的版本串 (宏拼接, 不用读代码表)
header_version() {
    local f="$1"
    [ -f "$f" ] || return 1
    local maj min rev bld
    maj=$(sed -n 's/^#define[ \t]*RGA_API_MAJOR_VERSION[ \t]*\([0-9]*\).*/\1/p' "$f" | head -1)
    min=$(sed -n 's/^#define[ \t]*RGA_API_MINOR_VERSION[ \t]*\([0-9]*\).*/\1/p' "$f" | head -1)
    rev=$(sed -n 's/^#define[ \t]*RGA_API_REVISION_VERSION[ \t]*\([0-9]*\).*/\1/p' "$f" | head -1)
    bld=$(sed -n 's/^#define[ \t]*RGA_API_BUILD_VERSION[ \t]*\([0-9]*\).*/\1/p' "$f" | head -1)
    [ -n "$maj" ] && [ -n "$min" ] && [ -n "$rev" ] && [ -n "$bld" ] || return 1
    printf '%s.%s.%s_[%s]' "$maj" "$min" "$rev" "$bld"
}

# 库文件里编进去的版本串 (与头文件是两处独立来源)。模式收在版本号的字符集内:
#   二进制里这个串是 NUL 结尾且后面紧跟别的字面量, 用 [^ ]* 会连下一段一起吞进去。
lib_version() {
    local f="$1"
    [ -f "$f" ] || return 1
    grep -ao 'rga_api version [0-9][0-9.]*_\[[0-9]*\]' "$f" 2>/dev/null | head -1 |
        sed 's/^rga_api version //'
}

# ============================= ① librga =============================
ensure_librga() {
    # 已装且版本正确: 这一段什么都不做 (但**只跳过这一段**, 后面两项前提照常检查)
    if [ "$(header_version "$PREFIX_INC/im2d_version.h" 2>/dev/null)" = "$RGA_VER_EXPECT" ] &&
       [ -f "$PREFIX_LIB/librga.so" ]; then
        echo "✅ librga 已是 $RGA_VER_EXPECT ($PREFIX_LIB/librga.so), 无需安装"
        return 0
    fi

    # 取上游
    if [ -d "$WORK/.git" ]; then
        echo "→ 更新上游检出 $WORK (ref $REF)"
        git -C "$WORK" fetch --depth=1 -q origin "$REF"
        git -C "$WORK" checkout -q FETCH_HEAD
    else
        echo "→ 克隆上游 $RGA_URL (ref $REF) 到 $WORK"
        mkdir -p "$(dirname "$WORK")"
        rm -rf "$WORK"
        if ! git clone --depth=1 -q --branch "$REF" "$RGA_URL" "$WORK"; then
            echo "❌ 克隆失败。网络受限时换镜像或代理后重试:"
            echo "   sudo bash $0 --url https://ghfast.top/https://github.com/airockchip/librga.git"
            echo "   (或先 export http_proxy/https_proxy 再跑)"
            exit 1
        fi
    fi

    local src_ver lib_ver
    src_ver="$(header_version "$WORK/include/im2d_version.h" 2>/dev/null || true)"
    lib_ver="$(lib_version "$WORK/$RGA_PREBUILT" 2>/dev/null || true)"
    if [ "$src_ver" != "$RGA_VER_EXPECT" ]; then
        echo "❌ 上游 $REF 的头文件版本是 \"$src_ver\", 期望 \"$RGA_VER_EXPECT\""
        echo "   (1.3.2 一类的旧版会被 6.1 的 rga3 驱动拒绝: invalid setting core by user)"
        exit 1
    fi
    if [ "$lib_ver" != "$RGA_VER_EXPECT" ]; then
        echo "❌ $RGA_PREBUILT 里的版本串是 \"$lib_ver\", 与头文件 \"$src_ver\" 不一致 (上游错配?)"
        exit 1
    fi
    echo "→ 上游版本核对通过: 头文件与预编译库都是 $src_ver"

    # 装到 /usr/local
    install -d -m 755 "$PREFIX_INC"
    install -m 644 "$WORK"/include/*.h "$PREFIX_INC"/
    install -m 755 "$WORK/$RGA_PREBUILT" "$PREFIX_LIB/librga.so"
    ldconfig
    echo "→ 已装: $PREFIX_INC/*.h → $PREFIX_LIB/librga.so"

    # 回读校验 (装完的文件而不是源目录里的)
    local inst_ver inst_lib_ver
    inst_ver="$(header_version "$PREFIX_INC/im2d_version.h" 2>/dev/null || true)"
    inst_lib_ver="$(lib_version "$PREFIX_LIB/librga.so" 2>/dev/null || true)"
    if [ "$inst_ver" != "$RGA_VER_EXPECT" ] || [ "$inst_lib_ver" != "$RGA_VER_EXPECT" ]; then
        echo "❌ 回读校验失败: 头文件 \"$inst_ver\" / 库 \"$inst_lib_ver\" (期望 $RGA_VER_EXPECT)"
        exit 1
    fi
    echo "✅ 回读校验: $PREFIX_INC/im2d_version.h = $inst_ver, $PREFIX_LIB/librga.so 内版本串 = $inst_lib_ver"

    # 运行期前提 (库能链接不等于能跑; 这里只报事实, 不代替谁去配)
    if [ -c /dev/rga ]; then
        echo "✅ /dev/rga 存在 ($(ls -l /dev/rga | awk '{print $1, $3":"$4}')) — RGA 作业需要它的读写权 (root)"
    else
        echo "❌ /dev/rga 不存在 — librga 能链接, 但 improcess() 会失败; 内核的 rga3 驱动没起来"
        exit 1
    fi
}

# =========================== ② raw_gadget ===========================
ensure_raw_gadget() {
    # 持久化: modules-load.d 让模块每次开机自动载入。内容相同就不重写 (幂等)。
    local conf=/etc/modules-load.d/raw-gadget.conf
    if [ "$(cat "$conf" 2>/dev/null)" != "raw_gadget" ]; then
        install -d -m 755 /etc/modules-load.d
        printf 'raw_gadget\n' > "$conf"
        echo "→ 已写 $conf (开机自动载入 raw_gadget)"
    else
        echo "✅ $conf 已就位 (开机自动载入 raw_gadget)"
    fi

    if ! modprobe raw_gadget 2>/dev/null; then
        echo "❌ raw_gadget 模块不可用 — 本机内核须先具备该模块 (发行版包, 或按内核文档"
        echo "   Documentation/usb/raw_gadget.rst 出树编译安装); 本脚本不替内核补模块"
        exit 1
    fi
    if [ ! -e /dev/raw-gadget ]; then
        echo "❌ /dev/raw-gadget 未出现 (模块已载入但设备节点缺失)"
        exit 1
    fi
    chmod 666 /dev/raw-gadget 2>/dev/null || echo "⚠ chmod 666 /dev/raw-gadget 失败"

    # UDC 独占腾空 — 遗留 gadget 写空 UDC 文件即解绑 (无 gadget 则整段跳过;
    #   模块未载时该目录不存在, 循环自然空转)
    for g in /sys/kernel/config/usb_gadget/*; do
        [ -e "$g/UDC" ] || continue
        if echo "" > "$g/UDC" 2>/dev/null; then
            echo "✅ 已解绑遗留 gadget: $(basename "$g")"
        else
            echo "⚠ 解绑 $g 失败 (手动: echo \"\" | sudo tee $g/UDC)"
        fi
    done
    echo "ℹ 若 UDC 仍被占 (aimbot 报 EBUSY): 先停占用 /dev/raw-gadget 的进程 (如另一 aimbot 实例)"

    local udc
    udc=$(ls /sys/class/udc 2>/dev/null | head -n 1)
    if [ -z "$udc" ]; then
        echo "❌ 找不到 UDC 控制器 (/sys/class/udc 为空) — USB 输出通道没有落点"
        exit 1
    fi
    echo "✅ raw_gadget 就绪: /dev/raw-gadget (UDC: $udc; 会话由固件自行绑定/解绑)"
}

# ============================== ③ axcl ==============================
AXCL_LIBDIR=/usr/lib/axcl
ensure_axcl() {
    local missing=""
    # 固件链接的就是这两个 (scripts/compile.sh 的 -laxcl_rt -laxcl_pkg)
    for l in libaxcl_rt.so libaxcl_pkg.so; do
        [ -e "$AXCL_LIBDIR/$l" ] || missing="$missing $AXCL_LIBDIR/$l"
    done
    if [ -n "$missing" ]; then
        echo "❌ axcl 运行时库缺失:$missing"
        echo "   axcl 是板级 BSP 的一部分 (随镜像提供), 本脚本不负责安装"
        exit 1
    fi

    # 设备节点由 axcl_host 内核模块给出。不在时试一次 modprobe —— 那是 BSP 自己的
    #   modules-load.d 条目被清掉的情形, 否则一次重启就没这条通道。
    if [ ! -e /dev/axcl_host ]; then
        modprobe axcl_host 2>/dev/null || true
    fi
    if [ ! -e /dev/axcl_host ]; then
        echo "❌ /dev/axcl_host 不存在 — NPU 会话打不开 (axcl_host 模块未载入?)"
        echo "   NPU 主机驱动是镜像自身的一部分, 本脚本只在它已具备时把它载入"
        exit 1
    fi
    echo "✅ axcl 就绪: $AXCL_LIBDIR (链接来源) + /dev/axcl_host ($(ls -l /dev/axcl_host | awk '{print $1, $3":"$4}'))"
}

ensure_librga
ensure_raw_gadget
ensure_axcl

echo "================================================="
echo "✅ 平台前提就绪: librga $RGA_VER_EXPECT + raw_gadget/USB 输出通道 + axcl (NPU)"
echo "   下一步: bash scripts/compile.sh → bin/aimbot"
echo "================================================="
