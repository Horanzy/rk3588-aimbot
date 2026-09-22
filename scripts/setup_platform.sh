#! /bin/bash
# ==============================================================================
#  平台准备脚本 — RK3588 板端执行一次, 幂等: 把 librga 装进 /usr/local, 使
#  scripts/compile.sh 一条 -lrga / -I/usr/local/include/rga 就能编译链接。
#
#  为什么需要装: librga 是**部署机自带的系统库** (与本库放在哪里无关), 而发行版
#  仓库里没有这个包; 上游发布形式是源码仓库 + 各架构的预编译 librga.so, 所以本脚本
#  做的就是"取上游 → 校验版本 → 装到 /usr/local → 回读校验"。
#
#  为什么必须 1.10.x: 6.1 内核的 rga3 驱动拒绝旧的 1.3.2 (dmesg 报
#  "rga: ID[n]: invalid setting core by user" / "job assign failed", improcess 返回
#  RGA_BLIT fail: Invalid argument)。本脚本钉在 1.10.6_[3] —— 板端实测可用的那个
#  上游提交 (2b32edc), 版本号从两处独立读出并比对: 头文件 im2d_version.h 的四个宏
#  与 libs/Linux/gcc-aarch64/librga.so 里编进去的 "rga_api version" 串。任何一处不
#  等于钉住的版本就报错退出, 不装一个没核对过的库。
#
#  用法: sudo bash scripts/setup_platform.sh [--ref <上游分支/提交>]
#    已是目标版本时什么都不做 (只打印版本), 故可以反复执行。
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
    echo "❌ 需要 root (要写 /usr/local 与 ldconfig)"
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

# ---- 已装且版本正确: 什么都不做 ----
if [ "$(header_version "$PREFIX_INC/im2d_version.h" 2>/dev/null)" = "$RGA_VER_EXPECT" ] &&
   [ -f "$PREFIX_LIB/librga.so" ]; then
    echo "✅ librga 已是 $RGA_VER_EXPECT ($PREFIX_LIB/librga.so), 无需安装"
    exit 0
fi

# ---- 取上游 ----
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

SRC_VER="$(header_version "$WORK/include/im2d_version.h" 2>/dev/null || true)"
LIB_VER="$(lib_version "$WORK/$RGA_PREBUILT" 2>/dev/null || true)"
if [ "$SRC_VER" != "$RGA_VER_EXPECT" ]; then
    echo "❌ 上游 $REF 的头文件版本是 \"$SRC_VER\", 期望 \"$RGA_VER_EXPECT\""
    echo "   (1.3.2 一类的旧版会被 6.1 的 rga3 驱动拒绝: invalid setting core by user)"
    exit 1
fi
if [ "$LIB_VER" != "$RGA_VER_EXPECT" ]; then
    echo "❌ $RGA_PREBUILT 里的版本串是 \"$LIB_VER\", 与头文件 \"$SRC_VER\" 不一致 (上游错配?)"
    exit 1
fi
echo "→ 上游版本核对通过: 头文件与预编译库都是 $SRC_VER"

# ---- 装到 /usr/local ----
install -d -m 755 "$PREFIX_INC"
install -m 644 "$WORK"/include/*.h "$PREFIX_INC"/
install -m 755 "$WORK/$RGA_PREBUILT" "$PREFIX_LIB/librga.so"
ldconfig
echo "→ 已装: $PREFIX_INC/*.h → $PREFIX_LIB/librga.so"

# ---- 回读校验 (装完的文件而不是源目录里的) ----
INST_VER="$(header_version "$PREFIX_INC/im2d_version.h" 2>/dev/null || true)"
INST_LIB_VER="$(lib_version "$PREFIX_LIB/librga.so" 2>/dev/null || true)"
if [ "$INST_VER" != "$RGA_VER_EXPECT" ] || [ "$INST_LIB_VER" != "$RGA_VER_EXPECT" ]; then
    echo "❌ 回读校验失败: 头文件 \"$INST_VER\" / 库 \"$INST_LIB_VER\" (期望 $RGA_VER_EXPECT)"
    exit 1
fi
echo "✅ 回读校验: $PREFIX_INC/im2d_version.h = $INST_VER, $PREFIX_LIB/librga.so 内版本串 = $INST_LIB_VER"

# ---- 运行期前提 (库能链接不等于能跑; 这里只报事实, 不代替谁去配) ----
if [ -c /dev/rga ]; then
    echo "✅ /dev/rga 存在 ($(ls -l /dev/rga | awk '{print $1, $3":"$4}')) — RGA 作业需要它的读写权 (root)"
else
    echo "⚠ /dev/rga 不存在 — librga 能链接, 但 improcess() 会失败; 内核的 rga3 驱动没起来"
fi
