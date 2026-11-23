#!/bin/bash
# ==============================================================================
#  RP 模式构建依赖准备脚本 — 在 RK3588 板端执行一次 (幂等)
#  校验并补装编译 vendored libchiaki (third_party/chiaki-ng) 所需的开发包;
#  chiaki-lib 静态链接进 bin/aimbot, 运行期无额外动态库依赖。
# ==============================================================================
set -e

if [ "$(id -u)" != "0" ]; then
    echo "❌ 需要 root 权限安装 apt 构建依赖 (sudo bash $0)"
    exit 1
fi

PKGS=(
    cmake
    protobuf-compiler
    python3-protobuf
    libssl-dev
    libcurl4-openssl-dev
    libjson-c-dev
    libminiupnpc-dev
    libevent-dev
    pkg-config
)

MISSING=()
for p in "${PKGS[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed"; then
        MISSING+=("$p")
    fi
done

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "→ 安装缺失的构建依赖: ${MISSING[*]}"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${MISSING[@]}"
else
    echo "✅ 全部 RP 构建依赖已就位 (${PKGS[*]})"
fi

if command -v curl-config >/dev/null 2>&1; then
    if ! curl-config --protocols | grep -qw "WS"; then
        echo "❌ 系统 libcurl 未启用 WebSocket (WS/WSS) 支持 — chiaki-ng 需要它"
        exit 1
    fi
    echo "✅ 系统 libcurl 支持 WS/WSS"
fi

echo "================================================="
echo "✅ RP 构建环境就绪; 下一步: bash scripts/compile.sh"
echo "================================================="
