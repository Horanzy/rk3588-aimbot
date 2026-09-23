#!/bin/bash
# ==============================================================================
#  安装/更新 WebUI 的 systemd 服务 (板端以 root 运行一次)。
#  路径从脚本自身位置解析 — 部署目录改名/移动后重跑一次本脚本即可。
#
#  依赖走发行版包 (python3-fastapi / python3-uvicorn / python3-websockets):
#  本库禁用 pip 的 --break-system-packages, 而这三个包板端源里就有 — 于是不建 venv,
#  unit 直接用系统解释器。真要换成 venv/pip, 见 webui/README.md 的等价路线。
# ==============================================================================
set -e
WEBUI_DIR="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
SERVICE=aimbot-webui
PY="${PY:-python3}"

echo "WebUI 目录: $WEBUI_DIR"

# ---- 平台前提: raw_gadget (USB 输出通道的内核模块与设备节点) ----
#  三个输出模式 (hid/pad/p5g) 都走 /dev/raw-gadget, 而内核默认不自动载入该模块,
#  故这里当场载入并写一份开机自载配置 (服务先于任何一次手动 modprobe 启动也成立)。
if modprobe raw_gadget 2>/dev/null; then
    echo "✅ raw_gadget: $(lsmod | awk '$1=="raw_gadget"{print "已载入"}')"
    echo raw_gadget > /etc/modules-load.d/raw-gadget.conf
else
    echo "⚠ raw_gadget 模块不可用 — USB 输出通道 (hid/pad/p5g) 会起不来;"
    echo "  内核须具备该模块 (发行版包或按 Documentation/usb/raw_gadget.rst 出树编译)"
fi
[ -e /dev/raw-gadget ] || echo "⚠ /dev/raw-gadget 未出现 (模块已载入却无节点?)"

# ---- 依赖 ----
if ! "$PY" -c "import fastapi, uvicorn, websockets" 2>/dev/null; then
    echo "[依赖] 缺 fastapi/uvicorn/websockets, 尝试发行版包…"
    if command -v apt-get >/dev/null 2>&1; then
        apt-get install -y python3-fastapi python3-uvicorn python3-websockets || true
    fi
fi
if ! "$PY" -c "import fastapi, uvicorn, websockets" 2>/dev/null; then
    echo "[依赖] 发行版包仍缺, 退回 pip (requirements.txt)…"
    "$PY" -m pip install -r "$WEBUI_DIR/requirements.txt" \
        || { echo "✗ 依赖装不上。externally-managed 时按 webui/README.md 的 venv 路线装好后重跑本脚本。"
             exit 1; }
fi

# ---- 生成 unit (User=root 是刻意的, 理由见 webui/README.md 权限模型一节) ----
# 排序只要求 network.target (弱前提): 本服务只**监听** 0.0.0.0, 通配绑定不需要任何
#   已配置的地址、也不需要任何上游可达 —— network-online.target 等的是
#   NetworkManager-wait-online, 即"地址配好且能上网"的强前提: 板端 journal 里
#   network.target 10.3s 就绪, 而 wait-online 自己等满 60s 才失败、面板的起点被
#   推到 70.3s。弱前提是本服务的正确形状: 它既不消费网络也不依赖上游。
cat > /etc/systemd/system/$SERVICE.service <<EOF
[Unit]
Description=aimbot WebUI (RK3588 aimbot management panel)
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$WEBUI_DIR
ExecStart=$PY $WEBUI_DIR/server.py
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable $SERVICE.service
systemctl restart $SERVICE.service
sleep 1
systemctl --no-pager -l status $SERVICE.service | head -n 10 || true

echo
echo "✅ 安装完成: sudo systemctl {status|restart|stop} $SERVICE"
echo "   日志:   journalctl -u $SERVICE -f   (token 打印在启动横幅里)"
echo "   浏览器: http://<board-ip>/?token=<token>"
