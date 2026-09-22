"""宿主遥测: /proc CPU/内存, 热区温度, AX 加速卡的 NPU 占用与卡温 (尽力而为)。

全部做成"读不到就返回 None": 非 Linux (Windows 开发镜像) 或字段缺失时
前端隐藏对应卡片, 不报错不阻塞。帧率不在这里 —— 它来自固件日志 [AI FPS] 行,
由 InstanceManager 解析。

NPU 侧只有一个数据源: `axcl-smi` (AXCL 运行时自带, 装在 /usr/bin/axcl/, 需要 root ——
面板本身以 root 跑)。它是一次打印就退出的命令而不是持续输出, 所以由后台线程按遥测推送
的周期 (2s) 起一次进程读数; 读不到 (命令不在、非 root、卡缺席) 就让 NPU 卡片消失。
"""
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

_cpu_prev = None
_npu = None
_npu_lock = threading.Lock()

# axcl-smi 的安装位置: PATH 里可能没有, 发行包把它放在 /usr/bin/axcl/ (实测本机)
AXCL_SMI_CANDIDATES = ("axcl-smi", "/usr/bin/axcl/axcl-smi")
# 读数周期 = 遥测推送周期 (api.py 每 2s 推一次); 一次调用实测 ≈0.5s, 放在线程里不挡请求
AXCL_POLL_S = 2.0
AXCL_TIMEOUT_S = 10
# 表列 (实测 axcl-smi V3.6.5 的输出; 每张卡两行, 上一行是身份/内存, 下一行是风扇/温度/
# 功耗/CPU/NPU/CMM):
#   |    0  AX650N                     V3.6.5 | 0000:01:00.0 |  147 MiB /  945 MiB |
#   |   --   66C                      -- / -- | 1%        0% |   18 MiB / 2944 MiB |
_RE_CARD = re.compile(r"^\|\s*(\d+)\s+(\S+)\s+(\S+)\s*\|\s*(\S+)\s*\|\s*(\d+)\s*MiB\s*/\s*(\d+)\s*MiB")
_RE_CARD_STAT = re.compile(r"^\|\s*(\S+)\s+(?:(\d+)C|--)\s+\S+\s*/\s*\S+\s*\|\s*(\d+)%\s+(\d+)%\s*"
                           r"\|\s*(\d+)\s*MiB\s*/\s*(\d+)\s*MiB")


def cpu_percent():
    """两次调用间的 CPU 占用率 (第一次无前置样本, 返回 None)。"""
    global _cpu_prev
    fields = None
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu "):
                    fields = [int(x) for x in line.split()[1:8]]
                    break
    except (OSError, ValueError, IndexError):
        return None
    if fields is None:
        return None
    idle = fields[3] + fields[4]
    total = sum(fields)
    if _cpu_prev is None:
        _cpu_prev = (idle, total)
        return None
    di, dt = idle - _cpu_prev[0], total - _cpu_prev[1]
    _cpu_prev = (idle, total)
    if dt <= 0:
        return None
    return round(100.0 * (1.0 - float(di) / dt), 1)


def mem_info():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    info[k] = int(v.strip().split()[0])
        total, avail = info.get("MemTotal"), info.get("MemAvailable")
        if total and avail is not None:
            return {"total_mb": total // 1024,
                    "used_mb": (total - avail) // 1024,
                    "percent": round(100.0 * (total - avail) / total, 1)}
    except (OSError, ValueError, IndexError):
        pass
    return None


def temps() -> list:
    out = []
    base = Path("/sys/class/thermal")
    try:
        zones = sorted(base.glob("thermal_zone*"))
    except OSError:
        return out
    for z in zones:
        try:
            t = int((z / "temp").read_text().strip())
            name = (z / "type").read_text().strip()
            out.append({"zone": z.name, "type": name, "temp": t / 1000.0})
        except (OSError, ValueError):
            continue
    return out


def _axcl_smi_path():
    for cand in AXCL_SMI_CANDIDATES:
        p = shutil.which(cand)
        if p:
            return p
    return None


def parse_axcl_smi(text: str):
    """axcl-smi 的一次输出 → 第一张卡的读数 (多卡时也只报第一张: 本部署每台一张)。"""
    card, stat = None, None
    for line in text.splitlines():
        if card is None:
            m = _RE_CARD.match(line)
            if m:
                card = {"index": int(m.group(1)), "name": m.group(2),
                        "firmware": m.group(3), "bus": m.group(4),
                        "mem_used_mb": int(m.group(5)), "mem_total_mb": int(m.group(6))}
                continue
        if card is not None and stat is None:
            m = _RE_CARD_STAT.match(line)
            if m:
                stat = {"fan": m.group(1),
                        "temp": int(m.group(2)) if m.group(2) else None,
                        "cpu": int(m.group(3)), "npu": int(m.group(4)),
                        "cmm_used_mb": int(m.group(5)), "cmm_total_mb": int(m.group(6))}
    if card is None or stat is None:
        return None
    out = dict(card)
    out.update(stat)
    return out


def _axcl_read():
    path = _axcl_smi_path()
    if not path:
        return None
    try:
        p = subprocess.run([path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, timeout=AXCL_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_axcl_smi((p.stdout or b"").decode("utf-8", "replace"))


def _axcl_loop() -> None:
    global _npu
    while True:
        val = _axcl_read()
        with _npu_lock:
            _npu = val
        time.sleep(AXCL_POLL_S)


def start_npu_poller() -> None:
    """AX 加速卡读数源: 命令与卡都在时后台按周期读, 否则静默 (前端隐藏 NPU 卡片)。"""
    if _axcl_smi_path():
        threading.Thread(target=_axcl_loop, daemon=True).start()


def npu_card():
    with _npu_lock:
        return dict(_npu) if _npu else None


def snapshot() -> dict:
    zs = temps()
    card = npu_card()
    return {
        "cpu": cpu_percent(),
        "mem": mem_info(),
        "npu": card["npu"] if card else None,
        "npu_temp": card["temp"] if card else None,
        "npu_card": card,
        "temps": zs[:8],
        "soc_temp": max((z["temp"] for z in zs), default=None),
    }
