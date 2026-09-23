"""结构化发现: 从部署根按约定目录派生 游戏 profile / 模型 / HDMI 接收器。

只按固定结构发现 (scripts/game/, engine/, /sys/class/video4linux/), 不递归扫全盘。
**脚本 = 唯一事实源**: profile 参数就是 game 脚本头部的 VAR=value 块, 每次扫描现场解析,
没有独立存储; WebUI 的【保存】通过 write_script_params 原子写回脚本 (只改目标变量的值,
注释/引号风格/其余行逐字保留)。脚本把手改量写成 `${VAR:-默认}` 守卫 (缺行也能起), 所以
读取时解析到守卫里的有效默认值, 见到的是值而不是字面量。唯一的另一处脚本写回是固件经
-S 的标定回写机制。

采集侧只有**一块内建接收器** (板载 HDMI RX, 驱动名 rk_hdmirx, sysfs 节点名
stream_hdmirx), 它没有 /dev/v4l/by-id 节点, 所以发现按固件同一条判据走: 扫
/sys/class/video4linux/video*/name 取名字含 `hdmirx` 的那个节点 (判据取自驱动名与节点名
共有的子串, 见 src/io/hdmi_in.h)。信号状态由 v4l2-ctl --all 现场读出 (锁定状态 / 分辨率 /
刷新率 / 像素格式) —— 接收器没锁定时固件起不来, 所以这条状态必须在启动前可见。
"""
import os
import re
import subprocess
import threading
import time
from pathlib import Path

VAR_RE = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")
SCRIPT_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# 输出模式: 三套输出互斥, 各自独占 UDC (顺序也是 UI 里分组的顺序)
OUTPUT_MODES = ("hid", "pad", "p5g")
# 模板约定: 每个手改量写成 ${VAR:-默认}, 缺行也能起。读脚本时必须解析到**有效值**,
#   否则 UI 看到的是 ${CLASS_ID:-0} 这样的字面量而不是 0。
_GUARD_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*:-?(.*)\}$", re.S)

# 参数定义: 类型/范围与服务端校验 (与 CLI/固件钳制一致, 双保险)
PARAM_DEFS = {
    "model":           dict(kind="path",  default=None),
    # 目标类别 -c: 固件**不钳制**它, 负数 = 不筛类别 (见 src/main.cpp 与 core/detect.cpp
    #   的 want_cls < 0), 模板里写着 CLASS_ID=-1 就是这个用法。故下界是 -1: 把它钳到 0
    #   会让面板启动 (传 -c 0 = 只留类 0) 与直跑脚本 (不筛类别) 变成两件事 —— 而脚本是
    #   唯一事实源。上界 255 是标签号的防误输入带。
    "class_id":        dict(kind="int",   lo=-1, hi=255, default=0),
    "conf":            dict(kind="float", lo=0.0, hi=1.0, default=0.5),
    "y_offset":        dict(kind="float", lo=0.0, hi=100.0, default=65.0),
    # 采集设备 = /dev/videoN (空 = 固件按驱动名解析接收器节点); 帧率不是参数:
    #   采集率由信号决定 (接收器锁定的时序), 检测率看日志 [AI FPS] 行
    "cam_dev":         dict(kind="str",   default=""),
    # 模型类数 -n: 0 = 由输出属性数自解 (网格头); **未折叠 DFL 头必须给** (公开 YOLO11 = 80)
    #   —— 该头的属性数 = 4·reg_max + 类数, 而 reg_max 不是可观测量。上界 1000 是防误输入,
    #   取值带是公开检测数据集的类数量级 (COCO 80, 千级即上限)。
    "class_n":         dict(kind="int",   lo=0, hi=1000, default=0),
    # 速度上限 -x 与 FOV 半径 -r 都是**像素量**, 默认值按部署源分辨率 (2560×1440)
    #   落位: 把 1080p 参考下的推导值 (2000 px/s / 150 px) 按 R = 1440/1080 = 4/3 换算
    #   (见 AGENTS.md 的 "分辨率规则")
    "max_speed":       dict(kind="float", lo=100.0, hi=20000.0, default=2667.0),
    # 输出模式 (冷: 它决定整条输出后端, 运行中不可换) 与各模式的设备选择
    "output_mode":     dict(kind="enum",  choices=OUTPUT_MODES, default="hid"),
    "mouse_keyword":   dict(kind="str",   default=""),
    "pad_keyword":     dict(kind="str",   default=""),
    # 手柄触发阈值 (% 满量程, RT/LT 共享; 只门控触发判定, 扳机模拟量仍 1:1 透传)
    "pad_trig_thr":    dict(kind="float", lo=0.0, hi=100.0, default=6.0),
    "pad_dump":        dict(kind="bool",  default=False),
    "aim_key":         dict(kind="enum",  choices=("fire", "ads", "both"), default="both"),
    "aim_enabled":     dict(kind="bool",  default=True),
    "fov":             dict(kind="float", lo=10.0, hi=1000.0, default=200.0),
    "preview":         dict(kind="bool",  default=False),
    "capture_enabled": dict(kind="bool",  default=False),
    "cap_fire":        dict(kind="bool",  default=True),
    "cap_det":         dict(kind="bool",  default=True),
    "cap_auto":        dict(kind="bool",  default=True),
    "capture_dir":     dict(kind="dsdir", default=None),
    "fire_ms":         dict(kind="int",   lo=50, hi=60000, default=800),
    "auto_s":          dict(kind="float", lo=1.0, hi=3600.0, default=10.0),
    "cooldown_ms":     dict(kind="int",   lo=0, hi=60000, default=800),
    "jpeg_q":          dict(kind="int",   lo=1, hi=100, default=95),
}
# 三套输出各占一组 5 个值: 延迟 1 + 拉枪倍率 4。分槽是必须的 —— 同一个倍率数在两套
#   口径下含义不同 (hid 基线 1.0 px/count, 手柄基线 3000 px/s 满偏屏速), 延迟也各是一个
#   物理量 (p5g 还多一跳真加密狗的签名往返), 共用一套等于让一个模式的取值去服务另一套
#   基线。切换 output_mode 就是整套自动换, 保存/标定回写只落进本模式那一格。
# 倍率刻度: 整数, 100 = 基线, 与有效灵敏度成反比 (有效灵敏度 = 基线/(倍率/100)),
#   范围 = 固件 spd_clamp 的夹取带 (防误输入), 有意义的带是 5..2000。
# 槽默认值是 **75** 而不是 100: 基线是按参考分辨率 (1080p) 定的一把尺, 而部署源是
#   2560×1440 —— 每度更多的像素意味着游戏的**真实**灵敏度 (px/count) 是参考的 4/3 倍,
#   倍率与灵敏度成反比, 故默认倍率 = 100/(4/3) = 75 (见 AGENTS.md 的 "分辨率规则";
#   模板里的同一批默认值与这条推导一致)。
SPEED_SLOTS = (("spd_x", "SPDX"), ("spd_y", "SPDY"),
               ("ads_spd_x", "ADS_SPDX"), ("ads_spd_y", "ADS_SPDY"))
for _m in OUTPUT_MODES:
    PARAM_DEFS["l_%s" % _m] = dict(kind="float", lo=0.0, hi=120.0, default=60.0)
    for _a, _v in SPEED_SLOTS:
        PARAM_DEFS["%s_%s" % (_m, _a)] = dict(kind="int", lo=1, hi=10000, default=75)
# 热参数白名单: param key → 固件通道 key (对应 src/io/hotctl.cpp hotctl_thread)。
#   固件侧的倍率热参只有一套 (spdx/spdy/adsspdx/adsspdy, 作用于运行中实例的当前输出
#   模式), 所以槽键→线上键的映射按模式取 (mode_spd_keys): 保存 pad 槽而实例跑 hid 时,
#   改的是下次启动的取值, 不外发热参。
HOT_WIRE_KEYS = {"conf": "t", "y_offset": "y", "max_speed": "x", "fov": "fov", "aim_key": "k",
                 "aim_enabled": "aim", "cap_fire": "cap_fire", "cap_det": "cap_det",
                 "cap_auto": "cap_auto", "pad_trig_thr": "padthr"}


def mode_spd_keys(mode) -> dict:
    """某输出模式的四个倍率参数键 → 热参线上键 (模式是槽唯一的身份)。"""
    m = mode if mode in OUTPUT_MODES else "hid"
    return {"%s_%s" % (m, a): w for a, w in
            (("spd_x", "spdx"), ("spd_y", "spdy"),
             ("ads_spd_x", "adsspdx"), ("ads_spd_y", "adsspdy"))}


SCRIPT_VARS = {
    "CLASS_ID": "class_id", "CLASS_N": "class_n", "CONF_THRESH": "conf",
    "Y_OFFSET": "y_offset",
    "CAM_DEV": "cam_dev", "MAX_SPEED": "max_speed",
    "OUTPUT_MODE": "output_mode",                        # 冷: 决定整条输出后端
    "MOUSE_KEYWORD": "mouse_keyword",                    # hid: -D
    "PAD_KEYWORD": "pad_keyword",                        # pad/p5g: -P
    "PAD_TRIG_THR": "pad_trig_thr",                      # pad/p5g: -T (热参 padthr)
    "PAD_DUMP": "pad_dump",                              # pad/p5g: --pad-dump
    "AIM_KEY": "aim_key", "AIM_ENABLED": "aim_enabled", "PREVIEW": "preview",
    "CAPTURE": "capture_enabled", "CAP_FIRE": "cap_fire", "CAP_DET": "cap_det",
    "CAP_AUTO": "cap_auto", "OUT_DIR": "capture_dir", "FIRE_MS": "fire_ms",
    "AUTO_S": "auto_s", "COOLDOWN_MS": "cooldown_ms", "JPEG_Q": "jpeg_q",
    "MODEL_PATH": "model", "FOV_R": "fov",
}
# 三套输出的 5 个值各自的 VAR 名: <模式>_L_EST (固件标定回写目标) + <模式>_SPDX/SPDY/
#   ADS_SPDX/ADS_SPDY; 模式前缀是必需的身份 —— 同一个数在两套口径下含义不同, 所以每个值
#   只有一个拼写, 且它自带模式。
for _m in OUTPUT_MODES:
    SCRIPT_VARS["%s_L_EST" % _m.upper()] = "l_%s" % _m
    for _a, _v in SPEED_SLOTS:
        SCRIPT_VARS["%s_%s" % (_m.upper(), _v)] = "%s_%s" % (_m, _a)
# 固件标定回写量 (脚本 VAR → 参数键): 标定只写延迟, 速度倍率是手动项。三套输出各一条,
#   互不覆盖 (delay 是三者各自的物理量: p5g 的回路还含一跳加密狗签名往返)。
CALIB_VARS = {"%s_L_EST" % m.upper(): "l_%s" % m for m in OUTPUT_MODES}
# 模式 → 本模式那条延迟的键 (冷参数 output_mode 决定, 与固件 -l 的来源同一条规则)
MODE_L_KEY = {m: "l_%s" % m for m in OUTPUT_MODES}


def mode_l_key(output_mode) -> str:
    return MODE_L_KEY.get(str(output_mode or "hid"), "l_hid")


def root_status(root: Path):
    """部署根有效性: 存在且含 scripts/game 结构。返回 (status, 错误说明)。"""
    if not root.is_dir():
        return "missing", "部署根不存在: %s" % root
    if not (root / "scripts").is_dir():
        return "invalid", "该目录不像部署根 (缺 scripts/): %s" % root
    if not (root / "scripts" / "game").is_dir():
        return "invalid", "缺 scripts/game/ (无游戏 profile 可启动)"
    return "ok", ""


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _relativize(p: Path, root: Path) -> str:
    """绝对路径且位于部署根内 → 相对 root 的 posix 串 (可移植); 否则原样。
    相对输入本身就是根相对约定, 禁止按服务端 CWD 解析。

    判定是**词法**的 (normpath 后比前缀), 不解析符号链接: 部署根里的 engine/ 指到别处
    是合法部署形态, 按 resolve() 判会把根内路径说成根外路径, 读侧与写侧就此不一致。"""
    if not p.is_absolute():
        return p.as_posix()
    root_s = os.path.normpath(str(root))
    p_s = os.path.normpath(str(p))
    if p_s == root_s:
        return "."
    if p_s.startswith(root_s + os.sep):
        return p_s[len(root_s) + 1:].replace(os.sep, "/")
    return Path(p_s).as_posix()


def _coerce(key: str, val: str, root: Path):
    d = PARAM_DEFS[key]
    try:
        if d["kind"] == "int":
            return int(_clamp(float(val), d["lo"], d["hi"]))
        if d["kind"] == "float":
            return _clamp(float(val), d["lo"], d["hi"])
        if d["kind"] == "bool":
            return val.strip().lower() in ("y", "yes", "true", "1")
        if d["kind"] == "enum":
            v = val.strip().lower()
            return v if v in d["choices"] else d["default"]
        if d["kind"] == "path":
            return _relativize(Path(val), root)
        if d["kind"] == "dsdir":
            return _relativize(Path(val), root)
    except (TypeError, ValueError):
        return d["default"]
    return val.strip()


def parse_script(path: Path, root: Path):
    """只读解析 game 脚本顶部 VAR=value 块 → 参数集。

    延迟与四个倍率一样是**可编辑参数** (三套输出各占一槽), 只是它的落点同时是固件标定
    的回写目标 (CALIB_VARS) —— 同一份值只解析一次, 没有第二处存储。"""
    params = {k: d["default"] for k, d in PARAM_DEFS.items()}
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")   # 脚本带 BOM
    except OSError:
        return params
    for line in text.splitlines():
        line = re.split(r"\s#", line, maxsplit=1)[0].strip()            # 脚本约定: # 前有空格的行内注释
        m = VAR_RE.match(line)
        if not m:
            continue
        name, raw = m.group(1), m.group(2)
        v = raw.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        g = _GUARD_RE.match(v)              # ${VAR:-默认} → 有效默认 (模板的守卫写法)
        if g:
            v = g.group(1)
        v = v.replace("$ROOT", str(root))
        key = SCRIPT_VARS.get(name)
        if key is not None:
            params[key] = _coerce(key, v, root)
    return params


def validate_params(user: dict, root: Path, partial: bool = False) -> dict:
    """UI 提交的参数 → 白名单化 + 类型化 + 钳制后的参数集 (非法项回退默认)。

    partial=True 只回传提交里出现的键 (逐项校验后就地夹取), 用于【保存】的补丁语义:
    没提交的键保持脚本现值, 不套默认值 —— 否则一次只改 spd 的提交会把其余 VAR 全部
    打回默认。"""
    out = {} if partial else {k: d["default"] for k, d in PARAM_DEFS.items()}
    for k, v in (user or {}).items():
        if k not in PARAM_DEFS:
            continue
        d = PARAM_DEFS[k]
        try:
            if d["kind"] == "int":
                out[k] = int(_clamp(float(v), d["lo"], d["hi"]))
            elif d["kind"] == "float":
                out[k] = _clamp(float(v), d["lo"], d["hi"])
            elif d["kind"] == "bool":
                out[k] = bool(v)
            elif d["kind"] == "enum":
                out[k] = v if v in d["choices"] else d["default"]
            elif d["kind"] == "path":
                s = str(v or "").strip()
                if s and ".." not in Path(s).parts:
                    out[k] = _relativize(Path(s), root)
            elif d["kind"] == "dsdir":
                s = str(v or "").strip()
                if s and ".." not in Path(s).parts:
                    out[k] = _relativize(Path(s), root)
            else:
                out[k] = str(v).strip()
        except (TypeError, ValueError):
            pass
    return out


# ---------- 脚本写回 (脚本 = 唯一事实源) ----------

_ASSIGN_RE = re.compile(r"^(\s*)([A-Z_][A-Z0-9_]*)(\s*=\s*)(.*)$")
_GUARD_NAME_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*):-?")
# 能原样放进 ${VAR:-…} 默认位的字面量 (无空白, 只含数字/字母与 _./$,:@%+-):
#   模板里的路径 ($ROOT/…) 与数值都合得上。合不上就退回裸值 —— 守卫只是模板的写法,
#   不能为保住它把值写坏。
_GUARD_SAFE_RE = re.compile(r"^[A-Za-z0-9_./$,:@%+-]*$")
VAR_OF = {v: k for k, v in SCRIPT_VARS.items()}


def _fmt_value(key: str, val, root: Path) -> str:
    """参数值 → 脚本里的字面量 (整数不带小数点; bool 用 y/n; 路径一律挂 $ROOT)。

    脚本尾部以路径变量直接展开传参, 相对路径会随 cwd 漂 —— 所以部署根内的路径一律写成
    `$ROOT/<相对>`, 解析时再还原。这一对是往返恒等的, 一次保存不会把用户写好的
    `$ROOT/engine/x.axmodel` 改写成绝对路径 (那会让"只改目标 VAR"变成两处改动)。

    延迟是唯一的浮点手改量, 写法固定一位小数 (= 固件标定回写的格式): 它是往返恒等的,
    于是保存 pad 槽时 hid/p5g 槽的延迟行逐字不动。"""
    d = PARAM_DEFS[key]
    if key in CALIB_VARS.values():          # 三套输出各一槽延迟 (L_MIN..L_MAX)
        return "%.1f" % float(val)
    if d["kind"] == "int":
        return str(int(val))
    if d["kind"] == "float":
        f = float(val)
        return str(int(f)) if f.is_integer() else repr(f)
    if d["kind"] == "bool":
        return "y" if val else "n"
    if d["kind"] in ("path", "dsdir"):
        s = str(val or "")
        if not s:
            return s
        # 与 _relativize 同一套词法规则, 保证 render(parse("$ROOT/x")) == "$ROOT/x":
        # 相对值 (含根内绝对路径被归一后的结果) 挂 $ROOT, 根外绝对路径原样。
        rel = _relativize(Path(s), root)
        return rel if rel.startswith("/") else "$ROOT/" + rel
    return str(val)


# 面板自身的写者互斥: 服务按契约是单进程 (uvicorn 单 worker), 而同步端点由线程池并发服务 ——
#   一把进程内锁让两个保存请求严格串行, 于是"两个面板保存"这一类竞争根本不存在 (临时名撞车、
#   后写者整段抹掉前写者都在这里关掉)。固件的回写是另一个进程, 锁管不到, 那一路靠重命名前的
#   重读比对 (见 write_script_params)。
_WRITE_LOCK = threading.Lock()

# 读-改-写的重读重试次数: 锁管不到的那个对手方 (固件经 -S 的标定回写, 见 src/core/calib.cpp)
#   每次重落都把自己那一行叠到对方刚写下的新底稿上, 而对方自己也可能正在重试。3 = 本写者
#   + 对手方一次 + 对手方自身的一次重试; 用尽即抛错给调用方: 既不静默丢更新, 也不在持续
#   竞争下活锁。
_RMW_ATTEMPTS = 3


def write_script_params(path: Path, params: dict, root: Path) -> None:
    """把 params 原子写回脚本头部的 VAR=value 块。

    只改目标变量的值: 行内注释、引号风格、其余每一行逐字保留;
    脚本里缺失的变量追加到最后一个已知变量行之后; 执行位不变。

    两条例外规则, 都是"不破坏脚本"优先:
    - 值为 None 的项跳过 (我们手上没有这个值, 不能把用户的模型路径/输出目录抹成空串);
    - 原行写成 `${VAR:-默认}` 守卫时照原样保回 (值落在默认位上), 模板的写法在一次网页
      保存后仍然成立。

    权限/属主随原子替换一起带回来: 服务通常以 root 跑, 不还属主的话一次保存就把脚本变成
    root 所有, 用户 SSH 上就再也改不动自己的启动脚本了。

    并发 (脚本的另一个写者是固件经 -S 的标定回写, 见 src/core/calib.cpp): 三层, 各管一段。
    ①临时名带本写者的身份 (pid + 线程) —— 固定名会让两个写者在重命名前共用同一个临时文件,
    两份内容互相穿插; 固件的回写由唯一线程驱动, pid 就是身份, 这里多带的线程标识与下面的锁
    一起把"面板自己撞自己"完全关掉, 也让它留下的临时文件名能看出是谁写的 (板端实测 20 轮并发
    写: 每轮两个在飞写者的临时路径互不相同)。②进程内锁 (_WRITE_LOCK): 服务按契约单进程, 而
    同步端点由线程池并发服务; 锁把"读现值 → 落笔"整段串行, 板端 40 轮两个并发 PUT
    /api/profiles/<stem> 两项都落笔 40/40 —— **前提是调用方交进来的是补丁** (api.py 的 PUT 就
    这么做)。先取现值、在锁外合并、再交全量参数是另一条被实测否掉的路: 那段合并落在锁外, 两个
    并发保存各拿一份旧现值, 后写者抹掉前写者的改动 (同样 20 轮并发 PUT: 两项都落笔 0 轮);
    把这把锁换成空实现、只压函数级并发写, 30 轮里 11 轮只剩后写者那一行 (最后一次读原文件与
    重命名之间本线程要重新拿回 GIL, 对面正好在这段里完成重命名)。③跨进程那一侧锁管不到, 靠
    重命名前的重读比对: 写临时文件之后、重命名之前再读一次原文件, 内容变了就换新底重来 (用尽
    _RMW_ATTEMPTS 抛错, 本写者要么落笔要么明说失败)。比对紧贴重命名, 残留窗口只剩这两步
    之间, 而它仍不是零: 板端实测 20 轮强制两个进程同时写同一 profile, 12 轮双方都落笔, 8 轮
    一方的改动被对方的重命名覆盖 (40 次调用没有一次比对来得及看见对方的改名 —— 两个写者
    同刻起跑、耗时也相同, 时间线因此对得上)。这一格是乐观比对法固有的, 关不掉
    (见 webui/README.md)。
    """
    with _WRITE_LOCK:
        for _ in range(_RMW_ATTEMPTS):
            if _write_once(path, params, root):
                return
        raise OSError("脚本在读-改-写窗口内被连续改动 %d 次: 本次保存未落笔, 请重新保存"
                      % _RMW_ATTEMPTS)


def _write_once(path: Path, params: dict, root: Path) -> bool:
    """一次读-改-写: True = 已重命名落笔, False = 窗口内被对方改过 (调用方换新底重来)。"""
    st = path.stat()
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig", errors="replace")
    trailing_nl = text.endswith("\n")
    body = text[:-1].split("\n") if trailing_nl else text.split("\n")

    seen = {}                            # VAR → 行号
    last_var_idx = -1
    for i, line in enumerate(body):
        m = _ASSIGN_RE.match(line)
        if m and m.group(2) in SCRIPT_VARS:   # SCRIPT_VARS 的键就是 VAR 名
            seen[m.group(2)] = i
            last_var_idx = i

    def render(key: str, quote: str, guard: str = "") -> str:
        s = _fmt_value(key, params[key], root)
        if guard and _GUARD_SAFE_RE.match(s):
            s = "${%s:-%s}" % (guard, s)
        return quote + s + quote if quote else s

    missing = []
    for key in params:
        var = VAR_OF.get(key)              # 参数名 → VAR 名 (SCRIPT_VARS 的逆映射)
        if var is None or params[key] is None:
            continue
        if var not in seen:
            missing.append(key)
            continue
        i = seen[var]
        m = _ASSIGN_RE.match(body[i])
        rhs = m.group(4)
        cm = re.search(r"\s+#", rhs)     # 脚本约定: 值后空白 + # 才是行内注释 (\s+ 保住对齐空格)
        comment = rhs[cm.start():] if cm else ""
        vs = (rhs if not cm else rhs[:cm.start()]).strip()
        quote = vs[0] if len(vs) >= 2 and vs[0] == vs[-1] and vs[0] in "\"'" else ""
        inner = vs[1:-1] if quote else vs
        gm = _GUARD_NAME_RE.match(inner) if quote else None
        guard = gm.group(1) if (gm and gm.group(1) == var) else ""
        body[i] = m.group(1) + m.group(2) + m.group(3) + render(key, quote, guard) + comment

    if missing:
        ins = []
        for k in missing:
            v = VAR_OF[k]
            s = _fmt_value(k, params[k], root)
            if _GUARD_SAFE_RE.match(s):
                s = "${%s:-%s}" % (v, s)
            ins.append('%s="%s"' % (v, s))
        if last_var_idx >= 0:
            body[last_var_idx + 1:last_var_idx + 1] = ins
        else:
            body.extend(ins)

    out = "\n".join(body) + ("\n" if trailing_nl else "")
    data = (b"\xef\xbb\xbf" if has_bom else b"") + out.encode("utf-8")
    tmp = path.with_name("%s.tmp.%d.%d" % (path.name, os.getpid(),
                                           threading.get_ident()))
    tmp.write_bytes(data)
    if path.read_bytes() != raw:             # 比对紧贴重命名: 见 write_script_params 的并发说明
        tmp.unlink()
        return False
    os.replace(tmp, path)
    os.chmod(path, st.st_mode)               # tmp 是新文件, 执行位要显式带回来
    try:
        os.chown(path, st.st_uid, st.st_gid)
    except (AttributeError, OSError):
        pass                                 # 非 root 且文件不属自己: 保持现状, 不改写流程结果
    return True


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return a == b


def scan_profiles(root: Path) -> list:
    """每次扫描现场解析脚本 —— 没有独立参数存储, SSH 侧改动天然可见。"""
    out = []
    game_dir = root / "scripts" / "game"
    for p in sorted(game_dir.glob("*.sh")):
        if not SCRIPT_STEM_RE.match(p.stem):
            continue
        script_params = parse_script(p, root)
        out.append({
            "file": p.stem,
            "display_name": p.stem,
            "script_params": script_params,   # 三套输出的 5 个值都在里面 (脚本 = 唯一事实源)
        })
    return out


# ---------- 其它发现 ----------

def list_models(root: Path) -> list:
    """engine/**/*.axmodel —— 模型转换在本仓库之外完成, 这里只列已经放好的模型。"""
    out = []
    edir = root / "engine"
    if edir.is_dir():
        for p in sorted(edir.rglob("*.axmodel")):
            try:
                st = p.stat()
            except OSError:
                continue
            out.append({"path": p.relative_to(root).as_posix(),
                        "size": st.st_size, "mtime": st.st_mtime})
    return out


# ---------- HDMI 接收器 (唯一采集源) ----------

# 判据与固件 io/hdmi_in.h 的 HDMIRX_NAME_SUBSTR 同一串: 驱动名 rk_hdmirx 与 sysfs
#   节点名 stream_hdmirx 共有的子串 —— 两处名字都换掉也不会认错设备。
HDMIRX_NAME_SUBSTR = "hdmirx"
V4L2_CTL = "v4l2-ctl"
# 一次 --all 是只读查询 (取不到锁时立刻返回), 放 5s 上限只为兜住设备卡住的极端情形
V4L2_TIMEOUT_S = 5
_RE_VIDEO_INPUT = re.compile(r"Video input\s*:\s*\d+\s*\(([^)]*)\)")
_RE_ACTIVE_W = re.compile(r"Active width:\s*(\d+)")
_RE_ACTIVE_H = re.compile(r"Active height:\s*(\d+)")
_RE_FPS = re.compile(r"\(([\d.]+) frames per second\)")
_RE_FMT = re.compile(r"Pixel Format\s*:\s*'([^']+)'\s*\(([^)]*)\)")


def sysfs_video_nodes() -> list:
    """扫 /sys/class/video4linux/video*/name → 名字含 hdmirx 的节点 (与固件同一判据)。

    节点下标跨重启不稳定 (同一张板子两次启动 hdmirx 落在不同号上), 所以按名字找节点、
    不按号记节点。返回 [{"node", "name", "driver"}], 按节点号排序 (确定性)。"""
    base = Path("/sys/class/video4linux")
    out = []
    try:
        entries = sorted(base.glob("video*"))
    except OSError:
        return out
    for d in entries:
        try:
            name = (d / "name").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if HDMIRX_NAME_SUBSTR not in name:
            continue
        driver = ""
        try:
            # device → ../../../fdee0000.hdmirx-controller, driver 符号链接指向 rk_hdmirx
            drv = (d / "device" / "driver").resolve()
            driver = drv.name
        except OSError:
            pass
        out.append({"node": "/dev/" + d.name, "name": name, "driver": driver})
    return out


def hdmi_signal_state(node: str) -> dict:
    """现场读接收器的信号状态: 锁定与否 / 分辨率 / 刷新率 / 像素格式。

    一次 `v4l2-ctl -d <node> --all` 全给 (实测输出, 各字段都从这里取):
        Driver name      : rk_hdmirx
        Video input : 0 (hdmirx: ok)          ← 锁定状态 (括号里就是驱动的自述)
        DV timings:
            Active width: 2560 / Active height: 1440
            Pixelclock: 497768000 Hz (120.00 frames per second)
        Format Video Capture Multiplanar:
            Pixel Format      : 'BGR3' (24-bit BGR 8-8-8)
    括号里的自述与 DV timings 是两处独立证据, 两者不一致时报未锁定 (有一处说没锁就是
    没锁)。取不到 (命令缺失/超时/节点不可打开) 时返回 ok=False 与原因, 不猜状态。"""
    out = {"ok": False, "error": "", "locked": False, "status": "",
           "width": None, "height": None, "fps": None,
           "pixel_format": "", "pixel_desc": "", "raw": ""}
    try:
        p = subprocess.run([V4L2_CTL, "-d", node, "--all"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=V4L2_TIMEOUT_S)
    except FileNotFoundError:
        out["error"] = "找不到 %s (v4l2-utils 未安装)" % V4L2_CTL
        return out
    except subprocess.TimeoutExpired:
        out["error"] = "%s 超时 (%ds): 设备无响应" % (V4L2_CTL, V4L2_TIMEOUT_S)
        return out
    except OSError as e:
        out["error"] = str(e)
        return out
    text = (p.stdout or b"").decode("utf-8", "replace")
    out["raw"] = text.strip()
    m = _RE_VIDEO_INPUT.search(text)
    if m:
        out["status"] = m.group(1).strip()
    mw, mh = _RE_ACTIVE_W.search(text), _RE_ACTIVE_H.search(text)
    if mw and mh:
        out["width"], out["height"] = int(mw.group(1)), int(mh.group(1))
    m = _RE_FPS.search(text)
    if m:
        try:
            out["fps"] = float(m.group(1))
        except ValueError:
            pass
    m = _RE_FMT.search(text)
    if m:
        out["pixel_format"], out["pixel_desc"] = m.group(1), m.group(2)
    timings = out["width"] is not None and out["height"] is not None and out["fps"] is not None
    stated_ok = "ok" in out["status"].lower()
    out["locked"] = bool(stated_ok and timings)
    if not out["status"] and not timings:
        out["error"] = "读不到接收器状态 (设备被占用或节点无响应)"
    elif not out["locked"]:
        out["error"] = "接收器未锁定信号" + (" (驱动自述: %s)" % out["status"] if out["status"] else "")
    out["ok"] = True
    return out


def list_hdmi_inputs() -> list:
    """接收器节点 + 现场信号状态 (下拉与状态卡共用一份读数)。

    label 里带上锁定状态与格式 —— 参数页的采集设备选择就是这个下拉, 于是"选中的接收器
    现在有没有信号"在选择处即可见。"""
    out = []
    for n in sysfs_video_nodes():
        sig = hdmi_signal_state(n["node"])
        item = dict(n)
        item.update({k: sig[k] for k in ("ok", "error", "locked", "status",
                                         "width", "height", "fps",
                                         "pixel_format", "pixel_desc")})
        item["label"] = hdmi_label(item)
        out.append(item)
    return out


def hdmi_label(item: dict) -> str:
    parts = [item["node"]]
    if not item.get("ok"):
        parts.append("状态未知 (%s)" % (item.get("error") or "读取失败"))
    elif item.get("locked"):
        parts.append("已锁定 %dx%d @%s Hz · %s" % (item["width"], item["height"],
                                                  _fmt_hz(item["fps"]), item["pixel_format"]))
    else:
        parts.append("无信号 (%s)" % (item.get("status") or "接收器未锁定"))
    if item.get("driver"):
        parts.append(item["driver"])
    return " · ".join(parts)


def _fmt_hz(v) -> str:
    """刷新率两位小数 —— 与 v4l2-ctl 自己的写法一致 (实测 120.00 / 60.00)。"""
    return "%.2f" % float(v) if v is not None else "—"


def list_dataset_dirs(root: Path) -> list:
    ds = root / "dataset"
    if not ds.is_dir():
        return []
    try:
        return sorted(d.name for d in ds.iterdir() if d.is_dir())
    except OSError:
        return []


_bin_info_cache = {}                     # (path, mtime_ns, size) → info; 重编译才失效


def binary_info(root: Path) -> dict:
    """bin/aimbot 存在性 + 热参能力探测 (二进制内查握手串, 用于认领实例的兜底)。
    结果按 (mtime, size) 缓存 —— WS 周期重扫时避免反复读大二进制。"""
    p = root / "bin" / "aimbot"
    if not p.is_file():
        return {"exists": False, "hot_capable": False}
    try:
        st = p.stat()
    except OSError:
        return {"exists": True, "hot_capable": False}
    key = (str(p), st.st_mtime_ns, st.st_size)
    hit = _bin_info_cache.get(key)
    if hit is not None:
        return hit
    hot = False
    needle = "热参数通道".encode("utf-8")
    try:
        with p.open("rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                if needle in chunk:
                    hot = True
                    break
    except OSError:
        return {"exists": True, "hot_capable": hot}
    info = {"exists": True, "hot_capable": hot, "size": st.st_size, "mtime": st.st_mtime}
    _bin_info_cache.clear()
    _bin_info_cache[key] = info
    return info


def scan(root: Path) -> dict:
    """全量发现一次。永不抛异常 —— 任何失败转成 status/error/warnings 在页面可见。"""
    status, err = root_status(root)
    out = {"status": status, "error": err, "scanned_at": time.time(),
           "profiles": [], "models": [], "hdmi_in": [],
           "dataset_dirs": [], "binary": {"exists": False, "hot_capable": False},
           "warnings": []}
    if status != "ok":
        return out
    out["profiles"] = scan_profiles(root)
    out["models"] = list_models(root)
    out["hdmi_in"] = list_hdmi_inputs()
    out["dataset_dirs"] = list_dataset_dirs(root)
    out["binary"] = binary_info(root)
    if not out["models"]:
        out["warnings"].append("engine/ 下没有 *.axmodel —— 模型转换在本仓库之外完成, "
                               "转好后放进 engine/ 才能选择模型")
    if not out["profiles"]:
        out["warnings"].append("scripts/game/ 下没有 *.sh —— 没有可启动的游戏 profile")
    if not out["hdmi_in"]:
        out["warnings"].append("未发现 HDMI 接收器 (扫 /sys/class/video4linux/video*/name "
                               "里含 \"%s\" 的节点), 启动会被固件拒绝" % HDMIRX_NAME_SUBSTR)
    else:
        for h in out["hdmi_in"]:
            if not h["ok"]:
                out["warnings"].append("%s: %s — 启动前先确认信号" % (h["node"], h["error"]))
            elif not h["locked"]:
                out["warnings"].append("%s: 接收器未锁定信号 (%s), 启动会被固件拒绝"
                                       % (h["node"], h["status"] or "无时序"))
    if not out["binary"]["exists"]:
        out["warnings"].append("bin/aimbot 不存在 —— 先执行 compile, 否则无法启动")
    elif not out["binary"]["hot_capable"]:
        out["warnings"].append("bin/aimbot 是旧版 (无热参数通道) —— 建议重编译以启用热参数")
    return out
