"""HTTP API + WebSocket 推送。

鉴权: 日常用密码登录 (首次设置密码需验证 token), token 保留为万能凭证
(URL ?token= 直达 / 忘记密码兜底); 登录成功换得 token, 之后所有请求带
X-WebUI-Token 头, WS 走 ?token=。
推送: 单条 WS 轮询合流 —— 实例快照+任务摘要每 0.4s、日志按序号增量
([SAVE]/[AI FPS] 行不推进页面流, 见 proc.HIDDEN_IN_STREAM)、FPS 读数增量、遥测每 2s;
客户端断线重连后用 /api/state 全量重建, 再以 log_seq 续传。
循环: HTTP 与 WS 共用**同一条事件循环** (uvicorn 单 worker), 所以心跳里的阻塞段 (重扫要开
接收器、实例快照要拿实例状态锁) 一律丢进线程池 —— 一处阻塞调用就是所有请求一起排队。
"""
import asyncio
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, discover, proc, tasks, telemetry

STATIC_DIR = config.WEBUI_DIR / "static"


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


class WebUIState:
    """进程级单例。uvicorn 必须单 worker (README 说明)。"""

    def __init__(self):
        self.cfg = config.load()
        self._scan = None
        self.inst = proc.InstanceManager(config.HISTORY_PATH)
        self.op = tasks.TaskManager()

    def root(self) -> Path:
        return Path(self.cfg["deploy_root"])

    def scan(self) -> dict:
        if self._scan is None:
            self._scan = discover.scan(self.root())
        return self._scan

    def rescan(self) -> dict:
        self._scan = discover.scan(self.root())
        return self._scan


S = WebUIState()
app = FastAPI(title="aimbot-webui", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


PUBLIC_PATHS = ("/", "/favicon.ico", "/static",
                "/api/auth/mode", "/api/auth/login", "/api/auth/setup")


@app.middleware("http")
async def _auth_mw(request, call_next):
    path = request.url.path
    if any(path == p or path.startswith(p + "/") for p in PUBLIC_PATHS):
        response = await call_next(request)
    else:
        token = request.headers.get("x-webui-token") or request.query_params.get("token")
        if not _token_ok(token or ""):
            return PlainTextResponse("unauthorized", status_code=401)
        response = await call_next(request)
    if path == "/" or path.startswith("/static"):
        # 页面/静态资源强制回源验证 (etag 304, 局域网开销可忽略):
        # 不然更新代码后浏览器还在吃旧缓存, 新界面出不来
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---------- 基础 ----------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/state")
def api_state():
    inst = S.inst.snapshot()
    return {
        "config": {"deploy_root": S.cfg["deploy_root"], "bind": S.cfg["bind"],
                   "port": S.cfg["port"], "hot_port": S.cfg["hot_port"],
                   "token": S.cfg["token"], "is_root": _is_root(),
                   "has_password": bool(S.cfg.get("password_hash"))},
        "scan": S.scan(),
        "instance": inst,
        "logs": S.inst.log_since(0),
        "fps_hist": S.inst.fps_hist_since(0),
        "tasks": S.op.summaries(),
        "history": S.inst.history,
        "telemetry": telemetry.snapshot(),
        "now": time.time(),
    }


class ConfigIn(BaseModel):
    deploy_root: Optional[str] = None
    bind: Optional[str] = None
    port: Optional[int] = None


@app.post("/api/config")
def api_config(inp: ConfigIn):
    kw = {}
    if inp.deploy_root is not None:
        root = Path(inp.deploy_root.strip()).expanduser()
        status, err = discover.root_status(root)
        if status != "ok":
            raise HTTPException(400, err)
        kw["deploy_root"] = str(root)
    if inp.bind is not None:
        if inp.bind not in ("0.0.0.0", "127.0.0.1"):
            raise HTTPException(400, "bind 仅支持 0.0.0.0 (局域网) 或 127.0.0.1 (仅本机)")
        kw["bind"] = inp.bind
    if inp.port is not None:
        if not (1 <= inp.port <= 65535):
            raise HTTPException(400, "端口无效 (1-65535)")
        kw["port"] = inp.port
    if not kw:
        raise HTTPException(400, "没有要修改的字段")
    S.cfg = config.update(**kw)
    S.rescan()
    return {"ok": True, "needs_restart": ("bind" in kw or "port" in kw),
            "config": {"deploy_root": S.cfg["deploy_root"], "bind": S.cfg["bind"],
                       "port": S.cfg["port"], "token": S.cfg["token"],
                       "has_password": bool(S.cfg.get("password_hash"))}}


@app.post("/api/scan")
def api_scan():
    return S.rescan()


# ---------- 认证 ----------

def _token_ok(tok: str) -> bool:
    """token 判据的唯一拼写 (HTTP 头/查询串、登录端点与 WS ?token= 共用一条)。"""
    return secrets.compare_digest(str(tok).encode("utf-8"),
                                  str(S.cfg.get("token", "")).encode("utf-8"))


# 登录/设密失败的固定延迟 (README 的"固定 1s 延迟", 只有一个定义点): 两个端点因此是 async 的
#   —— 延迟期间事件循环只挂起自己, 不占线程池, 未认证突发打不垮面板的响应性。同一条理由把
#   PBKDF2 那一段 (config 的 200k 轮) 放进线程: 它留在循环上, 等于把线程池里省下的等待换成
#   循环上的 CPU, 面板照样不响。
AUTH_FAIL_DELAY_S = 1.0


@app.get("/api/auth/mode")
def api_auth_mode():
    # 公开: 登录浮层要据此决定弹"首次设置密码"还是"输密码"; 是否设过密码不算敏感
    return {"has_password": bool(S.cfg.get("password_hash"))}


class LoginIn(BaseModel):
    password: Optional[str] = None
    token: Optional[str] = None


@app.post("/api/auth/setup")
async def api_auth_setup(inp: LoginIn):
    """首次设置密码 (或忘记密码后凭 token 重设): 必须先验证 token。"""
    if not inp.token or not _token_ok(inp.token):
        await asyncio.sleep(AUTH_FAIL_DELAY_S)
        raise HTTPException(401, "token 不对")
    if not inp.password or len(inp.password) < 4:
        raise HTTPException(400, "密码至少 4 位")
    digest = await asyncio.to_thread(config.hash_password, inp.password)
    S.cfg = config.update(password_hash=digest)
    return {"ok": True, "token": S.cfg["token"]}


@app.post("/api/auth/login")
async def api_auth_login(inp: LoginIn):
    """密码或 token 登录, 成功换回 token —— 之后所有请求仍走 X-WebUI-Token,
    鉴权管道不变。失败加固定小延迟 (单用户局域网工具, 不上重型防爆破)。"""
    ok = False
    if inp.token and _token_ok(inp.token):
        ok = True
    elif inp.password and S.cfg.get("password_hash"):
        ok = await asyncio.to_thread(config.verify_password, inp.password,
                                     S.cfg["password_hash"])
    if not ok:
        await asyncio.sleep(AUTH_FAIL_DELAY_S)
        raise HTTPException(401, "密码或 token 不对")
    return {"ok": True, "token": S.cfg["token"]}


class PasswordIn(BaseModel):
    password: str


@app.post("/api/auth/password")
def api_auth_password(inp: PasswordIn):
    """修改密码 (登录态内操作, 与重置 token 同一信任级)。"""
    if len(inp.password) < 4:
        raise HTTPException(400, "密码至少 4 位")
    S.cfg = config.update(password_hash=config.hash_password(inp.password))
    return {"ok": True}


@app.post("/api/token/reset")
def api_token_reset():
    S.cfg = config.update(token=secrets.token_urlsafe(16))
    return {"ok": True, "token": S.cfg["token"]}


# ---------- 游戏 profile ----------

def _find_profile(stem: str):
    for p in S.scan()["profiles"]:
        if p["file"] == stem:
            return p
    return None


@app.get("/api/profiles/{stem}")
def api_profile(stem: str):
    p = _find_profile(stem)
    if p is None:
        raise HTTPException(404, "未知的游戏 profile: %s (试一次重新扫描)" % stem)
    return p


class ProfileIn(BaseModel):
    params: Optional[dict] = None


def _wire_value(pk: str, v):
    """参数值 → 固件热参通道字面量 (bool=0/1, 触发键=枚举串, 其余=数字)。"""
    if pk == "aim_key":
        return str(v)
    if isinstance(v, bool):
        return "1" if v else "0"
    return proc.fmt_num(v)


@app.put("/api/profiles/{stem}")
def api_profile_put(stem: str, inp: ProfileIn):
    """保存 = 原子写回脚本 (脚本 = 唯一事实源), 然后与脚本现值做热参差量下发。

    提交体是**补丁**: 只校验并写回它点名的键, 其余 VAR 保持脚本现值 —— UI 只提交本模式
    那一槽的改动 (切换 output_mode 后原样保存不会碰另两槽的任何一行), 后端也照补丁语义
    写回, 于是"保存只写当前槽"在两头都成立。

    写回函数收到的就是这份补丁, 不是"现值 ∪ 补丁"的合并结果: 整段"读现值 → 落笔"在它自己
    的临界区里完成。在这里先合并再交出去是**一条被实测否掉的路** —— 那等于把读的那一半留在
    锁外, 两个同时在飞的保存各拿一份旧现值, 后写者的重命名把前写者的改动整段抹掉 (板端
    20 轮两个并发 PUT: 两项都落笔 0 轮)。下面的现值解析只服务于热参差量 (改了哪几项), 不参与
    写回。

    `rejected` = 提交了却没有写回脚本的项 (值非法 / 路径被清空 / 键不在白名单): 补丁语义下
    这类键在脚本里保持现值是正确的, 但"没写"必须说出来 —— 页面据此提示, 不给一句已保存的
    假话。"""
    p = _find_profile(stem)
    if p is None:
        raise HTTPException(404, "未知的游戏 profile: %s" % stem)
    script_path = S.root() / "scripts" / "game" / (stem + ".sh")
    old_params = discover.parse_script(script_path, S.root())
    patch, rejected = discover.validate_params(inp.params or {}, S.root(), partial=True)
    try:
        discover.write_script_params(script_path, patch, S.root())
    except OSError as e:
        raise HTTPException(500, "写回脚本失败: %s" % e)
    new_params = dict(old_params)
    new_params.update(patch)             # 热参差量: 只比"本次点名的键"的旧值与新值
    # 热参数: 本次保存中变化的热项 → 直接下发运行中实例 (即时生效, 不重启)。
    #   倍率热参只有一套 (作用于运行中实例的当前输出模式), 所以槽键按**运行中实例的模式**
    #   取: 改的是另两槽的值时它只是脚本改动, 下次【启动】才生效。
    applied, reason = {}, None
    inst = S.inst.snapshot()
    if inst["state"] == "running":
        if inst["adopted"]:
            reason = "认领实例 (非本 WebUI 启动): 不盲发热参; 按【启动】以新设置接管"
        elif not inst["hot_capable"]:
            reason = "运行中的二进制不支持热参数通道 (旧版固件, 重编译后启动即可)"
        elif inst["profile"] == stem:
            run_mode = str(old_params.get("output_mode") or "hid")   # 实例启动时的模式
            hot_keys = dict(discover.HOT_WIRE_KEYS)
            if run_mode in discover.OUTPUT_MODES:
                hot_keys.update(discover.mode_spd_keys(run_mode))
            for pk, wk in hot_keys.items():
                if not discover._same(old_params.get(pk), new_params.get(pk)):
                    applied[wk] = _wire_value(pk, new_params[pk])
            if applied:
                try:
                    proc.send_hot(inst["hot_port"] or S.cfg["hot_port"], applied)
                except OSError as e:
                    reason = "热参发送失败: %s (实例可能刚好退出)" % e
            elif new_params.get("output_mode") != run_mode:
                reason = "输出模式已改 (冷参数): 下次【启动】生效, 本次不发热参"
        else:
            reason = "运行中的是其它 profile, 本 profile 的改动将在下次【启动】生效"
    S.rescan()
    return {"ok": True, "hot_applied": applied, "hot_reason": reason,
            "rejected": rejected}


class CopyIn(BaseModel):
    file_name: Optional[str] = None


@app.post("/api/profiles/{stem}/copy")
def api_profile_copy(stem: str, inp: CopyIn):
    src_prof = _find_profile(stem)
    if src_prof is None:
        raise HTTPException(404, "未知的游戏 profile: %s" % stem)
    src = S.root() / "scripts" / "game" / (stem + ".sh")
    fname = (inp.file_name or (stem + "_copy")).strip()
    if fname.endswith(".sh"):
        fname = fname[:-3]
    if not discover.SCRIPT_STEM_RE.match(fname):
        raise HTTPException(400, "文件名只能以字母/数字开头, 含字母数字._-")
    dst = S.root() / "scripts" / "game" / (fname + ".sh")
    if dst.exists():
        raise HTTPException(400, "目标脚本已存在: %s.sh" % fname)
    dst.write_bytes(src.read_bytes())     # 参数随源 —— 脚本本身就是全部参数
    S.rescan()
    return {"ok": True, "file": fname, "display_name": fname}


# ---------- 实例 ----------

@app.get("/api/instance/cmd")
def api_instance_cmd(profile: str):
    p = _find_profile(profile)
    if p is None:
        raise HTTPException(404, "未知的游戏 profile: %s" % profile)
    argv = proc.build_argv(S.root(), p["script_params"],
                           S.root() / "scripts" / "game" / (profile + ".sh"))
    return {"cmd": proc.cmd_string(proc.wrapped_argv(argv))}


@app.post("/api/instance/start")
def api_instance_start(body: dict):
    stem = (body or {}).get("profile")
    p = _find_profile(stem)
    if p is None:
        raise HTTPException(404, "未知的游戏 profile: %s" % stem)
    ok, err = S.inst.start(S.root(), p["file"], p["display_name"], p["script_params"],
                           S.root() / "scripts" / "game" / (stem + ".sh"))
    if not ok:
        raise HTTPException(409, err)
    return {"ok": True}


@app.post("/api/instance/stop")
def api_instance_stop():
    ok, err = S.inst.stop(S.root())
    if not ok:
        raise HTTPException(409, err)
    return {"ok": True}


@app.post("/api/instance/calib")
def api_instance_calib():
    """请求运行中的实例跑一轮标定 (热参 padcalib=1, 固件一次消费即清)。

    只有手柄模式有这条入口 —— hid 的触发是鼠标双侧键长按 5 秒, 固件不给按钮路径。
    落地量永远是延迟, 写进**当前输出模式**那一格 (HID_L_EST / PAD_L_EST / P5G_L_EST),
    结论看日志的 `[标定]` 行。"""
    inst = S.inst.snapshot()
    if inst["state"] != "running":
        raise HTTPException(409, "实例未在运行 —— 标定要在运行中触发")
    stem = inst["profile"]
    p = _find_profile(stem) if stem else None
    if p is None:
        raise HTTPException(409, "认领实例 (非本 WebUI 启动): 不知道它读的是哪个 profile, 不盲发")
    if (p["script_params"] or {}).get("output_mode") not in ("pad", "p5g"):
        raise HTTPException(409, "hid 模式的触发是鼠标双侧键长按 5 秒; 热参标定入口只给手柄模式")
    ok, err = S.inst.request_calib(S.cfg["hot_port"])
    if not ok:
        raise HTTPException(409, err)
    return {"ok": True}


@app.get("/api/logs/current")
def api_logs_current():
    return PlainTextResponse(
        S.inst.log_all(),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="aimbot-webui.log"'})


# ---------- 采集统计 ----------

_JPG_SUFFIX = (".jpg", ".jpeg")


def _count_jpgs(d: Path) -> int:
    try:
        return sum(1 for e in os.scandir(d)
                   if e.name.lower().endswith(_JPG_SUFFIX) and e.is_file())
    except OSError:
        return 0


@app.get("/api/capture/count")
def api_capture_count(profile: str = ""):
    """输出目录里的截图总数: fire/det/auto 三个子目录 + 目录根部散图 (非递归,
    与固件 make_filepath 的落盘布局一致)。目录按 profile 参数动态解析
    (相对部署根或绝对路径), 不绑定 dataset/。"""
    stem = profile.strip()
    params = {}
    if stem:
        p = _find_profile(stem)
        if p is None:
            raise HTTPException(404, "未知的游戏 profile: %s" % stem)
        params = p["script_params"] or {}
    od = str(params.get("capture_dir") or "dataset")
    d = Path(od) if os.path.isabs(od) else S.root() / od
    out = {"dir": str(d), "dir_name": d.name or str(d), "exists": d.is_dir(),
           "total": 0, "fire": 0, "det": 0, "auto": 0}
    if out["exists"]:
        out["fire"] = _count_jpgs(d / "fire")
        out["det"] = _count_jpgs(d / "det")
        out["auto"] = _count_jpgs(d / "auto")
        out["total"] = out["fire"] + out["det"] + out["auto"] + _count_jpgs(d)
    return out


# ---------- 运维任务 ----------

@app.post("/api/tasks")
def api_task_start(body: dict):
    kind = (body or {}).get("kind")
    if kind != "compile":
        raise HTTPException(400, "kind 须为 compile (模型转换在本仓库之外完成)")
    script = S.root() / "scripts" / (kind + ".sh")
    block = None
    if S.inst.snapshot()["state"] in proc.RUNNING_STATES:
        block = ("实例正在运行: 运行中的 bin/aimbot 无法被覆盖 (text file busy)。"
                 "先【停止】实例再编译。")
    tid, err = S.op.start(kind, script, block)
    if tid is None:
        raise HTTPException(409, err)
    return {"ok": True, "id": tid}


@app.get("/api/tasks")
def api_tasks():
    return S.op.summaries()


@app.get("/api/tasks/{tid}/log")
def api_task_log(tid: str, since: int = 0):
    lines = S.op.log_since(tid, since)
    return {"lines": lines, "seq": lines[-1][0] if lines else since}


# ---------- WebSocket ----------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: str = Query(""), since: int = 0):
    if not _token_ok(token):
        await ws.close(code=4401)
        return
    await ws.accept()
    inst_seq = since
    fps_i = 0                            # 每连接独立的 FPS 读数游标 (from=0 → 前端整表替换)
    fps_ep = S.inst.snapshot()["fps_epoch"]   # 本连接见到的会话纪元 (重启/认领会换掉历史)
    task_seqs = {}
    last_tel = 0.0
    last_scan = None
    tick = 0
    try:
        while True:
            await asyncio.sleep(0.4)
            tick += 1
            if tick % 12 == 0:
                # ~5s 重扫: 脚本是唯一事实源, SSH 侧改动自动到达所有打开的页面。扫描要读脚本/
                #   模型清单、还要开一次接收器 (v4l2-ctl 子进程, 上限 5s), 所以整段放线程里 ——
                #   事件循环上的阻塞调用会把同一时刻的每个 HTTP 请求与每条 WS 消息一起卡住
                #   (面板是单进程单循环)。
                await asyncio.to_thread(S.rescan)
                if S.inst.state in ("stopped", "exited"):
                    # 认领不是一次性的: 面板起来**之后**才在别处 (SSH) 跑起来的实例, 状态停在
                    #   "已停止"时定期回看 /proc。漏掉它就轮到下一次【启动】撞上实例锁、新进程
                    #   rc=1 退出, 页面只会报"异常退出" —— 而这正是"绝不把存活实例显示成已停止"
                    #   要挡住的那件事 (adopt 没命中时不改任何状态, 所以这里可以反复调)。
                    await asyncio.to_thread(S.inst.adopt, S.root(), S.cfg["hot_port"])
            # 快照与启动序列共用实例状态锁: 放线程里拿, 免得面板的 HTTP/WS 响应性跟着别人
            #   进程的退出速度走
            inst = await asyncio.to_thread(S.inst.snapshot)
            lines = S.inst.log_since(inst_seq)
            if lines:
                inst_seq = lines[-1][0]
            msg = {"type": "tick", "instance": inst, "tasks": S.op.summaries()}
            if lines:
                msg["log"] = lines
            scan = S.scan()
            if scan != last_scan:
                last_scan = scan
                msg["scan"] = scan
            # FPS 读数增量: 游标是**本连接**的状态, 而历史会被重启/认领清空 (下标回到 0) ——
            #   不清零则本连接的游标停在旧会话的条数上, 页面从此收不到新读数 (旧会话的条数
            #   比新会话长时永远收不到)。清零后 from=0, 前端按既有约定整表替换 (与重连同一条
            #   路径); 同时先发一条空的 from=0 让页面立刻清表, 不然旧会话的读数要挂到新会话
            #   的第一条 [AI FPS] 行 (60s) 才被换掉。
            if inst["fps_epoch"] != fps_ep:
                fps_ep = inst["fps_epoch"]
                fps_i = 0
                msg["fps_new"] = {"from": 0, "items": []}
            fh = S.inst.fps_hist_since(fps_i)
            if fh:
                msg["fps_new"] = {"from": fps_i, "items": fh}
                fps_i += len(fh)
            now = time.time()
            if now - last_tel >= 2.0:
                last_tel = now
                msg["telemetry"] = telemetry.snapshot()
            tlog = []
            for t in msg["tasks"]:
                prev = task_seqs.get(t["id"], 0)
                new = S.op.log_since(t["id"], prev)
                if new:
                    task_seqs[t["id"]] = new[-1][0]
                    tlog.append({"id": t["id"], "lines": new})
            if tlog:
                msg["task_log"] = tlog
            await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


@app.on_event("startup")
def _startup():
    # 孤儿认领: WebUI 自身 (重) 启动时, 已存活的 aimbot 必须可见, 不允许显示"已停止"
    try:
        S.inst.adopt(S.root(), S.cfg["hot_port"])
    except Exception:
        pass
    telemetry.start_npu_poller()
