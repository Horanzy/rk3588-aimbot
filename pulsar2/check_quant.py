#!/usr/bin/env python3
"""按通道角色量量化误差 —— 工具链的余弦看不见这件事。

本库的这些导出把**像素坐标(0..W)与置信度(0..1)拼进同一张输出张量**, 而量化是**按张量**
定一个尺度: 量程由坐标定(640:1), 置信度只剩 1/码长 档。工具链自己的逐层表
(`quant/debug/precision_analysis_table.txt`) 按**整张张量**报 Cosine/MSE, 于是这张张量的
"合格"由坐标通道的大范数决定。实测同一次构建的两种配置(320 输入 4 类模型,
标定本游戏专用集 400 张, NPU3):

    配置        output0 位宽  码长        0..1 上档数  分数过阈值   整张 cosine
    单张 U8     8           1.55457     0(不足一档)   1/10        0.99625
    提 U16      16          0.006048901 165          9/10        0.99626

**两行的 cosine 只差 1e-5, 而可用的候选从 10 个掉到 1 个** —— 这就是为什么判据不能是余弦:
它由坐标通道的范数主导, 分数通道被压平了它照样很高(本库文档里记的那个模型更极端:
板上 0 候选, 输出层 cosine 0.99951)。本脚本把两类通道**分开**报。

两个转储点, 同一张 output0 的 MSE 不同源: `io/` 给出 78.1468, `numpy/` 给出 75.5055 ——
后者与工具链自己那张表里的 75.50555 逐位一致。按角色的表走 `io/`(它直接给图输出张量,
逐通道切分最直), 逐层扫描走 `numpy/`。`numpy/xrun` 侧两种 dtype 都有: 逐层激活是**原始码**
(u1/u2), 图输出已经是去量化的浮点 —— 脚本按 dtype 分辨, 只对整数码去量化(对已去量化的浮点
再乘一次 scale, MAE 会被放大到码长的倒数倍: 实测 output0 从 3.02 变成 56.2)。

U8 那半的分数角色网格残差恰好是 **0**: 它只有两个取值, 0 与正好一个码长 —— 本库文档里那句
"置信度通道最大值恰好等于一个 step"的精确形式。逐层扫描 251 张耗时约 2 s, 最差的一层是
`/model.23/Mul_2_output_0`(MAE 6.04), 即输出张量前一级的合并点。

判据(都是上面那些数的直接后果, 没有阈值):
    一张张量的码长 s 决定置信度能分辨成 1/s 档。s = 1.55457 时 0..1 上不足一档, 整条置信度
    通道只剩 0 与 1 两个码位(实测该通道最大值恰好等于一个码长); 把**产生输出张量的那些
    算子**提成 U16 后 s = 0.006048901, 0..1 上 165 档。两条都由 `quant_axmodel.json` 的
    scale 直接算出, 与工具链那条 "very big range" 提示是同一件事。

用法:
    python3 check_quant.py <产物目录>              # 例: python3 check_quant.py axmodel
    python3 check_quant.py <产物目录> --layers     # 追加逐层扫描(251 张约 2 s)
    python3 check_quant.py <产物目录> --box 5      # 头部带 objectness 的 v5 风格布局

复现上面那张表(占位名 mymodel, 与 README 的"一条命令"同一个写法):
    KEEP_QUANT=1 bash convert.sh -m onnx/mymodel.onnx -s 400
    KEEP_QUANT=1 NO_LAST_U16=1 bash convert.sh -m onnx/mymodel.onnx -s 400 \
        -c dataset/calib_mymodel.tar -o mymodel_u8 -O axmodel_u8
    python3 check_quant.py axmodel ; python3 check_quant.py axmodel_u8
(码长是 MinMax 的产物, 所以它带着那次标定的量程 —— 换标定集这个数就变, 结论(0 档还是
165 档)不变。)

读什么(都由 KEEP_QUANT=1 的那次构建留在产物目录里):
    quant/quant_axmodel.json                  每张张量的位宽与 scale/zero_point
    quant/debug/io/{float,quant}/<张量>.npy   FP32 参考 vs 量化后: 模型的真实输出
    quant/debug/numpy/{float,xrun}/<张量>.npy 逐层激活(仅 --layers; xrun 侧是原始码)

三条经实测确认的口径:

  * **码长与档数取自配置, 不取自转储。** 在单张图上数"出现了几个不同取值"得到的是**内容**
    (实测 U16 的分数通道只有 6 个不同取值, 而分辨率是 165 档), 那个数列只叫"观察档数"。
  * **scale 先在 `hash` 下找, 再在 `dominator` 下找。** 两者相同时条目在 `hash` 下; 不同时
    (实测 apex 的 U8 输出张量: hash 2024901455 没有条目, dominator 4251374706 有)条目在
    `dominator` 下。取到的 scale 与转储互证: U8 的 1.554567575 对上转储的最小非零相邻差
    1.5545654, U16 的 0.006048901 与 xrun 原始码去量化后的值在 16800 个元素上最大差
    1.37e-05(float32 的精度)。
  * **角色是张量上的一个划分。** 整张 MAE 必等于两角色按元素数加权的平均 —— 脚本拿这条当
    自检, 通道下标写错时不会静默给出一张看起来合理的表。

依赖: 只用标准库。转储是 numpy 写的 .npy(魔数 + 表头字典 + 裸缓冲), 这里直接解 ——
于是本脚本在没装 numpy 的机器上(本库的 WSL 宿主就没有)也跑得动, 不必进容器。
"""

import argparse
import array
import ast
import json
import math
import os
import struct
import sys

NPY_MAGIC = b"\x93NUMPY"

# numpy 的 descr -> (array 的 typecode, 字节数)。只收窄字节序('<')与单字节('|'):
#   转储都是小端写出来的, 收到大端声明时宁可报错也不静默读错数值。
NPY_DTYPES = {
    "<f4": ("f", 4), "<f8": ("d", 8),
    "<i4": ("i", 4), "<u4": ("I", 4),
    "<i2": ("h", 2), "<u2": ("H", 2),
    "|i1": ("b", 1), "|u1": ("B", 1),
}

# 固件侧的置信度阈值: CLI 的 -t 与启动模板的同一个默认值(两边都在 core/state.h 取值)。
#   拿它把"分数通道还活着吗"变成一个带操作含义的数 —— 过阈值的候选个数, 与 FP32 并排。
DEFAULT_THR = 0.5


def read_npy(path):
    """(shape, array) —— 只收 C 序的 .npy, 转储正是这么写的。"""
    shape, vals, _descr = read_npy_typed(path)
    return shape, vals


def read_npy_typed(path):
    """(shape, array, descr) —— descr 决定它是不是原始码, 见 report_layers。"""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:6] != NPY_MAGIC:
        raise ValueError("不是 .npy: %s" % path)
    major = raw[6]
    if major == 1:
        hlen, off = struct.unpack("<H", raw[8:10])[0], 10
    elif major == 2:
        hlen, off = struct.unpack("<I", raw[8:12])[0], 12
    else:
        raise ValueError("不支持的 .npy 版本 %d: %s" % (major, path))
    hdr = ast.literal_eval(raw[off:off + hlen].decode("latin1"))
    descr = hdr["descr"]
    if descr not in NPY_DTYPES:
        raise ValueError("不认识的 dtype %r: %s" % (descr, path))
    if hdr["fortran_order"]:
        raise ValueError("fortran_order=True 的 .npy 不支持: %s" % path)
    tc, size = NPY_DTYPES[descr]
    n = 1
    for d in hdr["shape"]:
        n *= int(d)
    a = array.array(tc)
    a.frombytes(raw[off + hlen:off + hlen + n * size])
    if size > 1 and sys.byteorder != "little":
        a.byteswap()
    return tuple(int(d) for d in hdr["shape"]), a, descr


def channel_axis(shape):
    """通道轴 = 除 batch 外最短且唯一短于其余各根的那一轴; 判不出返回 None。

    与 patch_onnx.py 同一条判据: 解码头每根锚点的属性向量远短于锚点数, 所以它在
    `[1, C, N]` 与 `[1, N, C]` 两种布局上落到同一根轴上, 不需要额外的布局信息。
    """
    if shape is None or len(shape) < 3:
        return None
    inner = list(shape[1:])
    lo = min(inner)
    if inner.count(lo) != 1:
        return None
    return 1 + inner.index(lo)


def take_channels(shape, values, axis, channels):
    """取通道轴上属于 channels 的那些元素。

    C 序(行主序)的下标展开: 扁平下标 f 落在通道 `(f // 其后各根之积) % C` 上 ——
    `[1, C, N]` 与 `[1, N, C]` 都由这一条覆盖。
    """
    n_chan = shape[axis]
    stride_after = 1
    for d in shape[axis + 1:]:
        stride_after *= d
    want = set(channels)
    return [v for f, v in enumerate(values) if ((f // stride_after) % n_chan) in want]


def distinct(values):
    return len(set(values))


def error(vals, refs):
    """(MAE, MSE)。"""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    sa = ss = 0.0
    for a, b in zip(vals, refs):
        d = a - b
        sa += abs(d)
        ss += d * d
    return sa / n, ss / n


def cosine(a, b):
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return dot / math.sqrt(na * nb)


def grid_residual(values, step):
    """取值偏离"码长整数倍"的最大量(真值单位)。

    它是 scale 取对了没有的自检: 反量化出来的每个值都必是码长的整数倍, 偏差只剩 float32
    的舍入。没有 scale 时返回 None —— 不拿观察值的最小间隔冒充码长。
    """
    if not step or step <= 0 or not values:
        return None
    return max(abs(v / step - round(v / step)) for v in values) * step


def min_gap(values):
    u = sorted(set(values))
    return min((b - a for a, b in zip(u, u[1:])), default=0.0)


def load_config(quant_dir):
    qj = os.path.join(quant_dir, "quant_axmodel.json")
    if not os.path.isfile(qj):
        sys.exit("读不到 %s —— 产物目录不对, 或这次构建没留 quant/(构建时 KEEP_QUANT=1)"
                 % qj)
    with open(qj, encoding="utf-8") as f:
        return json.load(f)


def tensor_params(cfg, name):
    """(bit_width, scale, zero_point) —— 缺哪项就是 None。

    `values` 的键是 hash 的十进制字符串而 `tensor_configs` 里是整数(不转字符串查不到);
    且该用 hash 还是 dominator 取决于这一层有没有 requant: 两者相同时条目在 hash 下,
    不同时在 dominator 下(实测 apex 的 U8 输出张量)。
    """
    for _op, tensors in cfg.get("tensor_configs", {}).items():
        v = tensors.get(name)
        if v is None:
            continue
        scale = zp = None
        for key in ("hash", "dominator"):
            h = v.get(key)
            e = cfg.get("values", {}).get(str(h)) if h is not None else None
            if e and e.get("scale") is not None:
                s, z = e["scale"], e.get("zero_point")
                scale = s[0] if isinstance(s, list) and s else s
                zp = z[0] if isinstance(z, list) and z else z
                break
        return v.get("bit_width"), scale, zp
    return None, None, None


def unique_bit_widths(cfg):
    """{张量名: 位宽} —— **按唯一张量名**计数(同一张张量挂在两个算子上只算一张)。

    本库的文档用这个口径记数("模型B 实测 3 张 16 位"), 按算子条目数会数出 5 张。
    """
    seen = {}
    for _op, tensors in cfg.get("tensor_configs", {}).items():
        for name, v in tensors.items():
            if name not in seen and v.get("bit_width") is not None:
                seen[name] = v.get("bit_width")
    return seen


def graph_outputs(quant_dir, cfg):
    io_dir = os.path.join(quant_dir, "debug", "io")
    if not os.path.isdir(os.path.join(io_dir, "float")):
        sys.exit("读不到 %s —— 这次构建没留转储(构建时 KEEP_QUANT=1)"
                 % os.path.join(io_dir, "float"))
    inputs = set(cfg["quant_config"].get("input_configs", {}))
    names = [n[:-4] for n in sorted(os.listdir(os.path.join(io_dir, "float")))
             if n.endswith(".npy")]
    return [n for n in names if n not in inputs], io_dir, sorted(inputs)


def report_config(cfg, names):
    seen = unique_bit_widths(cfg)
    hist = {}
    for bw in seen.values():
        hist[bw] = hist.get(bw, 0) + 1
    print("位宽(唯一张量名计数): " +
          " · ".join("%d 位 %d 张" % (b, hist[b])
                     for b in sorted(hist, key=lambda b: -hist[b])))
    u16 = sorted(n for n, b in seen.items() if b == 16)
    print("  16 位的张量: %s"
          % (", ".join(u16) if u16 else "(整图没有 —— 产生输出张量的算子没提 U16)"))
    for name in names:
        bw, scale, zp = tensor_params(cfg, name)
        if bw is None:
            print("  输出 %s: 配置里没有这张张量的量化参数" % name)
            continue
        line = "  输出 %s: %s 位" % (name, bw)
        if scale:
            line += " · 码长 %.8g · 零位 %s · 0..1 上 %d 档(1/码长)" % (scale, zp,
                                                                      int(1.0 / scale))
        else:
            line += " · 配置里没有 scale —— 码长取不到, 下面只报观察档数"
        print(line)


def report_tensor(name, io_dir, step, box, thr):
    """打一张输出张量的按角色表, 返回(f分数通道过阈值_量化, 过阈值_FP32, 用到的最高码位)。"""
    fsh, fval = read_npy(os.path.join(io_dir, "float", name + ".npy"))
    qsh, qval = read_npy(os.path.join(io_dir, "quant", name + ".npy"))
    if fsh != qsh:
        print("  ⚠ 参考与量化后的形状不同(%s vs %s)—— 跳过" % (fsh, qsh))
        return None
    ax = channel_axis(fsh)
    head = "  %s %s" % (name, list(fsh))
    if ax is None:
        print(head + " · 判不出通道轴(除 batch 外没有唯一最短的一根)—— 只报整张")
    else:
        n_chan = fsh[ax]
        print(head + " · 通道轴 %d, %d 通道 ⇒ box 0..%d, 分数 %d..%d(类数 %d)"
              % (ax, n_chan, box - 1, box, n_chan - 1, n_chan - box))
    mae_all, mse_all = error(qval, fval)
    print("    整张:  cosine %.5f   MAE %.6g   MSE %.6g"
          % (cosine(fval, qval), mae_all, mse_all))
    if ax is None:
        return None
    n_chan = fsh[ax]
    if n_chan <= box:
        print("    (通道数 %d 不大于 box 宽度 %d —— 没有分数通道可分开报)" % (n_chan, box))
        return None
    print("    %-4s %-8s %8s  %-23s %-11s %-11s %s"
          % ("角色", "通道", "观察档数", "量程", "MAE", "MSE", "网格残差"))
    parts = {}
    for role, chans in (("box", range(0, box)), ("分数", range(box, n_chan))):
        vals = take_channels(qsh, qval, ax, chans)
        refs = take_channels(fsh, fval, ax, chans)
        mae, mse = error(vals, refs)
        parts[role] = (len(vals), mae)
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
        res = grid_residual(vals, step)
        print("    %-4s %-8s %8d  [%-10.6g %-10.6g] %-11.6g %-11.6g %s"
              % (role, "%d..%d" % (min(chans), max(chans)), distinct(vals), lo, hi, mae, mse,
                 "—" if res is None else "%.3g" % res))
    # 角色是张量上的一个划分: 整张 MAE 必等于两角色按元素数加权的平均。通道下标写错时
    #   这条自检会当场报出来, 而不是给出一张看起来合理的表。
    tot = sum(n for n, _ in parts.values())
    wavg = sum(n * m for n, m in parts.values()) / tot if tot else 0.0
    if abs(wavg - mae_all) > 1e-6 * max(1.0, abs(mae_all)):
        print("    ⚠ 划分自检不过: 角色加权 MAE %.6g ≠ 整张 %.6g —— 通道下标可能错了"
              % (wavg, mae_all))
    svals = take_channels(qsh, qval, ax, range(box, n_chan))
    srefs = take_channels(fsh, fval, ax, range(box, n_chan))
    nq = sum(1 for v in svals if v > thr)
    nr = sum(1 for v in srefs if v > thr)
    print("    分数通道过阈值(>%.3g): 量化 %d 个 / FP32 参考 %d 个" % (thr, nq, nr))
    peak = None
    if step and svals:
        peak = round(max(svals) / step)
        print("    分数通道用到的码位: 0..%d(0..1 上共 %d 档)" % (peak, int(1.0 / step)))
    elif svals:
        print("    分数通道量程: [%g, %g], 观察档数 %d"
              % (min(svals), max(svals), distinct(svals)))
    return nq, nr, peak


def report_verdict(cfg, names, stats):
    print()
    print("判据")
    for name in names:
        bw, scale, _zp = tensor_params(cfg, name)
        st = stats.get(name)
        tail = ""
        if st and st[0] is not None:
            tail = "过阈值的候选从 FP32 参考的 %d 个掉到 %d 个。" % (st[1], st[0])
        if not scale:
            print("  %s: 配置里没有这张张量的 scale —— 码长取不到, 看上面分数角色的观察档数"
                  "与过阈值个数。%s" % (name, tail.strip()))
            continue
        n = int(1.0 / scale)
        if n < 2:
            print("  %s: 码长 %.6g ⇒ 0..1 上 %d 档(不足一档)—— 置信度只能落进 0 与 1 两个"
                  "码位, 板上没有可用的置信度分辨率。%s" % (name, scale, n, tail))
            print("     这张张量在 %s 位下量程被坐标占满(坐标量程/置信度量程 ≈ 640:1); 要的是"
                  "产生它的那层提 U16, 或把两类通道拆成两张张量。" % bw)
        else:
            print("  %s: 码长 %.6g ⇒ 0..1 上 %d 档, 置信度阈值只能落在 1/%d 的格子上 —— "
                  "可分辨。%s" % (name, scale, n, n, tail))
    print("  余弦不作判据: 整张的 cosine 由坐标通道的范数主导 —— 实测同一个模型的两次构建"
          "(U8 / 提 U16)cosine 只差 1e-5(0.99625 / 0.99626), 而可用的置信度档数是 0 与 165。")


def report_layers(quant_dir, cfg):
    """逐层扫描: xrun 侧是原始码, 用每层自己的 scale/zero_point 去量化后与 FP32 比。"""
    root = os.path.join(quant_dir, "debug", "numpy")
    flo, xru = os.path.join(root, "float"), os.path.join(root, "xrun")
    if not (os.path.isdir(flo) and os.path.isdir(xru)):
        print("\n(没有 numpy/{float,xrun} 转储 —— 跳过逐层扫描)")
        return
    # 转储文件名是 ONNX 名把 '/' 换成 '_'(如 /model.0/act/Mul_output_0 →
    #   _model.0_act_Mul_output_0), 这个变换不可逆, 所以从配置的张量名正向建表。
    by_file = {}
    for _op, tensors in cfg.get("tensor_configs", {}).items():
        for tname in tensors:
            by_file[tname.replace("/", "_")] = tname
    names = sorted(set(os.listdir(flo)) & set(os.listdir(xru)))
    print("\n逐层扫描: %d 个同名张量(xrun 侧按各层自己的 scale 去量化)" % len(names))
    rows = []
    for i, fn in enumerate(names):
        if i % 40 == 0:
            print("  ... %d/%d" % (i, len(names)), file=sys.stderr)
        tname = by_file.get(fn[:-4])
        if tname is None:
            continue
        _bw, scale, zp = tensor_params(cfg, tname)
        if not scale:
            continue
        try:
            _fsh, fv, _fd = read_npy_typed(os.path.join(flo, fn))
            _qsh, qv, qd = read_npy_typed(os.path.join(xru, fn))
        except (ValueError, OSError) as e:
            print("  ⚠ %s: %s" % (fn, e), file=sys.stderr)
            continue
        if len(fv) != len(qv):
            continue
        # xrun 侧两种都有: 逐层激活是**原始码**(u1/u2), 图输出已经被工具链去量化成浮点。
        #   只有整数码才该按该层的 scale/zero_point 去量化 —— 对已去量化的浮点再乘一次
        #   scale, MAE 会被放大到码长的倒数倍(实测 output0 会从 3.1 变成 56)。
        if qd.startswith(("<f", ">f")):
            dq = qv
        else:
            z = zp or 0.0
            dq = [(v - z) * scale for v in qv]
        mae, mse = error(dq, fv)
        rows.append((mae, mse, fn, scale))
    if not rows:
        print("  (没有一张张量能对上配置 —— 逐层扫描空)")
        return
    rows.sort(key=lambda r: -r[0])
    print("  %-40s %-12s %-12s %s" % ("张量(转储名)", "MAE", "MSE", "码长"))
    for mae, mse, fn, scale in rows[:12]:
        print("  %-40s %-12.6g %-12.6g %.6g" % (fn[:40], mae, mse, scale))
    print("  (MAE 最大的 12 张; 共 %d 张对上配置; 整图最大在 %s)" % (len(rows), rows[0][2]))


def main():
    ap = argparse.ArgumentParser(description="按通道角色量量化误差(读 Pulsar2 的转储)")
    ap.add_argument("out_dir", help="产物目录, 如 axmodel")
    ap.add_argument("--box", type=int, default=4,
                    help="box 通道宽度: 4 = v8/v11/v26 的 rank-3 导出(box 已在图里解好), "
                         "5 = 头部带 objectness 的 v5 风格布局")
    ap.add_argument("--thr", type=float, default=DEFAULT_THR,
                    help="置信度阈值(默认 %.2f, 与固件 -t 同一个默认值)" % DEFAULT_THR)
    ap.add_argument("--layers", action="store_true", help="追加逐层扫描(251 张约 2 s)")
    args = ap.parse_args()

    if args.box < 1:
        sys.exit("--box 要 ≥ 1")
    quant_dir = os.path.join(args.out_dir, "quant")
    cfg = load_config(quant_dir)
    names, io_dir, inputs = graph_outputs(quant_dir, cfg)
    print("产物 %s" % args.out_dir)
    print("输入张量: %s" % (", ".join(inputs) or "(配置里没记)"))
    report_config(cfg, names)
    print()
    print("按通道角色分开量(FP32 参考 vs 量化后, 同一张量同一次前向)")
    stats = {}
    for n in names:
        _bw, step, _zp = tensor_params(cfg, n)
        stats[n] = report_tensor(n, io_dir, step, args.box, args.thr)
    report_verdict(cfg, names, stats)
    if args.layers:
        report_layers(quant_dir, cfg)


if __name__ == "__main__":
    main()
