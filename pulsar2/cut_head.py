#!/usr/bin/env python3
"""把端到端导出的"挑候选"那一段切掉, 交回网格布局 —— 不需要 .pt, 不重新训练。

用法: python3 cut_head.py <in.onnx> <out.onnx> [--corners]

背景: 端到端导出在**坐标与分数合并成一张张量之后**才做 TopK 选候选, 再拿选中的下标去
Gather 框与分数、并用 Mod 把展平下标还原成网格坐标。那张合并张量之后的整段选择逻辑对量化
是无解的: 选择是分数的**离散函数**, 量化扰动分数就换掉选中的候选集, 下游每个元素都跟着指向
另一个目标 —— 实测同一张图上类别列 150/300 行与 FP32 不同、框的逐元素 MAE 168 px, 而按分数
重排后仍是 168 px(不是坐标量化误差, 是"换了个人")。所以这里把整段砍掉, 而不是想办法让它转。

**怎么砍: 从合并点往回退一步, 让输出从两条分支直接拼出来。**
合并张量本身**不能**拿来当输出用: 它是"坐标(0..W)与分数(0..1)混在一张"的张量, 一个张量只有
一个 scale, 而工具链对**搬运算子**(Concat/Split/Reshape)的输出不再往上升精度 —— 实测给产生
它的 Concat 配 U16、它的两个输入也确实是 16 位, 它自己仍是 8 位(步长由坐标定), 于是分数整列
读成 0.0(板上 0 候选, 而同一帧 FP32 是 0.93)。所以输出改为: 框走算出来的 cx,cy,w,h、分数直接
接它的生产者 —— 两条分支全程都是**计算算子**的输出(16 位可配), 最后拼一次 Concat 就是图输出。
这与网格导出那条一直能用的路是同一个形态(输出的生产者是最后那次 Concat)。

一条实测的约定差异: 端到端的框是 **x1,y1,x2,y2**(NMS 要用角点), 而主机 `src/core/detect` 的
YOLOv8-11 一支读的是 **cx,cy,w,h** —— 所以框这一支要补 4 个逐元素算子换成中心式(顺带让这条
分支落到计算算子上)。要留角点就加 `--corners`(代价: 框那一支退回搬运算子的 8 位, 分辨率 ~W/255)。

合并张量怎么认(全部由图的装配读出, 不是猜): 从最终输出回溯, 取**第一个**满足"rank 3、
batch=1、除 batch 外最短且唯一短的那根通道轴在位置 1、长度 >= 5"的 Concat 输出。端到端输出
[1,300,6] 因短轴在最后而不满足。它的输入按通道数分成两条分支: 4 通道的是框, 其余按原顺序
是类分数(通道顺序照抄合并算子, 主机的 argmax 依赖这个顺序)。

两张分支要提成 U16 的算子名写进 ONNX metadata(`pulsar2_cut_u16_ops`), `gen_config.py` 读它。
找不到合并张量、或它的输入不像"4 通道框 + 其余分数"时原样退出(rc=3), 不做任何猜测性改写。
"""

import sys
import heapq

import numpy as np
import onnx
from onnx import numpy_helper


def toposort(graph):
    """稳定拓扑排序(与 patch_onnx.py 同一支): 原序合法时逐位不动。"""
    n = len(graph.node)
    avail = {x.name for x in graph.input} | {x.name for x in graph.initializer}
    queued = [False] * n
    ready = []

    def try_push(j):
        if not queued[j] and all((not i) or i in avail for i in graph.node[j].input):
            queued[j] = True
            heapq.heappush(ready, j)

    for j in range(n):
        try_push(j)
    order = []
    while ready:
        k = heapq.heappop(ready)
        order.append(k)
        avail.update(graph.node[k].output)
        for j in range(n):
            if not queued[j]:
                try_push(j)
    return None if len(order) != n else [graph.node[k] for k in order]


def shapes(graph):
    d = {}
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        t = v.type.tensor_type
        if t.HasField("shape") and len(t.shape.dim) == 3:
            s = [x.dim_value if x.HasField("dim_value") else None for x in t.shape.dim]
            if all(x is not None for x in s):
                d[v.name] = s
    return d


def find_merge(graph, shp):
    """从输出回溯, 第一个"属性在最内层"的 Concat —— 坐标与分数刚合并的那张张量。"""
    prod = {}
    for nd in graph.node:
        for o in nd.output:
            prod[o] = nd
    seen, queue = set(), [o.name for o in graph.output]
    while queue:
        t = queue.pop(0)                      # 广度优先: 先碰到的是离输出最近的那张
        nd = prod.get(t)
        if nd is None or id(nd) in seen:
            continue
        seen.add(id(nd))
        s = shp.get(t)
        if nd.op_type == "Concat" and s and s[0] == 1 and s[1] >= 5 and s[1] < s[2]:
            inner = s[1:]
            if inner.count(min(inner)) == 1 and inner.index(min(inner)) == 0:
                return t, s, nd
        queue.extend(nd.input)
    return None, None, None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    corners = "--corners" in sys.argv
    if len(args) < 2:
        print(__doc__)
        return 1
    src, dst = args[0], args[1]

    m = onnx.load(src, load_external_data=True)
    try:
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as e:
        print("   形状推断失败 (%s) —— 靠图声明继续" % type(e).__name__)
    g = m.graph
    shp = shapes(g)

    # 已经是网格布局就不动它(标准 v8/v11 导出走这条)
    outs = list(g.output)
    if len(outs) == 1:
        t = outs[0].type.tensor_type
        s = [x.dim_value if x.HasField("dim_value") else None for x in t.shape.dim]
        if len(s) == 3 and s[1] is not None and s[2] is not None and s[1] >= 5 and s[1] < s[2]:
            print("   已经是网格布局 %s %s —— 不需要切" % (outs[0].name, s))
            return 3

    merge, mshape, mnode = find_merge(g, shp)
    if merge is None:
        print("   ✗ 找不到合并张量 (rank3、batch=1、除 batch 外最短唯一轴在位置 1 且 >=5 的 Concat)")
        print("     这不是端到端导出, 或结构与预期不同 —— 原样退出, 不改写")
        return 3

    prod = {}
    for nd in g.node:
        for o in nd.output:
            prod[o] = nd
    ins = [t for t in mnode.input if t]
    ch = {t: (shp.get(t) or [None, None, None])[1] for t in ins}
    if any(ch[t] is None for t in ins):
        print("   ✗ 合并算子的输入通道数判不出 —— 原样退出")
        return 3
    boxes = [t for t in ins if ch[t] == 4]
    scores = [t for t in ins if t not in boxes]
    if len(boxes) != 1 or not scores or 4 + sum(ch[t] for t in scores) != mshape[1]:
        print("   ✗ 合并算子的输入不是'一个 4 通道框 + 其余类分数'(通道数 %s) —— 原样退出"
              % [ch[t] for t in ins])
        return 3
    box_t = boxes[0]
    print("   合并张量 %s %s, 输入: 框 %s(%d 通道) + 分数 %s"
          % (merge, mshape, box_t, ch[box_t], [(t, ch[t]) for t in scores]))

    roots = [box_t] + list(scores)
    keep, stack = set(), list(roots)
    while stack:
        t = stack.pop()
        nd = prod.get(t)
        if nd is None or id(nd) in keep:
            continue
        keep.add(id(nd))
        stack.extend(nd.input)
    nodes = [nd for nd in g.node if id(nd) in keep]
    dropped = len(g.node) - len(nodes)

    # 两条分支的生产者算子: gen_config 要把它们的输出提成 U16
    lift = [prod[t].name for t in roots if prod.get(t) is not None and prod[t].name]

    inits, extra = [], []
    nc = sum(ch[t] for t in scores)
    if corners:
        # 保留角点: 直接按通道拆开接进输出(框那一支因此留在搬运算子上, 分辨率 ~W/255)
        parts = ["cut_b0", "cut_b1", "cut_b2", "cut_b3"]
        extra.append(onnx.helper.make_node("Split", [box_t], parts, name="cut_box_split",
                                           axis=1, split=[1, 1, 1, 1]))
    else:
        # x1,y1,x2,y2 -> cx,cy,w,h:  cx=(x1+x2)/2, cy=(y1+y2)/2, w=x2-x1, h=y2-y1
        inits.append(numpy_helper.from_array(np.array([0.5], dtype=np.float32), "cut_half"))
        extra.append(onnx.helper.make_node("Split", [box_t], ["cut_b0", "cut_b1", "cut_b2", "cut_b3"],
                                           name="cut_box_split", axis=1, split=[1, 1, 1, 1]))
        extra.append(onnx.helper.make_node("Add", ["cut_b0", "cut_b2"], ["cut_x12"], name="cut_x12"))
        extra.append(onnx.helper.make_node("Add", ["cut_b1", "cut_b3"], ["cut_y12"], name="cut_y12"))
        extra.append(onnx.helper.make_node("Mul", ["cut_x12", "cut_half"], ["cut_cx"], name="cut_cx"))
        extra.append(onnx.helper.make_node("Mul", ["cut_y12", "cut_half"], ["cut_cy"], name="cut_cy"))
        extra.append(onnx.helper.make_node("Sub", ["cut_b2", "cut_b0"], ["cut_w"], name="cut_w"))
        extra.append(onnx.helper.make_node("Sub", ["cut_b3", "cut_b1"], ["cut_h"], name="cut_h"))
        parts = ["cut_cx", "cut_cy", "cut_w", "cut_h"]
    for si, t in enumerate(scores):
        c = ch[t]
        if c == 1:
            parts.append(t)
        else:
            outs = ["cut_s%d_%d" % (si, j) for j in range(c)]
            extra.append(onnx.helper.make_node("Split", [t], outs, name="cut_score_split_%d" % si,
                                               axis=1, split=[1] * c))
            parts.extend(outs)
    extra.append(onnx.helper.make_node("Concat", parts, ["output0"], name="cut_out", axis=1))
    nodes = nodes + extra
    order = toposort(onnx.helper.make_graph(nodes, "g", g.input, g.output,
                                            initializer=list(g.initializer) + inits))
    if order is not None:
        nodes = order

    vi = onnx.helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 4 + nc, mshape[2]])
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(inits)
    del g.output[:]
    g.output.extend([vi])
    e = m.metadata_props.add()
    e.key = "pulsar2_cut_u16_ops"
    e.value = ",".join(lift)
    alive = {v.name for v in g.input} | {i.name for i in g.initializer}
    alive |= {o for nd in g.node for o in nd.output}
    viv = [v for v in g.value_info if v.name in alive and v.name != "output0"]
    del g.value_info[:]
    g.value_info.extend(viv)

    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, dst)
    print("   %s -> %s" % (src, dst))
    print("   切掉 %d 个算子, 留 %d 个; 输出 %s %s (框 %s)"
          % (dropped, len(nodes), "output0", [1, 4 + nc, mshape[2]],
             "保留角点 x1y1x2y2" if corners else "已换成中心式 cx,cy,w,h"))
    print("   记为 U16 的算子: %s" % e.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
