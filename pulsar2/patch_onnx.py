"""把 Pulsar2 不支持、或按它的量化口径处理不了的结构, 按语义精确改写 (在 pulsar2 容器内执行).

用法: python3 patch_onnx.py <in.onnx> <out.onnx>

改写前先把图整平: ONNX 要求节点按数据流排序, 而导出工具会把顺序打乱 —— 这本库里的
模型E 导出有 220 处"先引用后定义", 检查器据此拒绝整张图, 于是下面每一条重写都落不下去。
节点的先后不是数据流语义 (每个算子只由自己的输入定义), 稳定拓扑排序是同一张图的另一种
写法, 不改任何数值; 排不出序 (有环或悬空引用) 就保持原序并跳过检查。

三类问题, 都由工具链报错或实机读数定位得到, 不是猜测:

  1. `Mod(i, C)` —— 端到端解码里用 TopK 给出的**下标** i 还原网格坐标的一步
     (`i % C`)。C 是常量而 i 是运行期整数张量, 所以折叠不掉; Pulsar2 对 Mod
     报 OnnxOptimizationError。i 恒为非负 (TopK 下标), 因此
         i % C ≡ i − C·(i / C)
     整数除法向左截断, 在非负域上就是向下取整, 三个算子 (Div/Mul/Sub) 都是
     工具链支持的 —— 精确等价, 不引入任何新常量。

  2. 声明了多个输出、后端在其中一个上崩 (NPU 后端对高维原始头报
     `AssertionError("tname = ...")`)。当一张图同时声明了可用的低维输出与更高维的
     中间态输出时, 丢掉后者 —— 它们是同一份数据的前一级, 属于导出产物而不是网络语义。
     判据按**秩**: 存在 2 或 3 维输出时丢掉 4 维以上者。

  3. **混合量程输出** —— 融合解码头把像素坐标与置信度放进**同一张张量**, 而 Pulsar2
     的量化是**按张量**定一个 step, 于是坐标的量程压死了置信度的分辨率。实测: 同一张
     真实帧上通道 0..3 的量程是几十到 640 px, 通道 4 以后是 0..1 (见 `onnx/` 任一模型的
     任一输出); 现有 `.axmodel` 的 step 为 模型B 1.4915 / 模型F 1.6453 / Z320 1.3031 /
     模型D 1.2502 / 模型C 3.4864, 五个模型的置信度通道**最大值恰好等于一个 step** —— 整条
     置信度通道被压成 0/1, 模型C 的目标置信度与类别通道读到恒 0 (同一帧原始 ONNX 有 7765 个
     >0.5 的候选, axmodel 有 0 个)。`output_processors[].output_dtype = "FP32"` 只是在
     呈现时反量化, 产生该张量的算子仍是 U8, 所以不解决。

  **混合量程输出在这里一律不动**：一张张量里同时装坐标 (0..W) 与置信度 (0..1) 确实会被
  按张量定的 step 压死置信度，但这件事由转换侧的另一层解决 —— `gen_config.py` 把
  **产生输出的那些算子**提成 U16（见该文件的注释与实测数字）。这里曾经按量程分组把输出
  重新声明成多张（"Concat 视图拆分" / "Split 分组"），那套已删除：它要按族做图手术，而且在
  端到端族的输出上过不了 Pulsar2 自己的 parser e2e 检查（实测 模型A：分成
  `GatherElements_1_output_0` / `Cast_2_output_0` 两个视图后报
  `check onnx parser e2e failed`）。U16 那条零手术、对任何模型成立，代价是档数略低
  （实测 0..1 上 165 档 vs 拆分的 255 档）。

  通道轴的判定规则 (不猜): 解码头每根锚点的属性向量远短于锚点数, 所以通道轴是该张量
  **除 batch 以外最短的那根**, 并且它必须比其余每根都严格短 (最短者唯一)—— 这一条同时
  把 `[1, C, N]` 与 `[1, N, C]` 两种布局都判对, 不需要知道哪根是通道轴以外的信息。
  再要求它的长度是 `4 + k (k ≥ 1)`: 4 只是 box 组, 没有需要分开量化的对手。
  不满足就把该输出**原样放过并打一行说明** —— 认不出的结构一律不动。

  按形状认不如按装配认: 通道轴上的 `Concat` 把分组方式直接写在了图里, 所以它的各段长度
  全都已知、却**不是** box + score 时 (首段既不是 4 也不是 5), 那是图给出的反证, 不再退回
  按 `4 + k` 的形状去猜 (b)。分组长度有未知量时没有反证可言, 仍按形状判。形状与装配
  两条都不成立就原样放过。

  同理, 同一张图里已经**单独声明**了 box 组 (除通道轴外各根相同的 4/5 通道输出) 时, 与它
  同布局的那张张量是纯分数表, 拆它只会把一张分数表切成三份, 不是本规则要修的东西: (b) 不动
  它。这两条反证也顺带让改写幂等 —— 拆出来的分组要么长度不足 5, 要么与已声明的 box 组同布局。

  b) 还要能处理通道轴尺寸未知的情形 (模型C 就是: 工具链对它的 reshape 推不出形状, 只有
  `Concat` 输出宣言是已知的)。此时按 `Concat` 的定义 —— 输出在该轴上等于各输入之和 ——
  由**唯一**未知量解开; 有两个以上未知量就放弃。这仍是推导而不是猜测。

  两条改写都不需要知道哪一组是 box: 分开量程的收益只要求各组量程不同, 而 box 组的
  首件位置 (4/5 通道) 是导出工具的一致约定 —— 顺序即便反过来, 改写本身依然正确。
"""

import collections
import heapq
import os
import sys

import onnx
from onnx import TensorProto, shape_inference

src, dst = sys.argv[1:3]
m = onnx.load(src, load_external_data=True)
g = m.graph

# ---- 0. 拓扑序归一 ----
def toposort(graph):
    """稳定拓扑排序; 排不出 (有环/悬空引用) 返回 None。

    按下标最小的就绪节点优先出队: 原序已经合法时逐位不动 (合法序里第 i 个节点出队时,
    它之前的节点都已出队, 所以它的输入必已就绪, 于是下标 i 总是最小就绪者), 只有真的
    逆序了才动, 而且给出的仍是最接近原序的一种合法顺序。
    """
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


nodes = toposort(g)
resorted = 0
if nodes is not None and [id(n) for n in nodes] != [id(n) for n in g.node]:
    resorted = 1
    del g.node[:]
    g.node.extend(nodes)

# ---- 1. Mod -> Div/Mul/Sub ----
patched = 0
for node in list(g.node):
    if node.op_type != "Mod":
        continue
    x, c = node.input[0], node.input[1]
    base = (node.name or "mod").replace("/", "_")
    new = [
        onnx.helper.make_node("Div", [x, c], [base + "_div"], base + "_div_node"),
        onnx.helper.make_node("Mul", [c, base + "_div"], [base + "_mul"], base + "_mul_node"),
        onnx.helper.make_node("Sub", [x, base + "_mul"], list(node.output), base + "_sub_node"),
    ]
    idx = list(g.node).index(node)
    del g.node[idx]
    for k, nd in enumerate(new):
        g.node.insert(idx + k, nd)
    patched += 1

# ---- 2. 丢掉冗余的高维输出 ----
ranks = []
for o in g.output:
    t = o.type.tensor_type
    ranks.append(len(t.shape.dim) if t.HasField("shape") else 0)
dropped = []
if any(r in (2, 3) for r in ranks) and any(r >= 4 for r in ranks):
    keep = [o for o, r in zip(list(g.output), ranks) if r in (2, 3)]
    dropped = [o.name for o, r in zip(list(g.output), ranks) if r not in (2, 3)]
    del g.output[:]
    g.output.extend(keep)


# ---- 输出宣言：只做第 2 条的丢弃, 不做任何拆分(理由见文件头) ----
def channel_axis(shape):
    """通道轴 = 除 batch 外最短且唯一短于其余各根的那一轴; 判不出返回 None。

    解码头每根锚点的属性向量 (4 box + 置信度) 远短于锚点数, 所以这条判据在
    `[1, C, N]` 与 `[1, N, C]` 两种布局上落到同一根轴上。
    """
    if shape is None or len(shape) < 3 or any(d is None for d in shape):
        return None
    inner = shape[1:]
    lo = min(inner)
    if inner.count(lo) != 1:
        return None
    return 1 + inner.index(lo)


def piece_vi(name, out, axis, size):
    """把输出宣言复制一份, 只在通道轴上换成该分组的长度 (其余轴 Concat/Split 都不改)。"""
    tt = onnx.TypeProto()
    tt.tensor_type.elem_type = out.type.tensor_type.elem_type
    for i, d in enumerate(out.type.tensor_type.shape.dim):
        nd = tt.tensor_type.shape.dim.add()
        if i == axis:
            nd.dim_value = size
        else:
            nd.CopyFrom(d)
    vi = onnx.ValueInfoProto()
    vi.name = name
    vi.type.CopyFrom(tt)
    return vi


def unique(name, taken):
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def real_producer(name, producer):
    """穿过 Identity (纯拷贝) 找真正的生产者。"""
    nd = producer.get(name)
    for _ in range(8):
        if nd is None or nd.op_type != "Identity":
            break
        nxt = producer.get(nd.input[0])
        if nxt is None:
            break
        nd = nxt
    return nd


def layout(shp, ca):
    """除通道轴以外的各根 —— 同一张解码头无论怎么分组, 这部分都一样。"""
    return tuple(d for i, d in enumerate(shp) if i != ca)


try:
    m = shape_inference.infer_shapes(m, strict_mode=False)
    g = m.graph
except Exception:  # 形状推断不是必须的: 判不出就走"原样放过"
    pass

opset = 0
for o in m.opset_import:
    if o.domain in ("", "ai.onnx"):
        opset = max(opset, o.version)

dims = {}
for v in list(g.value_info) + list(g.input) + list(g.output):
    if not v.type.HasField("tensor_type") or not v.type.tensor_type.HasField("shape"):
        continue
    dims[v.name] = [d.dim_value if d.HasField("dim_value") else None
                    for d in v.type.tensor_type.shape.dim]

producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n

taken = {v.name for v in list(g.value_info) + list(g.input) + list(g.output)}
taken |= {i.name for i in g.initializer}
taken |= {x for nd in g.node for x in nd.output}

view_split = 0
box_split = 0
skipped = []
outs = []

# 图里已经**另有**一个单独声明的 box 组 (同锚点布局的 4/5 通道输出) —— 那 (b) 面对的是纯
# 分数张量, 拆它只会把一张分数表切成三份, 不是本规则要修的东西, 所以不拆。
box_layouts = {}
for q in g.output:
    q_shp = dims.get(q.name)
    q_ca = channel_axis(q_shp)
    if q_ca is not None and q_shp[q_ca] in (4, 5):
        box_layouts.setdefault(layout(q_shp, q_ca), set()).add(q.name)


def box_declared(name, shp, ca):
    """同锚点布局的 box 组是否由**另一个**输出单独声明; 自身不算。"""
    return bool(box_layouts.get(layout(shp, ca), set()) - {name})


for o in list(g.output):
    shp = dims.get(o.name)
    ca = channel_axis(shp)
    n = real_producer(o.name, producer)
    if ca is None:
        skipped.append(f"{o.name} (通道轴判不出)")
        outs.append(o)
        continue

    # 混合量程输出**不拆**：一张张量里同时装坐标(0..W)与置信度(0..1)确实会让按张量定的
    #   step 压死置信度，但这件事在转换侧有一层更省的处理 —— gen_config.py 把**产生输出的
    #   那些算子**提成 U16。实测(模型B, 320)：单张 U8 时 0..1 上 0 档(全 0)，提 U16 后 165 档；
    #   而拆成两张各拿自己的尺度是 255 档 —— 只差 1.5 倍，代价却是按族做图手术，且在有些族
    #   的输出上过不了 Pulsar2 自己的 parser e2e 检查(实测 模型A 的端到端族)。所以这里统一
    #   保留单张合并输出，量程问题只由那一层解决。
    outs.append(o)

# 回写输出宣言：这里已经不再做任何拆分，回写只为让第 2 条的"丢弃冗余输出"生效 —— 原先
#   的条件只看 view_split/box_split，于是没有拆分可做时丢弃也一并失效。
if dropped:
    del g.output[:]
    g.output.extend(outs)

onnx.checker.check_model(m, full_check=True)
onnx.save(m, dst)

c = collections.Counter(n.op_type for n in g.node)
print(f"   Mod 替换 {patched} 处, 输出 {len(g.output)} 个", end="")
if dropped:
    print(f" (丢弃 {dropped})", end="")
if resorted:
    print(" (拓扑序归一)", end="")
if view_split:
    print(f" (Concat 视图拆分 {view_split} 处)", end="")
if box_split:
    print(f" (Split 分组 {box_split} 处)", end="")
for s in skipped:
    print(f"\n   原样保留 {s}", end="")
print(f"; 剩余 Mod {c.get('Mod', 0)}, 节点 {sum(c.values())}")
