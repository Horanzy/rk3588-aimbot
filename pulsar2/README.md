# Pulsar2 转换套件（AX‑M1 / AX650N）


> 本文里的「模型A / 模型B / 模型C …」都是**占位名**：具体的游戏名与模型名不上传。
> 所有路径都是相对于本目录的。本目录只带脚本与说明 —— `onnx/`（输入模型）、`axmodel/`
> （产物）、`dataset/`（标定集）、`work/`（每次构建的配置与日志）都在 `.gitignore` 里，
> 换台机器时把它们从别处拷过来即可。
把 ONNX 转成 `.axmodel`，给 RK3588 板上那块 MX650N 用。转换在 **WSL 的 docker** 里做
（镜像 `pulsar2:7.0-patch1`），不在板上、也不在 Windows 侧。

---

## 目录

```
~/pulsar2/
├── onnx/            输入: 放 .onnx
├── axmodel/         产物: .axmodel
├── dataset/         标定集 tar
│   ├── calib_generic.tar      通用池(跨游戏混采, 第一遍用)
│   └── calib_<模型名>.tar     某个游戏专用(第二遍用)
├── work/            每个模型的配置与日志(可删, 重跑会再生成)
├── convert.sh       单模型转换入口  ← 只用这一个
├── gen_config.py    按 ONNX 自己声明的张量名/形状生成配置
├── patch_onnx.py    等价改写(折掉工具链不支持的算子)
└── prep_calib.py    标定集按模型输入尺寸中心裁剪
```

---

## 一条命令

```bash
cd ~/pulsar2

bash convert.sh -m onnx/mymodel.onnx                       # 自动挑标定集(专用集 → 通用池)
bash convert.sh -m onnx/mymodel.onnx -c dataset/calib_模型A.tar
bash convert.sh -m onnx/mymodel.onnx -n NPU1 -s 200        # 核数 / 标定张数
bash convert.sh -m onnx/mymodel.onnx -F                    # 已存在也重转
```

`convert.sh` 里的顺序是四步，每步都能单独看日志：

| 步 | 做什么 |
|---|---|
| `patch_onnx.py` | 等价改写：折掉工具链不支持的算子（`Mod` 等），逐值验证 |
| `prep_calib.py` | 标定集按该模型输入尺寸**中心裁剪**（见下） |
| `gen_config.py` | 现生成配置（按 ONNX 自己声明的张量名/形状，所以任何新模型丢进 `onnx/` 都能转） |
| `pulsar2 build` | 编 `.axmodel` |

产物 `axmodel/<名>.axmodel`；配置 `work/<名>.json`；完整日志 `work/<名>.log`。
`-n` 取 `NPU1/NPU2/NPU3`，只影响单帧延迟（**编译期**定，运行期改不了）。

---

## 标定集：两遍走的约定

| 文件 | 是什么 | 什么时候用 |
|---|---|---|
| `calib_generic.tar` | **通用池**，跨游戏混采（往里加图、重打包即可） | **第一遍**：先把模型跑起来 |
| `calib_<模型名>.tar` | 那个游戏自己采的帧 | **第二遍**：拿第一遍的模型去玩、去截图，攒够再转 |

省略 `-c` 时按 `calib_<模型名>*` → `calib_generic.tar` 的顺序找；两者都没有就直接退出并说明
——**没有标定集就没有激活量程，int8 无从谈起**（这点和 Jetson 的 fp16 不同，那边本来不需要标定）。

三条要点：

1. **不需要人工标注。** 标定只做前向、只统计每层激活的 min/max，不跟真值比 —— `gen_config.py`
   里连填标注的字段都没有。决定质量的是**图的分布**，不是标注。
2. **图要来自运行时那条链。** 最好的来源是**自瞄自己采的帧**（同一块采集卡、同一条裁剪链）。
   别塞极端帧：MinMax 取**绝对极值**，一张白闪/菜单/加载画面就能把某层量程拉宽、摊薄其余帧。
   打包：`cd <采集目录> && tar -cf calib_xxx.tar *.jpg`
3. **必须按模型输入尺寸中心裁一次，不能让工具链缩。** 采集窗口是**相机像素 1:1 的中心裁剪**、
   边长就是当前加载模型的输入边长，所以 640 的截图集**只对 640 的模型**等于运行期输入；
   对 320 的模型必须从中心**裁出** 320（不缩放）。交给工具链去缩会喂进"视野宽一倍、目标小一半"
   的分布，而模型运行期永远看不到那个分布。`prep_calib.py` 自动做这件事，实测产出与"中心裁剪"
   差 **0.20–0.57 灰阶**、与"整图缩放"差 **17–37 灰阶**（确实裁对了）。
   模型输入大于标定图时会告警并不中断（往上裁是假的）。

---

## 导出 ONNX（这一步决定了能不能转）

**工具链支持的算子是有限的**：AX650 的官方支持表一共 94 个算子，
**`TopK` 与 `Mod` 都不在其中**，`NonMaxSuppression` 也没有；`Cast` 的目标类型是
`uint8/int8/uint16/int16/uint32/int32/float32`，**没有 int64**。

所以**凡是把 NMS/端到端头烘进图的导出，都转不了** —— 报错长这样：

```
axnn.yamain.common.error.CodeException: (<ErrorCode.OnnxOptimizationError: 13>,
  AssertionError("check onnx parser e2e failed, ['output0']"))
op name: /model.23/Mod, Mod, {'fmod': 0, ...} get opr failed.
```

这两条都出自端到端头：`TopK` 选候选 → `Mod` 把展平下标还原成网格坐标。它在解析器的一致性
检查里会失败在**坐标那条支路**上（置信度那条反而能过），而 `Mod` 换写法也救不了 —— 因为
`TopK` 本身不支持，下标就是错的。

**YOLO26 / YOLOv10（默认 `end2end: True`）**：不用重训，导出时关掉那一支即可：

```python
from ultralytics import YOLO
m = YOLO("runs/detect/train/weights/xxx.pt")
m.export(format="onnx", imgsz=640, half=False, simplify=False,
         opset=17, dynamic=False, end2end=False)      # ← 关键
```

依据（ultralytics 8.4.x `engine/exporter.py`）：

```python
if hasattr(model, "end2end"):
    if self.args.end2end is not None:
        model.end2end = self.args.end2end        # end2end=False 时走 one2many(网格)分支
```

exporter 自己也有这句注释：*"Disable end2end branch for certain export formats as they does
not support **topk**"* —— 同一个原因。

**导出后先自检一眼，比什么都快**：

| 看什么 | 对的样子 | 不对的样子 |
|---|---|---|
| 输出形状 | `[1, 4+类数, N]`（640 / 2 类 → `[1, 6, 8400]`） | `[1, 300, 6]`（端到端，转不了） |
| 图里的算子 | `TopK` / `Mod` / `NonMaxSuppression` 计数 = 0 | 有 |
| opset | 17 | 更高版本可能带工具链不认的算子形态 |

`[1, 4+类数, N]` 正好是自瞄主机 `src/core/detect.h` 里 rank‑3 的 **YOLOv8‑11 布局**
（box 已在图里 `dist2bbox` 解好、分数已 sigmoid），所以**主机侧不用改代码**，
**也不用给 `-n`**（类数由属性数自解；`-n` 只在未折叠的 rank‑4 DFL 头上才必须给）。
顺带一句：YOLO26 的 `reg_max: 1` 让 DFL 退化成 `Identity`，所以它的 box 部分本来就是 4 通道，
不会落进"未折叠 DFL 头"那一支。

---

## 默认带的一层：输出算子提 U16

`gen_config.py` 会把**产生输出张量的那些算子**自动配成 `U16`（`quant.layer_configs`）。
这是针对"一个张量里混着两个量程"的唯一处理，不是可选项：

这些导出把**像素坐标（0..W）与置信度（0..1）拼进同一张输出张量**，而一张张量只有一个量化尺度
—— 量程由坐标定，置信度被分到约 1/255 甚至 0 格。实测（模型B, 320, 通用池, NPU3）：

| 配置 | 分数通道（0..1 上几档） | 后果 |
|---|---|---|
| 单张 U8（不提 U16） | **0 档（全 0）** | 板上 **0 候选** |
| **U16（现在的默认）** | **165 档**（步长 0.00606） | 可用；坐标通道也从 204 档升到 255 档 |

- **几乎不花钱**：被提上去的是 `Concat`/`Sigmoid`/`Mul` 这类**没有乘加的算子**，不吃算力。
  实测同一模型 median **1.247 → 1.230 ms**（NPU3），差异在噪声里。
- 每张张量的位数可以直接核：`quant/quant_axmodel.json` 里 `tensor_configs` 的 `bit_width`
  —— 只有输出那一段是 16，其余（含全部权重）仍是 8。模型B 实测 3 张 16 位 / 其余 8 位；
  CS 实测 13 张（整个输出尾）16 位 / 580 张 8 位。
  构建中间产物默认在成功后删掉，要留着核就加 `KEEP_QUANT=1`。
- 想 A/B 关掉它：`NO_LAST_U16=1`（传给 `gen_config.py`）。


---

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `opset` / `Unsupported` / `not support` 报错 | 该模型算子或 opset 超出工具链支持。**按 opset 17 重新导出**（模型A 原来 opset 23 就是这样解决的） |
| 图上带 `Mod` | 端到端解码里 `i%C` 那一步折不掉 → `patch_onnx.py` 会做等价改写；失败则退回原图 |
| `These op contains very big range, maybe cause low precision` | **提示，不是判据**。它说的就是上面那个混量程张量；提 U16 之后这条**仍会打印**（那张张量本身量程确实大），只有"拆成两张"能让它消失 |
| 设了 `quant.highest_mix_precision: true` 想自动混合精度 | **别用**：实测在这些图上直接崩（`OpXrunException: AxResize`），而且它与 `layer_configs` 互斥（工具链会警告 layer configs 不生效） |
| 容器内没有 PIL | 标定集无法裁剪，会告警并退回未裁剪的图（视野不对齐）。镜像里装 `pillow` 即可 |
| 找不到标定集 | 见上节；或用 `-c` 指定，或 `CALIB=` 环境变量 |

---

## 怎么验收（重要）

**别只看工具链的 cosine。** 实测那个"分数通道全 0、板上 0 候选"的坏模型，输出层
cosine 是 **0.99951** —— 按常见的 `>0.98` 口径是"合格"。原因是合并张量里坐标通道的范数远大于
分数通道，**余弦被坐标主导**；同一行的 MSE 是 9.3，而中间层只有 0.1，**MSE 才看得见**。

三层验收，从便宜到贵：

1. **构建日志**：`grep -i "very big range\|cosine\|similarity" work/<名>.log` —— 提示与余弦，只能当参考。
2. **按通道角色看误差**：读 Pulsar2 自己的 debug 转储（`<产物目录>/quant/debug/numpy/{float,xrun}/`，
   是 FP32 参考 vs 量化后跑的同一批张量），把坐标通道与分数通道**分开**算档数/MAE。这条不用上板。
3. **上板实测**：`sudo LD_LIBRARY_PATH=/usr/lib/axcl /usr/bin/axcl/axcl_run_model -m <模型> -r 200 -w 20`
   看延迟与能否加载；再放进自瞄看真实检出。

---

## 环境

- 镜像已 `docker load`，`convert.sh` 自动取版本最高的那个（可用 `IMG=` 指定）。
- 板上运行时是 **AXCL 3.6.5**，本套件用的是 **Pulsar2 7.0-patch1**（比它新几代）。若模型在板上
  加载失败，把 Pulsar2 降到同期版本重转即可，配置不用改（`IMG=` 指过去）。
- 该镜像的 ENTRYPOINT 就是 `bash`，所以脚本里必须显式 `--entrypoint /bin/bash`；直接写
  `IMG bash -c ...` 会变成 bash 去读一个叫 bash 的文件，报 `cannot execute binary file`。
