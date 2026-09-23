"""按 ONNX 自身声明的输入/输出张量名与形状生成 Pulsar2 配置 (在 pulsar2 容器内执行).

用法: python3 gen_config.py <src.onnx> <dst.json> <标定tar> <标定张数> <核数> <产物名>

配置口径的依据:

  * 运行期输入 U8 / NHWC / RGB + mean=0, std=255 —— 与本库既有前处理等价: 固件本来就是
    BGR→RGB 取通道后乘 1/255 送网络, 所以网络训练在 RGB 上; Pulsar2 **不会**代做
    RGB↔BGR 通道互换, 声明 RGB 并按 RGB 喂才是同一口径, mean/std 取 0/255 就是那次 ÷255.
    NCHW 是 ONNX 自身的布局 (tensor_layout), src_layout=NHWC 由工具链自己插 Transpose.
  * 输出留 FP32: 这些模型的输出把像素坐标 (0..640) 与置信度 (0..1) 放在同一张张量里,
    按张量做 U8 量化会用坐标的量程去压置信度 (动态范围差约 640:1), 置信度会被压成 0.
  * 量化用 MinMax, 标定集是实机采下来的游戏帧 (见 dataset/): 统计来自真实分布.
  * npu_mode=NPU3: 这块 AX-M1 是 3 核卡, 单帧延迟由编译期核数决定, 运行期改不了.
  * compiler.check=0: 官方 YOLO 示例的取值; 精度看 precision_analysis 的输出层余弦相似度.
"""

import json
import os
import sys

import onnx

src, dst, calib, calib_size, npu_mode, out_name = sys.argv[1:7]


def dims(t):
    if not t.HasField("tensor_type") or not t.tensor_type.HasField("shape"):
        return None
    return [d.dim_value if d.HasField("dim_value") else -1 for d in t.tensor_type.shape.dim]


m = onnx.load(src, load_external_data=False)
g = m.graph
ins = [(i.name, dims(i.type)) for i in g.input]
outs = [(o.name, dims(o.type)) for o in g.output]

# 逐模型打一句可核对的画像: 输入形状 / 每张输出的形状与由它推出的类别数
print(f"   输入  {ins[0][0]} {ins[0][1]}")
for name, d in outs:
    note = ""
    if d and len(d) == 3 and d[0] == 1:
        attrs, anchors = min(d[1], d[2]), max(d[1], d[2])
        if anchors == 300 and attrs == 6:
            note = "  (端到端: 300 候选 × [x1,y1,x2,y2,conf,cls])"
        elif attrs >= 5:
            note = f"  ({attrs - 4} 类)"
    print(f"   输出  {name} {d}{note}")
if any(v is not None and -1 in v for _, v in ins):
    print("   ⚠ 输入含动态维 — Pulsar2 需要静态形状")

cfg = {
    "model_type": "ONNX",
    "npu_mode": npu_mode,
    "output_name": out_name,
    "onnx_opt": {"enable_onnxsim": True, "model_check": True},
    "quant": {
        "calibration_method": "MinMax",
        "precision_analysis": True,
        "precision_analysis_method": "EndToEnd",
        "precision_analysis_mode": "NPUBackend",
        "input_configs": [
            {
                "tensor_name": ins[0][0],
                "calibration_dataset": calib,
                "calibration_format": "Image",
                "calibration_size": calib_size,
                "calibration_mean": [0.0, 0.0, 0.0],
                "calibration_std": [255.0, 255.0, 255.0],
            }
        ],
    },
    "input_processors": [
        {
            "tensor_name": ins[0][0],
            "tensor_layout": "NCHW",
            "tensor_format": "RGB",
            "src_layout": "NHWC",
            "src_dtype": "U8",
            "src_format": "RGB",
            "mean": [0.0, 0.0, 0.0],
            "std": [255.0, 255.0, 255.0],
        }
    ],
    "output_processors": [{"tensor_name": n, "output_dtype": "FP32"} for n, _ in outs],
    "compiler": {"check": 0},
}

# 产生输出张量的那些算子提到 U16 —— 这是"一个张量里混着两个量程"的通用止损。
#
# 这些导出把像素坐标(0..W)与置信度(0..1)拼进同一张输出张量, 而一张张量只有一个量化尺度:
# 量程由坐标定, 置信度被分到约 1/255 甚至 0 格。实测(模型B, 320, 通用标定集, NPU3):
#     单张 U8 : 0..1 上 0 档(全 0) —— 置信度被抹平, 板上 0 候选
#     U16     : 0..1 上 165 档(步长 0.00606), 坐标通道也从 204 频升到 255 档
# 而把输出拆成两张张量各拿自己的尺度是 255 档(见 work/exp/split_head.py) —— 但那要按族做
# 图手术; 这条规则零手术、对任何模型成立, 是转换侧默认该带的一层。
#
# 为什么不是 quant.highest_mix_precision(自动混合精度): 实测它在这些图上直接崩
#     (OpXrunException: AxResize), 且与 layer_configs 互斥(工具链会警告 layer configs
#      不生效)。所以走手动的 layer_configs。
#
# 这些算子都是逐元素/拼接/变形一类, 没有 MAC, 16 位几乎不增加延迟; 关掉它可设 NO_LAST_U16=1
# 做 A/B。
ops = [n.name for n in g.node if any(o in {x.name for x in g.output} for o in n.output)]
ops = [o for o in ops if o]
if ops and os.environ.get("NO_LAST_U16") != "1":
    cfg["quant"]["layer_configs"] = [
        {"layer_names": ops, "data_type": "U16", "output_data_type": "U16"}
    ]
    print("   输出算子提 U16: %s" % ", ".join(ops))

with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"   配置  {dst}")
