// ============================================================================
//  detect.h — 检测输出的可移植部分: 张量形状/元素类型的最小表述, 输出布局
//    解析, 三种输出口径 (端到端 / YOLOv5 / YOLOv8-11) 的候选解码与 NMS。
//    取帧与推理后端 (MPP/RGA 取帧、AXCL NPU 推理或任何别的后端) 各自成模块,
//    本文件只消费它们给出的**裸 float 张量**, 不包含任何平台相关实现, 也不
//    依赖任何后端头文件。
// ============================================================================

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// 张量形状与元素类型: 后端在调用前把自己的类型转换成这两者 (元素字节数语义不变)
struct Dims { int nbDims=0; int64_t d[8]={}; };
enum class DataType { kFLOAT, kHALF, kINT8, kINT32, kBOOL };

inline size_t elemSize(DataType dt) {
    switch (dt) {
        case DataType::kFLOAT: return 4; case DataType::kHALF: return 2;
        case DataType::kINT8: return 1;  case DataType::kINT32: return 4;
        case DataType::kBOOL: return 1;  default: return 0;
    }
}
inline size_t volume(const Dims& d) {
    size_t v=1; for (int i=0;i<d.nbDims;++i) v*=d.d[i]; return v;
}
inline bool hasDynamicDim(const Dims& d) {
    for (int i=0;i<d.nbDims;++i) if (d.d[i]<0) return true; return false;
}

struct OutputLayout { int attrs=0,num=0; bool attrs_first=true; };
inline bool parseOutputLayout(const Dims& d, OutputLayout& l) {
    if (d.nbDims==3 && d.d[0]==1) { int a=(int)d.d[1],b=(int)d.d[2];
        l.attrs=std::min(a,b); l.num=std::max(a,b); l.attrs_first=(a<=b); return true; }
    if (d.nbDims==2) { int a=(int)d.d[0],b=(int)d.d[1];
        l.attrs=std::min(a,b); l.num=std::max(a,b); l.attrs_first=(a<=b); return true; }
    return false;
}
inline float outVal(const float* o, const OutputLayout& l, int a, int i) {
    return l.attrs_first ? o[a*l.num+i] : o[i*l.attrs+a];
}
struct Detection { float cx,cy,w,h,conf; int class_id; };

// 解码口径 — 由布局与配置的类数判定, 判定规则见 detect.cpp
enum class HeadKind { kEnd2End, kYoloV5, kYoloV8 };
HeadKind head_kind(const OutputLayout& l, int num_classes);

// 逐候选解码为框 (**不含** NMS, **不含**裁剪平移 — 都由调用方在拿到同一批框之后做,
//   与逐帧循环里的先后顺序一致): 第 i 个缓冲按第 i 个布局解, conf_thr 以下与
//   want_cls 以外的候选丢弃。单输出后端传一个缓冲 + 一个布局即可。
std::vector<Detection> decode_outputs(const std::vector<const float*>& bufs,
                                      const std::vector<OutputLayout>& layouts,
                                      int num_classes, float conf_thr, int want_cls);

std::vector<Detection> nms(const std::vector<Detection>& dets, float thr);
