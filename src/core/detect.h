// ============================================================================
//  detect.h — 检测输出的可移植部分: 张量形状/元素类型的最小表述, 输出布局
//    解析, 四种输出口径 (端到端 / YOLOv5 / YOLOv8-11 / 未折叠 DFL 头) 的候选
//    解码与 NMS。
//    取帧与推理后端 (HDMI IN 取帧 + RGA 裁剪、AXCL NPU 推理或任何别的后端) 各自
//    成模块, 本文件只消费它们给出的**裸 float 张量**, 不包含任何平台相关实现, 也
//    不依赖任何后端头文件。
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

// 导出网格头三个特征层的步长: 8/16/32 (输入边长的 1/8、1/16、1/32)。三个候选数
//   实测值 (320→2100、416→3549、640→8400) 与 (S/8)²+(S/16)²+(S/32)² 逐个吻合,
//   10647 = 3×3549 亦然 —— 这就是"锚点数由模型几何给出"的那条物理关系。
constexpr int YOLO_LEVEL_STRIDE0 = 8;
constexpr int YOLO_LEVELS = 3;

// 网格头 (YOLOv5 / YOLOv8-11) 在一个正方输入下的候选数 G; 边长未知或不能被全部
//   步长整除时返回 0 (几何判不了, 由 head_kind 的退路口径接手)。
inline int anchor_count(int input_side) {
    if (input_side<=0) return 0;
    int g=0, s=YOLO_LEVEL_STRIDE0;
    for (int i=0;i<YOLO_LEVELS;++i,s*=2) {
        if (input_side%s) return 0;
        const int n=input_side/s; g+=n*n;
    }
    return g;
}

// 输出布局:
//   rank 2/3 — [num,attrs] 或 [1,attrs,num] / [1,num,attrs]; attrs_first 表示属性是
//      批次后第一个维度 (即内存里每个候选的属性连续); dfl=false。
//   rank 4   — [1,gh,gw,attrs], 未折叠 DFL 头 (AXERA 公开 YOLO11 模型的形状):
//      attrs = 4·reg_max + 类数, 每格一组 (gh,gw), 属性在最内层; dfl=true, stride =
//      输入边长/gh (框分布的单位), num = gh·gw。
struct OutputLayout {
    int attrs=0,num=0; bool attrs_first=true;
    bool dfl=false;                 // rank 4 (未折叠 DFL 头)
    int gw=0,gh=0,stride=0;         // 仅 dfl: 网格列数/行数与步长 (输入像素)
};

// 解析一个张量的布局: rank 2/3/4 各自成立, 其余返回 false。input_side = 模型方形
//   输入边长 (只被 rank 4 的步长用到; 非方形或未知时给 0, 则 dfl 布局的 stride=0,
//   解码遇到 stride==0 只能跳过该缓冲)。
inline bool parseOutputLayout(const Dims& d, OutputLayout& l, int input_side) {
    if (d.nbDims==3 && d.d[0]==1) { int a=(int)d.d[1],b=(int)d.d[2];
        l.attrs=std::min(a,b); l.num=std::max(a,b); l.attrs_first=(a<=b); return true; }
    if (d.nbDims==2) { int a=(int)d.d[0],b=(int)d.d[1];
        l.attrs=std::min(a,b); l.num=std::max(a,b); l.attrs_first=(a<=b); return true; }
    if (d.nbDims==4 && d.d[0]==1) {
        const int64_t h=d.d[1], w=d.d[2], c=d.d[3];
        // 只接受正方网格 (gh==gw): 网格头导出的三个特征层都是正方, 而 NCHW 的原始
        //   头 ([1,C,H,W]) 在这一判据下自动落空 (C≠H), 不会与 DFL 头混淆。
        if (h<=0||w<=0||h!=w||c<5) return false;
        if (input_side>0 && (input_side%(int)h)) return false;   // 步长必须是整数
        l.dfl=true; l.gh=(int)h; l.gw=(int)w; l.attrs=(int)c; l.num=(int)(h*w);
        l.attrs_first=true;
        l.stride = input_side>0 ? input_side/(int)h : 0;
        return true;
    }
    return false;
}
inline float outVal(const float* o, const OutputLayout& l, int a, int i) {
    return l.attrs_first ? o[a*l.num+i] : o[i*l.attrs+a];
}
struct Detection { float cx,cy,w,h,conf; int class_id; };

// 解码口径 — 由布局与模型的**物理几何**判定, 判定规则见 detect.cpp
enum class HeadKind { kEnd2End, kYoloV5, kYoloV8, kYoloDfl };
HeadKind head_kind(const OutputLayout& l, int num_classes, int input_side);
const char* head_kind_name(HeadKind k);

// 逐候选解码为框 (**不含** NMS, **不含**裁剪平移 — 都由调用方在拿到同一批框之后做,
//   与逐帧循环里的先后顺序一致): 第 i 个缓冲按第 i 个布局解, conf_thr 以下与
//   want_cls 以外的候选丢弃 (want_cls < 0 = 不筛类别)。单输出后端传一个缓冲 + 一个
//   布局即可。input_side 只被布局判定与 DFL 头的步长用到。
std::vector<Detection> decode_outputs(const std::vector<const float*>& bufs,
                                      const std::vector<OutputLayout>& layouts,
                                      int num_classes, float conf_thr, int want_cls,
                                      int input_side);

std::vector<Detection> nms(const std::vector<Detection>& dets, float thr);
