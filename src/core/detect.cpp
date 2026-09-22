// ============================================================================
//  detect.cpp — detect.h 的实现: 解码口径的判定, 四支候选解码与 NMS。
//    解码只做算术, 不含任何后端调用; 三支网格/提案口径的取属性路径都用 outVal,
//    因此属性在前/在后两种导出布局共用同一段代码; DFL 头的属性在最内层 (NHWC),
//    直接按下标取。
// ============================================================================

#include "core/detect.h"

#include <cmath>
#include <map>

// 解码口径的判定 — 只用**物理结构**, 不用配置的类数:
//   DFL (rank 4) — 布局本身是 [1,网格,网格,4·reg_max+类数] (未折叠的分布头), 解析期
//     即定, 无歧义 —— 判定发生在 head_kind 的第一行。
//   网格头 — 候选数由输入边长经三个特征层步长 (8/16/32) 给出: G = Σ(S/s)²。YOLOv5 在
//     每个网格点上挂 3 个锚 (候选数 3·G), YOLOv8-11 是 anchor-free (候选数 G)。实测
//     三档输入给出的 2100 (320)、1344 (256)、3549 (416)、8400 (640) 与 10647 = 3×3549
//     全部由这条式子复现, 故"候选数与 G 的关系"就是这两族的物理判别式: num==3G 判
//     v5 (每格 3 锚), num==G 判 v8-11 (anchor-free) —— 两族的差别在候选数上就是一个
//     3 倍。
//   端到端 — 候选表每行 6 个属性 (x1,y1,x2,y2,conf,class) 且**属性在后**; 候选数是
//     导出时的固定配额 (实测 300), 与任何网格数都不等。判据里**不含那个配额**: 它是
//     导出时的选择而不是物理量, 认定靠"属性在后且恰 6 个属性, 且候选数不是 G 或 3G"。
//   几何判不了时 (输入边长未知/非方形/不能被步长整除, 或候选数与 G、3·G 都不等) 的
//   退路口径, 按此顺序: 属性在后且恰 6 个属性 → 端到端提案表; 否则用"含 objectness"这
//   一物理差别 (v5 的属性数比 v8-11 恰多一个) 与配置的类数比。这一支只在几何信息缺席
//   时生效 (例如只有张量、没有模型输入边长可用的离线解码), 有几何时几何优先 —— 部署
//   路径上几何永远在 (运行时自述输入形状), 故退路口径的取舍不动部署语义: 属性数 6 与
//   "单类 v5"在几何缺席时同形, 本层把这一形判给端到端表 (该族的候选配额与网格数无关,
//   而单类 v5 的候选数恰是 3G, 有几何时先被上面那条判走)。
HeadKind head_kind(const OutputLayout& l, int num_classes, int input_side) {
    if (l.dfl) return HeadKind::kYoloDfl;
    const int G=anchor_count(input_side);
    if (G>0) {
        if (l.num==3*G) return HeadKind::kYoloV5;
        if (l.num==G)   return HeadKind::kYoloV8;
    }
    if (l.attrs==6 && !l.attrs_first) return HeadKind::kEnd2End;
    if (l.attrs==num_classes+5) return HeadKind::kYoloV5;
    return HeadKind::kYoloV8;
}

const char* head_kind_name(HeadKind k) {
    switch (k) {
        case HeadKind::kEnd2End: return "端到端提案表";
        case HeadKind::kYoloV5:  return "YOLOv5 (含 objectness)";
        case HeadKind::kYoloV8:  return "YOLOv8-11 (转置类分数)";
        case HeadKind::kYoloDfl: return "DFL 分布头 (未折叠)";
    }
    return "?";
}

std::vector<Detection> decode_outputs(const std::vector<const float*>& bufs,
                                      const std::vector<OutputLayout>& layouts,
                                      int num_classes, float conf_thr, int want_cls,
                                      int input_side) {
    std::vector<Detection> raw;
    const size_t nbuf=std::min(bufs.size(),layouts.size());
    for (size_t b=0;b<nbuf;++b) {
        const float* od=bufs[b];
        const OutputLayout& ol=layouts[b];
        const HeadKind kind=head_kind(ol,num_classes,input_side);
        if (kind==HeadKind::kYoloDfl) {
            // 类数配置在本支只用来定分布长度: attrs = 4·reg_max + 类数 → reg_max。它
            //   不参与口径判定 (rank 4 已判定)。契约不成立 (类数未给/非正, 步长未知, 或
            //   剩余通道不是 4 的倍数) 时整块跳过 —— 不猜分布长度, 宁可不解这一块。
            if (num_classes<1 || ol.stride<=0) continue;
            const int nb=ol.attrs-num_classes;
            if (nb<=0 || nb%4) continue;
            const int reg=nb/4;
            for (int i=0;i<ol.num;++i) {
                const float* p=od+(size_t)i*ol.attrs;      // 属性在最内层 (NHWC)
                // 类分数: 该头的导出是**未过激活的 logits**, 故这里过 sigmoid (等价于
                //   先取最大再 sigmoid, 取最大与 sigmoid 都是单调的)
                float mc=0; int ci=-1;
                for (int c=0;c<num_classes;++c) {
                    const float s=1.f/(1.f+std::exp(-p[4*reg+c]));
                    if (s>mc) { mc=s; ci=c; }
                }
                if (mc<conf_thr || (want_cls>=0 && ci!=want_cls)) continue;
                // 框: 每条边一组 reg_max 个分布值, 期望值 × 步长 = 该边到格心的距离。
                //   步长由输入边长与网格数给出 (解析期算好), 故框落在**模型输入像素**
                //   坐标里, 与另三支同一约定 (裁剪平移仍归调用方)。
                float ltrb[4];
                for (int k=0;k<4;++k) {
                    const float* d=p+(size_t)k*reg;
                    float mx=d[0];
                    for (int j=1;j<reg;++j) mx=std::max(mx,d[j]);
                    float sum=0, acc=0;
                    for (int j=0;j<reg;++j) { const float e=std::exp(d[j]-mx); sum+=e; acc+=j*e; }
                    ltrb[k]=(acc/sum)*(float)ol.stride;
                }
                const float gx=((float)(i%ol.gw)+0.5f)*(float)ol.stride;
                const float gy=((float)(i/ol.gw)+0.5f)*(float)ol.stride;
                const float x1=gx-ltrb[0], y1=gy-ltrb[1], x2=gx+ltrb[2], y2=gy+ltrb[3];
                Detection d; d.cx=(x1+x2)*.5f; d.cy=(y1+y2)*.5f; d.w=x2-x1; d.h=y2-y1;
                d.conf=mc; d.class_id=ci;
                raw.push_back(d);
            }
            continue;
        }
        // 类数仍由本缓冲的属性数导出 (与类数配置无关): 多输出时每个头各解各的
        const int nclass=ol.attrs-(kind==HeadKind::kYoloV5?5:4);
        for (int i=0;i<ol.num;++i) {
            Detection d;
            if (kind==HeadKind::kEnd2End) {
                d.conf=outVal(od,ol,4,i);
                d.class_id=(int)std::round(outVal(od,ol,5,i));
                if (d.conf<conf_thr||(want_cls>=0&&d.class_id!=want_cls)) continue;
                float x1=outVal(od,ol,0,i),y1=outVal(od,ol,1,i),
                      x2=outVal(od,ol,2,i),y2=outVal(od,ol,3,i);
                d.cx=(x1+x2)*.5f;d.cy=(y1+y2)*.5f;d.w=x2-x1;d.h=y2-y1;
            } else if (kind==HeadKind::kYoloV5) {
                float obj=outVal(od,ol,4,i),mc=0;int ci=-1;
                for(int c=0;c<nclass;++c){float s=obj*outVal(od,ol,5+c,i);if(s>mc){mc=s;ci=c;}}
                if(mc<conf_thr||(want_cls>=0&&ci!=want_cls))continue;
                d.cx=outVal(od,ol,0,i);d.cy=outVal(od,ol,1,i);
                d.w=outVal(od,ol,2,i);d.h=outVal(od,ol,3,i);d.conf=mc;d.class_id=ci;
            } else {
                float mc=0;int ci=-1;
                for(int c=0;c<nclass;++c){float s=outVal(od,ol,4+c,i);if(s>mc){mc=s;ci=c;}}
                if(mc<conf_thr||(want_cls>=0&&ci!=want_cls))continue;
                d.cx=outVal(od,ol,0,i);d.cy=outVal(od,ol,1,i);
                d.w=outVal(od,ol,2,i);d.h=outVal(od,ol,3,i);d.conf=mc;d.class_id=ci;
            }
            raw.push_back(d);
        }
    }
    return raw;
}

static float iou(const Detection& a, const Detection& b) {
    float ax1=a.cx-a.w*.5f,ay1=a.cy-a.h*.5f,ax2=a.cx+a.w*.5f,ay2=a.cy+a.h*.5f;
    float bx1=b.cx-b.w*.5f,by1=b.cy-b.h*.5f,bx2=b.cx+b.w*.5f,by2=b.cy+b.h*.5f;
    float ix1=std::max(ax1,bx1),iy1=std::max(ay1,by1),ix2=std::min(ax2,bx2),iy2=std::min(ay2,by2);
    if (ix2<=ix1||iy2<=iy1) return 0;
    return (ix2-ix1)*(iy2-iy1)/(a.w*a.h+b.w*b.h-(ix2-ix1)*(iy2-iy1));
}
std::vector<Detection> nms(const std::vector<Detection>& dets, float thr) {
    std::vector<Detection> out;
    std::map<int,std::vector<Detection>> per;
    for (auto& d:dets) per[d.class_id].push_back(d);
    for (auto& [cls,ds]:per) {
        std::sort(ds.begin(),ds.end(),[](auto&a,auto&b){return a.conf>b.conf;});
        std::vector<bool> sup(ds.size(),false);
        for (size_t i=0;i<ds.size();++i) { if (sup[i]) continue; out.push_back(ds[i]);
            for (size_t j=i+1;j<ds.size();++j) if (!sup[j]&&iou(ds[i],ds[j])>thr) sup[j]=true; } }
    return out;
}
