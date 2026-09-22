// ============================================================================
//  detect.cpp — detect.h 的实现: 解码口径的判定, 三支候选解码与 NMS。
//    解码只做算术, 不含任何后端调用; 三支的取属性路径都用 outVal, 因此属性在
//    前/在后两种导出布局共用同一段代码。
// ============================================================================

#include "core/detect.h"

#include <cmath>
#include <map>

// 解码口径由布局判定:
//   端到端 — 6 个属性 ([x1,y1,x2,y2,conf,class]) 且属性在后, 候选数为 300 (固定配额);
//   YOLOv5 — 属性里含 objectness, 故属性数 = 5 + 类数 (conf = objectness × 类分数);
//   YOLOv8/11 — 属性只有框 + **转置**的类分数, 故属性数 = 4 + 类数。
//   YOLOv5 与 YOLOv8/11 的候选数相同 (同一个网格), 故两者的区分落在属性数上:
//   "含 objectness" 这一物理差别正好是 +1。此判定只用布局与配置的类数, 不再依赖
//   模型输入边长导出的网格数。
HeadKind head_kind(const OutputLayout& l, int num_classes) {
    if (l.attrs==6 && !l.attrs_first && l.num==300) return HeadKind::kEnd2End;
    if (l.attrs==num_classes+5) return HeadKind::kYoloV5;
    return HeadKind::kYoloV8;
}

std::vector<Detection> decode_outputs(const std::vector<const float*>& bufs,
                                      const std::vector<OutputLayout>& layouts,
                                      int num_classes, float conf_thr, int want_cls) {
    std::vector<Detection> raw;
    const size_t nbuf=std::min(bufs.size(),layouts.size());
    for (size_t b=0;b<nbuf;++b) {
        const float* od=bufs[b];
        const OutputLayout& ol=layouts[b];
        const HeadKind kind=head_kind(ol,num_classes);
        // 类数仍由本缓冲的属性数导出 (与类数配置无关): 多输出时每个头各解各的
        const int nclass=ol.attrs-(kind==HeadKind::kYoloV5?5:4);
        for (int i=0;i<ol.num;++i) {
            Detection d;
            if (kind==HeadKind::kEnd2End) {
                d.conf=outVal(od,ol,4,i);
                d.class_id=(int)std::round(outVal(od,ol,5,i));
                if (d.conf<conf_thr||d.class_id!=want_cls) continue;
                float x1=outVal(od,ol,0,i),y1=outVal(od,ol,1,i),
                      x2=outVal(od,ol,2,i),y2=outVal(od,ol,3,i);
                d.cx=(x1+x2)*.5f;d.cy=(y1+y2)*.5f;d.w=x2-x1;d.h=y2-y1;
            } else if (kind==HeadKind::kYoloV5) {
                float obj=outVal(od,ol,4,i),mc=0;int ci=-1;
                for(int c=0;c<nclass;++c){float s=obj*outVal(od,ol,5+c,i);if(s>mc){mc=s;ci=c;}}
                if(mc<conf_thr||ci!=want_cls)continue;
                d.cx=outVal(od,ol,0,i);d.cy=outVal(od,ol,1,i);
                d.w=outVal(od,ol,2,i);d.h=outVal(od,ol,3,i);d.conf=mc;d.class_id=ci;
            } else {
                float mc=0;int ci=-1;
                for(int c=0;c<nclass;++c){float s=outVal(od,ol,4+c,i);if(s>mc){mc=s;ci=c;}}
                if(mc<conf_thr||ci!=want_cls)continue;
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
