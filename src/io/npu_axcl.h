// ============================================================================
//  npu_axcl.h — AXCL (AX650N 加速卡) 上的推理会话与一次推理 tick。
//
//  为什么是一条同步的"H2D → execute → D2H"直路: 同一 context 上先调过
//    axclrtEngineExecuteAsync 再调同步的 axclrtEngineExecute 会段错误 (RC=139) —— 两条
//    路不能共存, 所以本层只走同步路径, 一次 tick 就是三段之和; 这也是延迟预算的算法
//    (每段的百分位由 model_probe 给, 见下)。
//
//  输入契约: **U8 / NHWC / RGB, [1,S,S,3]**。转换后的模型自带
//    AxDequantizeLinear → AxNormalize(mean=0,std=255) → AxTranspose(NHWC→NCHW),
//    主机侧因此喂**原始 RGB 字节** (无需任何归一化/换轴)。实测 engine/ 下九个自转引擎与
//    HuggingFace 公开模型的输入逐个都是这个形态 (dtype=UINT8, dims=[1,S,S,3],
//    S∈{256,320,416,640}), 契约由本层在 open 时**读运行时自述**校验, 不是假设。
//    同一批实测确认通道序为 RGB: 把 model_probe 落盘的输入张量按 RGB/255 与原 ONNX 的
//    输入口径逐元素对齐, 输出余弦相似度 0.9989 (RGB) 对 0.9950 (BGR)。
//    注意运行时对输入的 layout 字段报的是 NCHW(=1) 而 dims 是 NHWC —— 两者对不上,
//    故判据只取 dims 与 dtype, layout 字段原值只记进日志 (见 NpuTensor::raw_layout)。
//
//  输出侧只搬**解码器能消费的那几个**: 有的模型会多声明用不上的输出 (未经 concat 的
//    原始头等), 本层对每个输出试着解出布局 (rank 2/3 的网格/提案表, rank 4 的未折叠
//    DFL 头), 解析得到的才分配主机暂存并逐帧 D2H, 其余只分配设备缓冲 (execute 要求
//    每个输出都有缓冲) 而**一个字节都不过 PCIe**, 跳过的原因记在 NpuTensor::note 里。
//    实测: 九个自转引擎各 1 个输出 (全部保留 —— 上游 ONNX 声明的原始头在转换期就被丢掉
//    了, 例如 R6 的 ONNX 有 4 个输出而转换产物只有 1 个), yolo11s 是 3 个 DFL 层 (全部
//    保留)。跳过的分支因此还没有真实模型走到, 只有单测覆盖。
//
//  输出张量的数值性质 (转换侧产物, 本层不改语义, 只报出来): 自转模型的输出把**像素坐标
//    (0..几百) 与置信度 (0..1) 装进同一张量**, 而设备侧那张张量是按一个量程量化的 ——
//    实测它的最小非零相邻差 (即码长) 就成了置信度的分辨率: apex 1.4915、codwz 1.6453、
//    Z320 1.3031、Dawan 1.2502、R6 3.4864 (坐标量程 [0, 256×码长] 的 8 位码), 而这几个
//    模型实测的**类分数通道最大值恰好等于一个码长** —— 置信度只剩"0 或 1 个码"这一位;
//    R6 更极端: obj/类通道 (量程 0..1) 整条落到码 0, 模型实测 0 候选, 而同帧、同一输入
//    张量喂原 ONNX 有 7765 条候选的类分数 >0.5。HuggingFace 公开模型不受这条影响: 它的
//    输出张量只装 DFL 分布与 logits (量程 ≤16), 实测码长 0.21, 分辨率够用。转换侧的修法
//    是把头拆成两张输出张量 (坐标一张、分数一张), 各自拿到自己的量程, 分数那张的码长就
//    回到 1/256 —— 在那之前, 本层照实取回 (码值原样, 不做任何补偿), 由 model_probe 的
//    "输出张量实测"段把码长与评分通道量程打进验收输出。
//
//  缓冲一次分配、逐帧复用: 每帧的分配/释放会把延迟抖动带进控制环, 而 PCIe 上真正的
//    成本是"每次传输的固定开销", 不是分配本身。传输尺寸扫描 (model_probe
//    --xfer-sweep, 每尺寸 200 次, 单位 µs) 的最小二乘拟合: H2D 每次调用 105–138 +
//    1854–1865×MB (562–566 MB/s), D2H 74–86 + 1494–1508×MB (695–702 MB/s), 残差最大
//    5.9–7.2% (多次运行间的差; 最远点在 1MB —— 小尺寸离这条直线最远, 故预算用带宽段
//    斜率加各尺寸自己的实测值, 不拿直线外推)。输入先打包进主机暂存再由它做 H2D 源: 采集侧的帧是 dma-buf 的 CPU
//    映射, 它的缓存一致性由采集层负责 (见 io/rga_pp.h), 让 H2D 的源永远是普通主机内存,
//    这条路径就不依赖别人有没有做失效化。
//
//  tick 的分段耗时 (µs) 由调用方按需取 (NpuTick): h2d / exec / d2h =
//    三段硬件路径, decode = core/detect 的解码。固件按固定间隔打日志, 探针取百分位。
// ============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/detect.h"

// 一个输入或输出张量: 名字/形状/元素类型/字节数 + 它在解码契约里的角色
struct NpuTensor {
    std::string name;
    Dims        dims{};
    DataType    dtype = DataType::kFLOAT;
    size_t      bytes = 0;
    int         raw_dtype = -1;    // 后端自述的元素类型原值 (日志用: 4=U8, 13=FP16, 15=FP32)
    int         raw_layout = -1;   // 后端自述的 layout 字段原值 (只作日志; 判据是 dims)
    bool        input = false;

    // 输出专用
    bool         layout_ok = false;  // 布局解得出来 (rank 2/3/4)
    bool         kept = false;       // 逐帧 D2H 并喂给解码器的输出
    OutputLayout layout{};
    std::string  note;               // 跳过该输出时的原因 (原样进日志)
};

// 一次 tick 的分段耗时 (µs)
//   逐输出 D2H 是一个定长小数组而不是容器: 一次 tick 里不做任何分配 (分配会把延迟抖动
//   带进控制环), 本平台的模型最多 3 个输出, 故上限取 16 —— 超过上限的输出仍计入
//   d2h_us 总和, 只是不进逐张量表 (表只给诊断用, 预算用的是总和)。
constexpr int NPU_MAX_OUT = 16;
struct NpuTick {
    double h2d_us = 0, exec_us = 0, d2h_us = 0, decode_us = 0;
    double d2h_each_us[NPU_MAX_OUT] = {};   // 逐保留输出的 D2H (下标 = 保留顺序)
    int    d2h_n = 0;                       // 上面有效的前多少项
    size_t dets = 0;       // 解码出的候选数 (未过 NMS)
    bool   ok = true;      // 三段任一步失败即 false (该 tick 不产生检测)
    double total_us() const { return h2d_us + exec_us + d2h_us + decode_us; }
};

// 进程级设备初始化 (幂等): axclInit(0) → axclrtGetDeviceList → SetDevice →
//   axclrtEngineInit(AXCL_VNPU_DISABLE)。/dev/axcl_host 是 root-only, 故调用它的程序
//   必须 root 运行。
bool npu_init(std::string* err = nullptr);
// 卡自述名 (AX650N); 未初始化返回空串
const char* npu_soc_name();
// 后端自述字段的人读化: 元素类型 (axclrtEngineDataType) 与 layout
//   (axclrtEngineDataLayout) 的枚举名, 只作日志 —— 契约判据是 dims/dtype 本身。
const char* npu_dtype_str(int raw_dtype);
const char* npu_layout_str(int raw_layout);

class NpuSession {
public:
    NpuSession() = default;
    ~NpuSession();
    NpuSession(const NpuSession&) = delete;
    NpuSession& operator=(const NpuSession&) = delete;

    // 加载模型并建立会话 (分配全部设备/主机缓冲)。失败时写 err 并返回 false。
    bool open(const std::string& model_path, std::string* err = nullptr);
    void close();
    bool ready() const { return ready_; }

    const std::vector<NpuTensor>& inputs() const { return in_; }
    const std::vector<NpuTensor>& outputs() const { return out_; }
    const std::string& model_path() const { return path_; }
    int  input_side() const { return input_side_; }   // 方形输入边长; 非方形 = 0
    size_t kept_outputs() const;
    size_t in_bytes() const { return in_bytes_; }     // 每 tick 的 H2D 字节数
    size_t out_bytes() const { return out_bytes_; }   // 每 tick 的 D2H 字节数 (只算保留的)

    // 保留输出的主机暂存 (最近一次 run 的内容): 原始字节, 供逐字节比对落盘;
    //   下标是 outputs() 里的位置, 非保留输出返回 nullptr。
    const void* output_host(size_t idx) const;

    // 一次推理 tick: H2D (直读 rgb_hwc, 无中间拷贝) → execute → 逐保留输出 D2H → 解码 (core/detect)。
    //   返回**候选框** (不含 NMS, 不含裁剪平移 —— 与 detect.h 的约定一致, 归调用方)。
    //   rgb_hwc = 边长 input_side() 的 RGB 字节流 (input_side()²×3 字节), 本函数**直接读它**
//   (它就是 RGA 目的缓冲的映射), 开头不再有中间打包拷贝。
    std::vector<Detection> run(const uint8_t* rgb_hwc, int num_classes, float conf_thr,
                               int want_cls, NpuTick* tick = nullptr);

private:
    bool alloc_buffers(std::string* err);
    void free_buffers();

    std::string path_;
    std::vector<NpuTensor> in_, out_;
    std::vector<void*> in_dev_, out_dev_;       // 设备缓冲 (全部输入/输出)
    std::vector<void*> out_host_;               // 主机暂存 (保留的输出)
    std::vector<int>   kept_;                   // 保留输出在 out_ 里的下标
    std::vector<const float*> keep_ptrs_;       // 保留输出的主机暂存 (喂给解码器)
    std::vector<OutputLayout> keep_layouts_;    // 与 keep_ptrs_ 一一对应的布局
    int input_side_ = 0;
    size_t in_bytes_ = 0, out_bytes_ = 0;
    uint64_t model_id_ = 0;
    uint64_t ctx_ = 0;
    void* io_ = nullptr;      // axclrtEngineIO (不透明句柄, 头文件不依赖 axcl)
    bool ready_ = false;
};
