// ============================================================================
//  npu_axcl.cpp — npu_axcl.h 的实现: AXCL 会话的建立 (设备初始化 → 载模型 →
//    枚举 IO 并按契约校验 → 一次性分配缓冲) 与一次推理 tick (打包/H2D/execute/
//    D2H/解码, 逐段计时)。
// ============================================================================

#include "io/npu_axcl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <time.h>

#include <axcl.h>
#include <axcl_rt.h>
#include <axcl_rt_device.h>
#include <axcl_rt_engine.h>
#include <axcl_rt_engine_type.h>
#include <axcl_rt_memory.h>
#include <axcl_rt_type.h>

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

// axcl 的 errno 风格返回值打印 (0x8xxxxxxx; 只看十进制数字会认不出来)
static std::string axcl_err(int ret) {
    char b[64];
    snprintf(b, sizeof b, "0x%08x", (unsigned)ret);
    return b;
}

static DataType to_detect_dtype(axclrtEngineDataType t) {
    switch (t) {
        case AXCL_DATA_TYPE_UINT8: return DataType::kINT8;   // 8 位无符号, 元素字节数与 kINT8 相同
        case AXCL_DATA_TYPE_INT8:  return DataType::kINT8;
        case AXCL_DATA_TYPE_FP16:  return DataType::kHALF;
        case AXCL_DATA_TYPE_INT32: return DataType::kINT32;
        case AXCL_DATA_TYPE_FP32:  return DataType::kFLOAT;
        default:                   return DataType::kFLOAT;
    }
}

// 进程级设备初始化: 幂等 (一次运行里多个会话共用一张卡)
bool npu_init(std::string* err) {
    static std::once_flag once;
    static bool ok = false;
    static std::string fail;
    std::call_once(once, [] {
        int ret = axclInit(0);
        if (ret) { fail = "axclInit 失败 " + axcl_err(ret); return; }
        axclrtDeviceList lst{};
        ret = axclrtGetDeviceList(&lst);
        if (ret || lst.num == 0) {
            fail = "取设备表失败 (" + axcl_err(ret) + "), 卡的驱动/固件未就绪?";
            return;
        }
        ret = axclrtSetDevice(lst.devices[0]);
        if (ret) { fail = "SetDevice 失败 " + axcl_err(ret); return; }
        ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
        if (ret) { fail = "EngineInit 失败 " + axcl_err(ret); return; }
        ok = true;
    });
    if (!ok && err) *err = fail;
    return ok;
}

const char* npu_soc_name() {
    static const char* n = axclrtGetSocName();
    return n ? n : "";
}

const char* npu_dtype_str(int raw) {
    switch ((axclrtEngineDataType)raw) {
        case AXCL_DATA_TYPE_UINT8:  return "U8";
        case AXCL_DATA_TYPE_INT8:   return "S8";
        case AXCL_DATA_TYPE_UINT16: return "U16";
        case AXCL_DATA_TYPE_INT16:  return "S16";
        case AXCL_DATA_TYPE_INT32:  return "S32";
        case AXCL_DATA_TYPE_FP16:   return "FP16";
        case AXCL_DATA_TYPE_BF16:   return "BF16";
        case AXCL_DATA_TYPE_FP32:   return "FP32";
        default: break;
    }
    static thread_local char b[24];
    snprintf(b, sizeof b, "?%d", raw);
    return b;
}

const char* npu_layout_str(int raw) {
    switch ((axclrtEngineDataLayout)raw) {
        case AXCL_DATA_LAYOUT_NHWC: return "NHWC";
        case AXCL_DATA_LAYOUT_NCHW: return "NCHW";
        default: break;
    }
    static thread_local char b[24];
    snprintf(b, sizeof b, "?%d", raw);
    return b;
}

NpuSession::~NpuSession() { close(); }

size_t NpuSession::kept_outputs() const { return kept_.size(); }

const void* NpuSession::output_host(size_t idx) const {
    return idx < out_host_.size() ? out_host_[idx] : nullptr;
}

bool NpuSession::open(const std::string& model_path, std::string* err) {
    auto bad = [&](const std::string& m) { if (err) *err = m; close(); return false; };
    close();
    path_ = model_path;
    if (!npu_init(err)) return false;

    int ret = axclrtEngineLoadFromFile(path_.c_str(), &model_id_);
    if (ret) return bad("载入模型失败: " + path_ + " " + axcl_err(ret));

    axclrtEngineIOInfo ioInfo = nullptr;
    if (axclrtEngineGetIOInfo(model_id_, &ioInfo) || !ioInfo)
        return bad("取 IO 信息失败 (模型与运行时版本不匹配?)");

    const uint32_t nin = axclrtEngineGetNumInputs(ioInfo);
    const uint32_t nout = axclrtEngineGetNumOutputs(ioInfo);
    // 输入契约: 单输入 (本层的 H2D 源是"一帧 RGB", 多输入模型不在本契约里)
    if (nin != 1) return bad("契约要求单输入, 该模型有 " + std::to_string(nin) + " 个");

    if (axclrtEngineCreateContext(model_id_, &ctx_))
        return bad("CreateContext 失败");
    if (axclrtEngineCreateIO(ioInfo, &io_))
        return bad("CreateIO 失败");

    // ---- 输入: 读运行时自述并校验 U8 / NHWC / 3 通道 ----
    {
        NpuTensor t;
        t.input = true;
        t.name = axclrtEngineGetInputNameByIndex(ioInfo, 0);
        t.bytes = axclrtEngineGetInputSizeByIndex(ioInfo, 0, 0);
        axclrtEngineIODims d{};
        if (axclrtEngineGetInputDims(ioInfo, 0, 0, &d)) return bad("取输入形状失败");
        axclrtEngineDataType dt = AXCL_DATA_TYPE_NONE;
        axclrtEngineGetInputDataType(ioInfo, 0, &dt);
        axclrtEngineDataLayout ly = AXCL_DATA_LAYOUT_NONE;
        axclrtEngineGetInputDataLayout(ioInfo, 0, &ly);
        t.raw_dtype = (int)dt;
        t.raw_layout = (int)ly;
        t.dtype = to_detect_dtype(dt);
        t.dims.nbDims = d.dimCount > 8 ? 8 : d.dimCount;
        for (int i = 0; i < t.dims.nbDims; ++i) t.dims.d[i] = d.dims[i];
        if (dt != AXCL_DATA_TYPE_UINT8 || t.dims.nbDims != 4 || t.dims.d[0] != 1
            || t.dims.d[3] != 3) {
            char b[256];
            snprintf(b, sizeof b,
                     "输入不是契约要求的 U8/NHWC/3 通道 (dtype=%d dims=%dD [%lld %lld %lld %lld])",
                     (int)dt, t.dims.nbDims, (long long)t.dims.d[0], (long long)t.dims.d[1],
                     (long long)t.dims.d[2], (long long)t.dims.d[3]);
            return bad(b);
        }
        input_side_ = (t.dims.d[1] == t.dims.d[2]) ? (int)t.dims.d[1] : 0;
        in_bytes_ = t.bytes;
        // H2D 直读调用方的缓冲 (见 run): 那条路要求输入字节数恰为 边长²×3, 否则会越读
        if (input_side_ > 0 && in_bytes_ != (size_t)input_side_ * (size_t)input_side_ * 3) {
            char b[192];
            snprintf(b, sizeof b,
                     "输入字节数 %zu 与边长 %d 的平方×3 (%zu) 不符: H2D 直读调用方缓冲, "
                     "尺寸必须与契约一致",
                     in_bytes_, input_side_, (size_t)input_side_ * (size_t)input_side_ * 3);
            return bad(b);
        }
        in_.push_back(t);
    }
    // 非方形输入: 网格步长推不出来 (DFL 头与端到端/网格头的几何判定都要方形边长),
    //   记 0 让解码走"几何判不了"的退路口径, 不当错误 —— 契约里本平台没有这种模型。
    if (input_side_ == 0)
        fprintf(stderr, "[NPU] ⚠ 输入非方形, 网格几何不可判 (解码走退路口径)\n");

    // ---- 输出: 布局解得出来的才逐帧取回; 其余只留设备缓冲 ----
    keep_layouts_.clear();
    for (uint32_t i = 0; i < nout; ++i) {
        NpuTensor t;
        t.name = axclrtEngineGetOutputNameByIndex(ioInfo, i);
        t.bytes = axclrtEngineGetOutputSizeByIndex(ioInfo, 0, i);
        axclrtEngineIODims d{};
        if (axclrtEngineGetOutputDims(ioInfo, 0, i, &d)) return bad("取输出形状失败");
        axclrtEngineDataType dt = AXCL_DATA_TYPE_NONE;
        axclrtEngineGetOutputDataType(ioInfo, i, &dt);
        axclrtEngineDataLayout ly = AXCL_DATA_LAYOUT_NONE;
        axclrtEngineGetOutputDataLayout(ioInfo, i, &ly);
        t.raw_dtype = (int)dt;
        t.raw_layout = (int)ly;
        t.dtype = to_detect_dtype(dt);
        t.dims.nbDims = d.dimCount > 8 ? 8 : d.dimCount;
        for (int k = 0; k < t.dims.nbDims; ++k) t.dims.d[k] = d.dims[k];

        if (dt != AXCL_DATA_TYPE_FP32) {
            char b[128];
            snprintf(b, sizeof b, "跳过: 元素类型不是 FP32 (dtype=%d), 解码器只吃 float", (int)dt);
            t.note = b;
        } else if (t.dims.nbDims < 2 || t.dims.nbDims > 4) {
            char b[128];
            snprintf(b, sizeof b, "跳过: rank=%d 不是可解布局 (rank 2/3 网格或提案表, rank 4 DFL 头)",
                     t.dims.nbDims);
            t.note = b;
        } else {
            t.layout_ok = parseOutputLayout(t.dims, t.layout, input_side_);
            if (!t.layout_ok) {
                char b[192];
                snprintf(b, sizeof b,
                         "跳过: rank=%d 的形状解不出布局 (rank 4 需 [1,H,W,C] 且 H=W、C≥5、"
                         "边长能被网格整除)", t.dims.nbDims);
                t.note = b;
            } else {
                t.kept = true;
                kept_.push_back((int)i);
                keep_layouts_.push_back(t.layout);
                out_bytes_ += t.bytes;
            }
        }
        out_.push_back(t);
    }

    if (!alloc_buffers(err)) return false;
    ready_ = true;
    return true;
}

bool NpuSession::alloc_buffers(std::string* err) {
    auto bad = [&](const std::string& m) { if (err) *err = m; return false; };
    in_dev_.assign(in_.size(), nullptr);
    out_dev_.assign(out_.size(), nullptr);
    out_host_.assign(out_.size(), nullptr);
    for (size_t i = 0; i < in_.size(); ++i) {
        if (axclrtMalloc(&in_dev_[i], in_[i].bytes, AXCL_MEM_MALLOC_HUGE_FIRST))
            return bad("输入设备缓冲分配失败");
        if (axclrtEngineSetInputBufferByIndex(io_, (uint32_t)i, in_dev_[i], in_[i].bytes))
            return bad("SetInputBuffer 失败");
    }
    for (size_t i = 0; i < out_.size(); ++i) {
        if (axclrtMalloc(&out_dev_[i], out_[i].bytes, AXCL_MEM_MALLOC_HUGE_FIRST))
            return bad("输出设备缓冲分配失败");
        if (axclrtEngineSetOutputBufferByIndex(io_, (uint32_t)i, out_dev_[i], out_[i].bytes))
            return bad("SetOutputBuffer 失败");
        if (out_[i].kept) {
            out_host_[i] = malloc(out_[i].bytes);
            if (!out_host_[i]) return bad("输出主机缓冲分配失败");
        }
    }
    keep_ptrs_.assign(kept_.size(), nullptr);
    for (size_t k = 0; k < kept_.size(); ++k) keep_ptrs_[k] = (const float*)out_host_[kept_[k]];
    return true;
}

void NpuSession::free_buffers() {
    for (void* p : in_dev_)  if (p) axclrtFree(p);
    for (void* p : out_dev_) if (p) axclrtFree(p);
    for (void* p : out_host_) free(p);
    in_dev_.clear(); out_dev_.clear(); out_host_.clear();
    keep_ptrs_.clear();
}

void NpuSession::close() {
    if (!ready_ && !model_id_) return;
    free_buffers();
    if (io_) { axclrtEngineDestroyIO(io_); io_ = nullptr; }
    if (model_id_) { axclrtEngineUnload(model_id_); model_id_ = 0; }
    ctx_ = 0;
    kept_.clear(); keep_layouts_.clear();
    in_.clear(); out_.clear();
    in_bytes_ = out_bytes_ = 0; input_side_ = 0;
    ready_ = false;
}

std::vector<Detection> NpuSession::run(const uint8_t* rgb_hwc, int num_classes, float conf_thr,
                                       int want_cls, NpuTick* tick) {
    std::vector<Detection> dets;
    if (!ready_) { if (tick) tick->ok = false; return dets; }
    NpuTick t{};

    // H2D 直接从调用方的缓冲读 —— 它就是 RGA 目的缓冲的映射, 与模型输入同尺寸同布局,
    //   中间再拷一份只是把同一批字节搬两遍, 而那一遍每帧都压在关键路径上。源缓冲的尺寸
    //   契约 (in_bytes_ == 边长²×3) 在 open 时已核 (见那里的注释)。
    const double t0 = now_us();
    if (axclrtMemcpy(in_dev_[0], (void*)rgb_hwc, in_bytes_, AXCL_MEMCPY_HOST_TO_DEVICE))
        { t.ok = false; if (tick) *tick = t; return dets; }
    const double t2 = now_us();
    if (axclrtEngineExecute(model_id_, ctx_, 0, io_))
        { t.ok = false; if (tick) *tick = t; return dets; }
    const double t3 = now_us();
    double t4_each = t3;
    for (size_t k = 0; k < kept_.size(); ++k) {
        const int i = kept_[k];
        if (axclrtMemcpy(out_host_[i], out_dev_[i], out_[i].bytes, AXCL_MEMCPY_DEVICE_TO_HOST))
            { t.ok = false; if (tick) *tick = t; return dets; }
        if (t.d2h_n < NPU_MAX_OUT) {          // 逐输出的 D2H (逐张量表, 见 NpuTick)
            t.d2h_each_us[t.d2h_n] = now_us() - t4_each;
            t4_each = now_us();
            ++t.d2h_n;
        }
    }
    const double t4 = now_us();
    dets = decode_outputs(keep_ptrs_, keep_layouts_, num_classes, conf_thr, want_cls,
                          input_side_);
    const double t5 = now_us();

    t.h2d_us = t2 - t0; t.exec_us = t3 - t2;
    t.d2h_us = t4 - t3; t.decode_us = t5 - t4; t.dets = dets.size();
    if (tick) *tick = t;
    return dets;
}
