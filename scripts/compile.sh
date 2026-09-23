#! /bin/bash
# ==============================================================================
#  编译脚本 — 在 RK3588 板端执行, 产物输出到 <根目录>/bin/
#  路径相对脚本自身解析, 与部署位置无关。
#  模块结构: main (入口) + core/ (共享状态/控制律/估计器/标定/检测输出解析)
#            + io/ (取帧与 NPU 推理/鼠标输入与 USB 输出/热参); 逐编译单元编译到
#            build/ 再链接。
# ==============================================================================
set -e
ROOT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
SRC="$ROOT/src"
BUILD="$ROOT/build"
BIN="$ROOT/bin"
mkdir -p "$BUILD" "$BIN"

CXX=g++
CXX_FLAGS="-O3 -DNDEBUG -std=c++17"
INCLUDES="-I$SRC -I/usr/include/opencv4 -I/usr/include/axcl -I/usr/local/include/rga"
OCV="-lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_imgcodecs"

# axcl (NPU 运行时) 与 librga (裁切/格式转换) 都是**部署机自带的系统库**: axcl 装在
#   /usr/lib/axcl 并已进 ldconfig (板级 BSP 的一部分), librga 装在 /usr/local (头文件
#   /usr/local/include/rga) 并由 scripts/setup_platform.sh 从上游装好 —— 与本库自身放在
#   哪里无关。
LIBS="-L/usr/lib/axcl -laxcl_rt -laxcl_pkg -lrga"
PTHREAD="-pthread"

# 模块清单 = src/ 下的全部可移植编译单元 (入口 main 与 core/io 各 .cpp 逐一对应)
MODULES="main core/control core/estimator core/calib core/detect core/state core/proc_util \
         io/hid_mouse io/usbraw io/hotctl io/calib_run io/pad_input io/pad_output \
         io/pad_xinput io/pad_p5g io/hdmi_in io/rga_pp io/npu_axcl io/capture"

OBJS=""
for m in $MODULES; do
    obj="$BUILD/$(echo "$m" | tr '/' '_').o"
    # shellcheck disable=SC2086
    $CXX -c "$SRC/$m.cpp" $CXX_FLAGS $INCLUDES -o "$obj"
    OBJS="$OBJS $obj"
done

# 单测统一链接除 main.o 外的全部模块对象 (与正式产物同一份目标码), 逐个执行;
#   断言失败 (退出码非 0) 时 set -e 终止整个编译。
TEST_OBJS=$(printf '%s\n' $OBJS | grep -v "build/main.o" | tr '\n' ' ')

# 正式产物: 入口 main.o + 全部模块对象 → bin/aimbot (它需要 root: /dev/axcl_host、
#   /dev/rga、raw_gadget 都是 root-only)
# shellcheck disable=SC2086
$CXX "$BUILD/main.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BIN/aimbot"

# 控制拍单测 (拉枪速度倍率 spd 的落点): spd=100 即基线 (1 count = 1 px), 逐轴独立,
#   ADS 键按住那一拍整套切换 + g_ads_down 导出, 注入换算/在飞补偿/估计器自身运动
#   补偿共用同一份逐轴有效灵敏度 (含随 spd 成比例变化), spd_clamp 的夹取带,
#   热参路径 (spdx 下一拍生效 / 非法值被拒绝), 以及标定回写 — 整个文件逐字节比对 —
#   值落在调用方给的 VAR, ${VAR:-…} 保护式与行内注释原样保留, 裸行写回裸行, 缺行追加。
# shellcheck disable=SC2086
$CXX -c "$SRC/core/control_test.cpp" $CXX_FLAGS $INCLUDES -o "$BUILD/control_test.o"
# shellcheck disable=SC2086
$CXX "$BUILD/control_test.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BUILD/control_test"
"$BUILD/control_test"

# 手柄模式单测 (输入映射 8→16 位 / 注入合并几何 / 账本与发布点契约 / XInput 与
#   P5G 两条线格式及设备字节 / P5G 认证状态机与签名流水线): 与控制拍单测同一链接
#   方式 (除 main.o 外的模块对象), 断言失败即 set -e 终止整个编译。
# shellcheck disable=SC2086
$CXX -c "$SRC/io/pad_test.cpp" $CXX_FLAGS $INCLUDES -o "$BUILD/pad_test.o"
# shellcheck disable=SC2086
$CXX "$BUILD/pad_test.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BUILD/pad_test"
"$BUILD/pad_test"

# 标定单测 (采样几何与块统计 / 一维投影与二维块相关的对照 / 状态机 / 合成闭环 e2e 的
#   偏差与散度 / 不可测的诚实性): 同一链接方式, 断言失败即终止整个编译。
# shellcheck disable=SC2086
$CXX -c "$SRC/core/calib_test.cpp" $CXX_FLAGS $INCLUDES -o "$BUILD/calib_test.o"
# shellcheck disable=SC2086
$CXX "$BUILD/calib_test.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BUILD/calib_test"
"$BUILD/calib_test"

# 检测输出解析单测 (布局解析与四种真实导出形状 / 锚点数由模型几何给出 / 口径判定与
#   它的退路口径 / 四支解码的手算候选 / DFL 头的分布期望 / NMS): 同一链接方式, 断言
#   失败即终止整个编译。
# shellcheck disable=SC2086
$CXX -c "$SRC/core/detect_test.cpp" $CXX_FLAGS $INCLUDES -o "$BUILD/detect_test.o"
# shellcheck disable=SC2086
$CXX "$BUILD/detect_test.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BUILD/detect_test"
"$BUILD/detect_test"

# 取帧层单测 (失败去向: 哪些要重建/哪些只是丢帧 / 连续 fence 超时的判定线 = 队列块数−1 /
#   重建节拍 / 常量的推导关系): 同一链接方式, 断言失败即终止整个编译。它不碰设备 ——
#   真机的失锁→重锁→画面回来由 scripts/test/hdmi_probe 与 aimbot 的 [HDMI] 行验收。
# shellcheck disable=SC2086
$CXX -c "$SRC/io/hdmi_test.cpp" $CXX_FLAGS $INCLUDES -o "$BUILD/hdmi_test.o"
# shellcheck disable=SC2086
$CXX "$BUILD/hdmi_test.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BUILD/hdmi_test"
"$BUILD/hdmi_test"

# HDMI IN 采集探针 (板端验收工具, 不是单测): 裸 V4L2 取帧 + RGA 裁剪/格式 + 与 mmap
#   采集缓冲的逐字节 CPU 对照 + 各阶段 PNG。同一链接方式 (除 main.o 外的模块对象);
#   它需要活动信号与 root, 故只构建不执行 —— 运行: sudo ./build/hdmi_probe 600
# shellcheck disable=SC2086
$CXX -c "$ROOT/scripts/test/hdmi_probe.cpp" $CXX_FLAGS $INCLUDES -o "$BUILD/hdmi_probe.o"
# shellcheck disable=SC2086
$CXX "$BUILD/hdmi_probe.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BUILD/hdmi_probe"

# NPU 推理探针 (板端验收工具, 不是单测): 一个 .axmodel 的 IO 契约 / 真图端到端解码 /
#   输入与原始输出的落盘 / 各段耗时百分位 / 传输尺寸扫描。它需要 root 与 AXCL 卡,
#   故只构建不执行 —— 运行: sudo LD_LIBRARY_PATH=/usr/lib/axcl ./build/model_probe <model>
# shellcheck disable=SC2086
$CXX -c "$ROOT/scripts/test/model_probe.cpp" $CXX_FLAGS $INCLUDES -o "$BUILD/model_probe.o"
# shellcheck disable=SC2086
$CXX "$BUILD/model_probe.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BUILD/model_probe"

# 取帧→NPU 端到端探针 (板端验收工具, 不是单测): 用生产路径本身 (io/capture 的
#   CapturePipeline) 跑活信号帧率/逐段耗时, 再把一张真图贴进一帧真实采集缓冲的中心窗口
#   走同一条链并落盘窗口与标注 PNG。要 root、卡与活动信号, 故只构建不执行 ——
#   运行: sudo LD_LIBRARY_PATH=/usr/lib/axcl ./build/pipeline_probe <model> --image <jpg>
# shellcheck disable=SC2086
$CXX -c "$ROOT/scripts/test/pipeline_probe.cpp" $CXX_FLAGS $INCLUDES -o "$BUILD/pipeline_probe.o"
# shellcheck disable=SC2086
$CXX "$BUILD/pipeline_probe.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BUILD/pipeline_probe"

echo "✅ 编译完成 (bin/aimbot + 可移植集 + 五个单测 + 采集探针 + NPU 探针 + 端到端探针) → $BUILD"
