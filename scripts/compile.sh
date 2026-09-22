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
#   /usr/lib/axcl 并已进 ldconfig, librga 装在 /usr/local (头文件 /usr/local/include/rga),
#   两者都由 scripts/setup_platform.sh 从上游装好 —— 与本库自身放在哪里无关。
LIBS="-L/usr/lib/axcl -laxcl_rt -laxcl_pkg -lrga"
PTHREAD="-pthread"

# 模块清单 = src/ 下的全部可移植编译单元 (core/io 各 .cpp 逐一对应)
MODULES="core/control core/estimator core/calib core/detect core/state \
         io/hid_mouse io/usbraw io/hotctl io/calib_run io/pad_input io/pad_output \
         io/pad_xinput io/pad_p5g io/hdmi_in io/rga_pp"

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

# HDMI IN 采集探针 (板端验收工具, 不是单测): 裸 V4L2 取帧 + RGA 裁剪/格式 + 与 mmap
#   采集缓冲的逐字节 CPU 对照 + 各阶段 PNG。同一链接方式 (除 main.o 外的模块对象);
#   它需要活动信号与 root, 故只构建不执行 —— 运行: sudo ./build/hdmi_probe 600
# shellcheck disable=SC2086
$CXX -c "$ROOT/scripts/test/hdmi_probe.cpp" $CXX_FLAGS $INCLUDES -o "$BUILD/hdmi_probe.o"
# shellcheck disable=SC2086
$CXX "$BUILD/hdmi_probe.o" $TEST_OBJS $LIBS $OCV $PTHREAD -o "$BUILD/hdmi_probe"

echo "✅ 编译完成 (可移植集 + 三个单测 + 采集探针) → $BUILD"
