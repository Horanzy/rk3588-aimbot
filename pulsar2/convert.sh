#!/bin/bash
# ==============================================================================
#  单模型转换 —— 一个 .onnx, 一个标定集, 一份 axmodel。
#
#      bash convert.sh -m onnx/mymodel.onnx                      # 自动挑标定集(专用集 → 通用池)
#      bash convert.sh -m onnx/mymodel.onnx -c dataset/calib_模型A.tar
#      bash convert.sh -m onnx/mymodel.onnx -n NPU1 -s 200       # 核数 / 标定张数
#      bash convert.sh -m onnx/mymodel.onnx -F                   # 已存在也重转
#
#  顺序: patch_onnx.py(等价改写, 折掉工具链不支持的 Mod) → prep_calib.py(标定集按模型输入
#        尺寸中心裁剪) → gen_config.py(现生成配置, 含"输出算子提 U16") → pulsar2 build
#
#  标定集怎么选(两遍走的约定):
#      dataset/calib_<模型名>.tar   那个游戏自己采的帧 —— 精度最好
#      dataset/calib_generic.tar    通用池, 跨游戏混采(往里加图重打包即可) —— 第一遍用它
#  省略 -c 时按上面的顺序找; 两者都没有就直接退出并说明(没有标定集就没有激活量程, int8
#  无从谈起 —— 这点与 Jetson 的 fp16 不同, 那边本来不需要标定)。
#
#  目录: onnx/ 输入 · axmodel/ 产物 · dataset/ 标定 tar · work/ 配置与日志(可删)
#  可覆盖: IMG= 镜像 · CALIB= 标定集 · CALIB_SIZE= 张数 · NPU_MODE= 核数
#          KEEP_QUANT=1 留下 Pulsar2 的 debug 转储(验收要用, 见 check_quant.py)
#          NO_LAST_U16=1 关掉"输出算子提 U16"(做 A/B)
# ==============================================================================
set -u
ROOT="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
MODEL=""; CALIB=""; OUT_NAME=""; OUT_DIR="axmodel"; FORCE=""
CALIB_SIZE="${CALIB_SIZE:-100}"
NPU_MODE="${NPU_MODE:-NPU3}"

# 用法横幅 = 开头连续的注释行(到第一行代码为止), 不写死行号 —— 改横幅不必再改这里。
usage() { awk 'NR>2 && /^#/ {print; next} NR>2 {exit}' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="${2:-}"; shift 2 ;;
        -c) CALIB="${2:-}"; shift 2 ;;
        -o) OUT_NAME="${2:-}"; shift 2 ;;
        -O) OUT_DIR="${2:-}"; shift 2 ;;
        -s) CALIB_SIZE="${2:-}"; shift 2 ;;
        -n) NPU_MODE="${2:-}"; shift 2 ;;
        -F) FORCE=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "不认识的参数: $1" >&2; usage 1 ;;
    esac
done

[ -n "$MODEL" ] || { echo "缺 -m <onnx>" >&2; usage 1; }
# 先把参数验完再去找文件, 免得参数写错却先看到一串路径解析的输出。
case "$NPU_MODE" in NPU1|NPU2|NPU3) ;; *) echo "-n 只接受 NPU1/NPU2/NPU3 (给了 $NPU_MODE)" >&2; exit 1 ;; esac
case "$CALIB_SIZE" in ''|*[!0-9]*) echo "-s 需要正整数 (给了 $CALIB_SIZE)" >&2; exit 1 ;; esac

# 容器只挂载套件目录, 所以模型与标定集都必须落在它里面。
to_container() {
    local p; p="$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1")" || return 1
    case "$p" in "$ROOT"/*) echo "/data${p#"$ROOT"}" ;; *) return 1 ;; esac
}
rel() { case "$1" in "$ROOT"/*) echo "${1#"$ROOT"/}" ;; *) echo "$1" ;; esac; }

MODEL_ABS="$(cd "$(dirname "$MODEL")" 2>/dev/null && pwd)/$(basename "$MODEL")" || true
[ -f "$MODEL_ABS" ] || { echo "模型不存在: $MODEL" >&2; exit 1; }
MODEL_C="$(to_container "$MODEL_ABS")" || {
    echo "模型必须在套件目录内(容器只挂载 $ROOT): $(rel "$MODEL_ABS")" >&2; exit 1; }
[ -n "$OUT_NAME" ] || OUT_NAME="$(basename "$MODEL_ABS" .onnx)"

if [ -n "$CALIB" ]; then
    CALIB_ABS="$(cd "$(dirname "$CALIB")" 2>/dev/null && pwd)/$(basename "$CALIB")" || true
    [ -f "$CALIB_ABS" ] || { echo "标定集不存在: $CALIB" >&2; exit 1; }
else
    CALIB_ABS="$(ls "$ROOT"/dataset/calib_"$OUT_NAME".tar "$ROOT"/dataset/calib_"$OUT_NAME"_*.tar \
                    "$ROOT"/dataset/calib_"$OUT_NAME"*.tar.gz "$ROOT"/dataset/calib_"$OUT_NAME"*.zip \
                    2>/dev/null | grep -v "calib_generic" | head -1)"
    if [ -n "$CALIB_ABS" ]; then
        echo "标定集: 专用 $(rel "$CALIB_ABS")"
    elif [ -f "$ROOT/dataset/calib_generic.tar" ]; then
        CALIB_ABS="$ROOT/dataset/calib_generic.tar"
        echo "标定集: 通用池 $(rel "$CALIB_ABS")  —  第一遍用它是为了先把模型跑起来;"
        echo "        拿它去玩、去截图, 攒够本游戏的帧打成 dataset/calib_$OUT_NAME.tar 再转第二遍,"
        echo "        量化统计才来自那个游戏真实的分布。"
    else
        echo "标定集两者都没有, 且没有指定 -c:" >&2
        echo "  dataset/calib_$OUT_NAME.tar(本游戏专用)与 dataset/calib_generic.tar(通用池)都不存在。" >&2
        exit 1
    fi
fi
CALIB_C="$(to_container "$CALIB_ABS")" || {
    echo "标定集必须在套件目录内: $(rel "$CALIB_ABS")" >&2; exit 1; }

mkdir -p "$ROOT/$OUT_DIR" "$ROOT/work"
if [ -f "$ROOT/$OUT_DIR/$OUT_NAME.axmodel" ] && [ -z "$FORCE" ]; then
    echo "已有 $OUT_DIR/$OUT_NAME.axmodel — 跳过 (要重转加 -F)"; exit 0
fi

IMG="${IMG:-}"
if [ -z "$IMG" ]; then
    CAND="$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i pulsar2)"
    [ -z "$CAND" ] && { echo "找不到 pulsar2 镜像 — 先 docker load -i ax_pulsar2_<版本>.tar.gz"; exit 1; }
    IMG="$(printf '%s\n' $CAND | sort -V | tail -1)"
    [ "$(printf '%s\n' $CAND | wc -l)" -gt 1 ] && echo "镜像候选: $(printf '%s ' $CAND)→ 取 $IMG"
fi
echo "模型 $(rel "$MODEL_ABS") → $OUT_DIR/$OUT_NAME.axmodel · $NPU_MODE · 标定 $CALIB_SIZE 张"
echo

# 该镜像的 ENTRYPOINT 就是 bash, 所以必须显式 --entrypoint /bin/bash; 直接写 "IMG bash -c ..."
#   会变成 bash 去读一个叫 bash 的文件, 读到 /usr/bin/bash 报 cannot execute binary file。
docker run --rm --net host -v "$ROOT":/data --entrypoint /bin/bash \
    -e MODEL="$MODEL_C" -e CALIB="$CALIB_C" -e CALIB_SIZE="$CALIB_SIZE" \
    -e NPU_MODE="$NPU_MODE" -e OUT_NAME="$OUT_NAME" -e OUT_DIR="$OUT_DIR" \
    -e KEEP_QUANT="${KEEP_QUANT:-}" -e NO_LAST_U16="${NO_LAST_U16:-}" \
    "$IMG" -c '
cd /data || exit 1
PY=python3; command -v python3 >/dev/null || PY=python
$PY -c "import onnx" 2>/dev/null || echo "⚠ 容器内无 onnx 包 — 配置生成会失败"
$PY -c "import PIL"   2>/dev/null || echo "⚠ 容器内无 PIL — 标定集无法按模型输入尺寸裁剪 (pip install pillow)"

# 等价改写: 折掉工具链不支持的算子(Mod 等)。规则与逐值验证写在 patch_onnx.py。
BUILD_IN="$MODEL"
if $PY /data/patch_onnx.py "$MODEL" "/data/work/$OUT_NAME.patched.onnx" \
        > "/data/work/$OUT_NAME.patch.log" 2>&1; then
    BUILD_IN="/data/work/$OUT_NAME.patched.onnx"
    sed "s/^/   /" "/data/work/$OUT_NAME.patch.log"
else
    echo "   (改写未生效, 用原图)"
fi

# 标定集的视野必须等于运行期视野: 采集窗口是相机像素 1:1 的中心裁剪, 边长就是模型输入
#   边长。640 的图对 320 的模型必须先中心裁到 320(不缩放) —— 交给工具链去缩会喂进
#   "视野宽一倍、目标小一半"的分布。依据见 prep_calib.py 头部。
CALIB_USE="$($PY /data/prep_calib.py "$CALIB" /data/work "$BUILD_IN" "$CALIB_SIZE" \
             2>"/data/work/$OUT_NAME.calib.log")"
rc=$?
sed "s/^/   /" "/data/work/$OUT_NAME.calib.log"
if [ "$rc" != "0" ]; then
    CALIB_USE="$CALIB"
    echo "   ⚠ 标定集视野未对齐 — 本次仍用它, 该模型的精度结论要打折"
fi

if ! $PY /data/gen_config.py "$BUILD_IN" "/data/work/$OUT_NAME.json" \
          "$CALIB_USE" "$CALIB_SIZE" "$NPU_MODE" "$OUT_NAME.axmodel"; then
    echo "配置生成失败"; exit 1
fi

if pulsar2 build --target_hardware AX650 --input "$BUILD_IN" --output_dir "/data/$OUT_DIR" \
                 --config "/data/work/$OUT_NAME.json" > "/data/work/$OUT_NAME.log" 2>&1; then
    echo "✅ $OUT_DIR/$OUT_NAME.axmodel  ($(du -h "/data/$OUT_DIR/$OUT_NAME.axmodel" | cut -f1))"
    echo "   配置 work/$OUT_NAME.json · 日志 work/$OUT_NAME.log"
    echo "   上板: sudo LD_LIBRARY_PATH=/usr/lib/axcl /usr/bin/axcl/axcl_run_model -m $OUT_NAME.axmodel -r 200 -w 20"
    # 构建中间产物(compiler/frontend/quant)只在排查时有用, 平时删掉免得堆满磁盘。
    #   quant/ 里还带着 Pulsar2 自己的 debug 转储(FP32 参考 vs 量化后跑), 那是按通道角色
    #   量量化误差的唯一来源 —— 要留就 KEEP_QUANT=1。
    if [ "${KEEP_QUANT:-}" != "1" ]; then
        rm -rf "/data/$OUT_DIR/compiler" "/data/$OUT_DIR/frontend" "/data/$OUT_DIR/quant"
    else
        echo "   转储留在 $OUT_DIR/quant/debug/ 下 — 按通道角色量: python3 check_quant.py $OUT_DIR"
    fi
    rm -f "/data/work/$OUT_NAME.patched.onnx"
    exit 0
fi
echo "❌ 失败 — 日志尾部 (全文 work/$OUT_NAME.log):"
tail -14 "/data/work/$OUT_NAME.log" | sed "s/^/   /"
grep -aoE "op name: [^,]*, [A-Za-z]+|opset|Unsupported|not support" "/data/work/$OUT_NAME.log" \
    | sort -u | head -3 | sed "s/^/   ↑ 工具链报的算子/opset: /"
exit 1
'
