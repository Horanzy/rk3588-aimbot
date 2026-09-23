#!/usr/bin/env python3
"""把标定集按模型的输入尺寸做一次中心裁剪, 让标定的视野等于运行期的视野。

为什么必须裁而不是让 Pulsar2 缩:

自瞄的采集窗口是**相机像素 1:1 的中心裁剪**, 边长就是当前加载模型的输入边长
(`src/io/capture.cu`: `crop_x=(cam_w-cap_w)/2` 裁 `cap_w x cap_h`, 输出同尺寸)。
所以 640 的模型看相机中心 640x640 像素, 320 的模型看中心 320x320 像素 —— 两者视野
差一倍(面积差四倍)。采集存下来的就是 `cap_img`, 因此 640 的截图集只对 640 的模型
**等于**运行期输入。

若把 640 的图交给 Pulsar2 让它缩到 320, 标定时看到的是"视野宽一倍、目标小一半"的画面,
而模型运行期永远看不到这个画面: 激活统计就来自一个错误的分布。从图像中心裁出
320x320(不缩放)得到的才是运行期那块区块本身。

用法:
    python3 prep_calib.py <源标定 tar/zip> <work 目录> <onnx> [<标定张数>]

输出约定(给调用脚本用):
    stdout 最后一行 = 应当交给 gen_config.py 的标定集路径
    stderr          = 人读的诊断
    退出码 0 = 视野已对齐(裁剪过, 或本来就同尺寸)
           2 = 无法对齐(模型输入大于源图, 或环境缺 PIL) —— stdout 给的是源标定集原样,
               调用方应当把这一次计入"未对齐"并告警, 但不该因此中断整个转换
           1 = 硬错误(源标定集读不了等)
"""

import io
import os
import sys
import tarfile
import zipfile

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def fallback(src, msg, size_want=None, size_have=None):
    print(msg, file=sys.stderr)
    if size_want and size_have:
        print("   模型输入 %s, 标定图 %s —— 需要重新采集一套同尺寸的帧, 或裁到不小于模型输入的图。"
              % (size_want, size_have), file=sys.stderr)
    print(src)
    sys.exit(2)


def input_dims(onnx_path):
    """模型的输入边长 (H, W), 取自它自己声明的形状。"""
    try:
        import onnx
    except ImportError:
        die("缺 onnx 包, 读不了模型输入形状 (gen_config.py 也需要它)")
    m = onnx.load(onnx_path, load_external_data=False)
    t = m.graph.input[0].type.tensor_type
    if not t.HasField("shape"):
        die("模型输入没有声明形状: %s" % onnx_path)
    d = [x.dim_value if x.HasField("dim_value") else 0 for x in t.shape.dim]
    if len(d) != 4 or d[2] == 0 or d[3] == 0:
        die("只支持静态 NCHW 输入, 读到的是 %s: %s" % (d, onnx_path))
    return d[2], d[3]


def iter_images(src):
    """(成员名, 读字节的可调用) —— 同时吃 tar / tar.gz / zip。"""
    if src.endswith(".zip"):
        zf = zipfile.ZipFile(src)
        for n in zf.namelist():
            if n.lower().endswith(IMG_EXT) and not n.endswith("/"):
                yield n, (lambda n=n: zf.read(n))
        return
    tf = tarfile.open(src, "r:*")
    for m in tf.getmembers():
        if m.isfile() and m.name.lower().endswith(IMG_EXT):
            yield m.name, (lambda m=m: tf.extractfile(m).read())


def main():
    if len(sys.argv) < 4:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("用法: prep_calib.py <源标定 tar/zip> <work 目录> <onnx> [<标定张数>]", file=sys.stderr)
        sys.exit(1)
    src, work, onnx_path = sys.argv[1:4]
    calib_size = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    if not os.path.isfile(src):
        die("标定集不存在: %s" % src)
    if not os.path.isfile(onnx_path):
        die("模型不存在: %s" % onnx_path)

    want_h, want_w = input_dims(onnx_path)

    try:
        from PIL import Image
    except ImportError:
        fallback(src, "环境里没有 PIL, 无法裁剪标定集 —— pip install pillow 后重跑; "
                      "本次直接使用未裁剪的标定集")

    first = next(iter_images(src), None)
    if first is None:
        die("标定集里没有图片: %s" % src)
    have_w, have_h = Image.open(io.BytesIO(first[1]())).size

    if (have_w, have_h) == (want_w, want_h):
        print("标定集已经是 %dx%d, 与模型输入同尺寸, 直接使用: %s"
              % (want_w, want_h, os.path.basename(src)), file=sys.stderr)
        print(src)
        return

    if want_w > have_w or want_h > have_h:
        fallback(src, "模型输入 %dx%d 大于标定图 %dx%d, 裁不出来(往上裁是假的)"
                      % (want_w, want_h, have_w, have_h),
                 size_want="%dx%d" % (want_w, want_h), size_have="%dx%d" % (have_w, have_h))

    os.makedirs(work, exist_ok=True)
    out = os.path.join(work, "calib_px%dx%d.tar" % (want_w, want_h))
    # 源标定集没变就复用上一轮的产物, 免得每加一个同尺寸的模型就重裁一遍。
    if os.path.isfile(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        print("复用已裁好的 %s (源标定集未变)" % os.path.basename(out), file=sys.stderr)
        print(out)
        return

    n = 0
    with tarfile.open(out, "w") as tf:
        for name, read in iter_images(src):
            im = Image.open(io.BytesIO(read()))
            w, h = im.size
            cx, cy = (w - want_w) // 2, (h - want_h) // 2   # 与运行期同一个中心裁剪
            buf = io.BytesIO()
            im.crop((cx, cy, cx + want_w, cy + want_h)).save(buf, format="JPEG", quality=95)
            data = buf.getvalue()
            ti = tarfile.TarInfo(os.path.basename(name))
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
            n += 1

    print("中心裁剪 %dx%d -> %dx%d, %d 张 -> %s"
          % (have_w, have_h, want_w, want_h, n, os.path.basename(out)), file=sys.stderr)
    if calib_size and calib_size > n:
        print("   注意: CALIB_SIZE=%d 大于本标定集的 %d 张, Pulsar2 会取不满" % (calib_size, n),
              file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
