"""把训练好的 best.pt 导出成两个 ONNX：FP32 与 FP16，并立即改名。

用法：
    python export_onnx.py                          # 默认导出 320 的 best.pt
    python export_onnx.py --imgsz 640 --weights runs/detect/train640/weights/best.pt --prefix best640

注意事项：
- ultralytics 两次导出都写到同名文件 best.onnx，第二次会静默覆盖第一次。
  所以导出后必须马上 shutil.move 改名，这就是下面 dst/shmove 那两行的原因。
- simplify=False：onnxsim 在 Windows 上会崩（segment fault），不要打开。
- dynamic=False：固定 batch=1，CPU 推理更快。
"""
import sys, os, shutil, time, argparse
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ["YOLO_CONFIG_DIR"] = os.path.join(BASE, "yolo_cfg")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE, "mpl_cfg")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import shim; shim.apply()
from ultralytics import YOLO

EXP = os.path.join(BASE, "exports")
os.makedirs(EXP, exist_ok=True)
BEST = os.path.join(BASE, "runs", "detect", "train", "weights", "best.pt")
IMGSZ = 320


def export(half, out_name, weights=None, imgsz=None):
    m = YOLO(weights or BEST)
    t = time.time()
    p = m.export(
        format="onnx",
        imgsz=imgsz or IMGSZ,
        opset=17,
        simplify=False,      # onnxsim 在 Windows 上会崩，保持关闭
        dynamic=False,       # 固定 batch=1，利于 CPU 推理
        half=half,
        device=0,
    )
    el = time.time() - t
    dst = os.path.join(EXP, out_name)
    shutil.move(str(p), dst)   # 两次导出都叫 best.onnx，必须立刻改名，否则第二次会覆盖第一次
    sz = os.path.getsize(dst)
    print(f"{out_name}  {sz} B  ({sz/1048576:.1f} MB)  export {el:.1f}s")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=BEST,
                    help="要导出的 .pt 权重，默认 runs/detect/train/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=IMGSZ,
                    help="导出分辨率，默认 320；与训练时的 imgsz 保持一致")
    ap.add_argument("--prefix", default="best", help="输出文件名前缀，如 best 或 best640（决定 exports/ 下的文件名）")
    a = ap.parse_args()

    f32 = export(False, f"{a.prefix}_fp32.onnx", a.weights, a.imgsz)
    f16 = export(True, f"{a.prefix}_fp16.onnx", a.weights, a.imgsz)
    print("RATIO", round(os.path.getsize(f16) / os.path.getsize(f32), 3))
