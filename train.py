"""YOLOv8n 训练入口：在 GC10-DET 上微调，产物落在 runs/detect/<name>/。

用法：
    python train.py --data data/gc10/data.yaml --epochs 100 --imgsz 320
    python train.py --resume runs/detect/train/weights/last.pt

注意事项：
- Windows 上必须 workers=0，多进程 dataloader 会因共享内存/管道错误崩溃。
- shim.apply() 必须在 import ultralytics 之前调用，否则替换不生效。
- 断点续训时只传 --resume <last.pt>，epochs/imgsz/batch 由 checkpoint 里的
  args 恢复，命令行再传会与存档参数冲突。
"""
import sys, os, argparse
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ["YOLO_CONFIG_DIR"] = os.path.join(BASE, "yolo_cfg")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE, "mpl_cfg")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import shim; shim.apply()          # 必须在 import ultralytics 之前
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml 的路径（绝对路径最稳）")
    ap.add_argument("--epochs", type=int, default=100, help="训练轮数，默认 100")
    ap.add_argument("--imgsz", type=int, default=320, help="输入分辨率，默认 320（GC10 原图为 2048x1000，320 是可用的精度/速度折中）")
    ap.add_argument("--batch", type=int, default=16, help="batch size，默认 16")
    ap.add_argument("--model", default="yolov8n.pt", help="初始权重，默认 yolov8n.pt（COCO 预训练）")
    ap.add_argument("--name", default="train", help="runs/detect/ 下的子目录名，默认 train")
    ap.add_argument("--resume", default=None,
                    help="断点续训：传上次的 last.pt 路径，其余参数从该 checkpoint 的 args 里恢复")
    a = ap.parse_args()

    if a.resume:
        # 续训时 ultralytics 会从 checkpoint 里读回 epochs/imgsz/batch 等参数，
        # 这里只传 resume=True，避免命令行参数与存档参数打架。
        if not os.path.exists(a.resume):
            raise FileNotFoundError(f"续训用的权重不存在: {a.resume}")
        m = YOLO(a.resume)
        r = m.train(resume=True)
        print("BEST", os.path.join(BASE, "runs", "detect", a.name, "weights", "best.pt"))
        print("BEST_MAP50", r.results_dict.get("metrics/mAP50(B)", -1))
        return

    m = YOLO(a.model)
    r = m.train(
        data=os.path.abspath(a.data),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=0,
        workers=0,             # Windows 上 workers>0 会因共享内存/管道错误崩溃
        project=os.path.join(BASE, "runs", "detect"),   # 绝对路径
        name=a.name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        # —— 工业缺陷语义相关的开关 ——
        flipud=0.0,            # 上下翻转：缺陷有方向语义，禁止
        fliplr=0.5,            # 左右翻转：钢带上可接受
        degrees=0.0,           # 旋转：会产生插值伪影，禁止
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.0,             # 灰度图，色相无意义
        hsv_s=0.0,             # 同上
        hsv_v=0.3,             # 亮度可抖（模拟光照波动）
        patience=30,
        plots=True,
        verbose=True,
    )
    print("BEST", os.path.join(BASE, "runs", "detect", a.name, "weights", "best.pt"))
    print("BEST_MAP50", r.results_dict.get("metrics/mAP50(B)", -1))


if __name__ == "__main__":
    main()
