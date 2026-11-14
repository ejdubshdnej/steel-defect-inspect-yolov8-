"""验证集评测：总体指标 + 逐类 mAP 表 + 混淆矩阵说明。

用法：
    python scripts/eval.py                          # 默认评 320 的 best.pt
    python scripts/eval.py --split val              # 键名必须是 val，不是 valid
    python scripts/eval.py --weights runs/detect/train640/weights/best.pt \
                           --imgsz 640 --name val640

产物：
- runs/detect/<name>/ 下的 PR/F1 曲线、混淆矩阵等图
- runs/detect/<name>/per_class.json（逐类 mAP@0.5，供 bench/compare_imgsz.py 用）
- stdout 一路打 markdown 表格，可直接贴进文档

已知坑：--split 必须是 val。ultralytics 会把 --split 的值直接拼成 data.yaml
里的键名去查路径，写成 valid 会抛
`FileNotFoundError: data.yaml 'valid:' images not found`
——GC10 的目录名是 valid，但 data.yaml 里的键名是 val，两者不一样。
"""
import sys, os, argparse
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["YOLO_CONFIG_DIR"] = os.path.join(BASE, "yolo_cfg")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE, "mpl_cfg")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
sys.path.insert(0, BASE)

import shim; shim.apply()
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val",
                    help="data.yaml 里的键名，默认 val。必须是 val 而不是 valid："
                         "目录名是 valid，但 data.yaml 里写的键是 val，"
                         "传 valid 会抛 FileNotFoundError")
    ap.add_argument("--name", default="val",
                    help="runs/detect/ 下的输出目录名，默认 val；640 对照常用 val640")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="评测分辨率，默认 320；必须与权重训练时的 imgsz 一致")
    ap.add_argument("--batch", type=int, default=16, help="batch size，默认 16")
    ap.add_argument("--weights", default=os.path.join(BASE, "runs", "detect", "train", "weights", "best.pt"),
                    help="要评测的 .pt 权重，默认 320 训练目录下的 best.pt")
    a = ap.parse_args()

    m = YOLO(a.weights)
    r = m.val(data=os.path.join(BASE, "data", "gc10", "data.yaml"),
              split=a.split,
              imgsz=a.imgsz, batch=a.batch, device=0, workers=0,
              project=os.path.join(BASE, "runs", "detect"), name=a.name, exist_ok=True,
              plots=True)

    print()
    print(f"SPLIT {a.split}")
    print("MAP50", round(float(r.box.map50), 4))
    print("MAP50_95", round(float(r.box.map), 4))
    print("MP", round(float(r.box.mp), 4))
    print("MR", round(float(r.box.mr), 4))

    names = r.names
    rows = []
    for i, cid in enumerate(r.box.ap_class_index):
        rows.append((names[int(cid)], float(r.box.p[i]), float(r.box.r[i]),
                     float(r.box.ap50[i]), float(r.box.ap[i])))
    rows.sort(key=lambda x: -x[4])
    print()
    print("| 类别 | P | R | mAP@0.5 | mAP@0.5:0.95 |")
    print("|---|---|---|---|---|")
    for n, p, rc, a50, ap_ in rows:
        print(f"| {n} | {p:.3f} | {rc:.3f} | {a50:.3f} | {ap_:.3f} |")
    print(f"| **all** | {float(r.box.mp):.3f} | {float(r.box.mr):.3f} | "
          f"{float(r.box.map50):.3f} | {float(r.box.map):.3f} |")

    # 落一份 json，给 bench/compare_imgsz.py 做逐类对比用
    import json
    out = os.path.join(BASE, "runs", "detect", a.name)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "per_class.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({n: round(a50, 6) for n, _, _, a50, _ in rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n逐类 mAP@0.5 已写入 {p}")


if __name__ == "__main__":
    main()
