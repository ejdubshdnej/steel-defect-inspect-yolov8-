"""在验证集上按固定置信度阈值比较两个模型的召回——这才是产线的实际工作点。

ultralytics 报的 P/R 是「F1 最大点」的值，会随模型自己漂移。
产线上操作员定的是固定阈值（上位机默认 0.25），所以要在同一 conf 下比召回：
同一条产线，同样的阈值，谁漏得少。

用法：
  python bench/recall_at_conf.py

依赖 exports/best_fp32.onnx 与 exports/best640_fp32.onnx 两份导出件，
以及 data/gc10 下的验证集图片与标注。缺一个会直接报错退出。
"""
import sys, os, glob
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "serve"))

import numpy as np
import cv2
from infer import Detector

VAL_IMG = os.path.join(BASE, "data", "gc10", "images", "valid")
VAL_LBL = os.path.join(BASE, "data", "gc10", "labels", "valid")
CONFS = [0.05, 0.10, 0.25, 0.40]


def load_gt(name):
    p = os.path.join(VAL_LBL, name.replace(".jpg", ".txt"))
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        s = line.split()
        if len(s) >= 5:
            out.append((int(s[0]), *map(float, s[1:5])))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def run(onnx_path, tag, imgsz_hint):
    # conf 是构造参数，这里取最低一档，之后自己按阈值过滤，
    # 这样一次推理就能同时算出所有阈值下的召回（省 4 倍时间）。
    det = Detector(onnx_path, conf=min(CONFS))
    files = sorted(glob.glob(os.path.join(VAL_IMG, "*.jpg")))
    # 每个 conf 一档：命中数 / GT 总数
    hit = {c: 0 for c in CONFS}
    tot = 0
    for f in files:
        img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_GRAYSCALE)
        gts = load_gt(os.path.basename(f))
        tot += len(gts)
        boxes = det.predict(img)["detections"]
        for c in CONFS:
            used = [False] * len(gts)
            for b in boxes:
                if b["conf"] < c:
                    continue
                best, bi = 0.0, -1
                for i, (cls, xc, yc, w, h) in enumerate(gts):
                    if used[i] or cls != b["cls"]:
                        continue
                    H, W = img.shape[:2]
                    g = [xc * W - w * W / 2, yc * H - h * H / 2,
                         xc * W + w * W / 2, yc * H + h * H / 2]
                    v = iou(b["xyxy"], g)
                    if v > best:
                        best, bi = v, i
                if best >= 0.5 and bi >= 0:
                    used[bi] = True
                    hit[c] += 1
    print(f"\n{tag}  (imgsz {imgsz_hint}, {len(files)} 张, GT {tot} 个框)")
    print(f"{'conf':>6}  {'命中':>5}  {'召回':>7}")
    for c in CONFS:
        print(f"{c:>6.2f}  {hit[c]:>5}  {hit[c]/max(tot,1):>7.4f}")
    return {c: hit[c] / max(tot, 1) for c in CONFS}


if __name__ == "__main__":
    a = run(os.path.join(BASE, "exports", "best_fp32.onnx"), "320 模型", 320)
    b = run(os.path.join(BASE, "exports", "best640_fp32.onnx"), "640 模型", 640)
    print("\n" + "=" * 52)
    print(f"{'conf':>6}  {'320':>8}  {'640':>8}  {'差(640-320)':>12}")
    for c in CONFS:
        print(f"{c:>6.2f}  {a[c]:>8.4f}  {b[c]:>8.4f}  {b[c]-a[c]:>+12.4f}")
