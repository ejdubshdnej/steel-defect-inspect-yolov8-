"""生成三份 splits 清单（train/val/test），并核对与磁盘图片数一致。

用法：
    python scripts/make_splits.py

产物：splits/train.txt、splits/val.txt、splits/test.txt，每行一个不含扩展名的
图片文件名（只包含「图片与标注都存在」的那部分）。

GC10-DET 的 hf-mirror 版本已经分好 train/valid/test，本脚本只做清点与落盘，
不重新划分——重新划分会与上游划分冲突，导致指标与公开结果不可比。

注意目录名与键名不一致：磁盘上是 valid，落盘的清单文件名是 val.txt，
这与 data.yaml 里 `val:` 的键名保持一致，不是笔误。
"""
import sys, os, glob
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(BASE, "data", "gc10")
OUT = os.path.join(BASE, "splits")
os.makedirs(OUT, exist_ok=True)

MAP = {"train": "train.txt", "valid": "val.txt", "test": "test.txt"}

total = 0
for split, fn in MAP.items():
    ip = os.path.join(DS, "images", split)
    lp = os.path.join(DS, "labels", split)
    imgs = sorted(glob.glob(os.path.join(ip, "*.jpg")))
    lbls = sorted(glob.glob(os.path.join(lp, "*.txt")))
    stems_img = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
    stems_lbl = {os.path.splitext(os.path.basename(p))[0] for p in lbls}
    paired = sorted(stems_img & stems_lbl)
    with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
        for s in paired:
            f.write(s + "\n")
    print(f"[{split:5s}] images {len(imgs):5d}  labels {len(lbls):5d}  "
          f"paired {len(paired):5d}  img-only {len(stems_img-stems_lbl)}  "
          f"lbl-only {len(stems_lbl-stems_img)}  -> splits/{fn}")
    total += len(paired)
print(f"合计写入 splits {total} 行")
