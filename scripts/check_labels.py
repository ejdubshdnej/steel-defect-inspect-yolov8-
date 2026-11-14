"""标注质检：越界框 / 微小框 / 缺失标注 / 空标注 / 坏格式 / 重复图（phash）。

用法：
    python scripts/check_labels.py

输出：
- 控制台摘要：每个 split 的图片数、框数、各类异常计数、逐类实例数、重复图对
- docs/label_check.csv：各 split 的统计表，便于随 PR 一起归档核对

依赖 Pillow 与 ImageHash（`pip install pillow imagehash`）。phash 阶段会读遍
所有图片，在 2300 张的规模上大约需要几十秒。

已知现象：GC10-DET 原图是 2048x1000 的单通道灰度图，部分标注框紧贴图像边界，
判越界时留了 ±1 像素的容差，否则会把浮点误差算成越界。
"""
import sys, os, glob, csv
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(BASE, "data", "gc10")
SPLITS = ("train", "valid", "test")


def img_size(p):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def check_split(split):
    imgs = sorted(glob.glob(os.path.join(DS, "images", split, "*.jpg")))
    stats = {"imgs": len(imgs), "boxes": 0, "oob": 0, "tiny": 0,
             "nolabel": 0, "empty": 0, "badcls": 0, "badfmt": 0}
    per_cls = {}
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(DS, "labels", split, stem + ".txt")
        if not os.path.exists(lp):
            stats["nolabel"] += 1
            continue
        lines = [l.strip() for l in open(lp, encoding="utf-8") if l.strip()]
        if not lines:
            stats["empty"] += 1
            continue
        W, H = img_size(ip)
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                stats["badfmt"] += 1
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError:
                stats["badfmt"] += 1
                continue
            stats["boxes"] += 1
            per_cls[c] = per_cls.get(c, 0) + 1
            if c < 0 or c > 9:
                stats["badcls"] += 1
            pw, ph = w * W, h * H
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            if x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1:
                stats["oob"] += 1
            if pw <= 1 or ph <= 1:
                stats["tiny"] += 1
    return stats, per_cls


def find_dupes(paths, threshold=5):
    import imagehash
    from PIL import Image
    hashes, dupes = {}, []
    for p in paths:
        try:
            h = imagehash.phash(Image.open(p))
        except Exception:
            continue
        for q, hq in hashes.items():
            if h - hq <= threshold:
                dupes.append((p, q, int(h - hq)))
        hashes[p] = h
    return dupes


NAMES = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
         "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"]


def main():
    tot = {"imgs": 0, "boxes": 0, "oob": 0, "tiny": 0, "nolabel": 0,
           "empty": 0, "badcls": 0, "badfmt": 0}
    grand_cls = {}
    rows = []
    for s in SPLITS:
        st, per = check_split(s)
        rows.append(dict(split=s, **st))
        print(f"[{s:5s}] 图 {st['imgs']:5d}  框 {st['boxes']:5d}  越界 {st['oob']:3d}  "
              f"微小 {st['tiny']:3d}  无标注 {st['nolabel']:3d}  空标注 {st['empty']:3d}  "
              f"坏类别 {st['badcls']:2d}  坏格式 {st['badfmt']:2d}")
        for k in tot:
            tot[k] += st[k]
        for c, n in per.items():
            grand_cls[c] = grand_cls.get(c, 0) + n
    print(f"[合计 ] 图 {tot['imgs']:5d}  框 {tot['boxes']:5d}  越界 {tot['oob']:3d}  "
          f"微小 {tot['tiny']:3d}  无标注 {tot['nolabel']:3d}  空标注 {tot['empty']:3d}")
    print("\n逐类实例数：")
    for c, n in sorted(grand_cls.items()):
        nm = NAMES[c] if 0 <= c < len(NAMES) else f"?{c}"
        print(f"  {c} {nm:15s} {n:5d}")

    all_imgs = []
    for s in SPLITS:
        all_imgs += sorted(glob.glob(os.path.join(DS, "images", s, "*.jpg")))
    dupes = find_dupes(all_imgs)
    print(f"\n重复图（phash 汉明距离 ≤5）：{len(dupes)} 对")
    for a, b, d in dupes[:10]:
        print(f"  d={d} {os.path.basename(a)} ~ {os.path.basename(b)}")

    csv_path = os.path.join(BASE, "docs", "label_check.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "imgs", "boxes", "oob", "tiny",
                                          "nolabel", "empty", "badcls", "badfmt"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n报告已写 {csv_path}")


if __name__ == "__main__":
    main()
