"""320 vs 640 消融对比：逐 epoch mAP 曲线 + 逐类指标 + 结论建议。

用法：
    python bench/compare_imgsz.py

读两份 results.csv（runs/detect/train 与 runs/detect/train640），产出：
  1. 收敛对比（best / final / 达标 epoch 数）
  2. 逐 epoch 差值（640 - 320），看 640 是稳定领先还是震荡
  3. 逐类 mAP 对比（需要先跑过两份 scripts/eval.py，把 per_class.json 存下来才有；否则跳过）
  4. 一句话建议

这个脚本不训练、不改模型，只读数据，可以随时重跑。
"""
import sys, os, json, csv
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(BASE, "runs", "detect", "train", "results.csv")
B = os.path.join(BASE, "runs", "detect", "train640", "results.csv")


def load(p):
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            e = int(row["epoch"])
            out[e] = {
                "m50": float(row["metrics/mAP50(B)"]),
                "m5095": float(row["metrics/mAP50-95(B)"]),
                "t": float(row["time"]),
            }
    return out


def stat(d, tag):
    """打印一组训练结果的关键统计。

    注意 best 的两套口径：
      best50  = mAP@0.5 峰值那一轮（看起来更漂亮，但不是存盘的那个）
      best95  = mAP@0.5:0.95 峰值那一轮 = **best.pt 的真实来源**
    ultralytics `utils/metrics.py:1016` 的 w = [0, 0, 0, 1.0] 决定 fitness 就是
    mAP@0.5:0.95，mAP@0.5 不参与选模。所以要比"最终发布哪个模型"，
    必须看 best95，看 best50 会得出错误结论。
    """
    e50 = max(d, key=lambda e: d[e]["m50"])
    e95 = max(d, key=lambda e: d[e]["m5095"])
    last_e = max(d)
    print(f"{tag:>6}  epochs {len(d):3d}  "
          f"best50 {d[e50]['m50']:.4f} @ep{e50:<3d}  "
          f"best95 {d[e95]['m5095']:.4f} @ep{e95:<3d}  "
          f"(m50 @best95 = {d[e95]['m50']:.4f})  "
          f"final {d[last_e]['m50']:.4f}  "
          f"{d[last_e]['t']/60:.1f} min")
    return e95, d[e95]["m50"], d[e95]["m5095"]


def main():
    if not os.path.exists(B):
        print(f"还没有 640 的结果：{B}")
        return 1

    a, b = load(A), load(B)
    print("=" * 104)
    print("（best95 = 按 mAP@0.5:0.95 选出，这就是 best.pt 的真实来源；best50 仅供对照）")
    be_a, m_a, m95_a = stat(a, "320")
    be_b, m_b, m95_b = stat(b, "640")
    print("=" * 104)

    n = min(max(a), max(b))
    print(f"\n共同完成的 epoch 数：{n}（两份都跑到的）\n")

    # ---- 逐 epoch 差值 ----
    print("逐 epoch 差值（640 - 320，单位 mAP@0.5）：")
    print(" epoch   320     640     diff    cum")
    wins, diffs = 0, []
    for e in sorted(b):
        if e > n:
            break
        d = b[e]["m50"] - a[e]["m50"]
        diffs.append(d)
        if d > 0:
            wins += 1
        flag = "  <<" if abs(d) >= 0.05 else ""
        if e % 5 == 0 or e == n:
            print(f" {e:5d}  {a[e]['m50']:.4f}  {b[e]['m50']:.4f}  {d:+.4f}  "
                  f"{sum(diffs)/len(diffs):+.4f}{flag}")
    print(f"\n  640 领先的 epoch：{wins}/{n}（{wins/n*100:.0f}%）")
    print(f"  平均差值：{sum(diffs)/len(diffs):+.4f}")
    print(f"  差值标准差：{(sum((x-sum(diffs)/len(diffs))**2 for x in diffs)/len(diffs))**0.5:.4f}")

    # ---- 结论 ----
    print("\n" + "=" * 96)
    dm = m_b - m_a
    rel = dm / m_a * 100
    print(f"最终对比：320 best {m_a:.4f}  vs  640 best {m_b:.4f}  "
          f"差 {dm:+.4f}（{rel:+.1f}%）")

    if dm >= 0.03:
        verdict = "640 明显更优（+3pp 以上），值得为了精度牺牲延迟"
    elif dm >= 0.01:
        verdict = "640 略优（+1~3pp），需要用延迟预算来权衡"
    elif dm > -0.01:
        verdict = "两者持平（±1pp 内），收益不足以抵消延迟翻倍"
    else:
        verdict = "320 反而更好，640 的额外分辨率没有转化为精度"
    print(f"判断：{verdict}")

    print("\n延迟对照（参考硬件上实测，见 docs/EVAL.md 的延迟小节）：")
    print("  320  e2e 2048x1000 p50 5.54 ms / p95 6.78 ms / p99 7.03 ms")
    print("  640  e2e 2048x1000 p50 23.49 ms / p95 24.66 ms / p99 25.14 ms"
          "（纯推理 12.73 ms）")
    print("=" * 96)

    # ---- 逐类对比（可选）----
    ja = os.path.join(BASE, "runs", "detect", "val", "per_class.json")
    jb = os.path.join(BASE, "runs", "detect", "val640", "per_class.json")
    if os.path.exists(ja) and os.path.exists(jb):
        with open(ja, encoding="utf-8") as f:
            pa = json.load(f)
        with open(jb, encoding="utf-8") as f:
            pb = json.load(f)
        print("\n逐类 mAP@0.5 对比：")
        print(f"| {'类别':<14} | {'320':>6} | {'640':>6} | {'diff':>7} |")
        print("|---|---|---|---|")
        for k in sorted(pa, key=lambda x: -pa[x]):
            d = pb.get(k, 0) - pa[k]
            print(f"| {k:<14} | {pa[k]:.3f} | {pb.get(k,0):.3f} | {d:+.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
