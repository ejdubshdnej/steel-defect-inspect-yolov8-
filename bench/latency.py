"""测 CPU 推理延迟。三个层次：纯推理 / 端到端 / 大图端到端。

用法：
  python bench/latency.py                       # 默认 exports/best_fp32.onnx（CPU 部署用这个）
  python bench/latency.py exports/best_fp16.onnx
  python bench/latency.py --repeats 3           # 多跑几遍取中位数，抑制冷启动与长跑降频的干扰

注意事项：
- 输入尺寸从 ONNX 的 input shape 自读，同一套代码既能测 320 也能测 640 的导出件。
- 首次跑偏高是冷启动污染；但更需要注意的是长跑劣化——前两遍稳定，第三遍起
  onnxruntime 线程池被系统降频，p50 会跳高。所以多轮测量要取中位数，不要取末轮。
"""
import sys, os, time, platform
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, onnxruntime as ort

IMGSZ = 320
N_WARM, N_RUN = 10, 100

# 从模型输入 shape 自读的实际尺寸，bench()/preprocess() 用这个而不是常量，
# 这样同一套代码既能测 320 也能测 640 的导出件。
SZ = IMGSZ


def letterbox(img, new=None, color=114):
    new = new or SZ
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (new - nh) // 2, (new - nw) // 2
    out = np.full((new, new, 3), color, np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return out, r, (left, top)


def preprocess(img, dtype=np.float32):
    lb, r, pad = letterbox(img)
    x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(dtype) / 255.0
    return np.ascontiguousarray(x), r, pad


def postprocess(out, r, pad, conf=0.25, nc=10):
    """out: [1, 4+nc, 2100]"""
    p = out[0].T                       # [2100, 4+nc]
    scores = p[:, 4:]
    cls = scores.argmax(1)
    sc = scores[np.arange(len(p)), cls]
    keep = sc > conf
    if not keep.any():
        return []
    box, sc, cls = p[keep, :4], sc[keep], cls[keep]
    cx, cy, w, h = box.T
    x1, y1 = cx - w / 2, cy - h / 2
    l, t = pad
    return np.stack([(x1 - l) / r, (y1 - t) / r,
                     (x1 + w - l) / r, (y1 + h - t) / r], 1), sc, cls


def stat(name, ms):
    a = np.array(ms)
    print(f"{name:28s} mean {a.mean():7.2f}  p50 {np.percentile(a,50):7.2f}  "
          f"p95 {np.percentile(a,95):7.2f}  p99 {np.percentile(a,99):7.2f} ms")
    return {"name": name, "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "n": len(ms)}


def bench(model_path, threads=4, tag=""):
    global SZ
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    it = sess.get_inputs()[0]
    iname = it.name
    dtype = np.float16 if "float16" in it.type else np.float32
    if isinstance(it.shape[2], int) and isinstance(it.shape[3], int):
        SZ = it.shape[2]
    print(f"model {os.path.basename(model_path)}  {tag}")
    print("cpu", platform.processor(), "| threads", so.intra_op_num_threads,
          "| input", it.type, "| imgsz", SZ, "| n", N_RUN)

    # --- 1 纯推理 ---
    x = np.random.rand(1, 3, SZ, SZ).astype(dtype)
    for _ in range(N_WARM):
        sess.run(None, {iname: x})
    t = []
    for _ in range(N_RUN):
        s = time.perf_counter(); sess.run(None, {iname: x}); t.append((time.perf_counter() - s) * 1e3)
    r1 = stat("inference only", t)

    # --- 2 端到端（小图 200x200）---
    img200 = (np.random.rand(200, 200) * 255).astype(np.uint8)
    r2 = e2e(img200, sess, iname, dtype, "e2e 200x200")

    # --- 3 端到端（大图 2048x1000）---
    imgbig = (np.random.rand(1000, 2048) * 255).astype(np.uint8)
    r3 = e2e(imgbig, sess, iname, dtype, "e2e 2048x1000")
    return [r1, r2, r3]


def e2e(img, sess, iname, dtype, name):
    for _ in range(N_WARM):
        x, r, pad = preprocess(img, dtype)
        out = sess.run(None, {iname: x})[0]
        postprocess(out, r, pad)
    t = []
    for _ in range(N_RUN):
        s = time.perf_counter()
        x, r, pad = preprocess(img, dtype)
        out = sess.run(None, {iname: x})[0]
        postprocess(out, r, pad)
        t.append((time.perf_counter() - s) * 1e3)
    return stat(name, t)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=os.path.join(BASE, "exports", "best_fp32.onnx"),
                    help="ONNX 模型路径，默认 exports/best_fp32.onnx")
    ap.add_argument("--threads", type=int, default=4,
                    help="onnxruntime intra-op 线程数，默认 4（与参考硬件的核数一致，便于复现）")
    ap.add_argument("--repeats", type=int, default=1,
                    help="跑几遍。注意：参考环境上是「前两遍稳定、第 3 遍起劣化」"
                         "（onnxruntime 线程池长跑被系统降频），故不是丢弃第一遍，"
                         "而是要丢弃末遍；跨模型对比请用交替测量取中位数")
    a = ap.parse_args()

    if a.repeats <= 1:
        bench(a.model, a.threads)
        return
    runs = []
    for i in range(a.repeats):
        print(f"\n===== pass {i+1}/{a.repeats} =====")
        runs.append(bench(a.model, a.threads))

    # 取各轮中位数作为稳定值。
    # 注意：不能取「最后一轮」——参考环境上第 3 轮起 p50 会跳高（onnxruntime
    # 线程池长跑被系统降频/抢占），取末轮等于把劣化当成稳定值。
    n = len(runs)
    print(f"\n===== stable (median of {n} passes) =====")
    import statistics
    for j in range(len(runs[0])):
        med = {k: statistics.median([p[j][k] for p in runs])
               for k in ("mean", "p50", "p95", "p99")}
        print(f"{runs[0][j]['name']:28s} mean {med['mean']:7.2f}  p50 {med['p50']:7.2f}  "
              f"p95 {med['p95']:7.2f}  p99 {med['p99']:7.2f} ms")

    if n > 1:
        print("\n===== pass-to-pass p50 drift =====")
        for j in range(len(runs[0])):
            vals = [f"{p[j]['p50']:.2f}" for p in runs]
            print(f"{runs[0][j]['name']:28s} p50 per pass: {' / '.join(vals)} ms")
        print("\n提示：若各轮 p50 呈单调上升，说明长跑劣化已发生，本次中位数可信度下降；"
              "\n      跨模型对比请用「两模型交替跑、每轮各测一遍」的方式，而不是各跑各的。")


if __name__ == "__main__":
    main()
