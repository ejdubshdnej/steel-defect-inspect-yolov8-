"""检查 exports/ 下 ONNX 的结构信息，并与 torch 输出做数值一致性对比。

用法：
    python verify_onnx.py                    # 只查 best_*
    python verify_onnx.py best640            # 只查 best640_*
    python verify_onnx.py best best640       # 两组都查

注意事项：
- imgsz 从 ONNX 的 input shape 里自读，不要写死 320：拿 640 的导出件核对时，
  写死 320 会直接抛 InvalidArgument（Got: 320 Expected: 640）。
- FP16 模型必须喂 float16 输入，否则 onnxruntime 报类型不匹配。
- 一致性只对 FP32 有严格意义（max abs diff ~6e-04）；FP16 的差异是量化的
  固有代价（max abs diff ~1e+00），不是 bug。
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
import onnx, numpy as np

EXP = os.path.join(BASE, "exports")


def shape_of(vi):
    tt = vi.type.tensor_type
    dt = onnx.TensorProto.DataType.Name(tt.elem_type)
    dims = [d.dim_value for d in tt.shape.dim]
    return dt, dims


def info(p):
    m = onnx.load(p)
    print(f"--- {os.path.basename(p)}  ({os.path.getsize(p)} B) ---")
    print("ir_version", m.ir_version, "| opset", m.opset_import[0].version)
    for i in m.graph.input:
        dt, dims = shape_of(i)
        print("input ", i.name, dims, dt)
    for o in m.graph.output:
        dt, dims = shape_of(o)
        print("output", o.name, dims, dt)
    print("nodes", len(m.graph.node))
    return m, shape_of(m.graph.input[0])


def numpy_dtype(dt):
    return np.float16 if dt == "FLOAT16" else np.float32


def check_consistency(pt_path, onnx_path, imgsz=None):
    """torch 输出与 onnxruntime 输出的最大/平均绝对差。

    imgsz 默认 None = 从 ONNX 的 input shape 自读。写死 320 的话，
    拿 640 导出件来核对会直接抛 InvalidArgument（Got: 320 Expected: 640）。
    """
    import numpy as np
    import onnxruntime as ort
    from ultralytics import YOLO

    _, (dt, dims) = info(onnx_path)
    if imgsz is None:
        imgsz = dims[2] if isinstance(dims[2], int) and dims[2] > 0 else 320
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, imgsz, imgsz)).astype(numpy_dtype(dt))
    o_ort = sess.run(None, {iname: x})[0]

    m = YOLO(pt_path)
    import torch
    t = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        o_t = m.model(t)[0].numpy()
    d = np.abs(o_t.astype(np.float32) - o_ort.astype(np.float32))
    print(f"  consistency vs torch: max {d.max():.4e}  mean {d.mean():.4e}  "
          f"shapes {o_t.shape} / {o_ort.shape}")
    return float(d.max())


if __name__ == "__main__":
    # 支持通过命令行指定要核对的前缀，例如：
    #   python verify_onnx.py                    # 只查 best_*
    #   python verify_onnx.py best640            # 只查 best640_*
    #   python verify_onnx.py best best640       # 两组都查
    # 每组的 .pt 从 --weights 推断，否则默认 320 训练目录（best640 用 640 目录）。
    import sys as _s
    prefixes = _s.argv[1:] or ["best"]
    for pre in prefixes:
        files = [f for f in (f"{pre}_fp32.onnx", f"{pre}_fp16.onnx")
                 if os.path.exists(os.path.join(EXP, f))]
        if not files:
            print(f"跳过 {pre}: exports/ 下没有 {pre}_*.onnx")
            continue
        for n in files:
            info(os.path.join(EXP, n))
            print()
        pt = os.path.join(BASE, "runs", "detect",
                          "train640" if "640" in pre else "train",
                          "weights", "best.pt")
        if os.path.exists(pt):
            for n in files:
                print(f"consistency {n}")
                check_consistency(pt, os.path.join(EXP, n))
                print()
