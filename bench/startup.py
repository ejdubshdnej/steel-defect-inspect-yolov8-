"""实测启动耗时：模型加载 + 首次推理 + 端口可用。

用法：
    python bench/startup.py

输出四段耗时（进程内视角）：
  1. import numpy + onnxruntime + opencv
  2. Detector 构造（模型加载 + 内部 warmup(3)）
  3. 加载后第一次真实推理
  4. 之后连续 20 次的平均

docs/DEPLOY_NOTES.md 里的启动时间表就是这份脚本的输出，不是估的。
注意它测的是「进程内」耗时；从进程被拉起、到 HTTP 端口可用，那一段用
bench/startup_http.py 测。
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

t0 = time.perf_counter()
import numpy as np
t_import = time.perf_counter() - t0

from serve.infer import Detector

NAMES = {0: "crease", 1: "crescent_gap", 2: "inclusion", 3: "oil_spot",
         4: "punching_hole", 5: "rolled_pit", 6: "silk_spot",
         7: "waist_folding", 8: "water_spot", 9: "welding_line"}

p = os.path.join(BASE, "exports", "best_fp32.onnx")
t = time.perf_counter()
det = Detector(p, names=NAMES)          # __init__ 内部已 warmup(3)，所以这段含预热开销
t_load = time.perf_counter() - t

z = np.zeros((det.imgsz[0], det.imgsz[1]), np.uint8)
t = time.perf_counter()
r = det.predict(z)
t_first = time.perf_counter() - t

t = time.perf_counter()
for _ in range(20):
    det.predict(z)
t_after = (time.perf_counter() - t) / 20

print(f"import numpy+onnxruntime+opencv   {t_import*1e3:8.2f} ms")
print(f"模型加载 + warmup(3)              {t_load*1e3:8.2f} ms   <- 对应 DEPLOY_NOTES 里的「模型加载」一项")
print(f"加载后首次推理                    {t_first*1e3:8.2f} ms")
print(f"之后平均每次推理                  {t_after*1e3:8.2f} ms")
print(f"imgsz {det.imgsz}  检出 {len(r['detections'])} 个框（零张量，预期 0）")
