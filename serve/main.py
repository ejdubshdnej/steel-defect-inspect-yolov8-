"""FastAPI 推理服务：POST /predict 上传图片，返回检测框。

用法：
    python serve/main.py                     # 默认监听 127.0.0.1:8000
    set DSI_PORT=8001 && python serve/main.py

环境变量（都可选）：
    DSI_ONNX   ONNX 模型路径，默认 exports/best_fp32.onnx；相对路径按项目根解析
    DSI_CONF   置信度阈值，默认 0.25，会被夹到 [0.01, 0.99]
    DSI_PORT   监听端口，默认 8000（README 与文档里所有实测数字都用这个端口）

接口：
    GET  /health   -> {"ok": true, "model": ..., "imgsz": 320, "conf": 0.25, "classes": 10}
    POST /predict  multipart 上传，字段名必须是 file
                   -> {"detections": [...], "elapsed_ms": 3.4, "image_size": [w, h]}

已知坑：/predict 的表单字段名是 file。写成 image 会直接返回 422
（FastAPI 校验失败），不是模型的问题。
"""
import sys, os, io, time
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import numpy as np, cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from serve.infer import Detector

NAMES = {0: "crease", 1: "crescent_gap", 2: "inclusion", 3: "oil_spot",
         4: "punching_hole", 5: "rolled_pit", 6: "silk_spot",
         7: "waist_folding", 8: "water_spot", 9: "welding_line"}

DET = None

# 模型路径可用环境变量覆盖，方便在 320（默认，快）和 640（高精度，慢 3.4 倍）
# 之间切换而不改代码。相对路径按项目根解析，也接受绝对路径。
#   set DSI_ONNX=exports\best640_fp32.onnx   (Windows cmd)
#   $env:DSI_ONNX="exports/best640_fp32.onnx" (PowerShell)
_onnx_env = os.environ.get("DSI_ONNX", "").strip()
MODEL_NAME = os.path.basename(_onnx_env) if _onnx_env else "best_fp32.onnx"
MODEL_PATH = (_onnx_env if os.path.isabs(_onnx_env) else os.path.join(BASE, _onnx_env)) \
    if _onnx_env else os.path.join(BASE, "exports", MODEL_NAME)

# 置信度阈值也可用环境变量覆盖。阈值不是纯技术参数而是成本参数：
# 漏检的代价（客户投诉/停线）与误检的代价（人工复判工时）哪个大，决定取多少。
#   set DSI_CONF=0.6   更严，只信高置信度，检出少误报少
#   set DSI_CONF=0.05  更松，尽量不漏，检出多误报多
try:
    CONF = float(os.environ.get("DSI_CONF", "").strip() or 0.25)
except ValueError:
    print("DSI_CONF 不是合法数字，回退到 0.25")
    CONF = 0.25
CONF = min(max(CONF, 0.01), 0.99)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global DET
    p = MODEL_PATH                            # CPU 部署用 FP32，比 FP16 快且无需类型适配
    if not os.path.exists(p):
        raise FileNotFoundError(f"模型不存在: {p}（检查 DSI_ONNX 环境变量）")
    t = time.time()
    DET = Detector(p, conf=CONF, names=NAMES)
    print(f"model loaded in {time.time()-t:.2f}s  imgsz {DET.imgsz[0]}  conf {CONF}  {MODEL_NAME}")
    yield
    DET = None


app = FastAPI(title="steel-defect-inspect", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    # imgsz 从模型自读，不写死——换 640 模型时这里会自动返回 640
    return {"ok": DET is not None,
            "model": MODEL_NAME,
            "imgsz": DET.imgsz[0] if DET is not None else None,
            "conf": CONF,
            "classes": len(NAMES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 表单字段名固定为 file；客户端写成 image 会 422
    if DET is None:
        raise HTTPException(503, "model not loaded")
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "cannot decode image")
    h, w = img.shape[:2]
    res = DET.predict(img)
    return JSONResponse({**res, "image_size": [w, h]})


if __name__ == "__main__":
    import uvicorn
    # 端口可用 DSI_PORT 覆盖，方便同时起多个实例做阈值/模型对比
    # （8000 是默认值，也是 README 与文档里所有实测数字所用的端口）
    try:
        _port = int(os.environ.get("DSI_PORT", "").strip() or 8000)
    except ValueError:
        _port = 8000
    uvicorn.run(app, host="127.0.0.1", port=_port)
