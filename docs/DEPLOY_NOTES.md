# 部署说明 · 工业表面缺陷检测（Deployment Notes）

> 这份文档回答的是「怎么把它装到产线上」，不是「怎么跑起来」——跑起来的命令见 [README](../README.md)。
>
> 全文按「先给结论，再说依据」组织。文中每一个延迟与体积数字都是**本测试环境实测**：AMD Ryzen 7 9800X3D（4 线程，CPU 推理）+ NVIDIA RTX 5070 12GB（仅训练用），Windows，Python 3.x，onnxruntime 1.31.1，torch 2.14.1+cu130，ultralytics 8.4.211。完整清单见第 7 节。

---

## 1 部署形态（Deployment topology）

### 1.1 三种可选形态（Three topologies）

| 形态 | 适用场景 | 优点 | 缺点 | 本项目的支持程度 |
|---|---|---|---|---|
| **A · 单机 exe** | 单条产线、一台工控机、离线环境 | 零网络依赖，装完即用；无鉴权和端口暴露风险 | 更新要搬 U 盘；无法集中管理 | 需自行 PyInstaller 打包，本项目未提供（见第 8 节末尾） |
| **B · HTTP 服务**（已实现） | 多工位共用一台推理机；或要接 MES | 一处更新全局生效；容易做监控和日志 | 需要鉴权、限流、健康检查；网络抖动敏感 | **`serve/main.py`，FastAPI + uvicorn** |
| **C · 嵌入产线 MES** | 已有 MES/SCADA 的成熟产线 | 与生产管理系统打通，缺陷数据直接进质量台账 | 改造成本最高，要和别人家的系统对接 | 提供 HTTP 接口，由 MES 侧调用 `/predict` |

**本项目的定位：B 为主，C 为延伸，A 作为离线兜底方案。**

选择 B 的理由：产线上相机采集和推理通常在**同一台工控机**上（图像不走网络，只传结果），此时 B 和 A 的延迟差别可以忽略；但 B 能集中更新模型、集中看日志，运维成本低一个数量级。

### 1.2 推荐架构（Reference architecture）

```text
[工业相机] --GigE/USB3--> [工控机 Windows]
                              |
                              +-- 采集服务（厂商提供，负责触发和存图）
                              |
                              +-- 本项目 FastAPI (127.0.0.1:8000)
                              |      |
                              |      +-- best_fp32.onnx  (12.0 MB)
                              |      +-- onnxruntime CPU 推理
                              |
                              +-- PyQt6 上位机（操作员界面）
                              |      +-- 显示原图 + 红框 + 表格
                              |      +-- 阈值滑块（工艺工程师调）
                              |
                              +-- 结果落库（SQLite / 推给 MES）
```

**关键设计**：推理服务与上位机**可以分离，也可以合在一起**。`hmi/app.py` 默认在进程内直接加载 `exports/best_fp32.onnx`（`from serve.infer import Detector`，不走 HTTP），所以单机形态下上位机不依赖服务活着；如果需要多工位共用一个推理服务，再把上位机改成 HTTP 客户端指向 `serve/main.py` 即可，检测逻辑都在 `serve/infer.py` 里，两种调用方式共用同一份实现。

这样分工的好处是：服务重启不影响已经打开的上位机，上位机关掉也不影响 `serve/main.py` 继续对外提供 `/predict`，两个组件可以独立升级。

---

## 2 数据集配置（Dataset setup）

### 2.1 先准备 `data/gc10/data.yaml`（Create the dataset config）

仓库里的 `data/` 目录被 `.gitignore` 排除（数据集 928 MB，版权归原作者，本项目不重新分发），所以 `data/gc10/data.yaml` **不在仓库里**，克隆后必须自己建一份。先用下载脚本把图拉下来：

```bash
python scripts/download_gc10.py        # 从 hf-mirror 拉 GC10-DET（2300 图 / 2300 标注，928 MB）
python scripts/make_splits.py          # 清点并生成 splits/{train,val,test}.txt
```

然后在 `data/gc10/` 下新建 `data.yaml`，内容如下（把 `path:` 换成使用者自己的 `data/gc10` 绝对路径）：

```yaml
# data/gc10/data.yaml
path: <你机器上 data/gc10 的绝对路径>     # 例如 D:/work/steel-defect-inspect/data/gc10
train: images/train
val: images/valid
test: images/test
nc: 10
names:
  0: crease            # 折痕
  1: crescent_gap      # 月牙形缺口
  2: inclusion         # 夹杂
  3: oil_spot          # 油斑
  4: punching_hole     # 冲孔
  5: rolled_pit        # 轧制坑
  6: silk_spot         # 丝斑
  7: waist_folding     # 腰部折
  8: water_spot        # 水斑
  9: welding_line      # 焊缝线
```

> 行尾的中文注释只是方便阅读，删掉不影响运行；`names:` 的**下标与顺序**才是模型认的类别编号，必须与标注 txt 首列严格对应。各类的中文定义见 [`LABEL_SPEC.md`](LABEL_SPEC.md) 第 1 节。

**为什么 `path:` 要用绝对路径**：Ultralytics 对相对的 `path` 有自己的解析规则——它会相对自己配置里的 `datasets_dir` 去展开，而不是相对 `data.yaml` 所在目录。结果就是同一份 yaml 换台机器、换个工作目录就跑不起来，报的还是「数据集路径不存在」这类看着像数据没下载完的错。写绝对路径最稳，也最利于排查。

**两个容易踩的细节**：

- **`val:` 写的是 `images/valid`，不是 `images/val`。** 数据集的上游划分目录名就是 `valid`，`data.yaml` 里的键名却必须叫 `val`（键名是 Ultralytics 的约定，值才是路径）。这两处都对不上就会报找不到验证集。
- **`nc: 10` 与 `names:` 的顺序即类别编号。** 顺序错一个，训练不报错但指标全歪。`scripts/check_labels.py` 会核对 label txt 首列是否落在 0–9。

> 本仓库开发时用的 `data.yaml` 除 `path:` 外与上面完全一致，`path:` 指向的是当时开发机上的目录，因此没有写进仓库。**换机器只需要改这一行。**

### 2.2 目录结构（Directory layout）

```text
data/gc10/
├── data.yaml
├── images/{train,valid,test}/   # 1840 / 230 / 230 张，2048×1000 单通道灰度
└── labels/{train,valid,test}/   # 同名 YOLO txt，共 3563 个框
```

### 2.3 训练与评测时的用法（Train & eval）

```bash
python train.py --data data/gc10/data.yaml --epochs 100 --batch 16 --imgsz 320 --name train
python scripts/eval.py --split val          # 注意是 val，不是 valid
```

---

## 3 依赖与体积（Runtime dependencies）

### 3.1 运行时依赖清单（Dependency list）

| 组件 | 版本 | 体积 | 说明 |
|---|---|---|---|
| **best_fp32.onnx** | — | **12.0 MB** | 主力模型，12,629,151 B。**不在仓库里**，从 Release 下载或自行导出 |
| best_fp16.onnx | — | 6.0 MB | 备选（GPU 场景），CPU 上不用，见 6.1 节。同样不入库 |
| onnxruntime | 1.31.1 | ~30 MB | CPU 版（`onnxruntime`，非 `onnxruntime-gpu`） |
| numpy | 2.x | ~20 MB | |
| opencv-python | 5.0.0 | ~40 MB | 只用 `cv2.resize / imdecode / dnn.NMSBoxes`，可用 `opencv-python-headless` 省 20 MB |
| fastapi + uvicorn | — | ~15 MB | 服务形态才需要 |
| PyQt6 | — | ~80 MB | 仅上位机需要（换 PySide6 体积接近） |
| **合计（服务形态）** | | **≈ 120 MB** | |
| **合计（服务 + 上位机）** | | **≈ 200 MB** | |

**体积控制建议**：

- 如果只要服务不要界面：去掉 PyQt6，约 120 MB。
- 如果用 `opencv-python-headless`：再省约 20 MB（生产环境无 GUI，headless 完全够用）。
- 不需要 PyTorch/ultralytics：**推理侧只要有 onnxruntime 即可**，PyTorch 那 2.5 GB 只在训练机上需要。

> 这条很重要：**训练环境和部署环境可以完全分离**。工控机上不需要装 CUDA，不需要装 PyTorch，一个 CPU 版 onnxruntime 就够。

### 3.2 安装命令（Install）

```bash
# 服务端（工控机）
pip install onnxruntime==1.31.1 numpy opencv-python-headless fastapi uvicorn python-multipart

# 上位机（可同一台，也可只装在操作员工位）
pip install PyQt6 requests
```

**不需要** `pip install ultralytics`，也不需要 CUDA。

### 3.3 运行时开关：640 与 320 怎么切（640 / 320 switch）

**结论：`imgsz` 不是编译期常量，是启动参数。** 默认走 320 的 `exports/best_fp32.onnx`；要切到 640，只改一个环境变量，代码一行不用动。

服务端读三个环境变量（都在进程启动时读一次，改完要重启）：

| 变量 | 作用 | 默认值 | 备注 |
|---|---|---|---|
| **`DSI_ONNX`** | 指定 ONNX 模型文件 | `exports/best_fp32.onnx`（320） | 相对路径按**项目根**解析；`/health` 返回的 `imgsz` 会跟着变成 640 |
| `DSI_CONF` | 置信度阈值 | 0.25 | 会被夹到 `[0.01, 0.99]`；非法数字回退 0.25 |
| `DSI_PORT` | 监听端口 | 8000 | 用于同时起多个实例做阈值/模型对比 |

```bash
# Windows cmd：切到 640
set DSI_ONNX=exports\best640_fp32.onnx
set DSI_CONF=0.25
python serve/main.py

# bash / PowerShell：切到 640
DSI_ONNX=exports/best640_fp32.onnx DSI_CONF=0.25 python serve/main.py

# 起第二个实例做 A/B 对比（320 与 640 并存）
set DSI_ONNX=exports\best640_fp32.onnx
set DSI_PORT=8001
python serve/main.py
```

切过去之后 `GET /health` 会报 `{"ok":true,"model":"best640_fp32.onnx","imgsz":640,"classes":10}`；切回默认则报 `imgsz: 320`。**这是验证切换是否生效的唯一可靠办法**——别靠看日志猜。

**320 与 640 的实际差别**（同一套超参，各 100 epochs，均为本测试环境实测）：

| 指标 | 320（默认） | 640 | 差距 |
|---|---|---|---|
| mAP@0.5 | 0.6248 | **0.6552** | +3.0 pp |
| 召回 | 0.6479 | 0.6562 | +0.8 pp |
| 端到端 p99（2048×1000） | 7.03 ms | 25.14 ms | **3.6 倍** |
| 训练显存 | 0.594 GB | 2.11 GB | 3.6 倍 |
| 训练耗时 | 36.7 min | 57.4 min | 1.6 倍 |
| ONNX 体积（FP32） | 12,629,151 B | 12,755,155 B | — |

**所以默认是 320 而不是 640**：3.0 pp 的 mAP 换 3.6 倍的延迟，在「产线节拍优先」的前提下不划算。但这个选择是可逆的——**如果产线的主要缺陷是小目标（inclusion / rolled_pit），就该切 640**，那 3.0 pp 里相当一部分正是补在这两类上（见第 6 节 6.2）。

> `DSI_CONF` 的取值逻辑：它不是技术参数而是**成本参数**。调高＝宁可漏检不误报（停线成本高时用），调低＝宁可误报不漏检（漏检会导致客诉时用）。默认 0.25 只是起点，投产前必须按当期质量目标重标（见第 8 节第 3 步）。

---

## 4 启动时间（Startup cost）

| 阶段 | 实测 | 说明 |
|---|---|---|
| **进程启动 → `/health` 可用** | **701.71 ms** | 含 Python 解释器启动 + 导入 + 模型加载 + warmup 全程 |
| `import numpy+onnxruntime+opencv` | **64.35 ms** | 三个包的首帧导入 |
| **ONNX 模型加载 + warmup(3)** | **71.87 ms** | 12 MB 模型读入 + onnxruntime 会话创建 + 3 次预热 |
| 加载完成后的首次真实推理 | 5.24 ms | warmup 已经把 JIT/内存分配开销吃掉 |
| 之后每次推理 | 4.70 ms | 零张量输入，无 NMS 开销，故略低于第 5 节的 3.37 ms 口径 |

（`bench/startup.py` 与 `bench/startup_http.py` 的输出，非估算。后者起子进程轮询 `/health`，从 `Popen` 到首次 200 的墙钟时间。）

**设计说明**：`serve/main.py` 在 lifespan 里做加载 + warmup，所以 `/health` 一旦返回 `ok:true`，服务就处于「随时可接受请求」的状态。这避免了第一次真实请求要等几百毫秒的抖动——产线上这个抖动会直接影响节拍。

```python
@asynccontextmanager
async def lifespan(app):
    det = Detector(ONNX)
    det.warmup(3)          # 主动预热，不把开销留给第一个真实请求
    app.state.det = det
    yield
```

---

## 5 并发能力（Throughput）

### 5.1 单实例实测（Single-instance latency）

| 场景 | p50 延迟 | 理论 QPS 上限 |
|---|---|---|
| 纯推理（320×320） | 3.37 ms | ≈ 297 |
| 端到端（200×200） | 4.43 ms | ≈ 226 |
| **端到端（2048×1000）** | **5.54 ms** | **≈ 181** |
| FastAPI `/predict` 实测 | 7.35 ms | ≈ 136 |

（QPS 上界 = 1000 / p50，实际会因调度、GC、网络栈打折扣。`/predict` 取五次实测 7.02–8.95 ms 的中位数 7.35，早期记的 9.02 ms 是在 640 训练抢 CPU 时测的。）

**关键约束：onnxruntime 会话内部已有 4 线程并行（`intra_op_num_threads=4`）。**

这意味着：

- **单进程跑满一个 4 核切片**。工控机如果是 8 核，还能再起一个进程，但意义不大——产线节拍通常是每秒几张图，不是几百张。
- **不要用多进程 + 模型副本**来提 QPS。12 MB 模型 × N 份内存，换来的吞吐在产线场景是浪费。**用队列串行处理更稳**：延迟可预测，不会有多个进程抢 CPU 导致的尾延迟抖动。

### 5.2 产线实际负载估算（Line-side load）

一条典型钢带产线的检测需求：

| 参数 | 典型值 | 说明 |
|---|---|---|
| 产线速度 | 1–3 m/s | |
| 相机线扫频率 | 50–200 Hz | 每帧对应 5–40 mm 长度 |
| 需要推理的帧率 | 10–30 fps | 通常隔帧推理，或只在触发信号后取图 |
| **本服务可提供** | **≈ 130–180 QPS** | 远超需求 |

**结论：单实例绰绰有余。** 真正的瓶颈从来不是推理速度，而是图像传输和预处理（大图解码本身就比推理慢）。

### 5.3 如果要提吞吐（Scaling options）

按代价从低到高：

1. **隔帧推理**——产线上相邻帧的缺陷高度相关，隔帧不影响检出率。直接 ×2。
2. **ROI 裁剪**——只推理钢带有缺陷风险的区域（例如边部）。延迟能降到 1/4。
3. **切片推理（SAHI）**——反而会变慢 4 倍，但换来小目标召回（见 [`LINE_ADAPTATION.md`](LINE_ADAPTATION.md) 的「分辨率」一条）。这是**精度和速度的交换，不是吞吐优化**。
4. **int8 量化**——`onnxruntime.quantization` 动态量化，体积减 4 倍、速度提 1.5–2 倍，但精度损失需要重新验证。本项目未做，因为 FP32 已经够快。

---

## 6 已知限制（Known limitations）

### 6.1 明确不支持的能力（Unsupported features）

| 限制 | 影响 | 为什么不做 / 怎么绕 |
|---|---|---|
| **不支持动态 batch** | 导出时 `dynamic=False`，固定 batch=1 | 产线逐张推理，batch 没意义。要批量离线跑，重新导一份 `dynamic=True` 的即可 |
| **不支持 GPU 推理** | 只提供 CPU 版部署说明 | 产线工控机通常没有独显；有独显就用 `onnxruntime-gpu` + `best_fp16.onnx`，代码不用改，换包和模型文件即可 |
| **无鉴权** | 任何能访问 8000 端口的人都能调 | **必须绑 `127.0.0.1` 或在内网防火墙后面**。要对外暴露，加一层 nginx + token 校验，或直接用形态 A（单机 exe） |
| **无请求限流** | 恶意/异常高频请求会占满 CPU | 同上，靠网络层挡。代码层面可以加 `slowapi`，本项目没加 |
| **无持久化** | 结果不落库，只返回 JSON | 由调用方（采集服务或 MES）负责落库。推理服务保持无状态，这是刻意的 |
| **单模型单类别集** | 只认这 10 类 | 换产品要重训。**不要试图让一个模型认所有产品**，类别语义会打架 |
| **不支持旋转框** | YOLO 是水平框（HBB） | 钢带缺陷多数是水平框够用的。要 OBB 得换 YOLOv8-OBB 重训 |

### 6.2 精度限制（Accuracy limits）

1. **小目标召回差**。inclusion（mAP@0.5 = 0.248）、rolled_pit（0.099）两类在 320 输入下基本失效。**如果产线的主要缺陷是这两类，本项目不能直接用**——必须先上切片推理或 640 训练。
2. **crease ↔ water_spot 互混 25%**。这是类别定义问题，不是模型问题。落地前必须和客户对齐「这两类是否合并」。
3. **彩色反光场景未验证**。训练数据是灰度图，产线彩色相机 + 磨砂反光的组合下，假阳性率会显著上升。**投产前必须用产线实拍图做一次验证**，不要拿公开数据集的指标当承诺。

### 6.3 可靠性限制（Reliability limits）

| 项 | 现状 | 建议 |
|---|---|---|
| 异常输入 | 非图片会返回 400；超大图会吃内存 | 加文件大小上限（如 20 MB） |
| 模型文件缺失 | 启动直接抛异常，服务起不来 | 可接受（快速失败比带病运行好） |
| 长时间运行内存泄漏 | 未做 72 小时压测 | 投产前跑一次长稳测试，或配 systemd / NSSM 做定期重启 |
| 日志 | 只有 uvicorn 访问日志 | 建议加结构化日志（请求 ID、耗时、检出数） |

---

## 7 测试环境（Reference environment）

| 项 | 值 |
|---|---|
| OS | Windows（本测试环境） |
| CPU | AMD Ryzen 7 9800X3D |
| GPU | NVIDIA RTX 5070 12GB（**仅训练用，推理走 CPU**） |
| Python | 3.x（见项目 venv） |
| onnxruntime | 1.31.1（CPUExecutionProvider，intra_op 4 线程） |
| PyTorch | 2.14.1+cu130（仅训练/导出阶段） |
| ultralytics | 8.4.211 |

**延迟数字的适用边界**：9800X3D 是高端桌面 CPU。工控机上常见的 i5-1235U / Atom 系列会慢 3–8 倍，即 p50 从 3.4 ms 变成 10–27 ms。**必须在目标硬件上重测**，不要拿本文的数字做承诺。

---

## 8 上线检查清单（Go-live checklist）

按顺序做，每步都要有实测证据：

- [ ] **1. 目标硬件延迟实测**——在工控机上跑 `bench/latency.py`，p95 必须进产线节拍预算
- [ ] **2. 产线实拍图验证**——取 200 张当班实拍图人工核对，重点看彩色反光下的假阳性率
- [ ] **3. 阈值标定**——和工艺工程师一起，按当期质量目标定默认阈值（不是 0.25 也不是 0.5，是成本最优点）
- [ ] **4. 灰度上线**——先做「辅助复判」模式：机器报框，人工确认，不直接判定。跑 2 周收集纠正数据
- [ ] **5. 长稳测试**——72 小时连续运行，观察内存曲线和延迟漂移
- [ ] **6. 故障演练**——拔网线、杀进程、断电，确认上位机能优雅降级（不能白屏，要提示「推理服务不可用」）
- [ ] **7. 打包与安装脚本**——形态 A 用 PyInstaller（注意要把 onnx 文件作为 data 打进去，不要漏）
- [ ] **8. 回滚预案**——模型文件保留上一版；出问题 30 秒内能切回去

**第 4 步是整份清单里最容易被跳过、也最该做的。** 直接上「机器判定」的视觉系统，一次误检停线就会让操作员彻底不信任它；而「辅助复判」模式下，机器只是提醒，报错了也没损失，人反而会慢慢发现它确实能抓到东西。信任是攒出来的。

---

## 9 出问题时怎么查（Troubleshooting）

| 现象 | 最可能的原因 | 排查方法 |
|---|---|---|
| 服务起不来 | 端口被占 / 模型文件路径错 | 看启动日志；`netstat -ano \| findstr 8000` |
| `/health` 通但 `/predict` 超时 | 上传的图超大（如 8000×8000） | 加尺寸校验；看 `elapsed_ms` |
| 检出框全在左上角 | letterbox 的坐标反变换写错 | 检查 `_pre` 的 `ratio` / `pad` 与 `_post` 的还原是否对称 |
| 中文显示成方块 | Qt 没加载中文字体 | `QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")`（本项目 `hmi/_shot.py` 踩过这个坑） |
| 某类突然全检不出 | 类别索引对不上 | 对比 `data/gc10/data.yaml` 的 `names` 顺序与 label txt 首列 |
| 训练/评测报「数据集路径不存在」 | `data.yaml` 的 `path:` 用了相对路径 | 改成使用者自己 `data/gc10` 的绝对路径，见第 2 节 |
| 延迟突然翻倍 | 后台有别的进程抢 CPU | 任务管理器看 CPU；`bench/latency.py` 重测 |
| 上线一段时间后误检率上升 | 灯管老化 / 镜头脏污 | 不是模型问题，先检查光学链路 |
