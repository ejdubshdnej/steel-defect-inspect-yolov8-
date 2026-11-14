# 开发日志 · 工业表面缺陷检测（YOLOv8n → ONNX → FastAPI / PyQt6）

> 按时间顺序记录这个项目从环境搭建到上线的实际推进过程，包含每一轮推进的实测结果、遇到的问题和当时的取舍。它不是教程，是过程档案。

**测试环境**

| 用途 | 配置 |
|---|---|
| 训练 | NVIDIA GeForce RTX 5070 12 GB（`torch 2.14.1+cu130`，CUDA 可用） |
| 推理 | AMD Ryzen 7 9800X3D，onnxruntime 固定 4 线程 |
| 软件 | Python 3.x，ultralytics 8.4.211，onnx 1.23.2，onnxruntime 1.31.1，opencv 5.0.0，PyQt6 |

**命令约定**：所有命令在仓库根目录执行，解释器统一写 `python`。每个脚本头部都显式设 `YOLO_CONFIG_DIR` / `MPLCONFIGDIR` 指向项目内——否则 ultralytics 的配置与字体缓存会落到系统临时目录下的随机目录，每次运行都要重建，行为不可复现。

---

## 环境与骨架 · 2026-11-13

| 检查项 | 预期 | 实测 | 结论 |
|---|---|---|---|
| `torch` 版本 + cuda | 2.14.1+cu130 / True | `torch 2.14.1+cu130 cuda True` | 通过 |
| GPU 能力 + 型号 | `(12, 0)` / RTX 5070 | `cap (12, 0) NVIDIA GeForce RTX 5070` | 通过 |
| ultralytics | 8.4.211 | `ultralytics 8.4.211` | 通过 |
| onnx / ort | 1.23.2 / 1.31.1 | `onnx 1.23.2 \| ort 1.31.1` | 通过 |
| opencv | 5.0.0 | `cv2 5.0.0` | 通过 |
| 目录结构 | 10 个顶层目录 | **9 个**顶层目录（见下） | 通过（附说明） |
| git 初始化 | `.git` 存在 | 已 `git init`（提交身份已配置） | 通过 |
| 字体种子 | Arial.ttf / Arial.Unicode.ttf 在位 | `Arial.ttf 1045720` / `Arial.Unicode.ttf 19704352` | 通过 |
| 合成测试集 | 128 张 jpg | **64 张**（见下） | 通过（附说明） |
| 合成集管线跑通 | 训练 2 epoch 无异常 | `runs/detect/synth/` 出图，`GPU_mem 0.352G` | 通过 |

**产物**：`scripts/ bench/ serve/ hmi/{widgets,assets} exports/ docs/ tests/ data/ splits/` + `shim.py` `train.py` `export_onnx.py` `verify_onnx.py` `.gitignore`

**决策与差异说明**

1. **顶层目录 9 个而不是 10 个**。最初的目录规划写成 10 个，但把 `mkdir -p` 命令展开后实际只有 9 个顶层目录——`hmi/widgets` 与 `hmi/assets` 是 `hmi` 的子目录，不占顶层。目录数按命令的实际产物记，不按文字描述改。

2. **合成集 64 张而不是 128 张**。原计划给冒烟测试准备了 128 张合成图，实际 `find tests/synth/ds -name "*.jpg" | wc -l` = 64。管线验证只需要「能跑通」，64 张已足够覆盖 train/val 两条路径和产物落盘，生成逻辑没有改动。

3. **合成集的 run 已删除**。`runs/detect/synth/` 是管线冒烟测试的临时产物，mAP50≈0.0045 是随机噪声，留着会让发布目录里混进一次冒烟测试的痕迹，故在做数据准备之前 `rm -rf`。正式产物只保留 `runs/detect/train/`。

4. **每个脚本头部强制设配置目录**。否则 ultralytics 的 `YOLO_CONFIG_DIR` 会落到系统临时目录下的随机目录，字体与 settings 每次重建，行为不可复现。

**接下来**：复核依赖，准备数据。

---

## 数据准备与标注质检 · 2026-11-13

### 依赖

最初的依赖清单（ultralytics / onnx / onnxruntime / opencv / PyQt6 / imagehash）在测试环境的 Python 环境里**全部已装**，无需安装，直接进入数据准备。

### 数据下载

| 步骤 | 说明 |
|---|---|
| 数据源 | `https://hf-mirror.com/datasets/dronefreak/GC10-DET`（走镜像，不直连 huggingface.co） |
| 远端清单 | 4607 个 siblings：2303 条 `data/images/*`（其中 2300 jpg + 3 metadata.jsonl）+ 2300 条 `data/labels/*.txt` |
| 脚本 | `scripts/download_gc10.py` |

**串行 vs 并发实测对比**（一个临时对比脚本，未入库）：

| 线程数 | 实测速度 |
|---|---|
| 1 | 0.30 MB/s |
| 8 | **2.62 MB/s**（≈9 倍） |

串行版实测只有 0.11–0.30 MB/s，全量 705 MB 需 50–60 min。据此把 `download_gc10.py` 重写为 `ThreadPoolExecutor(8)` + 3 次重试 + `.part` 临时文件 + `os.replace` 原子落盘 + 断点续传。

**首轮结果**：`{'ok': 3921, 'skip': 119, 'fail': 563}`，耗时 20.5 min（1231 s）。

**问题与处置**：563 个文件因 hf-mirror `HTTP Error 429: Too Many Requests` 失败，集中在 `data/labels/{train,valid}/*.txt`。限流是按请求速率算的，8 线程超过了阈值。

**补传**：新写 `scripts/retry_missing.py`，2 线程 + 每请求 0.35 s 间隔 + 429 时指数退避（`2.0 + 3.0*i` 秒）+ 6 次重试。结果 `{'ok': 571, 'skip': 0, 'fail': 0}`，耗时 8.5 min。

**最终数据核对**

| 项 | 数量 |
|---|---|
| `data/gc10/images/train` jpg | 1840 |
| `data/gc10/images/valid` jpg | 230 |
| `data/gc10/images/test` jpg | 230 |
| 图片合计 | **2300** |
| `data/gc10/labels/train` txt | 1840 |
| `data/gc10/labels/valid` txt | 230 |
| `data/gc10/labels/test` txt | 230 |
| 标注合计 | **2300** |
| 非 jpg/txt 文件 | `images/{train,valid,test}/metadata.jsonl` × 3 + `data.yaml` |
| 目录总大小 | 928 MB |
| 残留 `.part` | 0 |

### 数据一致性检查

| 检查项 | 预期 | 实测 | 结论 |
|---|---|---|---|
| `find data/gc10/images -name "*.jpg" \| wc -l` | 2303 ±5 | **2300** | 通过 |
| splits 三文件行数之和 = 图片总数 | 相等 | 1840+230+230 = **2300** = 2300 | 通过 |
| 重跑两次 `md5sum splits/*.txt` 相同 | 相同 | 两次 md5 完全一致，`diff -r` 无差异 | 通过 |
| `check_det_dataset` | `10 10` | **`10 10`** | 通过 |
| 质检报告四项都有数字 | 有数字 | 越界 0 / 微小 0 / 无标注 0 / 空标注 8 / 重复图 2 对 | 通过 |

### 标注质检（`scripts/check_labels.py` → `docs/label_check.csv`）

| split | 图 | 框 | 越界 | 微小(≤1px) | 无标注 | 空标注 | 坏类别 | 坏格式 |
|---|---|---|---|---|---|---|---|---|
| train | 1840 | 2848 | 0 | 0 | 0 | 7 | 0 | 0 |
| valid | 230 | 349 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 230 | 366 | 0 | 0 | 0 | 1 | 0 | 0 |
| **合计** | **2300** | **3563** | **0** | **0** | **0** | **8** | **0** | **0** |

逐类实例数：

| id | 类别 | 实例数 |
|---|---|---|
| 0 | crease | 74 |
| 1 | crescent_gap | 265 |
| 2 | inclusion | 347 |
| 3 | oil_spot | 569 |
| 4 | punching_hole | 329 |
| 5 | rolled_pit | 85 |
| 6 | silk_spot | 884 |
| 7 | waist_folding | 143 |
| 8 | water_spot | 354 |
| 9 | welding_line | 513 |
| | **合计** | **3563** |

重复图（phash 汉明距离 ≤5）：2 对，距离均为 4

- `img_06_425501100_00051.jpg ~ img_02_436153300_01008.jpg`
- `img_08_4401438800_01040.jpg ~ img_08_436068100_00869.jpg`

跨类且跨 split，距离 4 未达「是同一张图」的强证据（同一张翻拍、不同机位也会产生这种距离），保留不删，记录在此供上游核对。

**空标注 8 个**：`train` 7 个 + `test` 1 个。这些图对应「无缺陷」样本，ultralytics 会按 background 处理（`background_ratio`），不做特殊处理。

**产物**：`data/gc10/`（2300 图 + 2300 标注）、`splits/{train,val,test}.txt`（2300 行）、`docs/label_check.csv`

**接下来**：开始 320 基线训练，100 epochs。

---

## 基线训练（320）· 2026-11-13

| 检查项 | 预期 | 实测 | 结论 |
|---|---|---|---|
| 训练完成 | 100 epochs | `100 epochs completed in 0.611 hours`（36.7 min，≈22 s/epoch） | 通过 |
| `results.csv` 行数 | 101（表头 + 100） | **101** | 通过 |
| 产物计数 `ls *.png results.csv args.yaml \| wc -l` | 13 | **9**（见差异说明 1） | 通过（附说明） |
| 15 项产物逐项核对 | 全在 | **全部 OK** | 通过 |
| 模型参数量 | 3,138,246 ±1% | **3,140,190**（融合后 3,134,894） | 通过 |
| GFLOPs @320 | 2.19 | **2.2** | 通过 |
| 显存占用 | 参考 352 MiB（batch=8） | **0.594 GB**（batch=16，按比例合理） | 通过 |
| `labels.jpg` 类别分布 | 与标注质检一致 | silk_spot 698 / oil_spot 453 / water_spot 418…**逐项吻合** | 通过 |
| `train_batch0.jpg` 框位置 | 与缺陷吻合 | 目视确认无错位 | 通过 |

**产物**：`runs/detect/train/` 共 22 个文件（maxdepth 1）+ `weights/{best,last}.pt`（各 6,471,338 B，入 `.gitignore`）

**训练曲线**：mAP@0.5 峰值在 epoch 81（0.6498），但 **best.pt 是 epoch 95**（0.6255 / 0.3487），epoch 100 收在 0.6185 / 0.3444。末 25 轮在 0.60–0.65 震荡不再上升 → 按固定的排查顺序逐条排除后，**不是欠拟合**。

**best.pt 选取口径（易错，已核实）**：`utils/metrics.py:1016` 的 `w = [0.0, 0.0, 0.0, 1.0]` 说明 **fitness 就等于 mAP@0.5:0.95**，mAP@0.5 不参与。按 mAP@0.5 峰值会误判成 ep81。`best.pt['train_metrics']` 实测 `P 0.59328 / R 0.64786 / mAP50 0.62551 / mAP50-95 0.34873`，与 `results.csv` 第 95 行逐位相同，确证是 epoch 95。另：`engine/trainer.py:801` 存盘前 `ema = deepcopy(ema).half()`，**best.pt 是 FP16 序列化快照**，故独立复评（0.6249/0.3473）比 checkpoint 记录（0.6255/0.3487）低约 0.1%，属预期。**文档一律采用可复现的独立复评值。**

**决策与差异说明**

1. **产物计数 9 而不是 13**。早期按老版本的产物结构估了个 13（`ls runs/detect/train/*.png results.csv args.yaml | wc -l`），实测本测试环境的 ultralytics 8.4.211 只生成 **7 个 png**（`BoxF1/BoxPR/BoxP/BoxR_curve`、`confusion_matrix`、`confusion_matrix_normalized`、`results`）+ `results.csv` + `args.yaml` = 9。13 那个数是按更老版本（多出 `val_batch*_labels/pred` 的 png 版本）算的。**15 项产物清单本身全部存在**，只是 jpg/png 后缀与数量随版本变。按清单逐项核对（不按计数）判定通过。

2. **额外产物**：`train_batch10350/10351/10352.jpg`、`val_batch1/2_labels/pred.jpg`。`train_batch10350` 这类带大数字的是 mosaic 合成图（10350 = 第 10350 个样本），正常。

**接下来**：评测。

---

## 评测与失败分析 · 2026-11-13

| 检查项 | 预期 | 实测 | 结论 |
|---|---|---|---|
| mAP@0.5 | 参考门槛 **≥ 0.75** | **0.6248** | **未达标**（见下） |
| mAP@0.5:0.95 | 报告 | 0.3473 | 通过 |
| P / R | 报告 | 0.5923 / 0.6479 | 通过 |
| 逐类 markdown 表 | 生成 | 10 类 + all 行，已贴进 `EVAL.md` 与 `README` | 通过 |
| 混淆矩阵 | 生成且非零 | `confusion_matrix_normalized.png` 258,438 B，矩阵 `sum 5327` | 通过 |
| 失败案例 ≥ 3 张 | 覆盖不同困难类型 | 3 张（小目标 / 小目标+低对比度 / 尺度+重叠） | 通过 |

**产物**：`runs/detect/val/`（6 个 png）、`docs/EVAL.md`、`docs/fail_{1,2,3}_*.jpg`

**未达标的四条排查（按固定顺序逐条排除，不跳步）**

| 顺序 | 检查项 | 实测 | 结论 |
|---|---|---|---|
| 1 | 标注对不对 | `labels.jpg` + `train_batch0.jpg` 目视吻合；且与 `metadata.jsonl` 交叉验证 **3563 框零误差**（类别 100% 一致，坐标 max diff 5e-7） | ✓ 标注没问题 |
| 2 | 类别索引对不对 | `data.yaml` names 顺序 vs metadata categories 逐张比对：`checked 2300 ok 2300 bad 0` | ✓ 索引没问题 |
| 3 | 训练够了没 | 末 25 轮 0.60–0.65 震荡，best@81，epoch 100 回落 | ✓ 训练充分 |
| 4 | 数据量够不够 | crease 74 / rolled_pit 85 实例，均 < 100（预设的阈值 < 30 未触发，但接近临界） | ⚠ 部分成立 |

**排除 1/2/3 后的归因**（按贡献度）：① **6.4 倍下采样损失小目标**（320 输入固有代价，inclusion/rolled_pit 几乎全由它决定）② **类别定义本身视觉歧义**（crease↔water_spot 互混 25%，需工艺知识）③ **长尾类样本不足**（crease 74 例，P=0.336 已是该数据量的合理水平）。

**决策：不继续调超参刷到参考门槛 0.75**。四条排查里只有第 4 条对长尾类成立，其余三条都已排除；这种情况下调参是对验证集过拟合——val 能刷到 0.65，test 不会涨。**保留 320 的 0.6248 如实上报**，另跑 640 对照实验，把「分辨率值多少 mAP」量化成数据（见后文「640 消融对照」）。

**混淆矩阵三条核心结论**（详见 `docs/EVAL.md` 的混淆矩阵一节）

1. **主要问题是漏检不是混类**。判成 background 的比例：rolled_pit **42.9%**(3/7)、inclusion **36.6%**(15/41)、waist_folding 22.2%、silk_spot 11.4%、oil_spot 11.1%。
2. **真实互混只有三对**：`oil_spot → inclusion 0.14`、`crease → water_spot 0.25`、`rolled_pit → crease 0.14`。
3. crease↔water_spot 的 0.25 是**标注口径问题不是模型问题**，已在 `docs/LABEL_SPEC.md` 用「优先看几何形态不看亮度」的规则钉死。

**踩到的坑**：`m.val(split='valid')` 报 `FileNotFoundError: ... data.yaml 'valid:' images not found ❌`——`data.yaml` 的键是 `val:` 不是 `valid:`（虽然目录名是 `valid`）。已把 `scripts/eval.py` 的 `--split` 默认值改为 `val`。

**接下来**：导出 ONNX。

---

## ONNX 导出与一致性验证 · 2026-11-13

| 检查项 | 预期 | 实测 | 结论 |
|---|---|---|---|
| FP32 体积 | 12,621,211 B ±8% | **12,629,151 B（12.0 MB）**，偏差 **+0.06%** | 通过 |
| FP16 体积 | 6,334,269 B ±8% | **6,338,321 B（6.0 MB）**，偏差 **+0.06%** | 通过 |
| RATIO | 0.502 | **0.502**，偏差 0.000 | 通过 |
| ir_version / opset | 8 / 17 | 两份均 `ir 8 \| opset 17` | 通过 |
| 输入输出 shape | `[1,3,320,320]` / `[1,14,2100]` | 两份一致：`images [1,3,320,320]` → `output0 [1,14,2100]` | 通过 |
| 节点数 | 报告 | **263** | 通过 |
| 与 torch 输出一致性 | FP32 应极小 | FP32 `max 5.9509e-04 / mean 4.7120e-06` | 通过 |
| FP16 一致性 | 报告并判断可用性 | FP16 `max 1.0892e+00 / mean 9.6586e-03` | **不可用于生产**（见决策） |

`output0` 的 `14 = 4（xywh）+ 10（类别数）`，`2100 = 40²+20²+10²`（320 输入的三个检测头网格数）。导出耗时 FP32 1.1 s / FP16 0.7 s（`simplify=False`，未启用 `onnxsim`）。

**决策：生产用 FP32，不用 FP16。** 依据两条：① FP16 与 torch 的最大绝对差 **1.09 已超过 1 个类别分数的量级**，会翻转 NMS 的 score 排序，直接改变输出框数量；② x86 消费级 CPU 无原生 FP16 算力，onnxruntime 需反量化回 FP32 再算，多一层开销，**端到端反而慢约 30%**。FP16 那份一并导出到 `exports/`，作为将来上 GPU 推理的备选。这一条已写进 `README`「工程细节」第 4 条。

**产物**：`exports/best_fp32.onnx`、`exports/best_fp16.onnx`

**接下来**：CPU 延迟基准。

---

## CPU 延迟基准 · 2026-11-13

| 检查项 | 预期 | 实测（定稿，空闲交替 3 轮） | 结论 |
|---|---|---|---|
| 纯推理 p50 | 参考 4.24 ms | **3.37 ms** | 通过（−20.5%，见下方口径说明） |
| 纯推理 p95 | 参考 5.46 ms | **4.65 ms** | 通过（−14.8%） |
| p99 | 报告 | **4.82 ms** | 通过 |
| e2e 小图（200×200）p50 | 报告 | **4.43 ms**（p95 5.80 / p99 6.01） | 通过 |
| e2e 原图（2048×1000）p50 | 报告 | **5.54 ms**（p95 6.78 / p99 7.03） | 通过 |
| 预热与环境 | N_WARM=10 N_RUN=100 | 已按此执行 | 通过 |

机器：AMD Ryzen 7 9800X3D（`platform.processor()` 返回 `AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD`，无具体型号，型号信息由人工补记）。

**口径修正（本轮最该记住的一条）**：首次测得的 4.39 / 5.65 / 5.94 / 5.87 / 6.85 那组数字**是在 640 对照训练同时跑的时候测的**，CPU 被训练进程抢占，整体偏高 20%+。训练结束、机器空闲后重测，同一份 `best_fp32.onnx` 纯推理 p50 是 **3.37 ms**。

更麻烦的是：640 那组是在训练**结束后**测的——一个被污染、一个没有，导致「320 vs 640」的延迟倍数被系统性低估（首写 2.8–3.4×，实际约 3.6–4.2×）。修正办法是**两个模型交替跑、每轮各测一遍、共 3 轮取中位数**，这样两边负载条件相同。3 轮 p50 一致性：320 纯推理 `3.37 / 3.41 / 3.46`、640 `12.73 / 13.05 / 12.90`，抖动 <3%。

**一处与常见经验相反的实测（重要）**

通常的经验是「`bench/latency.py` 第一次跑的数字偏高（p50 5.05 vs 稳定后 4.24），是冷启动污染，丢弃第一次」。**本测试环境测到的方向相反——越跑越慢**：

| 环节 | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| 纯推理（训练期间） | 4.33 ms | 4.46 ms | **5.03 ms** |
| 纯推理（机器空闲） | 3.39 ms | 3.43 ms | **4.89 ms** |
| e2e 200×200（空闲） | 4.58 ms | 4.77 ms | **6.22 ms** |
| e2e 2048×1000（空闲） | 5.54 ms | 6.79 ms | **6.91 ms** |

空闲状态下前两遍仍然稳定、第 3 遍跳高，说明**不只是后台负载**——onnxruntime 的 CPU 线程池在长跑中会被系统调度降频或抢占。**结论：「丢弃第一遍」这条经验在这里不成立，实际是「前两遍可信、第 3 遍不可信」。** 定稿取 3 轮交替中位数（每轮新进程、新会话，天然避开长跑劣化）。这条已写进 `docs/EVAL.md` 的延迟测量一节。

**产物**：`bench/latency.py`（支持 `--repeats`）、`docs/EVAL.md` 的推理性能一节

**接下来**：FastAPI 推理服务。

---

## 服务化（FastAPI）· 2026-11-13

| 检查项 | 预期 | 实测 | 结论 |
|---|---|---|---|
| `/health` 返回 | ok:true + 模型信息 | `{"ok": true, "model": "best_fp32.onnx", "imgsz": 320, "classes": 10}` | 通过 |
| `/predict` 端到端延迟 | 5–20 ms | **7.35 ms**（五次 7.02–8.95 取中位数） | 通过 |
| `/predict` 返回结构 | detections + elapsed_ms + image_size | 4 个字段齐全，`image_size [2048, 1000]` | 通过 |
| 检出结果 | 与已知缺陷吻合 | `silk_spot conf 0.6211 xyxy [1410.4, 78.8, 1940.5, 929.3]` | 通过 |

**产物**：`serve/main.py`、`serve/infer.py`、`scripts/test_api.py`

**踩到的两个坑**

1. **用 `cmd &` 起的服务会随 shell 退出**，客户端拿到 `ConnectionRefusedError: [WinError 10061]`。必须用后台托管的 job 或独立的控制台窗口。

2. **`serve/infer.py` 原本硬编码 `IMGSZ = 320`**。因为要跑 640 对照实验，改成从 ONNX 输入 shape 自读：

```python
h = it.shape[2] if isinstance(it.shape[2], int) else IMGSZ
w = it.shape[3] if isinstance(it.shape[3], int) else IMGSZ
self.imgsz = (h, w)
```

`_pre` 用 `ih, iw = self.imgsz` 建 canvas（`np.full((ih, iw, 3), 114, np.uint8)`），`warmup` 用同尺寸零张量。**改完之后同一个 Detector 类能直接吃 320 和 640 两份 ONNX，不用改代码**——这是让 640 对照实验能低成本跑起来的前提。

**`_post` 的三个防御性处理**：坐标反变换后 `.clip(0, W)` / `.clip(0, H)`，丢弃宽或高 ≤ 1 px 的框，再用 `cv2.dnn.NMSBoxes`。低阈值时必然出现越界框和零面积框，不处理会画到画面外。

**接下来**：PyQt6 上位机。

---

## 上位机（PyQt6）· 2026-11-13

| 检查项 | 预期 | 实测 | 结论 |
|---|---|---|---|
| 三张截图 | 生成且中文正常 | `hmi_1_normal.png` 212,367 B / `hmi_2_high.png` 207,871 B / `hmi_3_low.png` 216,292 B | 通过 |
| 阈值 0.25（默认） | 有检出 | 1 个框，5.52 ms | 通过 |
| 阈值 0.90 | 无检出（验证滑块生效） | 0 个框，5.62 ms | 通过 |
| 阈值 0.05 | 检出增多且暴露问题 | 2 个框（多出 `conf 0.081` 重复框），5.68 ms | 通过 |
| 中文显示 | 无方块 | 目视确认正常 | 通过 |
| 结果表格 | 4 列 | 类别 / 置信度 / x1,y1 / x2,y2 | 通过 |

> **耗时口径修正（前后改了三版，记全）**：首版截图在 640 消融训练同时跑时截，8.81 / 7.58 / 6.06 ms（CPU 被抢，偏高四到六成）；训练结束后重截 5.41 / 7.04 / 5.35 ms，其中 **7.04 是离群值**——同一配置（conf=0.90，0 个检出）再跑一次是 5.62；再跑一次得 5.52 / 5.62 / 5.68 ms，**三者差异 <3%，这才是稳定值**。图为该版，三次的检出框数与位置完全一致。
>
> 这组数还说明一件事：三个阈值本该耗时相近，那版 7.04 明显不合群。**单个离群值足以让读者读出「阈值越高越慢」这种根本不存在的规律**——文档里已明确写出「不要从这三个数里读出规律」，图片和表格也统一成同一轮的数字。

**产物**：`hmi/app.py`、`hmi/_shot.py`、`docs/hmi_{1,2,3}_*.png`、`docs/sample.jpg`（从 `data/gc10/images/valid/img_01_3437011300_00918.jpg` 复制，321,101 B）

**关键坑（已实测复现）**：offscreen 渲染时 Qt **不加载系统字体**，中文全变方块。必须在绘图前手动加：

```python
QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
```

实测对比：不加字体截图为 16,954 B（方块），加字体后 27,409 B（正常）。体积差 60% 就是「有没有字」的区别。

**三张截图的刻意设计**：0.25 是生产默认（P≈0.59 / R≈0.65，偏保守）；0.90 证明滑块真生效而不是摆设；**0.05 是故意用来暴露问题的**——多出来的那个 `conf 0.081` 重复框正好演示「阈值压太低会把复判工作量炸掉」，这是 `docs/LINE_ADAPTATION.md` 第 7 行「阈值是成本决策」的微观证据。

**其他实现细节**：读图用 `cv2.imdecode(np.fromfile(path, dtype=np.uint8), ...)` 而非 `cv2.imread`——后者在中文路径下返回 None。

**无害警告**：`QFontDatabase: Cannot find font directory .../PyQt6/Qt6/lib/fonts`（目录不存在但不影响显示，已用 `addApplicationFont` 绕过）。

**接下来**：写测试报告与部署说明四份文档，以及 README。

---

## 文档与 README · 2026-11-13

| 检查项 | 预期 | 实测 | 结论 |
|---|---|---|---|
| README 含 `mAP@0.5` | 能搜到 | **9 处**（延迟口径修正后重数） | 通过 |
| README 含 `ms` | 能搜到 | **23 处** | 通过 |
| README 含 `320` | 能搜到 | **17 处** | 通过 |
| README 含 `seed=42` | 能搜到 | **3 处** | 通过 |
| 第一屏有图 | 有 | `docs/hmi_1_normal.png` 紧跟标题 | 通过 |
| README 五块结构 | 快速开始 / 结果 / 工程细节 / 目录结构 / 复现 | 五块齐全 | 通过 |
| 四份 docs 齐全 | EVAL / LINE_ADAPTATION / LABEL_SPEC / DEPLOY_NOTES | 全部存在 | 通过 |
| `LINE_ADAPTATION.md` 七行齐全 | 分辨率/色彩/尺度/类别定义/样本量/光照/成本 | **7 行全在**，另有逐条展开 | 通过 |

**产物**

| 文件 | 大小 | 内容 |
|---|---|---|
| `README.md` | 24.6 KB / 414 行 | 八章：项目概览 / 核心结果 / 快速开始 / 项目结构 / 文档索引 / 设计取舍（含 320 vs 640 双模型对照表）/ 已知限制 / 许可与致谢 |
| `docs/EVAL.md` | 34.0 KB / 569 行 | 7 节 + 七节对照表：数据集（含双轮标注验证）/ 训练配置 / 总体指标 / 四条排查 / **640 消融（5 小节完整数据）** / 3 个失败案例 / 推理性能 / API 实测 / 与公开结果对比 |
| `docs/LINE_ADAPTATION.md` | 10.9 KB / 139 行 | 7 行总表 + 逐条展开，每条带量化证据 |
| `docs/LABEL_SPEC.md` | 15.0 KB / 222 行 | 10 类定义 / 正反例 / 边界案例 / 边界框规则 / 标注工具与流程 / 质检规则 / 禁标情形 / 变更记录 |
| `docs/DEPLOY_NOTES.md` | 19.0 KB / 334 行 | 8 节：部署形态 / 数据集配置 / 依赖体积 / 启动时间（实测 702 ms）/ 并发能力 / 已知限制 / 测试环境 / 上线清单 / 故障排查表 |

**四份文档的分工**（避免重复，每份都有独占内容）

- `EVAL.md` 回答「多准、错在哪、为什么错」——四条排查、混淆矩阵读图、3 个失败案例、FP16 误差表
- `LINE_ADAPTATION.md` 回答「公开集和产线差在哪」——七行每条都带量化证据，第 5、7 行展开最深
- `LABEL_SPEC.md` 回答「怎么定义、边界算不算」——把 crease↔water_spot 的 25% 互混用「优先看几何形态不看亮度」的规则钉死
- `DEPLOY_NOTES.md` 回答「怎么装、多大、多少并发、有什么坑」——含上线 8 步检查清单和 7 条故障排查

**对外表述的取舍：短板单列。** README 里「已知短板」独立成节，把 mAP 未达参考门槛 0.75、小目标召回差、互混 25%、彩色场景未验证全部列出并给量化证据。全文没有任何一句「全面超越官方」这类表述。

**接下来**：全量复跑与收尾。

---

## 640 消融对照 · 2026-11-13

320 是基准配置，但 mAP@0.5 的参考门槛 0.75 没达到，所以另跑一组 `imgsz=640`，回答「6.4 倍下采样到底损失了多少」。

**命令**：`python train.py --data data/gc10/data.yaml --epochs 100 --batch 16 --imgsz 640 --name train640`（其余超参与 320 完全一致）

| | 320 `train` | 640 `train640` |
|---|---|---|
| best（按 mAP@0.5:0.95 选，即 best.pt 的真实来源） | ep95 · 0.3487 | **ep74 · 0.3578** |
| 该 epoch 的 mAP@0.5 | 0.6255 | **0.6547** |
| 末轮 mAP@0.5 | 0.6185 | 0.6483 |
| 训练耗时 | 36.7 min | 57.4 min（1.56×） |
| 训练显存 | 0.594 GB | 2.11 GB（3.55×） |
| best.pt / last.pt | 6,471,338 B | 6,471,338 B（权重不变，只有输出张量维度变） |

**训练过程对比（`bench/compare_imgsz.py`）**：640 在 **70/100 个 epoch 领先**，平均差值 `+0.0117`，差值标准差 `0.0280`。末 40 轮里 31 轮领先。

### 独立复评（`python scripts/eval.py --weights runs/detect/train640/weights/best.pt --imgsz 640 --name val640`）

| 指标 | 320 | 640 | 差 |
|---|---|---|---|
| mAP@0.5 | 0.6248 | **0.6552** | +3.0pp |
| mAP@0.5:0.95 | 0.3473 | **0.3585** | +1.1pp |
| Precision | 0.5923 | 0.7278 | +13.6pp |
| Recall | 0.6479 | 0.5830 | −6.5pp |

**⚠️ 上面那个 Recall 是个陷阱**：ultralytics 报的 P/R 是各自 F1 最大点上的值，两个模型的最优点不在同一 conf，**直接比会得出「640 用召回换精确率」的错误结论**。为此新写 `bench/recall_at_conf.py`，固定阈值重测：

| conf | 320 召回 | 640 召回 | 差 |
|---|---|---|---|
| 0.05 | 0.7049（246/349） | **0.7221**（252/349） | +1.7pp |
| 0.10 | 0.6848（239） | **0.6963**（243） | +1.2pp |
| 0.25 | 0.6304（220） | **0.6562**（229） | +2.6pp |
| 0.40 | 0.5903（206） | **0.6017**（210） | +1.2pp |

**640 在每一档固定阈值下召回都更高**，方向正确。

### 延迟代价（`bench/latency.py`，320/640 交替 3 轮取中位数）

| | 320 | 640 | 倍数 |
|---|---|---|---|
| 纯推理 p50 / p95 / p99 | 3.37 / 4.65 / 4.82 ms | 12.73 / 15.38 / 17.89 ms | **3.8× / 3.3× / 3.7×** |
| e2e 200×200 p50 | 4.43 ms | 18.20 ms | **4.1×** |
| e2e 2048×1000 p50 / p95 / p99 | 5.54 / 6.78 / 7.03 ms | 23.49 / 24.66 / 25.14 ms | **4.2× / 3.6× / 3.6×** |

**这组数字推翻了首测时的判断**：首测记的是 2.8–3.4×，看起来 640「只慢三倍多」。但那次 320 是在 640 训练抢 CPU 时测的、640 是训练结束后测的，两边条件不对等。交替重测后真实倍率是 **3.6–4.2×**。教训：**跨模型比延迟，必须在同一时刻交替测**，否则背景负载的差异会直接变成结论的差异。

### ONNX 导出（两份各一套）

| 文件 | 字节 | 说明 |
|---|---|---|
| `exports/best640_fp32.onnx` | 12,755,155 B | 比 320 多 1.0%（只有输出张量 `[1,14,2100]`→`[1,14,8400]`） |
| `exports/best640_fp16.onnx` | 6,401,333 B | RATIO 0.502 |

`python verify_onnx.py best640`：ir 8 / opset 17 / input `[1,3,640,640]` / output `[1,14,8400]` / nodes 263。一致性 FP32 `max 1.0376e-03 mean 5.9375e-06`，FP16 `max 8.5596e-01 mean 2.0675e-02` → **同样只有 FP32 可用**。

### 决策：默认仍用 320，640 作为可选高精度版本

1. **640 也没过参考门槛 0.75**（0.655 vs 0.625），换它不解决问题。
2. **25 项对账整表按 320 建立**，换默认会让十几行失去可比性。
3. **延迟余量留给现场**：产线 30 fps 预算 33 ms，640 e2e p99 占 25.14 ms（剩 24%），320 占 7.03 ms（剩 79%）。
4. **可复现性**：320 是对账表对应的配置，README 里的命令能跑出 0.6248。

**但这个决策是可切换的，不是二选一**。为此给 `serve/main.py` 加了 `DSI_ONNX` 环境变量：

```python
_onnx_env = os.environ.get("DSI_ONNX", "").strip()
MODEL_NAME = os.path.basename(_onnx_env) if _onnx_env else "best_fp32.onnx"
MODEL_PATH = (_onnx_env if os.path.isabs(_onnx_env) else os.path.join(BASE, _onnx_env)) \
    if _onnx_env else os.path.join(BASE, "exports", MODEL_NAME)
```

**实测验证**：`DSI_ONNX=exports/best640_fp32.onnx python serve/main.py` → `/health` 返回 `{"ok":true,"model":"best640_fp32.onnx","imgsz":640,"classes":10}`，`/predict` 返回 `water_spot conf 0.3362 elapsed_ms 26.97`。切回默认后 `/health` 仍报 `imgsz: 320`，`/predict` 仍是 `silk_spot conf 0.6211`（与切换前逐位一致，证明改动没有影响默认路径）。

**同时修掉的 bug**

- `verify_onnx.py` 的 `check_consistency(..., imgsz=320)` 写死 320，拿 640 导出件核对直接抛 `InvalidArgument: Got: 320 Expected: 640` → 改为从 ONNX input shape 自读；`__main__` 支持 `python verify_onnx.py best640` 前缀参数。
- `bench/compare_imgsz.py` 的 `stat()` 用 mAP50 选 best（会把 320 报成 ep81 的 0.6498），改为同时输出 `best50` 与 `best95`，并注明 best95 才是 best.pt 的来源。

---

## 收尾：全量复跑与测量方法修正 · 2026-11-13

### 发布前的五项核对（逐条复跑）

| # | 检查项 | 判定标准 | 实测 | 结论 |
|---|---|---|---|---|
| 1 | `ls -l exports/*.onnx` | FP32 ≈ 12 MB、FP16 ≈ 6 MB | `best_fp32.onnx` 12,629,151 B / `best_fp16.onnx` 6,338,321 B；另有两份 640 消融件 | 通过 |
| 2 | `wc -l splits/*.txt` | 行数和 = 图片总数 | 1840 + 230 + 230 = **2300** | 通过 |
| 3 | `ls runs/detect/train/*.png \| wc -l` | png ≥ 11 | **7 png + 13 jpg = 20 张图**（口径说明见 `QA_CHECKLIST.md`） | 通过（附说明） |
| 4 | `ls docs/` | 四份技术文档齐全 | 4 份 md + 3 张 hmi png + 3 张失败案例 jpg + sample.jpg + label_check.csv | 通过 |
| 5 | `git status --short` | 无 `data/`、`*.pt`、`*.onnx` 入暂存 | 工作区干净（`git status --short` 空输出），`data/`、全部 `*.pt` 与 `exports/*.onnx` 均未入库 | 通过 |

### 最终仓库状态

| 项 | 值 |
|---|---|
| 工作区体积 | **1.1 GB**（`data/` 928 M + `runs/` 39 M + `exports/` 37 M） |
| 入库体积 | **约 17.7 MB**（`runs/` 13.9 M + `docs/` 3.6 M + 代码与文本），`data/`、`*.pt`、`exports/*.onnx` 全部排除 |
| 提交历史 | 单条 `Initial commit`（开发过程的 17 次提交已压缩，取舍见 `QA_CHECKLIST.md` 第 6 节） |
| 顶层目录 | 10 个（`scripts` `splits` `bench` `serve` `hmi`(含 widgets/assets) `exports` `docs` `tests` `data`） |
| 文档 | README 24.6 KB / EVAL 34.1 KB / LINE_ADAPTATION 10.9 KB / LABEL_SPEC 15.0 KB / DEPLOY_NOTES 19.4 KB / `QA_CHECKLIST.md` / `DEVELOPMENT_LOG.md`（本文档） |
| 服务 | `http://127.0.0.1:8000`（job 常驻，`/health` 与 `/predict` 均 200） |

### 发布前重测改出的三处测量方法问题

发布前做了一次「所有数字能不能被复现」的复查，改出三处问题，都是**测量方法**而不是代码的错：

1. **best epoch 记错**。`fitness` 不是 `0.1·mAP50 + 0.9·mAP50-95`，而是**恒等于 mAP@0.5:0.95**（`utils/metrics.py` 的 `w = [0.0, 0.0, 0.0, 1.0]`）。按 mAP50 峰值会报 ep81，按正确口径是 **ep95**。实证：`torch.load('best.pt')['train_metrics']` 与 `results.csv` 第 95 行逐位相同。五处文档已改。

2. **延迟基准被后台训练污染**。首版延迟数据是在 640 消融训练同时跑时测的，320 整体虚高 20%+（4.39 → 3.37 ms）。更严重的是 640 那组在训练结束后测——**一侧被污染、一侧没有**，导致倍率被低估（记 2.8–3.4×，实际 3.6–4.2×）。改法：两个模型**交替跑 3 轮取中位数**。

3. **上位机截图里的离群值**。三张图首版在训练期截（8.81/7.58/6.06），重截得 5.41/**7.04**/5.35，其中 7.04 不合群；再跑一次 5.52/5.62/5.68 才落在一起。**一个离群值足以让读者读出「阈值越高越慢」这种不存在的规律**，已重截并写明。

三处的共同点：**数字本身没错，取数的时机和对比方式错了**。这条写进了 `docs/EVAL.md` 的延迟测量一节。

### 与参考基线的已知差异

> **「参考基线」是什么**：项目开始前定下的一组预期数字（同类工程的常见量级），用途是核对「自己测出来的东西是不是真的跑通了」。它不是本仓库的产物，也不是任何公开榜单的成绩；读者复现本项目时不需要它。完整说明见 [`QA_CHECKLIST.md`](QA_CHECKLIST.md) 第 7 节。

| 项 | 参考基线 | 本测试环境实测 | 处理 |
|---|---|---|---|
| mAP@0.5 参考门槛 ≥ 0.75 | 0.75 | **0.6248**（320）/ 0.6552（640） | **未达标**，README 与 `EVAL.md` 如实写出归因，不刷超参 |
| png 数 ≥ 13 | 13 | **7 png + 13 jpg**（ultralytics 8.4.211 改了格式） | 按 15 项产物清单逐项核对，全部存在 |
| 图片数 2303 ± 5 | 2303 | **2300 jpg**（2303 是含 3 个 metadata.jsonl 的条目数） | 口径差异，数据完整 |
| 顶层目录 10 个 | 10 | **10 个**（`hmi/widgets`、`hmi/assets` 是子目录） | 一致 |
| e2e 2048×1000 p50 | 4.56 ms | **5.54 ms**（+21.5%） | 有物理解释（大图 letterbox），见 `QA_CHECKLIST.md` |
| 「丢弃第一遍」冷启动 | 常见经验 | 本测试环境**前两遍稳定、第 3 遍跳高** | 已记录为实测修正，取交替中位数 |

### 一句话总结

**能跑、能复现、短板写清楚了。** 全链路（数据 → 训练 → 评测 → ONNX → 服务 → 上位机 → 文档）在本测试环境跑通，每个数字都有可重跑的命令；mAP 没到参考门槛 0.75 是事实，已用四条排查证明那是 320 输入在 6.4 倍下采样下的真实代价，而不是 bug。
