# Changelog

本文件记录仓库中值得注意的变更，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

**版本号说明**：`1.0.0` 对应服务端 `serve/main.py` 中声明的应用版本 `version="1.0"`，也是本仓库首次公开发布的代码状态。此前的开发历史见 `git log`，下面的条目按功能块归并，不逐条对应提交。

## [Unreleased]

暂无。

## [1.0.0] - 2026-11-14

首次公开发布。仓库包含一条完整链路：数据准备 → 训练 → 评测 → ONNX 导出 → CPU 推理服务 → PyQt6 上位机，全部代码与实测数据同批公开。

### Added

**数据准备**

- `scripts/download_gc10.py`：并发下载 GC10-DET（2300 图 / 2300 标注 / 928 MB），8 线程实测 2.62 MB/s，串行 0.30 MB/s；`scripts/retry_missing.py` 用于低并发补齐限流遗漏的文件。
- `scripts/make_splits.py`：生成 `splits/{train,val,test}.txt`，按 1840 / 230 / 230 划分，共 3563 个标注框。
- `scripts/check_labels.py`：六项标注质检（越界框、微小框、缺失标注、空标注、越界类别、格式错误）+ 逐类实例数 + pHash 重复图检测（汉明距离 ≤5），结果落盘 `docs/label_check.csv`。

**训练与评测**

- `train.py`：YOLOv8n 微调入口，默认 `imgsz=320 / batch=16 / epochs=100 / seed=42 / deterministic=true`。针对灰度工业缺陷关掉了上下翻转与旋转（缺陷有方向语义）、`hsv_h`/`hsv_s`（灰度图色相无意义），保留左右翻转与亮度抖动。
- `scripts/eval.py`：验证集评测，输出总体指标、逐类 mAP 表与混淆矩阵。
- `shim.py`：把 ultralytics 数据加载使用的 `ThreadPool` 替换为串行实现，避开 Windows 上共享内存/管道相关的原生异常（表现为训练或导出在第一个 batch 之前直接退出，栈里没有 Python 层错误）；`train.py`、`scripts/eval.py`、`export_onnx.py` 在导入 ultralytics 之前调用。
- `bench/`：`latency.py`（纯推理 / 端到端 / 大图端到端三层延迟，输出 mean/p50/p95/p99）、`startup.py` 与 `startup_http.py`（进程到 `/health` 可用耗时）、`compare_imgsz.py`（不同输入尺寸的消融对比）、`recall_at_conf.py`（固定置信度下的召回对比）。

**导出**

- `export_onnx.py`：一次导出 FP32 与 FP16 两份，`opset=17`、静态 shape、`simplify=False`。产物 12,629,151 B / 6,338,321 B。
- `verify_onnx.py`：校验 IR 版本、opset、输入输出 shape，并逐元素比对 ONNX Runtime 与 PyTorch 的输出差。

**服务**

- `serve/infer.py`：ONNX Runtime 推理封装，输入尺寸从模型自身读取（320 与 640 导出件共用同一份代码），自动识别 FP16 输入类型，启动时预热 3 轮。后处理把框裁剪回图像边界并丢弃裁剪后宽高不足 1 像素的框。
- `serve/main.py`：FastAPI 应用。`GET /health` 返回模型名、imgsz、阈值与类别数；`POST /predict` 接收 multipart 上传，字段名为 `file`。
- `scripts/test_api.py`：不依赖 `requests` 的 multipart 接口测试。

**上位机**

- `hmi/app.py`：PyQt6 单文件界面，左侧原图叠加红框，右侧置信度阈值滑块与结果表格，支持从剪贴板粘贴图片。与 `serve/infer.py` 共用同一套推理代码。
- `hmi/_shot.py`：批量生成界面截图，供文档引用。

**仓库配套**

- `docs/`：评测结论、产线适配的七条差距、标注规范、部署说明、实操验证指南、开发日志与质量核对表。
- `docs/PAPER.md` 与对应 DOCX：技术报告，含 16 个表题、三线表排版。

### Changed

- 默认部署规格定为 `imgsz=320`：CPU 端到端 p50 5.54 ms。640 版本精度更高（mAP@0.5 0.6552 vs 0.6248，召回 0.6562），但端到端 p99 升到 25.14 ms，训练耗时 57.4 min、显存 2.11 GB。两者通过 `DSI_ONNX` 环境变量切换，无需改代码。
- `serve/main.py` 支持启动期注入：`DSI_ONNX` 指定模型文件、`DSI_CONF` 指定置信度阈值（钳制在 0.01–0.99）、`DSI_PORT` 指定端口。阈值被当作成本参数而非技术参数，理由写在代码注释与部署文档里。
- `train.py` 新增 `--resume`，从 checkpoint 恢复训练，其余参数从存档中读回，避免命令行与存档打架。
- `/health` 返回的 `imgsz` 改为从模型自读，换 640 模型时不再需要同步修改常量。
- 启动流程加入 3 轮 warmup，把 ONNX Runtime 的图优化与内存分配开销提前吃掉；`/health` 一旦返回 ok 即为可服务状态。进程启动到 `/health` 可用 701.71 ms，其中模型加载与 warmup 71.87 ms。
- `bench/latency.py` 取稳定值的方式从「最后一轮」改为「各轮中位数」：同一配置下前两轮稳定、第 3 轮起 p50 跳高（长时间占用后系统降频/抢占），取末轮等于把劣化当成稳定值。

### Fixed

- `README` 中 `/predict` 的上传字段名曾写成 `image`，实际为 `file`，按文档调用会返回 422；已修正并补 `curl` 示例。
- 延迟基准首版在 640 训练占用 CPU 时测得，320 的结果整体偏高 20% 以上；已全部重测。
- 上位机截图中的耗时首版同样受训练抢占影响，偏高约一倍；重截后稳定在 5.52 / 5.62 / 5.68 ms，三者差异小于 1.5%。
- 逐类指标表的口径从「验证集实例数」修正为「训练集实例数」，与正文表述一致。
- `verify_onnx.py` 的输入尺寸改为从 ONNX 输入 shape 自读，避免拿 640 导出件核对时因写死 320 而抛 `InvalidArgument`。

### Docs

- 训练结果：验证集 mAP@0.5 **0.6248**、mAP@0.5:0.95 0.3473、P 0.5923 / R 0.6479，best.pt = epoch 95（Ultralytics 以 mAP@0.5:0.95 选模，fitness 即该指标）；mAP@0.5 的单轮峰值在 epoch 81（0.6498）。相对常见工程门槛 0.75 未达标，原因与四条排查记录写在 `docs/EVAL.md`，文档中不回避该差距。
- 逐类结果与失败案例：`crescent_gap` 0.907、`punching_hole` 0.942、`welding_line` 0.890、`water_spot` 0.844 较高；`inclusion` 0.248、`rolled_pit` 0.099 偏低，失败样例图附在 `docs/`。
- 精度一致性：FP32 与 PyTorch 的最大绝对差 5.95e-04、平均 4.71e-06；FP16 最大绝对差 1.09e+00、平均 9.66e-03。CPU 部署因此选 FP32。
- 补充短周期训练的对照：5 轮训练 mAP@0.5 为 0.559，反而高于 100 轮训练在第 5 轮的 0.433，原因是学习率调度按 `epochs` 摊平，短周期在第 5 轮已接近收敛。
- 文档中不再硬编码提交次数，这类数字每次修正都会过期。
