# Contributing · 贡献指南

本文说明如何搭建开发环境、本仓库的代码风格、提交 PR 前该跑什么，以及反馈问题需要提供哪些信息。

## 1 Environment · 环境准备

两套依赖刻意分开：部署侧永远不需要装 PyTorch。

```bash
# 仅推理 / 服务（CPU，无 CUDA，无 PyTorch）
pip install -r requirements.txt

# 训练 + 评测 + 导出
pip install -r requirements-dev.txt
```

- Python 3.x。本项目在 Windows 上开发与验证，但代码不依赖 Windows 专有接口。
- `torch` 的 CUDA 12 轮子不在 PyPI，需按 `requirements-dev.txt` 里的注释改用 PyTorch 官方源安装。
- 数据集 GC10-DET **不在本仓库重新分发**。先跑 `python scripts/download_gc10.py`，再跑 `python scripts/make_splits.py` 生成划分。
- Windows 上 `train.py` 固定 `workers=0`，改成大于 0 会报 `WinError 5`。这行不要「顺手修掉」。

## 2 Code style · 代码风格

- PEP 8，4 空格缩进，不用制表符。推送前 `python -m py_compile <file>` 必须通过。
- **只改必要处**。在同一个 PR 里重排一个没碰过的函数，会让改动无法评审。
- 注释用文件里已有的语言；注释解释**为什么**这么选（约束是什么、代价是什么），不重复代码已经说清楚的事。
- 不要写死可以推导的值。检测分辨率、类别数、置信度阈值从模型或环境变量（`DSI_ONNX` / `DSI_CONF` / `DSI_PORT`）读取，而不是钉在代码里。
- 文件统一 UTF-8 + LF；支持 EditorConfig 的编辑器会按 `.editorconfig` 自动处理。
- 文档里的数字都来自实测。改动指标、延迟或模型体积时，在同一次提交里同步改掉报告它的文档，并写清测的是哪台硬件、哪种配置。

## 3 Before opening a pull request · 提交前

跑一遍你改动涉及的那部分链路：

```bash
python -m py_compile train.py export_onnx.py verify_onnx.py

# 标注质检：越界框 / 微小框 / 缺失标注 / 重复图
python scripts/check_labels.py

# ONNX 结构 + 与 PyTorch 的数值一致性
python verify_onnx.py            # 也可指定前缀，如 `verify_onnx.py best640`
```

`scripts/check_labels.py` 会写 `docs/label_check.csv`；`verify_onnx.py` 拿 `exports/*.onnx` 与对应的 `runs/detect/*/weights/best.pt` 比对。

PR 描述请覆盖四点：

1. **改了什么、为什么** —— 讲问题，不要只贴 diff。
2. **怎么验证的** —— 跑了哪些命令、输出是什么、在什么硬件上。
3. **代价是什么** —— 对延迟、体积、内存、精度的影响（如果有）。
4. **还有什么不确定** —— 未测试的平台、数据集或边界情况。

如果改动与文档里的某个数字冲突（mAP、延迟、产物大小），在描述里明说，不要悄悄改数字。

## 4 Bug reports · 问题反馈

请提供：

- 操作系统与 Python 版本，以及 `onnxruntime` / `torch` / `ultralytics` 版本（`pip show onnxruntime ultralytics torch`）。
- 完整命令与完整 traceback，不要转述报错。
- 加载的是哪份产物（`exports/best_fp32.onnx`、`exports/best_fp16.onnx`、某个 `best.pt`，还是自己导出的），以及字节数。
- 精度或检出异常：输入图、期望的框，以及当前使用的 `conf` / `nms_iou`。FP16 与 FP32 的输出差异是可测量的，先跑 `verify_onnx.py` 再判断是不是回归。
- 服务异常：`/health` 与 `/predict` 的响应体与状态码。

## 5 License · 许可

贡献以本仓库的 MIT 协议接收（见 `LICENSE`），提交 PR 即表示同意按该协议发布你的贡献。

以下两项第三方约束**不在 MIT 范围内**：

- 训练与导出使用 **Ultralytics YOLOv8（AGPL-3.0）**。由它衍生的 `runs/**/weights/*.pt` 与 `exports/*.onnx` 仍为 AGPL-3.0，不适用 MIT。闭源商用需另行评估 Ultralytics 商业授权。
- **GC10-DET** 版权归原作者，本仓库只提供下载脚本、不重新分发图像。

完整说明见 `NOTICE`。若你的改动需要引入第三方内容，请在同一个 PR 里补上许可说明。
