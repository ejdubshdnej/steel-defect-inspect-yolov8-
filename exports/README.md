# exports/

模型文件目录。**仓库不包含任何 `.onnx` 文件**（`.gitignore` 已排除 `exports/*.onnx`）。

## 为什么是空的

四个 ONNX 合计 37 MB：`best_fp32.onnx` 12.0 MB、`best_fp16.onnx` 6.0 MB、`best640_fp32.onnx` 12.2 MB、`best640_fp16.onnx` 6.1 MB。二进制模型跟代码一起进版本库会带来三个问题：

- clone 体积翻倍，且每次换模型都在历史里留一份不可压缩的副本
- diff 里看不到有意义的变更，评审无从判断模型是否被换过
- 模型该按版本号分发，而不是跟着代码一起滚

## 怎么拿到

**路线 A：下载预导出件**

在仓库的 Releases 页面下载所需文件，放进本目录：

| 文件 | 字节数 | 说明 |
|---|---|---|
| `best_fp32.onnx` | 12,629,151 | **默认**，imgsz 320，CPU 部署用这份 |
| `best_fp16.onnx` | 6,338,321 | 压体积用，CPU 上更慢且误差大，见 `docs/EVAL.md` |
| `best640_fp32.onnx` | 12,755,155 | 可选高精度版，imgsz 640，延迟约 3.6–4.2× |
| `best640_fp16.onnx` | 6,401,333 | 同上，CPU 不可用 |

**路线 B：自己训练并导出**

```bash
python train.py --data data/gc10/data.yaml --epochs 100 --batch 16 --imgsz 320 --name train
python export_onnx.py      # 导出 FP32 + FP16 两份到本目录
python verify_onnx.py      # 核对 shape / opset / 与 torch 的一致性
```

## 下游怎么找模型

`serve/main.py`、`hmi/app.py`、`bench/*.py` 都只认**路径**，不关心文件是下载的还是导出的。默认相对路径是 `exports/best_fp32.onnx`（按项目根解析），所以两条路线的产物放进来即可，下游命令不用区分。

要换模型时用环境变量，代码一行不用改：

```bash
set DSI_ONNX=exports/best640_fp32.onnx     # Windows cmd
python serve/main.py                        # /health 会报 imgsz: 640
```

## 许可提示

`.onnx` 由 Ultralytics YOLOv8 衍生而来，适用 **AGPL-3.0**，不受本仓库 MIT 许可覆盖。详见仓库根目录的 `NOTICE`。
