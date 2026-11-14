"""Detector：封装 ONNX 推理，供 HTTP 服务与 PyQt6 上位机共用。

用法：
    from serve.infer import Detector
    det = Detector("exports/best_fp32.onnx", conf=0.25, names=NAMES)
    r = det.predict(img_bgr)     # -> {"detections": [...], "elapsed_ms": 3.4}

设计要点：
- 输入尺寸从 ONNX 的 input shape 自读，训练分辨率变了这里自动跟上，不写死。
- 构造时主动 warmup，把 onnxruntime 的图优化与内存分配开销提前吃掉。
- 灰度图、BGR 图、2048x1000 的大图都直接吃，前处理内部统一做 letterbox。
"""
import os, time
import numpy as np, cv2, onnxruntime as ort

IMGSZ = 320


class Detector:
    def __init__(self, onnx_path, conf=0.25, nms_iou=0.45, threads=4, names=None):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so,
                                         providers=["CPUExecutionProvider"])
        it = self.sess.get_inputs()[0]
        self.iname = it.name
        # 输入尺寸从模型自身读，不写死：训练分辨率变了这里自动跟上
        # shape 形如 ['batch', 3, 320, 320] 或 [1, 3, 320, 320]
        h = it.shape[2] if isinstance(it.shape[2], int) else IMGSZ
        w = it.shape[3] if isinstance(it.shape[3], int) else IMGSZ
        self.imgsz = (h, w)
        # FP16 模型必须喂 float16，否则 onnxruntime 直接抛
        # InvalidArgument: Unexpected input data type. Actual: tensor(float), expected: (tensor(float16))
        self.dtype = np.float16 if "float16" in it.type else np.float32
        self.conf, self.nms_iou = conf, nms_iou
        self.names = names or {}
        self.warmup()

    # ---- 预处理 ----
    def _pre(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ih, iw = self.imgsz
        h, w = img.shape[:2]
        r = min(ih / h, iw / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (ih - nh) // 2, (iw - nw) // 2
        canvas = np.full((ih, iw, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(self.dtype) / 255.0
        return np.ascontiguousarray(x), r, (left, top)

    # ---- 后处理 ----
    def _post(self, out, r, pad, shape):
        p = out[0].T
        scores = p[:, 4:]
        cls = scores.argmax(1)
        sc = scores[np.arange(len(p)), cls]
        keep = sc > self.conf
        if not keep.any():
            return []
        box, sc, cls = p[keep, :4], sc[keep], cls[keep]
        cx, cy, w, h = box.T
        l, t = pad
        H, W = shape
        xyxy = np.stack([(cx - w / 2 - l) / r, (cy - h / 2 - t) / r,
                         (cx + w / 2 - l) / r, (cy + h / 2 - t) / r], 1)
        # 必须裁到图内：letterbox 反变换后坐标可能为负或超出边界，
        # 不裁会让上位机画出跑出画面的框（现象见 docs/VERIFICATION.md）
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
        # 裁完可能宽高归零，丢掉
        valid = (xyxy[:, 2] - xyxy[:, 0] > 1.0) & (xyxy[:, 3] - xyxy[:, 1] > 1.0)
        if not valid.all():
            xyxy, sc, cls = xyxy[valid], sc[valid], cls[valid]
            if len(xyxy) == 0:
                return []
        idx = cv2.dnn.NMSBoxes(xyxy.tolist(), sc.tolist(), self.conf, self.nms_iou)
        out_list = []
        for i in np.array(idx).flatten():
            out_list.append({
                "cls": int(cls[i]),
                "name": self.names.get(int(cls[i]), str(int(cls[i]))),
                "conf": round(float(sc[i]), 4),
                "xyxy": [round(float(v), 1) for v in xyxy[i]],
            })
        out_list.sort(key=lambda d: -d["conf"])
        return out_list

    # ---- 对外 ----
    def warmup(self, rounds=3):
        """主动预热：把 onnxruntime 的图优化与内存分配开销提前吃掉。

        不做这一步，第一个真实请求要多等几十毫秒——产线上这个抖动
        会直接影响节拍。/health 一旦返回 ok，就必须是随时可服务的状态。
        """
        ih, iw = self.imgsz
        z = np.zeros((ih, iw), np.uint8)
        for _ in range(rounds):
            self.predict(z)

    def predict(self, img):
        s = time.perf_counter()
        x, r, pad = self._pre(img)
        out = self.sess.run(None, {self.iname: x})[0]
        dets = self._post(out, r, pad, img.shape[:2])
        return {"detections": dets, "elapsed_ms": round((time.perf_counter() - s) * 1e3, 2)}
