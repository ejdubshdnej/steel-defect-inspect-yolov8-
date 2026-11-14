"""PyQt6 上位机：左侧显示图片与检测框，右侧调阈值、看结果表。

用法：
    python hmi/app.py                 # 有控制台时用这个，能看日志
    launch_hmi.bat                    # 双击启动，走 pythonw.exe，不弹黑框

功能：
- 上传图片（按钮 / 拖拽 / Ctrl+V 粘贴剪贴板）
- 拖动置信度滑块即时重跑，结果表按置信度排序
- 支持 jpg / png / bmp / tif / webp，灰度、带 alpha、16bit 都会统一转成 8bit BGR

注意事项：
- 模型固定读 exports/best_fp32.onnx，换模型请改此处或改用 HTTP 服务。
- pythonw 下没有控制台，sys.stdout/stderr 是 None，所以下面的 reconfigure
  必须先判空，否则启动即崩。
"""
import sys, os
# pythonw 下没有控制台，stdout/stderr 是 None，直接 reconfigure 会崩
for _s in (sys.stdout, sys.stderr):
    if _s is not None:
        _s.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import numpy as np, cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPainter, QPen,
                         QKeySequence, QShortcut)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QSlider, QTableWidget, QTableWidgetItem,
                             QHBoxLayout, QVBoxLayout, QFileDialog, QHeaderView,
                             QMessageBox)
from serve.infer import Detector

NAMES = {0: "crease", 1: "crescent_gap", 2: "inclusion", 3: "oil_spot",
         4: "punching_hole", 5: "rolled_pit", 6: "silk_spot",
         7: "waist_folding", 8: "water_spot", 9: "welding_line"}
COLOR = QColor(220, 40, 40)

# 上传相关的常量：视图样式（拖拽时高亮）与允许的图片后缀
VIEW_QSS = "background:#1e1e1e;color:#bbb;border:1px solid #333"
VIEW_QSS_DROP = "background:#1e1e1e;color:#bbb;border:2px dashed #6ea8ff"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
IMG_FILTER = "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)"


def imdecode_any(path):
    """读图并统一成 8bit BGR。cv2.imread 在中文路径下返回 None，必须走 imdecode。"""
    buf = np.fromfile(path, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if buf.size else None
    if img is None:
        return None
    if img.ndim == 2:                                  # 灰度
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:                              # 带 alpha 的 png
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 2:                            # 灰度+alpha
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:                          # 16bit tif 等
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


class ImageView(QLabel):
    """图像显示区，同时是上传的落点：支持拖入文件、拖拽时高亮边框。"""

    def __init__(self, on_drop):
        super().__init__()
        self._on_drop = on_drop
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(VIEW_QSS)
        self.setText("点击「上传图片」\n或把图片拖到这里（Ctrl+V 粘贴剪贴板）")

    @staticmethod
    def _image_from(mime):
        if not mime.hasUrls():
            return None
        for u in mime.urls():
            p = u.toLocalFile()
            if p and p.lower().endswith(IMG_EXTS) and os.path.isfile(p):
                return p
        return None

    def dragEnterEvent(self, e):
        if self._image_from(e.mimeData()):
            e.acceptProposedAction()
            self.setStyleSheet(VIEW_QSS_DROP)
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self.setStyleSheet(VIEW_QSS)

    def dropEvent(self, e):
        self.setStyleSheet(VIEW_QSS)
        p = self._image_from(e.mimeData())
        if p:
            e.acceptProposedAction()
            self._on_drop(p)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("钢带表面缺陷检测 · 上位机")
        self.resize(1280, 760)
        self.det = Detector(os.path.join(BASE, "exports", "best_fp32.onnx"), names=NAMES)
        self.img = None
        self.img_path = None
        self._dir = os.path.join(BASE, "data", "gc10", "images", "valid")
        self._build()

    def _build(self):
        root = QWidget(); self.setCentralWidget(root)
        h = QHBoxLayout(root)

        # 左：图像
        left = QVBoxLayout()
        self.view = ImageView(self.load)
        self.view.setMinimumSize(760, 620)
        left.addWidget(self.view)
        row = QHBoxLayout()
        b1 = QPushButton("上传图片"); b1.clicked.connect(self.open_img)
        b2 = QPushButton("重新检测"); b2.clicked.connect(self.run)
        row.addWidget(b1); row.addWidget(b2)
        left.addLayout(row)
        h.addLayout(left, 4)

        # 右：参数
        right = QVBoxLayout()
        right.addWidget(QLabel("置信度阈值"))
        self.sl = QSlider(Qt.Orientation.Horizontal)
        self.sl.setRange(5, 95); self.sl.setValue(25)
        self.sl.valueChanged.connect(self.on_conf)
        right.addWidget(self.sl)
        self.lb = QLabel("0.25")
        right.addWidget(self.lb)

        right.addWidget(QLabel("检测结果"))
        self.tb = QTableWidget(0, 4)
        self.tb.setHorizontalHeaderLabels(["类别", "置信度", "x1,y1", "x2,y2"])
        self.tb.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right.addWidget(self.tb)

        self.stat = QLabel("就绪 · 支持拖拽图片或 Ctrl+V 粘贴")
        self.stat.setStyleSheet("color:#666")
        right.addWidget(self.stat)
        h.addLayout(right, 2)

        # 快捷键：Ctrl+V 粘贴剪贴板、Ctrl+O 打开文件
        QShortcut(QKeySequence.StandardKey.Paste, self, activated=self.paste_img)
        QShortcut(QKeySequence.StandardKey.Open, self, activated=self.open_img)

    def on_conf(self, v):
        self.lb.setText(f"{v/100:.2f}")
        self.det.conf = v / 100.0
        if self.img is not None:
            self.run()

    def open_img(self):
        start = self._dir if os.path.isdir(self._dir) else BASE
        p, _ = QFileDialog.getOpenFileName(self, "上传图片", start, IMG_FILTER)
        if p:
            self.load(p)

    def paste_img(self):
        mime = QApplication.clipboard().mimeData()
        if mime.hasImage():
            img = QImage(mime.imageData())
            # QImage → numpy BGR，与 imdecode 的产出保持一致，后续流程不用分支
            img = img.convertToFormat(QImage.Format.Format_RGB888)
            w, h = img.width(), img.height()
            ptr = img.constBits()
            ptr.setsize(img.sizeInBytes())
            arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 3, 3)[:, :w]
            self.set_image(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), "剪贴板")
            return
        if mime.hasUrls():
            for u in mime.urls():
                p = u.toLocalFile()
                if p:
                    self.load(p)
                    return
        self.stat.setText("剪贴板里没有图片")

    def load(self, path):
        """统一入口：文件对话框、拖拽、粘贴的文件路径都走这里。"""
        img = imdecode_any(path)
        if img is None:
            QMessageBox.warning(self, "上传失败",
                                f"无法读取这张图片：\n{path}\n\n"
                                f"支持格式：{' '.join(IMG_EXTS)}")
            return
        self.set_image(img, path)

    def set_image(self, img, name):
        self.img = img
        self.img_path = name
        if name != "剪贴板":
            d = os.path.dirname(name)
            if d:
                self._dir = d
        self.run()
        if name != "剪贴板":
            self.stat.setText(f"{self._base_stat} · {os.path.basename(name)}")
        else:
            self.stat.setText(f"{self._base_stat} · 来自剪贴板")

    def run(self):
        if self.img is None:
            return
        r = self.det.predict(self.img)
        self._base_stat = (f"{len(r['detections'])} 个缺陷 · {r['elapsed_ms']} ms · "
                           f"{self.img.shape[1]}x{self.img.shape[0]}")
        self.stat.setText(self._base_stat)
        self.draw(r["detections"])
        self.fill(r["detections"])

    def draw(self, dets):
        img = self.img.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 220), 2)
            cv2.putText(img, f'{d["name"]} {d["conf"]:.2f}', (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 220), 2)
        h, w = img.shape[:2]
        q = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self.view.setPixmap(QPixmap.fromImage(q).scaled(
            self.view.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def fill(self, dets):
        self.tb.setRowCount(len(dets))
        for i, d in enumerate(dets):
            self.tb.setItem(i, 0, QTableWidgetItem(d["name"]))
            self.tb.setItem(i, 1, QTableWidgetItem(f'{d["conf"]:.3f}'))
            self.tb.setItem(i, 2, QTableWidgetItem(f'{d["xyxy"][0]:.0f},{d["xyxy"][1]:.0f}'))
            self.tb.setItem(i, 3, QTableWidgetItem(f'{d["xyxy"][2]:.0f},{d["xyxy"][3]:.0f}'))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec())
