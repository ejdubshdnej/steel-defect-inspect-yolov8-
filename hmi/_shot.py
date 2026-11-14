"""无头环境下截图上位机界面，产出 docs/ 下的三张图：正常 / 高阈值 / 低阈值。

用法：
    python hmi/_shot.py

产物：docs/hmi_1_normal.png（conf 0.25）、docs/hmi_2_high.png（0.90）、
docs/hmi_3_low.png（0.05），用于 README 与文档。

注意事项：
- 用 QT_QPA_PLATFORM=offscreen 在没有显示器的环境里渲染窗口。
- offscreen 平台插件**不加载系统字体**，不显式注册字体的话中文会变成方块，
  所以下面必须调 QFontDatabase.addApplicationFont 并 setFont。
- 依赖 C:/Windows/Fonts/msyh.ttc（微软雅黑），非 Windows 系统需换字体路径。
- 示例图优先用 docs/sample.jpg；不存在时从验证集挑一张并写回该路径。
"""
import os, sys, glob
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import numpy as np, cv2
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

app = QApplication(sys.argv)
fid = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
print("font id", fid, "families", QFontDatabase.applicationFontFamilies(fid))
app.setFont(QFont("Microsoft YaHei", 9))
print("app font family:", app.font().family())

from hmi.app import Main

w = Main()
w.show()

# 选一张 GC10 验证图（优先挑检测数适中的，展示效果好）
cands = sorted(glob.glob(os.path.join(BASE, "data", "gc10", "images", "valid", "*.jpg")))
if not cands:
    cands = sorted(glob.glob(os.path.join(BASE, "data", "gc10", "images", "test", "*.jpg")))
pick = os.path.join(BASE, "docs", "sample.jpg")
if os.path.exists(pick):
    img = cv2.imdecode(np.fromfile(pick, np.uint8), cv2.IMREAD_UNCHANGED)
else:
    img = cv2.imdecode(np.fromfile(cands[0], np.uint8), cv2.IMREAD_UNCHANGED)
    cv2.imencode(".jpg", img)[1].tofile(pick)
w.img = img

docs = os.path.join(BASE, "docs")
os.makedirs(docs, exist_ok=True)
for conf, name in ((0.25, "hmi_1_normal.png"), (0.90, "hmi_2_high.png"), (0.05, "hmi_3_low.png")):
    w.det.conf = conf
    w.sl.blockSignals(True)
    w.sl.setValue(int(conf * 100))
    w.lb.setText(f"{conf:.2f}")
    w.sl.blockSignals(False)
    w.run()
    p = os.path.join(docs, name)
    ok = w.grab().save(p)
    print(f"conf={conf:.2f} stat: {w.stat.text()} rows {w.tb.rowCount()} "
          f"-> {name} saved={ok} {os.path.getsize(p)} B")
print("img size", img.shape)
