import sys, time, threading
import cv2 
import numpy as np 
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

CAP_INDEX = 0
REQ_W, REQ_H = 320, 240
REQ_FPS = 60

class CamWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.fps = 0 
        
    def run(self):
        cap = cv2.VideoCapture(CAP_INDEX, cv2.CAP_ANY)
        
        # Request mode
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
        cap.set(cv2.CAP_PROP_FPS, REQ_FPS)
                
        t0 = time.time()
        cnt = 0
        
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            
            if not ok:
                time.sleep(0.001) 
                continue 
            
            # ensure size 
            if frame.shape[1] != REQ_W or frame.shape[0] != REQ_H:
                frame = cv2.resize(frame, (REQ_W, REQ_H), interpolation=cv2.INTER_NEAREST)
            
            with self.lock:
                self.frame = frame 
            
            cnt += 1
            # FPS meter (1-sec window)
            now = time.time()
            if now - t0 > 1.0:
                self.fps = cnt / (now - t0)
                t0 = now 
                cnt = 0
                
        cap.release()
        
    def get_latest(self):
        with self.lock:
            return (None,None) if self.frame is None else self.frame.copy(), self.fps
        

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live retina")
        
        
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        h = QtWidgets.QHBoxLayout(cw)
        
        # image view (fast)
        pg.setConfigOptions(useOpenGL=False, imageAxisOrder='row-major')
        self.gv = pg.GraphicsLayoutWidget()
        h.addWidget(self.gv, 3)
        self.vb = self.gv.addViewBox(lockAspect=True)
        self.vb.invertY(True)                  # make (0,0) top-left
        self.img_item = pg.ImageItem(axisOrder='row-major')
        self.vb.addItem(self.img_item)
        self.img_item.setAutoDownsample(True)
        
        # ---- controls
        panel = QtWidgets.QWidget(); h.addWidget(panel, 1)
        form = QtWidgets.QFormLayout(panel)

        self.alpha = QtWidgets.QDoubleSpinBox()
        self.alpha.setRange(0.1, 4.0); self.alpha.setSingleStep(0.1); self.alpha.setValue(1.0)

        self.ksize = QtWidgets.QSpinBox()
        self.ksize.setRange(1, 31); self.ksize.setSingleStep(2); self.ksize.setValue(1)  # odd only

        self.invert = QtWidgets.QCheckBox(); self.invert.setChecked(False)

        self.fps_label = QtWidgets.QLabel("FPS: –")

        form.addRow("Alpha (float)", self.alpha)
        form.addRow("Blur ksize (int, odd)", self.ksize)
        form.addRow("Invert (bool)", self.invert)
        form.addRow(self.fps_label)

        # ---- worker
        self.worker = CamWorker(); self.worker.start()

        # ---- UI timer (try to refresh near 120 Hz; will drop gracefully)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / REQ_FPS))  # ~8 ms
        
        time.sleep(1.0)

    
    def update_frame(self):
        frm, cam_out_fps = self.worker.get_latest()
        if frm is None: return

        # BGR -> Gray [0..1]
        gray = cv2.cvtColor(frm, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # int slider: Gaussian blur (force odd)
        k = int(self.ksize.value())
        if k % 2 == 0: k += 1
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0, borderType=cv2.BORDER_REPLICATE)

        # float slider: alpha
        gray = np.clip(gray * float(self.alpha.value()), 0.0, 1.0)

        # bool: invert
        if self.invert.isChecked():
            gray = 1.0 - gray

        # show (uint8 is fine; avoid autoLevels recalcs each frame)
        img8 = (gray * 255.0).astype(np.uint8)
        self.img_item.setImage(img8, autoLevels=False)

        # fps readout from capture thread
        self.fps_label.setText(f"FPS: {self.worker.fps:.1f}")

    def closeEvent(self, e):
        self.worker.stop_evt.set()
        self.worker.join(timeout=1.0)
        super().closeEvent(e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = App(); w.resize(900, 600); w.show()
    sys.exit(app.exec())