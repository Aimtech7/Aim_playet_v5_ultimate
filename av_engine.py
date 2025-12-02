# av_engine.py
# Audio and Video engine (simplified)
import cv2, numpy as np, threading, time
import torch
import pyaudio
from PyQt5.QtCore import pyqtSignal, QObject

class VideoEngine(QObject):
    frame_time_signal = pyqtSignal(float)
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.running = False
        self.brightness = 0
        self.contrast = 1.0
        self.saturation = 1.0
        self.filters = {'sharpen':False, 'blur':False, 'lut':None}
        self.ai_model = None
    def load_ai(self, model_path):
        try:
            self.ai_model = torch.load(model_path, map_location='cpu')
            self.ai_model.eval()
        except Exception as e:
            print("AI load failed:", e)
    def grab_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
    def stop(self):
        self.running = False
    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self.frame_time_signal.emit(t)
            time.sleep(1/30)
    def apply_filters(self, frame):
        # brightness/contrast/saturation
        frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness*50)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(float)
        hsv[:,:,1] = np.clip(hsv[:,:,1]*self.saturation,0,255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if self.filters.get('sharpen'):
            k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            frame = cv2.filter2D(frame,-1,k)
        if self.filters.get('blur'):
            frame = cv2.GaussianBlur(frame,(5,5),0)
        return frame
    def ai_enhance(self, frame):
        return frame
    def set_brightness(self,v): self.brightness = v
    def set_contrast(self,v): self.contrast = v
    def set_saturation(self,v): self.saturation = v
    def set_lut(self, name):
        self.filters['lut'] = name
    def get_time(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

class AudioEngine:
    def __init__(self, path):
        self.path = path
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.eq = [0]*31
        self.latest_spectrum = None
        self.ai_eq_enabled = False
    def set_eq(self, vals):
        self.eq = vals
    def toggle_ai_eq(self, v):
        self.ai_eq_enabled = v
    def start_stream(self):
        # We'll spawn a thread that reads audio using ffmpeg or pyaudio in real product.
        # Here, simulate spectrum generation for visualizer.
        import threading, time, numpy as np
        def loop():
            while True:
                # fake spectrum
                self.latest_spectrum = np.abs(np.random.randn(256))
                time.sleep(0.05)
        t = threading.Thread(target=loop, daemon=True)
        t.start()