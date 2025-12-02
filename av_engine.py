# av_engine.py
# Audio and Video engine (simplified)
import cv2, numpy as np, threading, time
import torch
import pyaudio
from PyQt5.QtCore import pyqtSignal, QObject

class VideoEngine(QObject):
    frame_time_signal = pyqtSignal(float)
    frame_signal = pyqtSignal(object)  # emits numpy BGR frame
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.running = False
        self._thread = None
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
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    def pause(self):
        # stop emitting frames but keep capture open (so resume keeps position)
        self.running = False
    def stop(self):
        self.running = False
        # release capture so repeated loads work
        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass
    def _loop(self):
        # ensure capture exists
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.path)
        # compute fps-based sleep
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = 1.0 / fps
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # apply filters before emitting
            try:
                frame_proc = self.apply_filters(frame)
            except Exception:
                frame_proc = frame
            # emit time and raw frame (BGR numpy array)
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            try:
                self.frame_time_signal.emit(t)
                self.frame_signal.emit(frame_proc)
            except Exception:
                pass
            time.sleep(delay)
        # cleanup if thread exits
        try:
            # keep capture open for pause; on full stop user should call stop()
            pass
        except:
            pass
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
        self._spec_thread = None
        self._spec_running = False
        self.volume = 0.8  # 0.0..1.0
    def set_eq(self, vals):
        self.eq = vals
    def toggle_ai_eq(self, v):
        self.ai_eq_enabled = v
    def set_volume(self, v):
        # v expected 0..100 from UI slider; store normalized
        try:
            self.volume = max(0.0, min(1.0, float(v)/100.0))
        except:
            pass
    def start_stream(self):
        # spawn a thread that simulates spectrum for visualizer and can be stopped
        import threading, time, numpy as np
        if self._spec_running:
            return
        self._spec_running = True
        def loop():
            while self._spec_running:
                # fake spectrum (volume affects amplitude)
                self.latest_spectrum = np.abs(np.random.randn(256)) * (0.2 + 0.8*self.volume)
                time.sleep(0.05)
        self._spec_thread = threading.Thread(target=loop, daemon=True)
        self._spec_thread.start()
    def stop_stream(self):
        try:
            self._spec_running = False
            self._spec_thread = None
        except:
            pass