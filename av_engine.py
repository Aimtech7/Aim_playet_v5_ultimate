# av_engine.py
# Audio and Video engine (simplified)
import cv2, numpy as np, threading, time
import torch
import pyaudio
from PyQt5.QtCore import pyqtSignal, QObject
import subprocess, shutil

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
        self.running = False

    def stop(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass

    def _loop(self):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.path)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = 1.0 / fps
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # try to loop video; catch seek errors gracefully
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception as e:
                    # seek failed (multi-stream video issue); stop loop
                    print(f"Seek/loop failed (expected for multi-stream): {e}")
                    break
                continue
            try:
                frame_proc = self.apply_filters(frame)
            except Exception:
                frame_proc = frame
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            try:
                self.frame_time_signal.emit(t)
                self.frame_signal.emit(frame_proc)
            except Exception:
                pass
            time.sleep(delay)

    def apply_filters(self, frame):
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
        self._thread = None
        self._running = False
        self._proc = None
        # audio format params (match ffmpeg args below)
        self.rate = 44100
        self.channels = 2
        self.format = pyaudio.paInt16
        self.frames_per_buffer = 4096
        self.volume = 80  # 0..100
        # check ffmpeg availability
        self._have_ffmpeg = bool(shutil.which("ffmpeg"))

    def set_eq(self, vals):
        self.eq = vals

    def toggle_ai_eq(self, v):
        self.ai_eq_enabled = v

    def start_stream(self):
        if self._running:
            return
        self._running = True

        def run():
            if self._have_ffmpeg:
                cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error',
                    '-i', self.path, '-vn',
                    '-f', 's16le', '-acodec', 'pcm_s16le',
                    '-ar', str(self.rate), '-ac', str(self.channels),
                    'pipe:1'
                ]
                try:
                    self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except Exception:
                    self._proc = None

            if self._proc:
                try:
                    self.stream = self.p.open(format=self.format,
                                              channels=self.channels,
                                              rate=self.rate,
                                              output=True,
                                              frames_per_buffer=self.frames_per_buffer)
                except Exception:
                    try:
                        self._proc.kill()
                    except:
                        pass
                    self._proc = None

            while self._running:
                if self._proc and self._proc.stdout:
                    try:
                        chunk_bytes = self.frames_per_buffer * self.channels * 2
                        data = self._proc.stdout.read(chunk_bytes)
                        if not data:
                            # EOF
                            break
                        # apply volume scaling if needed
                        if self.stream:
                            if self.volume != 100:
                                try:
                                    arr = np.frombuffer(data, dtype=np.int16).astype(np.int32)
                                    factor = float(self.volume) / 100.0
                                    arr = np.clip((arr * factor), -32768, 32767).astype(np.int16)
                                    out = arr.tobytes()
                                except Exception:
                                    out = data
                            else:
                                out = data
                            try:
                                self.stream.write(out)
                            except Exception:
                                # ignore write errors
                                pass
                        # compute simple spectrum from left channel
                        try:
