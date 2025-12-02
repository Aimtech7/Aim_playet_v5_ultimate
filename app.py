# app.py
# AIM Player v5 Ultimate — Integrated PyQt5 application (simplified production bundle)
# Note: This is a functional scaffold assembled from the project conversation.

# Set environment variable BEFORE importing cv2 to increase ffmpeg read attempts for multi-stream videos
import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

import sys, random, numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QAction, QMenu,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QPushButton,
                             QListWidget, QDockWidget, QListWidgetItem, QInputDialog, QDialog,
                             QColorDialog, QToolBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QColor
import pyaudio
from av_engine import AudioEngine, VideoEngine
from utils.subtitles import SubtitleParser

# add optional VLC support
try:
    import vlc
    _HAS_VLC = True
except Exception:
    _HAS_VLC = False

class SubtitleOverlay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("color:white; background-color:transparent; font-size:18px;")
        self.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
        self.subs_tracks = []
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        # ensure fully translucent background on systems that need the flag
        self.setAttribute(Qt.WA_TranslucentBackground, True)
    def load(self, path):
        sp = SubtitleParser(path)
        self.subs_tracks.append(sp)
    def update_time(self, t):
        text = ""
        for s in self.subs_tracks:
            line = s.get_text(t)
            if line:
                text += line + "\n"
        self.setText(text)

class HotkeysEditor(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hotkeys Editor")
        self.setGeometry(500,200,400,300)

class SkinEditor(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skin Editor")
        self.setGeometry(500,200,400,300)

class LUTDialog(QDialog):
    def __init__(self, video_engine):
        super().__init__()
        self.setWindowTitle("Select LUT")
        self.video_engine = video_engine
        self.setGeometry(500,200,300,200)
        layout = QVBoxLayout()
        for lut in ["None","Warm","Cool","Cinematic","Vintage"]:
            btn = QPushButton(lut)
            btn.clicked.connect(lambda _, l=lut: self.apply(l))
            layout.addWidget(btn)
        self.setLayout(layout)
    def apply(self, lut):
        if self.video_engine:
            self.video_engine.set_lut(lut)
        self.close()

class BookmarksDialog(QDialog):
    def __init__(self, video_engine):
        super().__init__()
        self.video_engine = video_engine
        self.setWindowTitle("Bookmarks")
        self.setGeometry(450,200,400,300)
        layout = QVBoxLayout()
        self.listw = QListWidget()
        layout.addWidget(self.listw)
        add = QPushButton("Add Bookmark")
        add.clicked.connect(self.add)
        layout.addWidget(add)
        self.setLayout(layout)
    def add(self):
        if self.video_engine:
            t = self.video_engine.get_time()
            name, ok = QInputDialog.getText(self, "Name", "Bookmark name:")
            if ok and name:
                self.listw.addItem(f"{name} — {t:.2f}s")

class VisualizerWidget(QWidget):
    def __init__(self, audio_engine=None):
        super().__init__()
        self.audio_engine = audio_engine
        self.setMinimumHeight(140)
        self.freq_data = np.zeros(128)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(60)
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0,0,0))
        w = self.width()/len(self.freq_data)
        for i, v in enumerate(self.freq_data):
            h = self.height()*float(v)
            painter.setBrush(QColor.fromHsv(int(i*3)%360,255,200))
            painter.drawRect(int(i*w), int(self.height()-h), int(w-1), int(h))
    def update(self):
        if self.audio_engine and self.audio_engine.latest_spectrum is not None:
            spec = self.audio_engine.latest_spectrum
            # normalize to 0..1 length 128
            spec = np.array(spec)
            if spec.size >= 128:
                self.freq_data = spec[:128] / (spec.max()+1e-9)
            else:
                padded = np.zeros(128)
                padded[:spec.size] = spec
                self.freq_data = padded / (padded.max()+1e-9)
        else:
            # idle animation
            self.freq_data = np.roll(self.freq_data, -1)
            self.freq_data[-1] = random.random()*0.8

class MiniPlayer(QMainWindow):
    """Floating mini-player window for video playback"""
    def __init__(self, video_engine):
        super().__init__()
        self.video_engine = video_engine
        self.setWindowTitle("Mini Player")
        self.setGeometry(100, 100, 400, 300)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.video_label = QLabel("Mini Player")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

class AIMPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIM Player v5 Ultimate")
        self.setGeometry(50,50,1200,720)
        icon_path = os.path.join("assets","icons","player_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.audio_engine = None
        self.video_engine = None
        self.current_file = None
        # VLC player handles audio playback when available
        self.vlc_instance = vlc.Instance() if _HAS_VLC else None
        self.vlc_player = None
        # Central video layout
        central = QWidget()
        self.setCentralWidget(central)
        self.vlay = QVBoxLayout(central)
        # create a video container so overlay can be parented to it
        self.video_container = QWidget()
        self.video_container.setMinimumHeight(360)
        self.video_container_layout = QVBoxLayout(self.video_container)
        self.video_display = QLabel("No video loaded")
        self.video_display.setAlignment(Qt.AlignCenter)
        # make video area clearly visible so controls are not hidden visually
        self.video_display.setStyleSheet("background-color:black; color:white;")
        self.video_display.setMinimumSize(640, 360)
        self.video_display.setScaledContents(False)
        self.video_container_layout.addWidget(self.video_display)
        self.vlay.addWidget(self.video_container)
        # subtitle overlay (will be shown on top of video_display)
        self.subtitle_overlay = None
        # Playlist
        self.playlist_dock = QDockWidget("Playlist", self)
        self.playlist = QListWidget()
        self.playlist.setDragDropMode(QListWidget.InternalMove)
        self.playlist.itemDoubleClicked.connect(self.play_item)
        self.playlist_dock.setWidget(self.playlist)
        self.addDockWidget(Qt.RightDockWidgetArea, self.playlist_dock)
        # Visualizer
        self.visualizer = VisualizerWidget(None)
        self.vlay.addWidget(self.visualizer)
        # Controls bar (play/pause, stop, seek, volume)
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        self.play_btn = QPushButton("Play")
        self.stop_btn = QPushButton("Stop")
        # keep transport buttons compact so they remain visible
        self.play_btn.setFixedWidth(80)
        self.stop_btn.setFixedWidth(80)
        self.seek_slider = QSlider(Qt.Horizontal); self.seek_slider.setRange(0,1000)
        self.vol_slider = QSlider(Qt.Horizontal); self.vol_slider.setRange(0,100); self.vol_slider.setValue(80)
        ctrl_layout.addWidget(self.play_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.seek_slider, 1)
        ctrl_layout.addWidget(self.vol_slider)
        # connect volume changes
        self.vol_slider.valueChanged.connect(self.set_volume)
        self.vlay.addWidget(ctrl)
        # small status label for playback state/time
        self.status_label = QLabel("Ready")
        self.status_label.setFixedHeight(18)
        self.vlay.addWidget(self.status_label)
        self.play_btn.clicked.connect(self.toggle_play)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.seek_slider.sliderReleased.connect(self.seek_to_slider)
        # Sliders area (left dock)
        self.sliders_dock = QDockWidget("Sliders", self)
        sbox = QWidget()
        sLayout = QVBoxLayout()
        sbox.setLayout(sLayout)
        self.eq_sliders = []
        for i in range(31):
            s = QSlider(Qt.Horizontal)
            s.setMinimum(-15); s.setMaximum(15); s.setValue(0)
            s.valueChanged.connect(self.eq_changed)
            sLayout.addWidget(s)
            self.eq_sliders.append(s)
        # video filters sliders
        self.bright = QSlider(Qt.Horizontal); self.bright.setRange(-50,50); self.bright.setValue(0)
        self.bright.valueChanged.connect(self.video_filters_changed)
        sLayout.addWidget(self.bright)
        self.contrast = QSlider(Qt.Horizontal); self.contrast.setRange(10,300); self.contrast.setValue(100)
        self.contrast.valueChanged.connect(self.video_filters_changed)
        sLayout.addWidget(self.contrast)
        self.sat = QSlider(Qt.Horizontal); self.sat.setRange(0,200); self.sat.setValue(100)
        self.sat.valueChanged.connect(self.video_filters_changed)
        sLayout.addWidget(self.sat)
        self.sliders_dock.setWidget(sbox)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sliders_dock)
        # menus
        self.init_menus()
        # position timer to update seek slider
        self.position_timer = QTimer()
        self.position_timer.setInterval(200)
        self.position_timer.timeout.connect(self._update_seek)
        self.position_timer.start()

    def init_menus(self):
        mb = self.menuBar()
        media = mb.addMenu("Media")
        openf = QAction("Open File", self); openf.triggered.connect(self.open_file)
        media.addAction(openf)
        playback = mb.addMenu("Playback")
        play = QAction("Play/Pause", self); play.triggered.connect(self.toggle_play)
        playback.addAction(play)
        audio = mb.addMenu("Audio")
        ai_eq = QAction("AI Auto-EQ", self, checkable=True); ai_eq.triggered.connect(self.toggle_ai_eq)
        audio.addAction(ai_eq)
        subtitle = mb.addMenu("Subtitle")
        loads = QAction("Load Subtitle", self); loads.triggered.connect(self.load_sub)
        subtitle.addAction(loads)
        tools = mb.addMenu("Tools")
        hotkeys = QAction("Hotkeys Editor", self); hotkeys.triggered.connect(self.open_hotkeys)
        tools.addAction(hotkeys)
        skin = QAction("Skin Editor", self); skin.triggered.connect(self.open_skin)
        tools.addAction(skin)
        lut = QAction("Select LUT", self); lut.triggered.connect(self.open_lut)
        tools.addAction(lut)
        bookmarks = mb.addMenu("Bookmarks")
        bm = QAction("Manage Bookmarks", self); bm.triggered.connect(self.open_bookmarks)
        bookmarks.addAction(bm)
        view = mb.addMenu("View")
        pip = QAction("Mini-Player", self, checkable=True); pip.triggered.connect(self.toggle_mini)
        view.addAction(pip)
    def open_file(self):
        f,_ = QFileDialog.getOpenFileName(self,"Open media","","Media files (*.mp4 *.mkv *.mp3 *.wav *.ogg)")
        if f:
            self.playlist.addItem(f)
            self.play_media(f)
    def play_item(self, item):
        self.play_media(item.text())
    def play_media(self, path):
        self.current_file = path
        # initialize engines
        # stop previous
        try:
            if self.audio_engine:
                self.audio_engine.stop_stream()
        except:
            pass
        try:
            if self.video_engine:
                self.video_engine.stop()
        except:
            pass
        # stop/cleanup previous vlc player
        try:
            if self.vlc_player:
                try:
                    self.vlc_player.stop()
                except:
                    pass
                self.vlc_player = None
        except:
            pass

        self.audio_engine = AudioEngine(path)
        self.visualizer.audio_engine = self.audio_engine
        # set initial volume on engine
        self.audio_engine.volume = self.vol_slider.value()
        self.video_engine = VideoEngine(path)

        # create VLC media player for audio/video if available
        if _HAS_VLC:
            try:
                self.vlc_player = vlc.MediaPlayer(path)
                # apply initial volume if VLC supports it
                try:
                    self.vlc_player.audio_set_volume(self.vol_slider.value())
                except Exception:
                    pass
            except Exception as e:
                print("VLC media player creation failed:", e)
                self.vlc_player = None

        # connect signals
        try:
            if hasattr(self.video_engine, 'frame_time_signal'):
                self.video_engine.frame_time_signal.connect(self.on_frame_time)
        except Exception as e:
            print(f"Warning: Could not connect frame_time_signal: {e}")
        try:
            if hasattr(self.video_engine, 'frame_signal'):
                self.video_engine.frame_signal.connect(self.update_frame)
        except Exception as e:
            print(f"Warning: Could not connect frame_signal: {e}")

        # display first frame
        try:
            frame = self.video_engine.grab_frame()
            if frame is not None:
                h,w,c = frame.shape
                img = QImage(frame.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
                self.video_display.setPixmap(QPixmap.fromImage(img).scaled(self.video_display.size(), Qt.KeepAspectRatio))
        except Exception as e:
            print(f"Warning: Could not grab initial frame: {e}")

        # set duration info if possible (used by seek)
        try:
            fps = self.video_engine.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = self.video_engine.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration = (frames / fps) if fps else 0
            self._duration = max(duration, 0.0)
        except:
            self._duration = 0.0

        # start playback automatically
        self.start_playback()

    def start_playback(self):
        try:
            # start video engine only if not already running
            if self.video_engine and not getattr(self.video_engine, "running", False):
                self.video_engine.start()
            # start VLC player if available
            if self.vlc_player:
                try:
                    self.vlc_player.play()
                except Exception as e:
                    print("VLC play failed:", e)
                self.status_label.setText("Playing (VLC)")
            else:
                self.status_label.setText("Playing (no VLC)")
             # start simulated audio spectrum (kept running regardless to drive visualizer)
            if self.audio_engine:
                self.audio_engine.start_stream()
            # make sure volume applied to engine if using ffmpeg fallback
            try:
                if self.audio_engine:
                    self.audio_engine.volume = self.vol_slider.value()
            except:
                pass
            self.play_btn.setText("Pause")
        except Exception as e:
            print("Error starting playback:", e)

    def toggle_play(self):
        if not self.video_engine:
            return
        # If VLC is available prefer its state to determine action
        if self.vlc_player:
            try:
                if self.vlc_player.is_playing():
                    try:
                        self.vlc_player.pause()
                    except Exception as e:
                        print("VLC pause failed:", e)
                    # pause local video loop too
                    try:
                        self.video_engine.pause()
                    except:
                        pass
                    self.play_btn.setText("Play")
                else:
                    try:
                        self.vlc_player.play()
                    except Exception as e:
                        print("VLC resume failed:", e)
                    # resume video loop
                    try:
                        self.video_engine.start()
                    except:
                        pass
                    self.play_btn.setText("Pause")
            except Exception as e:
                print("VLC toggle error:", e)
            return
        # Fallback when VLC not present: toggle VideoEngine and simulated audio
        if getattr(self.video_engine, 'running', False):
            try:
                self.video_engine.pause()
                if self.audio_engine:
                    self.audio_engine.stop_stream()
                self.play_btn.setText("Play")
            except Exception as e:
                print("Error pausing:", e)
        else:
            try:
                self.video_engine.start()
                if self.audio_engine:
                    self.audio_engine.start_stream()
                self.play_btn.setText("Pause")
            except Exception as e:
                print("Error resuming:", e)

    def stop_playback(self):
        # stop video engine
        if self.video_engine:
            try:
                self.video_engine.stop()
            except:
                pass
        # stop simulated audio
        if self.audio_engine:
            try:
                self.audio_engine.stop_stream()
            except:
                pass
        # stop vlc if present
        if self.vlc_player:
            try:
                self.vlc_player.stop()
            except:
                pass
            # optionally release player object
            self.vlc_player = None
        self.play_btn.setText("Play")
        self.status_label.setText("Stopped")

    def update_frame(self, frame):
        # frame is BGR numpy array
        try:
            h,w,c = frame.shape
            # convert BGR->RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img).scaled(self.video_display.size(), Qt.KeepAspectRatio)
            self.video_display.setPixmap(pix)
            # ensure subtitle overlay is on top and parented to video_display
            if not self.subtitle_overlay:
                self.subtitle_overlay = SubtitleOverlay(self.video_display)
                self.subtitle_overlay.resize(self.video_display.size())
                self.subtitle_overlay.move(0,0)
                self.subtitle_overlay.raise_()
            else:
                self.subtitle_overlay.resize(self.video_display.size())
                self.subtitle_overlay.move(0,0)
        except Exception:
            pass

    def _update_seek(self):
        if self.video_engine and hasattr(self, '_duration') and self._duration > 0:
            try:
                pos = self.video_engine.get_time()
                val = int((pos / self._duration) * 1000)
                self.seek_slider.blockSignals(True)
                self.seek_slider.setValue(max(0, min(1000, val)))
                self.seek_slider.blockSignals(False)
                # update status with current time
                try:
                    self.status_label.setText(f"{pos:.1f}s / {self._duration:.1f}s")
                except:
                    pass
            except:
                pass

    def seek_to_slider(self):
        if not self.video_engine:
            return
        v = self.seek_slider.value()
        if hasattr(self, '_duration') and self._duration > 0:
            sec = (v/1000.0) * self._duration
            # seek video capture (OpenCV)
            try:
                self.video_engine.cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000.0)
            except Exception as e:
                print("Seek failed for video:", e)
            # seek vlc if available (set_time expects milliseconds)
            if self.vlc_player:
                try:
                    self.vlc_player.set_time(int(sec*1000.0))
                except Exception as e:
                    print("VLC seek failed:", e)
            # update status immediately after seeking
            try:
                self.status_label.setText(f"Seek to {sec:.1f}s")
            except:
                pass
    def eq_changed(self):
        if self.audio_engine:
            vals = [s.value() for s in self.eq_sliders]
            self.audio_engine.set_eq(vals)

    def video_filters_changed(self):
        if self.video_engine:
            self.video_engine.set_brightness(self.bright.value()/50.0)
            self.video_engine.set_contrast(self.contrast.value()/100.0)
            self.video_engine.set_saturation(self.sat.value()/100.0)
    def toggle_ai_eq(self, checked):
        if self.audio_engine:
            self.audio_engine.toggle_ai_eq(checked)
    def load_sub(self):
        f,_ = QFileDialog.getOpenFileName(self,"Load subtitle","","Subs (*.srt *.ass *.vtt)")
        if f and self.video_display:
            # parent overlay to video_display so it sits above video
            if not self.subtitle_overlay:
                self.subtitle_overlay = SubtitleOverlay(self.video_display)
                self.subtitle_overlay.resize(self.video_display.size())
                self.subtitle_overlay.move(0,0)
                self.subtitle_overlay.raise_()
            self.subtitle_overlay.load(f)
    def open_bookmarks(self):
        if self.video_engine:
            dlg = BookmarksDialog(self.video_engine)
            dlg.exec_()
    def open_lut(self):
        if self.video_engine:
            dlg = LUTDialog(self.video_engine)
            dlg.exec_()
    def open_hotkeys(self):
        dlg = HotkeysEditor(); dlg.exec_()
    def open_skin(self):
        dlg = SkinEditor(); dlg.exec_()
    def toggle_mini(self, checked):
        if checked and self.video_engine:
            self.mini = MiniPlayer(self.video_engine); self.mini.show()
        elif hasattr(self, 'mini') and self.mini:
            self.mini.close(); self.mini = None
    def on_frame_time(self, t):
        if hasattr(self, 'subtitle_overlay'):
            self.subtitle_overlay.update_time(t)

    def set_volume(self, val):
        # val: 0..100
        if self.vlc_player:
            try:
                # python-vlc uses 0..100
                self.vlc_player.audio_set_volume(int(val))
            except Exception:
                pass
        # apply to our AudioEngine fallback
        if self.audio_engine:
            try:
                self.audio_engine.volume = int(val)
            except Exception:
                pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AIMPlayer()
    win.show()
    sys.exit(app.exec_())