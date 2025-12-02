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
import json
from pathlib import Path
from datetime import datetime

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

class FileInfoDialog(QDialog):
    """Display media file information"""
    def __init__(self, path, video_engine=None):
        super().__init__()
        self.setWindowTitle("File Information")
        self.setGeometry(400, 300, 500, 300)
        layout = QVBoxLayout()
        
        # basic info
        try:
            fsize = Path(path).stat().st_size / (1024*1024)  # MB
            cdate = datetime.fromtimestamp(Path(path).stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        except:
            fsize, cdate = 0, "N/A"
        
        fname = Path(path).name
        layout.addWidget(QLabel(f"<b>Filename:</b> {fname}"))
        layout.addWidget(QLabel(f"<b>File Size:</b> {fsize:.2f} MB"))
        layout.addWidget(QLabel(f"<b>Created:</b> {cdate}"))
        
        # video info from VideoEngine
        if video_engine:
            try:
                w = int(video_engine.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(video_engine.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = video_engine.cap.get(cv2.CAP_PROP_FPS) or 0
                frames = int(video_engine.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps else 0
                layout.addWidget(QLabel(f"<b>Resolution:</b> {w}×{h}"))
                layout.addWidget(QLabel(f"<b>FPS:</b> {fps:.2f}"))
                layout.addWidget(QLabel(f"<b>Duration:</b> {duration:.1f}s ({int(duration//60)}m {int(duration%60)}s)"))
            except:
                pass
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        self.setLayout(layout)

# new Settings dialog
class SettingsDialog(QDialog):
	def __init__(self, settings):
		super().__init__()
		self.setWindowTitle("Settings")
		self.setGeometry(450,250,350,180)
		self.settings = settings
		l = QVBoxLayout(self)
		# default volume
		l.addWidget(QLabel("Default volume"))
		self.vol = QSlider(Qt.Horizontal); self.vol.setRange(0,100); self.vol.setValue(self.settings.get("default_volume",80))
		l.addWidget(self.vol)
		# theme selection
		l.addWidget(QLabel("Theme"))
		from PyQt5.QtWidgets import QComboBox
		self.theme = QComboBox(); self.theme.addItems(["light","dark"])
		self.theme.setCurrentText(self.settings.get("theme","light"))
		l.addWidget(self.theme)
		# auto-hide controls
		self.auto_hide_cb = QPushButton("Auto-hide controls in fullscreen: " + ("On" if self.settings.get("auto_hide", True) else "Off"))
		self.auto_hide_cb.setCheckable(True)
		self.auto_hide_cb.setChecked(self.settings.get("auto_hide", True))
		self.auto_hide_cb.clicked.connect(lambda: self.auto_hide_cb.setText("Auto-hide controls in fullscreen: " + ("On" if self.auto_hide_cb.isChecked() else "Off")))
		l.addWidget(self.auto_hide_cb)
		# buttons
		btns = QHBoxLayout()
		ok = QPushButton("OK"); ok.clicked.connect(self.accept)
		cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
		btns.addWidget(ok); btns.addWidget(cancel)
		l.addLayout(btns)
	def get_values(self):
		return {"default_volume": self.vol.value(), "theme": self.theme.currentText(), "auto_hide": self.auto_hide_cb.isChecked()}

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
        self.current_playlist_index = -1  # track current item in playlist
        self.repeat_mode = 0  # 0=off, 1=one, 2=all
        self.shuffle_enabled = False
        self.playback_speed = 1.0
        # VLC player handles audio playback when available
        self.vlc_instance = vlc.Instance() if _HAS_VLC else None
        self.vlc_player = None
        # Recent files tracking
        self.recent_files = []  # list of recent file paths
        self.recent_files_path = Path(__file__).parent / "recent_files.json"
        self.load_recent_files()
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
        # Controls bar (play/pause, stop, seek, volume, speed)
        self.ctrl = QWidget()  # replaced local ctrl variable with self.ctrl
        self.ctrl_layout = QHBoxLayout(self.ctrl)
        self.play_btn = QPushButton("Play")
        self.stop_btn = QPushButton("Stop")
        self.prev_btn = QPushButton("◄◄")  # Previous
        self.next_btn = QPushButton("►►")  # Next
        self.repeat_btn = QPushButton("Repeat: Off")
        self.shuffle_btn = QPushButton("Shuffle: Off")
        # keep transport buttons compact
        self.play_btn.setFixedWidth(80)
        self.stop_btn.setFixedWidth(80)
        self.prev_btn.setFixedWidth(50)
        self.next_btn.setFixedWidth(50)
        self.repeat_btn.setFixedWidth(100)
        self.shuffle_btn.setFixedWidth(100)
        # speed slider
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 20)  # 0.5x to 2.0x (multiply by 0.1)
        self.speed_slider.setValue(10)  # 1.0x
        self.speed_slider.setMaximumWidth(80)
        self.speed_label = QLabel("1.0x")
        self.speed_label.setFixedWidth(30)
        # seek and volume sliders
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0,1000)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0,100)
        self.vol_slider.setValue(80)
        # add to layout
        self.ctrl_layout.addWidget(self.prev_btn)
        self.ctrl_layout.addWidget(self.play_btn)
        self.ctrl_layout.addWidget(self.stop_btn)
        self.ctrl_layout.addWidget(self.next_btn)
        self.ctrl_layout.addWidget(self.repeat_btn)
        self.ctrl_layout.addWidget(self.shuffle_btn)
        self.ctrl_layout.addWidget(QLabel("Speed:"))
        self.ctrl_layout.addWidget(self.speed_slider)
        self.ctrl_layout.addWidget(self.speed_label)
        self.ctrl_layout.addWidget(self.seek_slider, 1)
        self.ctrl_layout.addWidget(self.vol_slider)
        # connect signals
        self.vol_slider.valueChanged.connect(self.set_volume)
        self.speed_slider.valueChanged.connect(self.set_playback_speed)
        self.vlay.addWidget(self.ctrl)
        # small status label for playback state/time
        self.status_label = QLabel("Ready")
        self.status_label.setFixedHeight(18)
        self.vlay.addWidget(self.status_label)
        self.play_btn.clicked.connect(self.toggle_play)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.prev_btn.clicked.connect(self.play_previous)
        self.next_btn.clicked.connect(self.play_next)
        self.repeat_btn.clicked.connect(self.cycle_repeat)
        self.shuffle_btn.clicked.connect(self.toggle_shuffle)
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
        # Recent files submenu
        self.recent_menu = media.addMenu("Recent Files")
        self.update_recent_menu()
        media.addSeparator()
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
        info = QAction("File Information", self); info.triggered.connect(self.open_file_info)
        tools.addAction(info)
        capture = QAction("Capture Frame", self); capture.triggered.connect(self.capture_frame)
        tools.addAction(capture)
        bookmarks = mb.addMenu("Bookmarks")
        bm = QAction("Manage Bookmarks", self); bm.triggered.connect(self.open_bookmarks)
        bookmarks.addAction(bm)
        view = mb.addMenu("View")
        pip = QAction("Mini-Player", self, checkable=True); pip.triggered.connect(self.toggle_mini)
        view.addAction(pip)
        settings = QAction("Settings", self); settings.triggered.connect(self.open_settings); tools.addAction(settings)
    def open_file(self):
        f,_ = QFileDialog.getOpenFileName(self,"Open media","","Media files (*.mp4 *.mkv *.mp3 *.wav *.ogg)")
        if f:
            self.add_recent_file(f)  # track in recent files
            self.playlist.addItem(f)
            self.play_media(f)

    def add_recent_file(self, path):
        """Add file to recent list (max 10 items)"""
        # remove if already in list
        if path in self.recent_files:
            self.recent_files.remove(path)
        # add to front
        self.recent_files.insert(0, path)
        # keep only 10 most recent
        self.recent_files = self.recent_files[:10]
        # save to file
        self.save_recent_files()
        # update menu
        self.update_recent_menu()

    def update_recent_menu(self):
        """Rebuild recent files menu"""
        self.recent_menu.clear()
        if not self.recent_files:
            self.recent_menu.addAction("(empty)")
            return
        for i, fpath in enumerate(self.recent_files):
            fname = Path(fpath).name
            action = QAction(f"{i+1}. {fname}", self)
            action.triggered.connect(lambda _, p=fpath: self.play_recent(p))
            self.recent_menu.addAction(action)

    def play_recent(self, path):
        """Play file from recent menu"""
        if Path(path).exists():
            self.play_media(path)
        else:
            # remove non-existent file
            self.recent_files.remove(path)
            self.save_recent_files()
            self.update_recent_menu()

    def save_recent_files(self):
        """Save recent files list to JSON"""
        try:
            with open(self.recent_files_path, 'w') as f:
                json.dump(self.recent_files, f)
        except Exception as e:
            print(f"Failed to save recent files: {e}")

    def load_recent_files(self):
        """Load recent files list from JSON"""
        try:
            if self.recent_files_path.exists():
                with open(self.recent_files_path, 'r') as f:
                    self.recent_files = json.load(f)
        except Exception as e:
            print(f"Failed to load recent files: {e}")
            self.recent_files = []

    def open_file_info(self):
        """Open file info dialog for current file"""
        if not self.current_file:
            QInputDialog.getText(self, "Info", "No file loaded")
            return
        dlg = FileInfoDialog(self.current_file, self.video_engine)
        dlg.exec_()

    def capture_frame(self):
        """Save current video frame as PNG"""
        if not self.video_engine or not self.current_file:
            return
        try:
            frame = self.video_engine.grab_frame()
            if frame is None:
                return
            # create filename: original_name_TIMESTAMP.png
            base = Path(self.current_file).stem
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = Path(self.current_file).parent / f"{base}_capture_{ts}.png"
            # save frame (BGR to RGB for PNG)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.status_label.setText(f"Frame captured: {output.name}")
        except Exception as e:
            print(f"Capture failed: {e}")
            self.status_label.setText("Capture failed")

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
        # check if repeat mode is "One" and restart same track
        if self.repeat_mode == 1:  # repeat one
            try:
                self.play_media(self.current_file)
            except:
                pass

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

    # helper: load/save settings
    def load_settings(self):
		try:
			if self.settings_path.exists():
				with open(self.settings_path, "r") as f:
					self.settings.update(json.load(f))
		except Exception:
			pass
    def save_settings(self):
		try:
			with open(self.settings_path, "w") as f:
				json.dump(self.settings, f)
		except Exception:
			pass

    # apply simple theme
    def apply_theme(self, theme):
		if theme == "dark":
			# minimal dark stylesheet
			self.setStyleSheet("QWidget{background-color:#222; color:#ddd;} QSlider::groove:horizontal{background:#444;} QPushButton{background:#333; color:#fff}")
		else:
			self.setStyleSheet("")  # default

    # open settings dialog
    def open_settings(self):
		dlg = SettingsDialog(self.settings)
		if dlg.exec_():
			new = dlg.get_values()
			self.settings.update(new)
			# apply and persist
			self.apply_theme(self.settings.get("theme","light"))
			self.vol_slider.setValue(self.settings.get("default_volume",80))
			self.save_settings()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        if key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Left:
            self.seek_relative(-5.0)  # seek back 5s
        elif key == Qt.Key_Right:
            self.seek_relative(5.0)  # seek forward 5s
        elif key == Qt.Key_Up:
            self.vol_slider.setValue(min(100, self.vol_slider.value() + 5))
        elif key == Qt.Key_Down:
            self.vol_slider.setValue(max(0, self.vol_slider.value() - 5))
        elif key == Qt.Key_M:
            # mute/unmute: store previous volume
            if not hasattr(self, '_muted_vol'):
                self._muted_vol = self.vol_slider.value()
                self.vol_slider.setValue(0)
            else:
                self.vol_slider.setValue(self._muted_vol)
                del self._muted_vol
        elif key == Qt.Key_F:
            # toggle fullscreen
            if self.isFullScreen():
                self.showNormal()
                self.show_controls()
                self.auto_hide_timer.stop()
            else:
                self.showFullScreen()
                # start auto-hide if enabled
                if self.settings.get("auto_hide", True):
                    self.auto_hide_timer.start()
        elif key == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
        else:
            super().keyPressEvent(event)

    # mouse movement shows controls and restarts timer
    def mouseMoveEvent(self, event):
		try:
			if self.isFullScreen():
				self.show_controls()
				if self.settings.get("auto_hide", True):
					self.auto_hide_timer.start()
			# pass to base
		except:
			pass
		return super().mouseMoveEvent(event)

    def hide_controls(self):
		try:
			if self.isFullScreen():
				# hide toolbars/controls
				if hasattr(self, "ctrl"):
					self.ctrl.hide()
				try:
					self.menuBar().hide()
				except:
					pass
		except:
			pass

    def show_controls(self):
		try:
			if hasattr(self, "ctrl"):
				self.ctrl.show()
			try:
				self.menuBar().show()
			except:
				pass
		except:
			pass

    def seek_relative(self, delta):
        """Seek relative to current position"""
        if not self.video_engine or self._duration <= 0:
            return
        try:
            current = self.video_engine.get_time()
            target = max(0, min(self._duration, current + delta))
            # seek video
            self.video_engine.cap.set(cv2.CAP_PROP_POS_MSEC, target*1000.0)
            # seek vlc if available
            if self.vlc_player:
                try:
                    self.vlc_player.set_time(int(target*1000.0))
                except:
                    pass
            self.status_label.setText(f"Seek to {target:.1f}s")
        except:
            pass

    def play_next(self):
        """Play next item in playlist"""
        if self.playlist.count() == 0:
            return
        if self.shuffle_enabled:
            import random
            idx = random.randint(0, self.playlist.count() - 1)
        else:
            idx = (self.current_playlist_index + 1) % self.playlist.count()
        self.current_playlist_index = idx
        item = self.playlist.item(idx)
        self.play_item(item)

    def play_previous(self):
        """Play previous item in playlist"""
        if self.playlist.count() == 0:
            return
        idx = (self.current_playlist_index - 1) % self.playlist.count()
        self.current_playlist_index = idx
        item = self.playlist.item(idx)
        self.play_item(item)

    def cycle_repeat(self):
        """Cycle through repeat modes: Off -> One -> All -> Off"""
        self.repeat_mode = (self.repeat_mode + 1) % 3
        modes = ["Off", "One", "All"]
        self.repeat_btn.setText(f"Repeat: {modes[self.repeat_mode]}")

    def toggle_shuffle(self):
        """Toggle shuffle mode"""
        self.shuffle_enabled = not self.shuffle_enabled
        state = "On" if self.shuffle_enabled else "Off"
        self.shuffle_btn.setText(f"Shuffle: {state}")

    def set_playback_speed(self, val):
        """Set playback speed (val: 5-20, multiply by 0.1 to get 0.5x-2.0x)"""
        self.playback_speed = val * 0.1
        self.speed_label.setText(f"{self.playback_speed:.1f}x")
        # if VLC available, apply speed
        if self.vlc_player:
            try:
                self.vlc_player.set_rate(self.playback_speed)
            except:
                pass
        # TODO: implement speed control for VideoEngine (requires audio resampling)

    def play_item(self, item):
        """Play item from playlist (override to track index)"""
        idx = self.playlist.row(item)
        if idx >= 0:
            self.current_playlist_index = idx
        self.play_media(item.text())