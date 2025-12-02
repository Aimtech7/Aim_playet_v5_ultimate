# app.py
# AIM Player v5 Ultimate — Integrated PyQt5 application (simplified production bundle)
# Note: This is a functional scaffold assembled from the project conversation.
import sys, os, random, numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QAction, QMenu,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QPushButton,
                             QListWidget, QDockWidget, QListWidgetItem, QInputDialog, QDialog,
                             QColorDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QColor
import pyaudio
from av_engine import AudioEngine, VideoEngine
from utils.subtitles import SubtitleParser

class SubtitleOverlay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("color:white; background-color:transparent; font-size:18px;")
        self.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
        self.subs_tracks = []
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
        # Central video layout
        central = QWidget()
        self.setCentralWidget(central)
        self.vlay = QVBoxLayout(central)
        self.video_display = QLabel("No video loaded")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.vlay.addWidget(self.video_display)
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
    def init_menus(self):
        mb = self.menuBar()
        media = mb.addMenu("Media")
        openf = QAction("Open File", self); openf.triggered.connect(self.open_file)
        media.addAction(openf)
        playback = mb.addMenu("Playback")
        play = QAction("Play/Pause", self); playback.addAction(play)
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
        self.audio_engine = AudioEngine(path)
        self.visualizer.audio_engine = self.audio_engine
        self.video_engine = VideoEngine(path)
        # connect video frame updates for subtitle timing
        if hasattr(self.video_engine, 'frame_time_signal'):
            try:
                self.video_engine.frame_time_signal.connect(self.on_frame_time)
            except Exception as e:
                print(f"Warning: Could not connect frame_time_signal: {e}")
        # display first frame
        try:
            frame = self.video_engine.grab_frame()
            if frame is not None:
                h,w,c = frame.shape
                img = QImage(frame.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
                self.video_display.setPixmap(QPixmap.fromImage(img).scaled(800,450, Qt.KeepAspectRatio))
        except Exception as e:
            print(f"Warning: Could not grab initial frame: {e}")
        # start playback loops
        try:
            self.video_engine.start()
            self.audio_engine.start_stream()
        except Exception as e:
            print(f"Error starting playback: {e}")
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
        if f and self.video_engine:
            if not hasattr(self, 'subtitle_overlay'):
                self.subtitle_overlay = SubtitleOverlay(self.video_display)
                self.vlay.addWidget(self.subtitle_overlay)
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AIMPlayer()
    win.show()
    sys.exit(app.exec_())