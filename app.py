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
        self.font_size = 18
        self.color = "white"
        self.delay = 0.0
    def set_style(self, font_size=None, color=None, delay=None):
        if font_size is not None:
            self.font_size = int(font_size)
        if color is not None:
            self.color = color
        if delay is not None:
            self.delay = float(delay)
        self.setStyleSheet(f"color:{self.color}; background-color:transparent; font-size:{self.font_size}px;")
    def load(self, path):
        sp = SubtitleParser(path)
        self.subs_tracks.append(sp)
    def update_time(self, t):
        t_eff = t + getattr(self, 'delay', 0.0)
        text = ""
        for s in self.subs_tracks:
            line = s.get_text(t_eff)
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
    """Settings dialog for theme, volume, and other preferences"""
    def __init__(self, settings):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(450, 250, 360, 200)
        layout = QVBoxLayout()
        # volume
        layout.addWidget(QLabel("Default Volume (0-100)"))
        self.vol_spin = QSlider(Qt.Horizontal)
        self.vol_spin.setRange(0,100)
        self.vol_spin.setValue(settings.get("volume", 80))
        layout.addWidget(self.vol_spin)
        # playback speed
        layout.addWidget(QLabel("Default Playback Speed (0.5x-2.0x)"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5,20)
        self.speed_slider.setValue(int(settings.get("playback_speed",1.0)*10))
        layout.addWidget(self.speed_slider)
        # theme selection
        layout.addWidget(QLabel("Theme"))
        self.theme_sel = QInputDialog()  # placeholder widget used for simple prompt below
        self.theme_combo = QLabel(settings.get("theme","Default"))
        layout.addWidget(self.theme_combo)
        # simple theme toggle buttons
        theme_row = QHBoxLayout()
        btn_default = QPushButton("Default"); btn_default.clicked.connect(lambda: self.theme_combo.setText("Default"))
        btn_dark = QPushButton("Dark"); btn_dark.clicked.connect(lambda: self.theme_combo.setText("Dark"))
        theme_row.addWidget(btn_default); theme_row.addWidget(btn_dark)
        layout.addLayout(theme_row)
        # buttons
        row = QHBoxLayout()
        ok = QPushButton("OK"); ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        row.addWidget(ok); row.addWidget(cancel)
        layout.addLayout(row)
        self.setLayout(layout)
    def values(self):
        return {
            "volume": int(self.vol_spin.value()),
            "playback_speed": float(self.speed_slider.value())/10.0,
            "theme": self.theme_combo.text()
        }

class AIMPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        # Recent files tracking
        self.recent_files = []  # list of recent file paths
        self.recent_files_path = Path(__file__).parent / "recent_files.json"
        self.settings_path = Path(__file__).parent / "settings.json"
        self.load_recent_files()
        self.load_settings()
        # ...existing code for video container, playlist, visualizer, controls...
        # keep reference to controls widget for show/hide and enable drag/drop
        self.controls_widget = ctrl
        self.setAcceptDrops(True)
        # fullscreen auto-hide timer
        self.hide_controls_timer = QTimer(self)
        self.hide_controls_timer.setInterval(3000)  # 3s inactivity
        self.hide_controls_timer.timeout.connect(self.hide_controls)
        # ensure controls visible initially
        self.show_controls()

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
        play = QAction("Play/Pause", self)
        play.triggered.connect(self.toggle_play)
        playback.addAction(play)
        audio = mb.addMenu("Audio")
        ai_eq = QAction("AI Auto-EQ", self, checkable=True)
        ai_eq.triggered.connect(self.toggle_ai_eq)
        audio.addAction(ai_eq)
        subtitle = mb.addMenu("Subtitle")
        loads = QAction("Load Subtitle", self)
        loads.triggered.connect(self.load_sub)
        subtitle.addAction(loads)
        # Add audio tracks submenu (after Load Subtitle)
        self.audio_track_menu = audio.addMenu("Audio Tracks")
        self.update_audio_tracks_menu()
        
        # Add subtitle customize
        subtitle.addSeparator()
        customize = QAction("Customize", self); customize.triggered.connect(self.customize_subtitles)
        subtitle.addAction(customize)
        
        tools = mb.addMenu("Tools")
        hotkeys = QAction("Hotkeys Editor", self); hotkeys.triggered.connect(self.open_hotkeys)
        tools.addAction(hotkeys)
        settings_act = QAction("Settings", self); settings_act.triggered.connect(self.open_settings)
        tools.addAction(settings_act)
        eq_menu = tools.addMenu("EQ Presets")
        eq_save = QAction("Save Preset", self); eq_save.triggered.connect(self.save_eq_preset_dialog)
        eq_menu.addAction(eq_save)
        eq_load = QAction("Load Preset", self); eq_load.triggered.connect(self.load_eq_preset_dialog)
        eq_menu.addAction(eq_load)
        # ...existing tools/menu items (info, capture)...

    def update_audio_tracks_menu(self):
        """Build audio tracks menu from VLC if available"""
        self.audio_track_menu.clear()
        
        if not self.vlc_player or not _HAS_VLC:
            self.audio_track_menu.addAction("(VLC not available)")
            return
        
        try:
            # Get available audio tracks
            media = self.vlc_player.get_media()
            if not media:
                self.audio_track_menu.addAction("(no media loaded)")
                return
            
            # Enumerate tracks
            track_count = 0
            for track_id in range(10):  # VLC limits reasonable number of tracks
                try:
                    # Attempt to get track info (VLC API varies)
                    action = QAction(f"Track {track_id + 1}", self)
                    action.triggered.connect(lambda _, tid=track_id: self.set_audio_track(tid))
                    self.audio_track_menu.addAction(action)
                    track_count += 1
                except:
                    break
            
            if track_count == 0:
                self.audio_track_menu.addAction("(no audio tracks)")
        except Exception as e:
            print(f"Audio tracks menu error: {e}")
            self.audio_track_menu.addAction("(error reading tracks)")

    def set_audio_track(self, track_id):
        """Switch to audio track by ID"""
        if not self.vlc_player or not _HAS_VLC:
            return
        try:
            self.vlc_player.audio_set_track(track_id)
            self.status_label.setText(f"Audio track: {track_id + 1}")
        except Exception as e:
            print(f"Failed to switch audio track: {e}")

    # EQ presets management
    def save_eq_presets(self):
        try:
            with open(self.eq_presets_path, 'w') as f:
                json.dump(self.eq_presets, f, indent=2)
        except Exception as e:
            print("Failed to save EQ presets:", e)

    def load_eq_presets(self):
        try:
            if self.eq_presets_path.exists():
                with open(self.eq_presets_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print("Failed to load EQ presets:", e)
        return {}

    def save_eq_preset_dialog(self):
        name, ok = QInputDialog.getText(self, "Save EQ Preset", "Preset name:")
        if not ok or not name:
            return
        vals = [s.value() for s in self.eq_sliders]
        self.eq_presets[name] = vals
        self.save_eq_presets()
        self.status_label.setText(f"EQ preset '{name}' saved")

    def load_eq_preset_dialog(self):
        if not self.eq_presets:
            self.status_label.setText("No EQ presets")
            return
        items = list(self.eq_presets.keys())
        name, ok = QInputDialog.getItem(self, "Load EQ Preset", "Preset:", items, 0, False)
        if ok and name:
            vals = self.eq_presets.get(name)
            if vals:
                for s, v in zip(self.eq_sliders, vals):
                    s.setValue(v)
                self.audio_engine.set_eq(vals) if self.audio_engine else None
                self.status_label.setText(f"EQ preset '{name}' loaded")

    class SubtitleCustomizeDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Subtitle Customize")
            self.setGeometry(480, 260, 320, 180)
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Font Size"))
            self.font_slider = QSlider(Qt.Horizontal); self.font_slider.setRange(10,48); self.font_slider.setValue(18)
            layout.addWidget(self.font_slider)
            layout.addWidget(QLabel("Delay (ms)"))
            self.delay_slider = QSlider(Qt.Horizontal); self.delay_slider.setRange(-500,500); self.delay_slider.setValue(0)
            layout.addWidget(self.delay_slider)
            layout.addWidget(QLabel("Color"))
            self.color_btn = QPushButton("Choose Color")
            self.color_btn.clicked.connect(self.pick_color)
            self.selected_color = "(255,255,255)"
            layout.addWidget(self.color_btn)
            row = QHBoxLayout()
            ok = QPushButton("OK"); ok.clicked.connect(self.accept)
            cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
            row.addWidget(ok); row.addWidget(cancel)
            layout.addLayout(row)
            self.setLayout(layout)
        def pick_color(self):
            col = QColorDialog.getColor()
            if col.isValid():
                self.selected_color = f"rgb({col.red()},{col.green()},{col.blue()})"
                self.color_btn.setStyleSheet(f"background:{self.selected_color}")

    def customize_subtitles(self):
        """Open subtitle customization dialog"""
        dlg = SubtitleCustomizeDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            font_size = dlg.font_slider.value()
            color = dlg.selected_color
            delay_sec = dlg.delay_slider.value() / 1000.0
            if self.subtitle_overlay:
                self.subtitle_overlay.set_style(font_size=font_size, color=color, delay=delay_sec)
            # persist subtitle prefs
            self.settings['subtitle'] = {"font_size": font_size, "color": color, "delay": delay_sec}
            self.save_settings()

    def toggle_statistics(self, checked):
        """Toggle statistics overlay visibility"""
        self.stats_visible = checked
        if checked:
            self.stats_label.show()
            # Start FPS counter timer
            if not hasattr(self, '_stats_timer'):
                self._stats_timer = QTimer()
                self._stats_timer.timeout.connect(self._update_statistics)
            self._stats_timer.start(500)  # update every 500ms
        else:
            self.stats_label.hide()
            if hasattr(self, '_stats_timer'):
                self._stats_timer.stop()

    def _update_statistics(self):
        """Update statistics display"""
        if not self.video_engine:
            return
        try:
            fps = self.video_engine.cap.get(cv2.CAP_PROP_FPS) or 0
            # Get current resolution
            w = int(self.video_engine.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.video_engine.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            speed_str = f"{self.playback_speed:.1f}x"
            self.stats_label.setText(f"FPS: {fps:.1f} | {w}×{h} | Speed: {speed_str}")
        except:
            pass

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
                    # apply saved volume/speed if present
                    if hasattr(self, 'settings') and self.settings:
                        try:
                            self.vlc_player.audio_set_volume(int(self.settings.get("volume", self.vol_slider.value())))
                            self.vlc_player.set_rate(float(self.settings.get("playback_speed", self.playback_speed)))
                        except:
                            pass
                    self.vlc_player.play()
                except Exception as e:
                    print("VLC play failed:", e)
                self.status_label.setText("Playing (VLC)")
            else:
                self.status_label.setText("Playing (no VLC)")
            # start audio engine (ffmpeg fallback)
            if self.audio_engine:
                # apply volume setting
                try:
                    self.audio_engine.volume = int(self.settings.get("volume", self.vol_slider.value()))
                except:
                    pass
                self.audio_engine.start_stream()
            # show controls and start hide timer if fullscreen
            self.show_controls()
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
        self.settings = {}
        try:
            if self.settings_path.exists():
                with open(self.settings_path, 'r') as f:
                    self.settings = json.load(f)
                # restore geometry if present
                geom = self.settings.get('geometry')
                if geom:
                    try:
                        self.restoreGeometry(bytes.fromhex(geom))
                    except:
                        pass
                vol = int(self.settings.get("volume", self.vol_slider.value()))
                self.vol_slider.setValue(vol)
                sp = float(self.settings.get("playback_speed", self.playback_speed))
                self.playback_speed = sp
                try:
                    self.speed_slider.setValue(int(sp*10))
                    self.speed_label.setText(f"{sp:.1f}x")
                except:
                    pass
                # apply theme
                theme = self.settings.get('theme', 'Default')
                if theme == 'Dark':
                    self.apply_dark_theme()
            else:
                self.settings = {"volume": self.vol_slider.value(), "playback_speed": self.playback_speed, "theme":"Default"}
                self.save_settings()
        except Exception as e:
            print("Failed to load settings:", e)

    def save_settings(self):
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print("Failed to save settings:", e)

    # Fullscreen auto-hide helpers
    def hide_controls(self):
        if self.isFullScreen():
            try:
                self.controls_widget.hide()
                self.status_label.hide()
            except:
                pass

    def show_controls(self):
        try:
            self.controls_widget.show()
            self.status_label.show()
            # restart hide timer only in fullscreen
            if self.isFullScreen():
                self.hide_controls_timer.start()
            else:
                self.hide_controls_timer.stop()
        except:
            pass

    def mouseMoveEvent(self, event):
        # show controls on mouse move and restart timer
        try:
            self.show_controls()
        except:
            pass
        return super().mouseMoveEvent(event)

    # Drag & drop support
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        # take first file: add to playlist and play
        for u in urls:
            path = u.toLocalFile()
            if path:
                self.playlist.addItem(path)
        # play first dropped file
        first = urls[0].toLocalFile()
        if first:
            self.add_recent_file(first)
            self.play_media(first)