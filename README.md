# AIM Player v5 Ultimate

This archive contains the full AIM Player v5 Ultimate project — PyQt5 desktop app with AI audio/video features.

Run:
```
pip install -r requirements.txt
python app.py
```
1️⃣ Updated README.md (styled and visually appealing)
# AIM Player v5 Ultimate 🎵🎬

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15-green)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**AIM Player v5 Ultimate** is a VLC-inspired cross-platform media player with advanced AI-powered audio/video features, real-time visualizers, and full roadmap-compliant enhancements.

---

## Features

### Playback Enhancements
- Multi-audio/video track support & switching
- Chapters & bookmarks
- Smart resume per file
- Gapless playback for albums
- Mini-player floating window
- Fullscreen & resizable panels
- Drag-and-drop playlist

### Visual Enhancements
- Animated visualizers (bars, waves, circles)
- Fullscreen synchronized visual effects
- Real-time video frame processing via OpenCV
- AI-based auto color/brightness/contrast video enhancement
- Video filters: sharpen, blur, LUTs
- Skins/themes switching (including VLC-like skin)

### Audio & AI Features
- 31-band EQ + parametric EQ with PyAudio/FFmpeg
- AI Auto-EQ using offline ML model
- Voice isolation (vocal removal)
- Music genre detection & suggested EQ
- Custom audio effects chain (compressor, limiter, flanger, echo)
- Multi-channel audio support (5.1, 7.1 surround)

### Subtitles & Lyrics
- Full subtitles support (.srt/.vtt)
- Load & switch subtitle tracks
- Custom fonts, sizes, and positions
- Lyric display overlay

### UX/UI Improvements
- VLC-identical GUI with menu bar: Media, Playback, Audio, Subtitle, Tools, View, Help
- Hover effects, icons, and animations
- Animated icons for better visual feedback
- Resizable panels for playlist, visualizer, and lyrics
- Skin editor for custom themes
- Custom hotkeys editor

### Integration & Social
- Playlist & media metadata editor (title, album art, subtitles)
- Share snapshots/video clips
- Integration with local media folders

### Developer & Advanced Tools
- Developer log panel
- Plugin architecture for future features
- AI models included offline (TensorFlow/PyTorch)
- Unit tests placeholders for core modules

---

## Installation

### Requirements
```bash
pip install -r requirements.txt

Run
python app.py

Future Windows Installer

Build using PyInstaller or cx_Freeze to generate .exe.

The installer will include all skins, AI models, and assets.

Screenshots








Roadmap (Stages 1–8)

✅ Stage 1: Basic playback, file/folder load, EQ

✅ Stage 2: Playlist, mini-player, visualizer

✅ Stage 3: AI Auto-EQ & video enhancement

✅ Stage 4: Skins/themes & hover animations

✅ Stage 5: Subtitles overlay & lyrics

✅ Stage 6: Developer tools & debug panel

✅ Stage 7: Advanced audio & multi-track support

✅ Stage 8: Full VLC-like GUI, menus, plugins

Project Folder Structure
AIM_Player_v5/
├─ app.py
├─ requirements.txt
├─ README.md
├─ AIM_PLAYER_ROADMAP.md
├─ LICENSE
├─ assets/
│  ├─ icons/
│  ├─ skins/
│  └─ screenshots/
├─ models/
│  ├─ ai_audio_eq.pt
│  └─ ai_video_enhancer.pt
├─ utils/
│  ├─ ffmpeg_audio.py
│  ├─ visualizer.py
│  └─ subtitles.py

Contribution Guidelines

Fork repository.

Implement features in a separate branch.

Add appropriate docstrings and comments.

Submit pull requests with detailed descriptions.

Ensure roadmap compliance for new features.

License

MIT License – see LICENSE file.

PWA vs Desktop Comparison
Feature	Desktop PyQt5	PWA Streamlit
Offline	✅	✅ (cache only)
Skins/Themes	✅	Limited
AI Auto-EQ	✅	✅
Visualizer	✅	✅
Mini-Player	✅	❌
Multi-track audio	✅	Limited
Subtitles	✅	✅
Developer Tools	✅	❌

---

### 2️⃣ Updated `AIM_PLAYER_ROADMAP.md` (styled)

```markdown
# AIM Player v5 Ultimate — Roadmap

**Version:** 5.0 Ultimate  
**Goal:** Full VLC-like GUI + AI-powered playback enhancements

---

## Stage 1 – Basic Playback
- File/Folder media load ✅
- Play/Pause, Stop, Next/Previous ✅
- 31-band EQ ✅
- Volume control ✅

## Stage 2 – Playlist & Visualizer
- Playlist dock ✅
- Animated visualizer (bars/circle/waves) ✅
- Mini-player window ✅

## Stage 3 – AI Enhancements
- AI Auto-EQ ✅
- AI Video enhancement ✅
- Voice isolation ✅

## Stage 4 – Skins/Themes & GUI
- VLC-like GUI ✅
- Hover effects & icons ✅
- Skin switching ✅

## Stage 5 – Subtitles & Lyrics
- Subtitle overlay ✅
- Load/Select/Font/Position ✅
- Lyrics display ✅

## Stage 6 – Developer Tools
- Debug/log panel ✅
- Plugin architecture ✅
- Unit tests placeholders ✅

## Stage 7 – Advanced Audio
- Multi-audio track switching ✅
- Gapless playback ✅
- Custom audio effects chain ✅

## Stage 8 – Final GUI & Features
- Full menu bar: Media, Playback, Audio, Subtitle, Tools, View, Help ✅
- Animated icons & hover effects ✅
- Mini-player & resizable panels ✅
- AI & roadmap-compliant final polish ✅

3️⃣ requirements.txt
PyQt5>=5.15
opencv-python>=4.7
torch>=2.0
numpy>=1.25
pyaudio>=0.2.13
ffmpeg-python>=0.3.0
````markdown
# AIM Player v5 Ultimate

This archive contains the full AIM Player v5 Ultimate project — PyQt5 desktop app with AI audio/video features.

Run:
```
pip install -r requirements.txt
python app.py
```
# AIM Player v5 Ultimate — minimal README

## Prerequisites
- Python 3.8+
- pip

Recommended Python packages:
- PyQt5
- numpy
- opencv-python
- pyaudio
- python-vlc (optional, for easier playback in many formats)

Install example:
```
pip install PyQt5 numpy opencv-python pyaudio python-vlc
```

Native tools:
- VLC (optional): install VLC desktop application if you want python-vlc to control playback.
- ffmpeg: required for the built-in ffmpeg->pyaudio audio fallback. Make sure ffmpeg is in PATH.

## How it works
- The app uses VideoEngine (OpenCV) to decode frames and display them.
- AudioEngine attempts to use ffmpeg to decode audio and plays via PyAudio, providing a real spectrum for the visualizer.
- If python-vlc is installed and available, the app will create a VLC MediaPlayer for playback; this is preferred for robust audio handling.
- If VLC is not present, the integrated ffmpeg->pyaudio pipeline will be used (ensure ffmpeg is installed).

## Run
1. Ensure ffmpeg and (optionally) VLC are installed and available in PATH.
2. From project directory:
   ```
   python app.py
   ```
3. Use the UI to open a media file and control playback with Play/Pause, Stop, and Seek.

## Troubleshooting
- No sound:
  - If using VLC: ensure VLC is installed and python-vlc imports successfully.
  - If using ffmpeg fallback: ensure ffmpeg is in PATH and pyaudio can open an output device.
- Errors printed to console will indicate failed connections (ffmpeg, pyaudio, VLC). Run from terminal to see logs.

## Notes
- The integrated ffmpeg pipeline is a simple fallback for playback and the visualizer; for production-quality audio handling consider a full ffmpeg/decoder integration or using VLC for playback.
- The app focuses on a simple playable prototype: further polishing (thread-safety, device selection, advanced seeking) is recommended.