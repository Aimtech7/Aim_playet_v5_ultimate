# AIM Player v5 Ultimate — Development Roadmap

## Project Overview
A PyQt5-based multimedia player with video/audio playback, visualizer, subtitle support, and advanced playback controls.

---

## Implementation Timeline

### Phase 1: Core Playback (Completed)
**Date:** Initial Setup
- VideoEngine (OpenCV-based frame extraction and filtering)
- AudioEngine (ffmpeg → PyAudio with volume control)
- VLC integration (optional, for robust audio)
- Basic UI: video display, playlist, visualizer

---

### Phase 2: Playback Controls & Shortcuts (Completed)
**Date:** Latest session

#### Feature 2.1: Keyboard Shortcuts
- **File:** app.py
- **Lines Added:** 596-632 (keyPressEvent method)
- **Changes:**
  - Added `keyPressEvent()` override to handle keyboard input
  - Space: Play/Pause
  - Left/Right arrows: Seek ±5 seconds
  - Up/Down arrows: Volume ±5%
  - M: Mute/Unmute (toggle with memory)
  - F: Toggle fullscreen
  - Esc: Exit fullscreen
- **Reason:** Improve UX by supporting standard media player shortcuts
- **Dependencies:** None (uses existing Qt Key enum)

#### Feature 2.2: Seek Relative
- **File:** app.py
- **Lines Added:** 634-649 (seek_relative method)
- **Changes:**
  - New method `seek_relative(delta)` to seek by offset
  - Syncs VideoEngine (OpenCV) and VLC positions
  - Clamps target to valid range [0, duration]
- **Reason:** Support arrow key seeking; reusable for other features
- **Dependencies:** VideoEngine, VLC (optional)

#### Feature 2.3: Playlist Navigation (Next/Previous)
- **File:** app.py
- **Lines Added:** 651-676 (play_next, play_previous methods)
- **Changes:**
  - `play_next()`: advance to next item, with shuffle support (random selection if shuffle enabled)
  - `play_previous()`: go to previous item
  - Wraps around playlist (circular navigation)
  - Tracks current index in `self.current_playlist_index`
- **Reason:** Essential playlist control for batch playback
- **Dependencies:** Shuffle mode, playlist widget

#### Feature 2.4: Repeat Mode (Off/One/All)
- **File:** app.py
- **Lines Added:** 678-684 (cycle_repeat method) + stop_playback enhancement
- **Changes:**
  - New method `cycle_repeat()`: cycles through modes (Off → One → All → Off)
  - Updates button text: "Repeat: Off|One|All"
  - In `stop_playback()` (line ~719): if repeat_mode==1, restart current file
- **Reason:** Standard media player feature for continuous playback
- **Dependencies:** Playlist, file path tracking

#### Feature 2.5: Shuffle Mode
- **File:** app.py
- **Lines Added:** 686-691 (toggle_shuffle method)
- **Changes:**
  - New method `toggle_shuffle()`: toggles `self.shuffle_enabled`
  - Updates button: "Shuffle: Off|On"
  - Used in `play_next()` for random track selection
- **Reason:** Randomize playback order for variety
- **Dependencies:** Next/Previous navigation, random module

#### Feature 2.6: Playback Speed Control
- **File:** app.py
- **Lines Added:** 693-704 (set_playback_speed method) + UI additions (lines ~90-98)
- **Changes:**
  - UI: Speed slider (range 5-20, multiply by 0.1 = 0.5x to 2.0x) + label display
  - New method `set_playback_speed(val)`: converts slider value to speed
  - Applies speed to VLC via `vlc_player.set_rate()`
  - TODO: VideoEngine speed requires audio resampling (not yet implemented)
- **Reason:** Common playback feature for accessibility and fast-forward/slow-mo
- **Dependencies:** VLC (for actual speed control); VideoEngine (partially)

#### Feature 2.7: UI Button Additions
- **File:** app.py
- **Lines Added:** 77-98 (in __init__)
- **Changes:**
  - Added Previous button (◄◄), Next button (►►)
  - Added Repeat and Shuffle buttons with fixed widths
  - Added Speed slider with label
  - Connected all buttons to respective methods
- **Reason:** Provide visual controls for new features
- **Dependencies:** QPushButton, QSlider, QLabel

---

### Phase 3: File Information & Metadata (Planned)

#### Feature 3.1: File Info Dialog
- **File:** app.py (to be added)
- **Planned Lines:** New method `open_file_info()` (~30 lines)
- **Changes:**
  - New dialog class `FileInfoDialog` displaying:
    - Filename, duration, resolution, codec info
    - File size, creation date
  - Added menu entry: Tools → File Info
- **Reason:** Help users inspect media properties
- **Dependencies:** os.path, datetime, cv2 properties

#### Feature 3.2: Recent Files Menu
- **File:** app.py (to be added)
- **Planned Lines:** New method `add_recent_file()`, update `init_menus()` (~50 lines)
- **Changes:**
  - Track last 10 opened files in memory/config
  - Add "Recent Files" submenu in Media menu
  - Quick-load files from recent list
- **Reason:** Fast access to frequently played files
- **Dependencies:** JSON config file or in-memory list

---

### Phase 4: Capture & Export (Planned)

#### Feature 4.1: Screenshot Capture
- **File:** app.py (to be added)
- **Planned Lines:** New method `capture_frame()` (~20 lines)
- **Changes:**
  - Save current video frame as PNG
  - Option to save to clipboard or file
  - Added menu: Tools → Capture Frame
- **Reason:** Quick export of interesting moments
- **Dependencies:** cv2.imwrite, PIL (optional for clipboard)

---

### Phase 5: Advanced (Future)

- Fullscreen improvements (auto-hide controls, mouse tracking)
- Audio track selection (VLC)
- Subtitle font/color customization
- Statistics overlay (FPS, bitrate, dropped frames)
- Drag & drop file support
- Settings/Preferences dialog
- Dark/Light theme toggle
- EQ presets
- Chapter marks

---

## Bug Fixes & Optimizations

### Fix: Multi-stream Video Seek Error
- **Date:** Latest session
- **File:** app.py (line 7-8), av_engine.py (line 61-67)
- **Issue:** OpenCV's ffmpeg wrapper fails seeking on multi-stream videos (video+audio)
- **Solution:** 
  - Set `OPENCV_FFMPEG_READ_ATTEMPTS=10000` before cv2 import
  - Wrap `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)` in try/except
  - Gracefully break loop on seek failure
- **Reason:** Prevents thread crash on playback end
- **Impact:** Players can now play multi-stream files without crashing

---

## Dependencies

### Core
- PyQt5 (UI framework)
- numpy (array operations)
- opencv-python (video decoding, frame extraction)
- pyaudio (audio playback)
- ffmpeg (system; used by cv2 and AudioEngine)

### Optional
- python-vlc (robust audio playback)
- VLC (system app; required for python-vlc)

### Utils (Project)
- av_engine.py (VideoEngine, AudioEngine)
- utils/subtitles.py (SubtitleParser)

---

## Testing Checklist

- [ ] Keyboard shortcuts: Space, arrows, M, F, Esc
- [ ] Playlist navigation: Next, Previous with wrap-around
- [ ] Repeat modes: Off, One, All
- [ ] Shuffle: Random order
- [ ] Playback speed: 0.5x to 2.0x (VLC)
- [ ] Multi-stream video playback (no crashes)
- [ ] Audio playback (ffmpeg fallback)
- [ ] Subtitle overlay
- [ ] Volume control
- [ ] Seek slider

---

## Performance Notes

- VideoEngine loop: 30 FPS (configurable via fps detection)
- AudioEngine spectrum: updated every 50ms
- Seek updates: 200ms timer
- Minimal CPU for idle animation on visualizer

---

## Known Limitations

1. VideoEngine playback speed not implemented (requires audio resampling)
2. No audio track selection UI (VLC capable, not exposed)
3. Subtitle delay adjustment not implemented
4. No recording/export (frame capture only)
5. Fullscreen controls auto-hide not implemented

---

## Future Enhancements (Priority Order)

1. **High:** Speed control for VideoEngine, Recent files menu
2. **Medium:** File info dialog, Screenshot with clipboard, Settings persistence
3. **Low:** Advanced filtering, EQ presets, Chapter marks, Themes

