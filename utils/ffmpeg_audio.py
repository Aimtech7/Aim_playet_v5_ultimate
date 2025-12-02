# utils/ffmpeg_audio.py
import subprocess
def list_audio_tracks(path):
    try:
        cmd = ["ffprobe","-v","error","-select_streams","a","-show_entries","stream=index:stream_tags=language","-of","csv=p=0", path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        lines = [l for l in out.splitlines() if l.strip()]
        return lines
    except Exception as e:
        return []
def switch_audio_track(input_file, track_index, output_file):
    cmd = ["ffmpeg","-i", input_file, "-map", "0:v", "-map", f"0:a:{track_index}", "-c", "copy", output_file, "-y"]
    subprocess.call(cmd)
    return output_file