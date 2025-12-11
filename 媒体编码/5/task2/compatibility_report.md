# Compatibility & Performance Report (Template)

System:
- CPU: ____________________
- Cores: 4
- OS: ____________________
- Compiler: ____________________
- FFmpeg version: ____________________

Build:
- Build command and flags used:
  - `make` or: ____________________

Test data:
- File: test_cif.yuv
- Resolution: 352x288 (CIF)
- Frames: 150
- Content: moving square (synthetic)

Encoding config:
- Encoder: libx264 via libavcodec
- Bitrate: 500 kbps
- Profile: baseline
- GOP: 30
- B-frames: 0
- Tune: zerolatency
- Preset: veryfast

Results:
- Scene: synthetic moving square
  - Avg FPS (encoder): ______
  - Avg encode latency per frame (ms): ______
  - 95th percentile latency (ms): ______
  - CPU usage (% avg): ______
  - Memory usage (MB): ______
  - Output file: out_lowlatency.h264
  - FFmpeg decode: PASS/FAIL
  - Notes: ______

Compatibility logs:
- Attach compat_log.txt (from validate_compat.sh)
- Attach decoded_frame.png

FFmpeg log excerpts:
- Paste the relevant ffmpeg log lines indicating successful decode or errors.

Conclusion:
- Does the stream decode in FFmpeg and VLC? ______
- Any observed playback artifacts? ______
- Recommended next steps for further latency reduction: ______
