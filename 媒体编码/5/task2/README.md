H.264 Low-Latency Encoder Demo

Quick steps:

1. Generate test YUV:
   pip3 install pillow numpy
   python3 test_generator.py
   -> produces test_cif.yuv (352x288, YUV420p)

2. Build encoder:
   # Ensure FFmpeg dev libs are installed (libavcodec, libavformat, libavutil, libswscale)
   make

3. Run encoder:
   ./encoder_libx264
   -> writes out_lowlatency.h264

4. Validate compatibility:
   ./validate_compat.sh out_lowlatency.h264
   -> produces compat_log.txt, decoded_frame.png, remuxed.mp4

Notes:
- For real-time performance evaluation, measure:
  * per-frame encoding latency (instrument code or use timestamps)
  * CPU and memory usage (top/pidstat)
  * end-to-end latency with network: send via RTP/UDP to ffmpeg receiver and measure timestamps (embed PTS or SEI).
- To reduce latency further: reduce GOP, use faster preset, disable lookahead, avoid B-frames, use smaller frame sizes, and consider hardware encoders (VAAPI, NVENC) or tune libx264 parameters.
