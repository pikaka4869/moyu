#!/bin/bash
set -e
INPUT_H264=${1:-out_lowlatency.h264}
LOG=compat_log.txt
> $LOG

echo "==== FFmpeg decode test ====" | tee -a $LOG
ffmpeg -v verbose -i "$INPUT_H264" -f null - 2>&1 | tee -a $LOG

echo "==== Extract frame (for screenshot) ====" | tee -a $LOG
ffmpeg -v quiet -i "$INPUT_H264" -frames:v 1 decoded_frame.png
if [ -f decoded_frame.png ]; then
  echo "Screenshot saved: decoded_frame.png" | tee -a $LOG
fi

# Remux to mp4
ffmpeg -v quiet -y -i "$INPUT_H264" -c:v copy remuxed.mp4 2>> $LOG || echo "mp4 remux failed" | tee -a $LOG

echo "Compatibility log saved to $LOG"
