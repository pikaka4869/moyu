# Qwen3-Omni-30B 环境搭建

选用优云智算的A800 80GB 显存

conda create -n qwen3 python=3.12

pip install -U modelscope

modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Instruct --local_dir ./Qwen3-Omni-30B-A3B-Instruct

pip install transformers==4.57.3

pip install accelerate

sudo apt update
sudo apt install ffmpeg

pip install qwen-omni-utils -U

pip install gradio==5.23.1