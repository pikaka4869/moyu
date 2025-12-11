# Qwen2.5-Omni-3B 环境搭建

## 1. 创建 Python 3.11 虚拟环境

```bash
conda create -n qwen python=3.11
```

## 2. 安装依赖库

### 2.1 安装 transformers 库

```bash
pip install transformers==4.57.3
```

### 2.2 安装 PyTorch 和 Torchvision

#### 下载链接
- [PyTorch 2.6.0+cu124](https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl)
- [Torchvision 0.21.0+cu124](https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp311-cp311-win_amd64.whl)

#### 安装命令

```bash
pip install torch-2.6.0+cu118-cp311-cp311-win_amd64.whl
pip install torchvision-0.21.0+cu118-cp311-cp311-win_amd64.whl
```

也可以直接使用 `pip install torch torchvision` 安装
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 验证 PyTorch 安装

安装完成后，建议先验证 PyTorch 是否正确安装并能够调用 GPU：

```bash
python -c "import torch; print(f'版本: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'可用: {torch.cuda.is_available()}')"
```

### 2.3 安装 accelerate 库

```bash
pip install accelerate
```

### 2.4 安装 qwen-omni-utils 库

该库提供便捷的 API 风格接口，可统一处理 base64、URL 及交错的音频、图像、视频等多模态输入。  
使用前请确保系统已安装 `ffmpeg`。

```bash
pip install qwen-omni-utils[decord] -U
```

### 2.5 安装 flash-attn 库

https://github.com/kingbri1/flash-attention/releases

然后 Pip 安装 flash-attn 库

```bash
pip install flash_attn-2.8.3+cu124torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl
```

## 3. 验证安装

### 3.1 验证 PyTorch GPU 是否可用

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. 获取 Qwen2.5-Omni-3B 模型

Qwen2.5-Omni-3B 模型建议通过魔塔社区 ModelScope 获取，这也是官方推荐的方式。

- **下载链接**：[Qwen2.5-Omni-3B 在 ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-3B/files)

## 5. 运行 demo.py 验证安装

```python
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen2.5-Omni-3B", torch_dtype="auto", device_map="auto")

# 我们建议启用 flash_attention_2 以获取更快的推理速度以及更低的显存占用.
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen2.5-Omni-3B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen2.5-Omni-3B")

prompt = '早上好'

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            # {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
            {"type": "text", "text": prompt},
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = False

# set return audio
RETURN_AUDIO = False

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
# text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=RETURN_AUDIO)
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=RETURN_AUDIO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text[0])
# sf.write(
#     "output.wav",
#     audio.reshape(-1).detach().cpu().numpy(),
#     samplerate=24000,
# )
```

##
