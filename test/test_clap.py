import numpy as np
import librosa
import torch
import laion_clap
import sys
import os

# 添加父目录到系统路径
sys.path.append('..')

# 设置设备
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"Using device: {device}")

# 加载模型
model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base', device=device)
model.load_ckpt('../checkpoint/music_speech_epoch_15.pt')  # 这会下载默认的预训练模型

# 准备一些示例文本
texts = [
    "This is a sound of a dog barking",
    "This is a piano playing",
    "This is a person speaking",
    "This is a sound of rain"
]

# 获取文本嵌入
print("Getting text embeddings...")
text_embeds = model.get_text_embedding(texts)
print(f"Text embedding shape: {text_embeds.shape}")

# 如果你有音频文件，可以获取音频嵌入
# 替换为你的音频文件路径
audio_files = ["../audio/short_riff.wav", "../audio/蛄蛹者_short.wav"]
# 加载音频文件并转换为numpy数组
audio_data = []
for audio_file in audio_files:
    if os.path.exists(audio_file):
        waveform, _ = librosa.load(audio_file, sr=48000)  # CLAP模型需要48kHz采样率
        audio_data.append(waveform)
    else:
        print(f"警告：找不到音频文件 {audio_file}")

# 使用get_audio_embedding_from_data获取音频嵌入
if audio_data:
    audio_embeds = model.get_audio_embedding_from_data(audio_data)
    print(f"Audio embedding shape: {audio_embeds.shape}")
else:
    print("没有找到有效的音频文件，跳过音频嵌入测试")

print("CLAP model loaded and tested successfully!")