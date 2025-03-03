import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加父目录到系统路径，以便导入text2fx模块
sys.path.append('..')
from text2fx import Text2FX

def main():
    """测试Text2FX中的apply_fx_chain方法"""
    print("测试Text2FX中的apply_fx_chain方法...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载或生成测试音频
    audio_path = "../audio/short_riff.wav"
    if os.path.exists(audio_path):
        audio, sr = torchaudio.load(audio_path)
        print(f"加载音频: {audio_path}, 形状: {audio.shape}, 采样率: {sr}")
    else:
        print("生成测试音频...")
        # 生成一个简单的测试音频（正弦波）
        sr = 44100
        duration = 3  # 3秒
        t = torch.linspace(0, duration, int(sr * duration))
        # 生成一个简单的吉他音色模拟（多个正弦波叠加）
        audio = torch.sin(2 * torch.pi * 220 * t) * 0.5  # A3音符
        audio += torch.sin(2 * torch.pi * 440 * t) * 0.3  # A4音符
        audio += torch.sin(2 * torch.pi * 660 * t) * 0.1  # E5音符
        audio = audio.unsqueeze(0)  # 添加通道维度
        print(f"生成的测试音频形状: {audio.shape}")
        
        # 保存生成的测试音频
        os.makedirs("../audio", exist_ok=True)
        torchaudio.save("../audio/short_riff.wav", audio, sr)
        print("生成的测试音频已保存为 ../audio/short_riff.wav")
    
    # 确保音频格式正确
    if audio.dim() == 2:  # (channels, samples)
        audio = audio.unsqueeze(0)  # 添加批次维度 (batch, channels, samples)
    
    audio = audio.to(device)
    
    # 初始化Text2FX
    text2fx = Text2FX(device=device)
    
    # 创建效果参数
    fx_params = {
        'eq_params': {
            'gains': torch.tensor([[3.0, -2.0, 4.0, -1.0, 2.0, -3.0]], device=device),  # 6个频段的增益
            'freqs': torch.tensor([[100.0, 300.0, 1000.0, 3000.0, 6000.0, 10000.0]], device=device),  # 频率
            'qs': torch.tensor([[0.7, 1.0, 1.2, 1.5, 2.0, 0.7]], device=device)  # Q因子
        },
        'reverb_params': {
            'band0_gain': torch.tensor([0.8], device=device),
            'band1_gain': torch.tensor([0.7], device=device),
            'band2_gain': torch.tensor([0.6], device=device),
            'band3_gain': torch.tensor([0.5], device=device),
            'band4_gain': torch.tensor([0.4], device=device),
            'band5_gain': torch.tensor([0.3], device=device),
            'band6_gain': torch.tensor([0.2], device=device),
            'band7_gain': torch.tensor([0.1], device=device),
            'band8_gain': torch.tensor([0.05], device=device),
            'band9_gain': torch.tensor([0.02], device=device),
            'band10_gain': torch.tensor([0.01], device=device),
            'band11_gain': torch.tensor([0.005], device=device),
            'band0_decay': torch.tensor([0.9], device=device),
            'band1_decay': torch.tensor([0.85], device=device),
            'band2_decay': torch.tensor([0.8], device=device),
            'band3_decay': torch.tensor([0.75], device=device),
            'band4_decay': torch.tensor([0.7], device=device),
            'band5_decay': torch.tensor([0.65], device=device),
            'band6_decay': torch.tensor([0.6], device=device),
            'band7_decay': torch.tensor([0.55], device=device),
            'band8_decay': torch.tensor([0.5], device=device),
            'band9_decay': torch.tensor([0.45], device=device),
            'band10_decay': torch.tensor([0.4], device=device),
            'band11_decay': torch.tensor([0.35], device=device),
            'mix': torch.tensor([0.7], device=device)
        }
    }
    
    # 应用效果链
    processed_audio = text2fx.apply_fx_chain(audio, sr, fx_params)
    
    print(f"处理后音频形状: {processed_audio.shape}")
    
    # 保存处理后的音频
    torchaudio.save("fx_chain_processed_audio.wav", processed_audio.squeeze(0).cpu(), sr)
    
    print("效果链处理后的音频已保存为 fx_chain_processed_audio.wav")
    
    # 绘制频谱图比较
    plot_spectrogram_comparison(audio.squeeze(0).cpu(), processed_audio.squeeze(0).cpu(), sr)

def plot_spectrogram_comparison(original_audio, processed_audio, sr):
    """绘制原始音频和处理后音频的频谱图比较"""
    plt.figure(figsize=(12, 8))
    
    # 原始音频频谱图
    plt.subplot(2, 1, 1)
    spec = torchaudio.transforms.Spectrogram(
        n_fft=2048,
        win_length=1024,
        hop_length=512,
        power=2.0
    )(original_audio)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    plt.imshow(spec_db[0].numpy(), aspect='auto', origin='lower', 
               extent=[0, original_audio.shape[1]/sr, 0, sr/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title('原始音频频谱图')
    plt.ylabel('频率 (Hz)')
    
    # 处理后音频频谱图
    plt.subplot(2, 1, 2)
    spec = torchaudio.transforms.Spectrogram(
        n_fft=2048,
        win_length=1024,
        hop_length=512,
        power=2.0
    )(processed_audio)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    plt.imshow(spec_db[0].numpy(), aspect='auto', origin='lower', 
               extent=[0, processed_audio.shape[1]/sr, 0, sr/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title('处理后音频频谱图')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')
    
    plt.tight_layout()
    plt.savefig('fx_chain_spectrogram_comparison.png')
    print("频谱图比较已保存为 fx_chain_spectrogram_comparison.png")

if __name__ == "__main__":
    main() 