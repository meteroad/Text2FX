import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import dasp_pytorch.functional as F
import os

def test_parametric_eq():
    """测试参数化均衡器功能"""
    print("测试参数化均衡器...")
    
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
    
    # 确保音频格式正确
    if audio.dim() == 2:  # (channels, samples)
        audio = audio.unsqueeze(0)  # 添加批次维度 (batch, channels, samples)
    
    audio = audio.to(device)
    
    # 先应用均衡器
    # 简化的均衡器参数
    low_shelf_gain_db = torch.tensor([4.0], device=device)
    low_shelf_cutoff_freq = torch.tensor([100.0], device=device)
    low_shelf_q_factor = torch.tensor([0.7], device=device)
    
    band0_gain_db = torch.tensor([-2.0], device=device)
    band0_cutoff_freq = torch.tensor([300.0], device=device)
    band0_q_factor = torch.tensor([1.0], device=device)
    
    band1_gain_db = torch.tensor([3.0], device=device)
    band1_cutoff_freq = torch.tensor([1000.0], device=device)
    band1_q_factor = torch.tensor([1.2], device=device)
    
    band2_gain_db = torch.tensor([-1.0], device=device)
    band2_cutoff_freq = torch.tensor([3000.0], device=device)
    band2_q_factor = torch.tensor([1.5], device=device)
    
    band3_gain_db = torch.tensor([2.0], device=device)
    band3_cutoff_freq = torch.tensor([6000.0], device=device)
    band3_q_factor = torch.tensor([2.0], device=device)
    
    high_shelf_gain_db = torch.tensor([-2.0], device=device)
    high_shelf_cutoff_freq = torch.tensor([10000.0], device=device)
    high_shelf_q_factor = torch.tensor([0.7], device=device)
    
    eq_audio = F.parametric_eq(
        audio,
        sr,
        low_shelf_gain_db,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        band0_gain_db,
        band0_cutoff_freq,
        band0_q_factor,
        band1_gain_db,
        band1_cutoff_freq,
        band1_q_factor,
        band2_gain_db,
        band2_cutoff_freq,
        band2_q_factor,
        band3_gain_db,
        band3_cutoff_freq,
        band3_q_factor,
        high_shelf_gain_db,
        high_shelf_cutoff_freq,
        high_shelf_q_factor
    )
    
    print(f"处理后音频形状: {eq_audio.shape}")
    
    # 保存处理前后的音频
    torchaudio.save("original_audio.wav", audio.squeeze(0).cpu(), sr)
    torchaudio.save("eq_processed_audio.wav", eq_audio.squeeze(0).cpu(), sr)
    
    print("音频已保存为 original_audio.wav 和 eq_processed_audio.wav")
    
    # 绘制频谱图比较
    plot_spectrogram_comparison(audio.squeeze(0).cpu(), eq_audio.squeeze(0).cpu(), sr)
    
    return eq_audio

def test_reverb():
    """测试混响功能"""
    print("\n测试混响...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载或生成测试音频
    audio_path = "short_riff.wav"
    if os.path.exists(audio_path):
        audio, sr = torchaudio.load(audio_path)
    else:
        # 生成一个简单的测试音频
        sr = 44100
        duration = 3  # 3秒
        t = torch.linspace(0, duration, int(sr * duration))
        audio = torch.sin(2 * torch.pi * 440 * t) * 0.5  # A4音符
        audio = audio.unsqueeze(0)  # 添加通道维度
    
    # 确保音频格式正确
    if audio.dim() == 2:  # (channels, samples)
        audio = audio.unsqueeze(0)  # 添加批次维度
    
    audio = audio.to(device)
    
    # 创建混响参数 - 使用正确的参数格式
    # 12个频段的增益参数
    band0_gain = torch.tensor([0.8], device=device)
    band1_gain = torch.tensor([0.7], device=device)
    band2_gain = torch.tensor([0.6], device=device)
    band3_gain = torch.tensor([0.5], device=device)
    band4_gain = torch.tensor([0.4], device=device)
    band5_gain = torch.tensor([0.3], device=device)
    band6_gain = torch.tensor([0.2], device=device)
    band7_gain = torch.tensor([0.1], device=device)
    band8_gain = torch.tensor([0.05], device=device)
    band9_gain = torch.tensor([0.02], device=device)
    band10_gain = torch.tensor([0.01], device=device)
    band11_gain = torch.tensor([0.005], device=device)
    
    # 12个频段的衰减参数
    band0_decay = torch.tensor([0.9], device=device)
    band1_decay = torch.tensor([0.85], device=device)
    band2_decay = torch.tensor([0.8], device=device)
    band3_decay = torch.tensor([0.75], device=device)
    band4_decay = torch.tensor([0.7], device=device)
    band5_decay = torch.tensor([0.65], device=device)
    band6_decay = torch.tensor([0.6], device=device)
    band7_decay = torch.tensor([0.55], device=device)
    band8_decay = torch.tensor([0.5], device=device)
    band9_decay = torch.tensor([0.45], device=device)
    band10_decay = torch.tensor([0.4], device=device)
    band11_decay = torch.tensor([0.35], device=device)
    
    # 混合参数 (干湿比)
    mix = torch.tensor([0.7], device=device)  # 70% 湿信号, 30% 干信号
    
    # 应用混响
    processed_audio = F.noise_shaped_reverberation(
        audio,
        sr,
        band0_gain,
        band1_gain,
        band2_gain,
        band3_gain,
        band4_gain,
        band5_gain,
        band6_gain,
        band7_gain,
        band8_gain,
        band9_gain,
        band10_gain,
        band11_gain,
        band0_decay,
        band1_decay,
        band2_decay,
        band3_decay,
        band4_decay,
        band5_decay,
        band6_decay,
        band7_decay,
        band8_decay,
        band9_decay,
        band10_decay,
        band11_decay,
        mix,
        num_samples=65536,  # 默认值
        num_bandpass_taps=1023  # 默认值
    )
    
    print(f"处理后音频形状: {processed_audio.shape}")
    
    # 保存处理后的音频
    torchaudio.save("reverb_processed_audio.wav", processed_audio.squeeze(0).cpu(), sr)
    
    print("混响处理后的音频已保存为 reverb_processed_audio.wav")
    
    return processed_audio

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
    plt.savefig('spectrogram_comparison.png')
    print("频谱图比较已保存为 spectrogram_comparison.png")

def main():
    """主函数"""
    print("测试DASP-PyTorch库的功能...")
    
    # 测试参数化均衡器
    eq_audio = test_parametric_eq()
    
    # 测试混响
    reverb_audio = test_reverb()
    
    # 测试均衡器和混响的组合
    print("\n测试均衡器和混响的组合...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载或生成测试音频
    audio_path = "short_riff.wav"
    if os.path.exists(audio_path):
        audio, sr = torchaudio.load(audio_path)
    else:
        # 使用之前生成的音频
        sr = 44100
        duration = 3  # 3秒
        t = torch.linspace(0, duration, int(sr * duration))
        audio = torch.sin(2 * torch.pi * 220 * t) * 0.5  # A3音符
        audio += torch.sin(2 * torch.pi * 440 * t) * 0.3  # A4音符
        audio += torch.sin(2 * torch.pi * 660 * t) * 0.1  # E5音符
        audio = audio.unsqueeze(0)  # 添加通道维度
    
    # 确保音频格式正确
    if audio.dim() == 2:  # (channels, samples)
        audio = audio.unsqueeze(0)  # 添加批次维度
    
    audio = audio.to(device)
    
    # 再应用混响
    # 12个频段的增益参数
    band0_gain = torch.tensor([0.7], device=device)
    band1_gain = torch.tensor([0.6], device=device)
    band2_gain = torch.tensor([0.5], device=device)
    band3_gain = torch.tensor([0.4], device=device)
    band4_gain = torch.tensor([0.3], device=device)
    band5_gain = torch.tensor([0.2], device=device)
    band6_gain = torch.tensor([0.1], device=device)
    band7_gain = torch.tensor([0.05], device=device)
    band8_gain = torch.tensor([0.02], device=device)
    band9_gain = torch.tensor([0.01], device=device)
    band10_gain = torch.tensor([0.005], device=device)
    band11_gain = torch.tensor([0.002], device=device)
    
    # 12个频段的衰减参数
    band0_decay = torch.tensor([0.85], device=device)
    band1_decay = torch.tensor([0.8], device=device)
    band2_decay = torch.tensor([0.75], device=device)
    band3_decay = torch.tensor([0.7], device=device)
    band4_decay = torch.tensor([0.65], device=device)
    band5_decay = torch.tensor([0.6], device=device)
    band6_decay = torch.tensor([0.55], device=device)
    band7_decay = torch.tensor([0.5], device=device)
    band8_decay = torch.tensor([0.45], device=device)
    band9_decay = torch.tensor([0.4], device=device)
    band10_decay = torch.tensor([0.35], device=device)
    band11_decay = torch.tensor([0.3], device=device)
    
    # 混合参数 (干湿比)
    mix = torch.tensor([0.6], device=device)  # 60% 湿信号, 40% 干信号
    
    combined_audio = F.noise_shaped_reverberation(
        eq_audio,
        sr,
        band0_gain,
        band1_gain,
        band2_gain,
        band3_gain,
        band4_gain,
        band5_gain,
        band6_gain,
        band7_gain,
        band8_gain,
        band9_gain,
        band10_gain,
        band11_gain,
        band0_decay,
        band1_decay,
        band2_decay,
        band3_decay,
        band4_decay,
        band5_decay,
        band6_decay,
        band7_decay,
        band8_decay,
        band9_decay,
        band10_decay,
        band11_decay,
        mix,
        num_samples=65536,  # 默认值
        num_bandpass_taps=1023  # 默认值
    )
    
    # 保存处理后的音频
    torchaudio.save("combined_eq_reverb_audio.wav", combined_audio.squeeze(0).cpu(), sr)
    
    print("均衡器和混响组合处理后的音频已保存为 combined_eq_reverb_audio.wav")
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    main()
