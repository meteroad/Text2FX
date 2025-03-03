import torch
import torchaudio
from text2fx import Text2FX

def main():
    # 设置参数
    audio_path = "short_riff.wav"  
    target_text = "音色温暖，带有轻微混响"
    
    # 加载音频
    try:
        audio, sr = torchaudio.load(audio_path)
        print(f"加载音频: {audio_path}, 形状: {audio.shape}, 采样率: {sr}")
    except Exception as e:
        print(f"无法加载音频文件: {e}")
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
    
    # 初始化Text2FX
    text2fx = Text2FX(device='cpu')
    
    # 使用余弦相似度方法优化
    print("\n使用余弦相似度方法优化...")
    cosine_params, cosine_audio = text2fx.optimize_cosine(
        audio, sr, target_text, 
        num_iterations=300,  # 减少迭代次数以加快示例运行
        num_runs=1  # 只运行一次以加快示例运行
    )
    
    # 保存处理后的音频
    torchaudio.save("output_cosine.wav", cosine_audio.squeeze(0).cpu(), sr)
    print("余弦方法处理后的音频已保存至: output_cosine.wav")
    
    # 使用方向性方法优化
    print("\n使用方向性方法优化...")
    contrast_text = "冷酷的吉他声音，没有混响"
    directional_params, directional_audio = text2fx.optimize_directional(
        audio, sr, target_text, contrast_text,
        num_iterations=300,  # 减少迭代次数以加快示例运行
        num_runs=1  # 只运行一次以加快示例运行
    )
    
    # 保存处理后的音频
    torchaudio.save("output_directional.wav", directional_audio.squeeze(0).cpu(), sr)
    print("方向性方法处理后的音频已保存至: output_directional.wav")
    
    # 打印最佳参数
    print("\n余弦方法最佳效果参数:")
    for fx_name, params in cosine_params.items():
        print(f"  {fx_name}:")
        for param_name, param_value in params.items():
            print(f"    {param_name}: {param_value.detach().cpu().numpy()}")
    
    print("\n方向性方法最佳效果参数:")
    for fx_name, params in directional_params.items():
        print(f"  {fx_name}:")
        for param_name, param_value in params.items():
            print(f"    {param_name}: {param_value.detach().cpu().numpy()}")

if __name__ == "__main__":
    main() 