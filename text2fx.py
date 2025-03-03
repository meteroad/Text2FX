import torch
import torchaudio
import numpy as np
import laion_clap
import dasp_pytorch
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse
import os

class Text2FX:
    def __init__(self, device=None):
        """初始化Text2FX模型
        
        Args:
            device: 运行设备，如果为None则自动选择
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        print(f"使用设备: {self.device}")
        
        # 加载CLAP模型
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=device)
        self.clap_model.load_ckpt('checkpoint/music_speech_epoch_15.pt')
        print("CLAP模型加载完成")
        
    def random_shift(self, audio, sr, max_shift_ms=1500):
        """对音频进行随机移位，防止模型过度关注音频内容
        
        Args:
            audio: 输入音频张量，形状为(batch_size, channels, samples)
            sr: 采样率
            max_shift_ms: 最大移位时间（毫秒）
            
        Returns:
            移位后的音频张量
        """
        if max_shift_ms <= 0:
            return audio
            
        max_shift = int(sr * max_shift_ms / 1000)
        shift = random.randint(0, max_shift)
        
        # 创建移位后的音频
        shifted_audio = torch.zeros_like(audio)
        shifted_audio[..., shift:] = audio[..., :(audio.shape[-1] - shift)]
        
        return shifted_audio
    
    def apply_fx_chain(self, audio, sr, fx_params):
        """应用音频效果链
        
        Args:
            audio: 输入音频张量，形状为(batch_size, channels, samples)
            sr: 采样率
            fx_params: 效果参数字典
            
        Returns:
            处理后的音频张量
        """
        processed_audio = audio
        
        # 应用参数均衡器
        if 'eq_params' in fx_params:
            eq_params = fx_params['eq_params']
            processed_audio = dasp_pytorch.functional.parametric_eq(
                processed_audio,
                sr,
                eq_params['low_shelf_gain_db'],
                eq_params['low_shelf_cutoff_freq'],
                eq_params['low_shelf_q_factor'],
                eq_params['band0_gain_db'],
                eq_params['band0_cutoff_freq'],
                eq_params['band0_q_factor'],
                eq_params['band1_gain_db'],
                eq_params['band1_cutoff_freq'],
                eq_params['band1_q_factor'],
                eq_params['band2_gain_db'],
                eq_params['band2_cutoff_freq'],
                eq_params['band2_q_factor'],
                eq_params['band3_gain_db'],
                eq_params['band3_cutoff_freq'],
                eq_params['band3_q_factor'],
                eq_params['high_shelf_gain_db'],
                eq_params['high_shelf_cutoff_freq'],
                eq_params['high_shelf_q_factor']
            )
        
        # 应用混响
        if 'reverb_params' in fx_params:
            reverb_params = fx_params['reverb_params']
            processed_audio = dasp_pytorch.functional.noise_shaped_reverberation(
                processed_audio,
                sr,
                reverb_params['band0_gain'],
                reverb_params['band1_gain'],
                reverb_params['band2_gain'],
                reverb_params['band3_gain'],
                reverb_params['band4_gain'],
                reverb_params['band5_gain'],
                reverb_params['band6_gain'],
                reverb_params['band7_gain'],
                reverb_params['band8_gain'],
                reverb_params['band9_gain'],
                reverb_params['band10_gain'],
                reverb_params['band11_gain'],
                reverb_params['band0_decay'],
                reverb_params['band1_decay'],
                reverb_params['band2_decay'],
                reverb_params['band3_decay'],
                reverb_params['band4_decay'],
                reverb_params['band5_decay'],
                reverb_params['band6_decay'],
                reverb_params['band7_decay'],
                reverb_params['band8_decay'],
                reverb_params['band9_decay'],
                reverb_params['band10_decay'],
                reverb_params['band11_decay'],
                reverb_params['mix'],
                num_samples=65536,  # 默认值
                num_bandpass_taps=1023  # 默认值
            )
            
        return processed_audio
    
    def get_audio_embedding(self, audio_tensor):
        """自定义函数，从音频张量获取嵌入
        
        Args:
            audio_tensor: 音频张量，形状为(channels, samples)或(batch_size, channels, samples)
            
        Returns:
            音频嵌入张量
        """
        # 确保音频张量在正确的设备上
        audio_tensor = audio_tensor.to(self.device)
        
        # 处理音频张量的形状
        if audio_tensor.dim() == 3 and audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor[0]
        
        # 多通道音频转单通道
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)  # (1, samples)
        audio_tensor = audio_tensor[0]
        
        audio_list = [audio_tensor]
        
        # 获取音频嵌入
        with torch.enable_grad():
            audio_embed = self.clap_model.get_audio_embedding_from_data(audio_list, use_tensor=True)
            if audio_embed.device != self.device:
                audio_embed = audio_embed.to(self.device)
        
        return audio_embed
    
    def _init_random_params(self):
        # 参数均衡器参数
        eq_params = {
            'low_shelf_gain_db': torch.nn.Parameter(torch.tensor([4.0 * torch.rand(1).item() - 2.0], device=self.device)),
            'low_shelf_cutoff_freq': torch.nn.Parameter(torch.tensor([80.0 + 40.0 * torch.rand(1).item()], device=self.device)),
            'low_shelf_q_factor': torch.nn.Parameter(torch.tensor([0.5 + 0.5 * torch.rand(1).item()], device=self.device)),
            'band0_gain_db': torch.nn.Parameter(torch.tensor([4.0 * torch.rand(1).item() - 2.0], device=self.device)),
            'band0_cutoff_freq': torch.nn.Parameter(torch.tensor([200.0 + 200.0 * torch.rand(1).item()], device=self.device)),
            'band0_q_factor': torch.nn.Parameter(torch.tensor([0.8 + 0.4 * torch.rand(1).item()], device=self.device)),
            'band1_gain_db': torch.nn.Parameter(torch.tensor([4.0 * torch.rand(1).item() - 2.0], device=self.device)),
            'band1_cutoff_freq': torch.nn.Parameter(torch.tensor([800.0 + 400.0 * torch.rand(1).item()], device=self.device)),
            'band1_q_factor': torch.nn.Parameter(torch.tensor([1.0 + 0.4 * torch.rand(1).item()], device=self.device)),
            'band2_gain_db': torch.nn.Parameter(torch.tensor([4.0 * torch.rand(1).item() - 2.0], device=self.device)),
            'band2_cutoff_freq': torch.nn.Parameter(torch.tensor([2500.0 + 1000.0 * torch.rand(1).item()], device=self.device)),
            'band2_q_factor': torch.nn.Parameter(torch.tensor([1.3 + 0.4 * torch.rand(1).item()], device=self.device)),
            'band3_gain_db': torch.nn.Parameter(torch.tensor([4.0 * torch.rand(1).item() - 2.0], device=self.device)),
            'band3_cutoff_freq': torch.nn.Parameter(torch.tensor([5000.0 + 2000.0 * torch.rand(1).item()], device=self.device)),
            'band3_q_factor': torch.nn.Parameter(torch.tensor([1.8 + 0.4 * torch.rand(1).item()], device=self.device)),
            'high_shelf_gain_db': torch.nn.Parameter(torch.tensor([4.0 * torch.rand(1).item() - 2.0], device=self.device)),
            'high_shelf_cutoff_freq': torch.nn.Parameter(torch.tensor([8000.0 + 4000.0 * torch.rand(1).item()], device=self.device)),
            'high_shelf_q_factor': torch.nn.Parameter(torch.tensor([0.5 + 0.4 * torch.rand(1).item()], device=self.device))
        }
        
        # 混响参数
        reverb_params = {}
        # 增益参数
        for i in range(12):
            if i == 0:
                reverb_params[f'band{i}_gain'] = torch.nn.Parameter(torch.rand(1, device=self.device) * 0.8 + 0.2)
            elif i < 7:
                reverb_params[f'band{i}_gain'] = torch.nn.Parameter(torch.rand(1, device=self.device) * (0.9 - i * 0.1) + 0.1)
            elif i < 9:
                reverb_params[f'band{i}_gain'] = torch.nn.Parameter(torch.rand(1, device=self.device) * 0.15 + 0.05)
            elif i < 10:
                reverb_params[f'band{i}_gain'] = torch.nn.Parameter(torch.rand(1, device=self.device) * 0.1 + 0.01)
            elif i < 11:
                reverb_params[f'band{i}_gain'] = torch.nn.Parameter(torch.rand(1, device=self.device) * 0.05 + 0.01)
            else:
                reverb_params[f'band{i}_gain'] = torch.nn.Parameter(torch.rand(1, device=self.device) * 0.02 + 0.005)
        
        # 衰减参数
        for i in range(12):
            reverb_params[f'band{i}_decay'] = torch.nn.Parameter(torch.rand(1, device=self.device) * 0.3 + (0.7 - i * 0.05))
        
        # 混合参数
        reverb_params['mix'] = torch.nn.Parameter(torch.rand(1, device=self.device) * 0.8 + 0.2)
        
        return {'eq_params': eq_params, 'reverb_params': reverb_params}
    
    def optimize_cosine(self, audio, sr, target_text, fx_init_params=None, 
                        num_iterations=600, lr=1e-2, max_shift_ms=1500, 
                        num_runs=3, verbose=True):
        """使用余弦相似度优化音频效果参数
        
        Args:
            audio: 输入音频张量，形状为(batch_size, channels, samples)
            sr: 采样率
            target_text: 目标文本描述
            fx_init_params: 初始效果参数（如果为None则随机初始化）
            num_iterations: 优化迭代次数
            lr: 学习率
            max_shift_ms: 最大音频移位（毫秒）
            num_runs: 运行次数，选择损失最低的结果
            verbose: 是否显示进度条
            
        Returns:
            优化后的效果参数和处理后的音频
        """
        # 确保音频格式正确
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        audio = audio.to(self.device).float()
        
        # 获取目标文本嵌入
        target_text_embed = torch.tensor(self.clap_model.get_text_embedding([target_text]), device=self.device)
        target_text_embed = target_text_embed / target_text_embed.norm(dim=-1, keepdim=True)
        
        best_loss = float('inf')
        best_params = None
        best_audio = None
        
        for run in range(num_runs):
            if verbose:
                print(f"运行 {run+1}/{num_runs}")
            
            # 初始化或随机化效果参数
            params = self._init_random_params() if fx_init_params is None else fx_init_params
                
            # 设置优化器
            optimizer_params = []
            for param_group in params.values():
                for param in param_group.values():
                    optimizer_params.append(param)
            
            optimizer = torch.optim.Adam(optimizer_params, lr=lr)
            
            # 优化循环
            pbar = tqdm(range(num_iterations)) if verbose else range(num_iterations)
            losses = []
            
            for i in pbar:
                # 随机移位音频
                shifted_audio = self.random_shift(audio, sr, max_shift_ms)
                
                # 应用音频效果
                processed_audio = self.apply_fx_chain(shifted_audio, sr, params)
                
                # 使用自定义函数获取音频嵌入
                with torch.enable_grad():
                    # 创建一个临时的音频张量，确保它有梯度
                    temp_audio = processed_audio.clone()
                    temp_audio.requires_grad_(True)
                    
                    # 获取音频嵌入
                    audio_embed = self.get_audio_embedding(temp_audio)
                    
                    # 计算余弦相似度损失
                    similarity = torch.sum(audio_embed * target_text_embed, dim=-1)
                    loss = 1 - similarity
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                losses.append(loss.item())
                
                if verbose:
                    pbar.set_description(f"Loss: {loss.item():.4f}")
            
            # 应用最终参数处理音频
            final_audio = self.apply_fx_chain(audio, sr, params)
            final_loss = losses[-1]
            
            # 保存最佳结果
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = params
                best_audio = final_audio
                
            if verbose:
                plt.figure(figsize=(10, 4))
                plt.plot(losses)
                plt.title(f"Run {run+1} Loss Curve")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.show()
        
        return best_params, best_audio
    
    def optimize_directional(self, audio, sr, target_text, contrast_text, 
                            fx_init_params=None, num_iterations=600, lr=1e-2, 
                            max_shift_ms=1500, num_runs=3, verbose=True):
        """使用方向性优化方法优化音频效果参数
        
        Args:
            audio: 输入音频张量，形状为(batch_size, channels, samples)
            sr: 采样率
            target_text: 目标文本描述
            contrast_text: 对比文本描述
            fx_init_params: 初始效果参数（如果为None则随机初始化）
            num_iterations: 优化迭代次数
            lr: 学习率
            max_shift_ms: 最大音频移位（毫秒）
            num_runs: 运行次数，选择损失最低的结果
            verbose: 是否显示进度条
            
        Returns:
            优化后的效果参数和处理后的音频
        """
        # 确保音频格式正确
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        audio = audio.to(self.device).float()
        
        # 获取目标文本和对比文本嵌入
        target_text_embed = torch.tensor(self.clap_model.get_text_embedding([target_text]), device=self.device)
        contrast_text_embed = torch.tensor(self.clap_model.get_text_embedding([contrast_text]), device=self.device)
        
        # 计算方向向量
        direction_vector = target_text_embed - contrast_text_embed
        direction_vector = direction_vector / direction_vector.norm(dim=-1, keepdim=True)
        
        best_loss = float('inf')
        best_params = None
        best_audio = None
        
        for run in range(num_runs):
            if verbose:
                print(f"运行 {run+1}/{num_runs}")
            
            # 初始化或随机化效果参数
            params = self._init_random_params() if fx_init_params is None else fx_init_params
                
            # 设置优化器
            optimizer_params = []
            for param_group in params.values():
                for param in param_group.values():
                    optimizer_params.append(param)
            
            optimizer = torch.optim.Adam(optimizer_params, lr=lr)
            
            # 优化循环
            pbar = tqdm(range(num_iterations)) if verbose else range(num_iterations)
            losses = []
            
            for i in pbar:
                # 随机移位音频
                shifted_audio = self.random_shift(audio, sr, max_shift_ms)
                
                # 应用音频效果
                processed_audio = self.apply_fx_chain(shifted_audio, sr, params)
                
                # 使用自定义函数获取音频嵌入
                with torch.enable_grad():
                    # 创建一个临时的音频张量，确保它有梯度
                    temp_audio = processed_audio.clone()
                    temp_audio.requires_grad_(True)
                    
                    # 获取音频嵌入
                    audio_embed = self.get_audio_embedding(temp_audio)
                    
                    # 计算方向性损失
                    projection = torch.sum(audio_embed * direction_vector, dim=-1)
                    loss = -projection  # 最大化投影等于最小化负投影
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                losses.append(loss.item())
                
                if verbose:
                    pbar.set_description(f"Loss: {loss.item():.4f}")
            
            # 应用最终参数处理音频
            final_audio = self.apply_fx_chain(audio, sr, params)
            final_loss = losses[-1]
            
            # 保存最佳结果
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = params
                best_audio = final_audio
                
            if verbose:
                plt.figure(figsize=(10, 4))
                plt.plot(losses)
                plt.title(f"Run {run+1} Loss Curve")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.show()
        
        return best_params, best_audio

def main():
    parser = argparse.ArgumentParser(description='Text2FX: 文本引导的音频效果控制')
    parser.add_argument('--audio_path', type=str, required=True, help='输入音频文件路径')
    parser.add_argument('--target_text', type=str, required=True, help='目标文本描述')
    parser.add_argument('--contrast_text', type=str, default=None, help='对比文本描述（用于方向性优化）')
    parser.add_argument('--method', type=str, default='cosine', choices=['cosine', 'directional'], 
                        help='优化方法: cosine或directional')
    parser.add_argument('--iterations', type=int, default=600, help='优化迭代次数')
    parser.add_argument('--output_path', type=str, default='output.wav', help='输出音频文件路径')
    
    args = parser.parse_args()
    
    # 处理音频路径，支持相对路径
    audio_path = args.audio_path
    if not os.path.exists(audio_path) and os.path.exists(os.path.join('audio', os.path.basename(audio_path))):
        audio_path = os.path.join('audio', os.path.basename(audio_path))
        print(f"在audio文件夹中找到音频文件: {audio_path}")
    
    # 加载音频
    audio, sr = torchaudio.load(audio_path)
    print(f"加载音频: {audio_path}, 形状: {audio.shape}, 采样率: {sr}")
    
    # 初始化Text2FX
    text2fx = Text2FX()
    
    # 根据选择的方法进行优化
    if args.method == 'cosine':
        best_params, best_audio = text2fx.optimize_cosine(
            audio, sr, args.target_text, 
            num_iterations=args.iterations
        )
    else:  # directional
        if args.contrast_text is None:
            raise ValueError("方向性优化方法需要提供对比文本描述")
        
        best_params, best_audio = text2fx.optimize_directional(
            audio, sr, args.target_text, args.contrast_text,
            num_iterations=args.iterations
        )
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # 保存处理后的音频
    torchaudio.save(args.output_path, best_audio.squeeze(0).cpu(), sr)
    print(f"处理后的音频已保存至: {args.output_path}")
    
    # 打印最佳参数
    print("最佳效果参数:")
    for fx_name, params in best_params.items():
        print(f"  {fx_name}:")
        for param_name, param_value in params.items():
            print(f"    {param_name}: {param_value.detach().cpu().numpy()}")

if __name__ == "__main__":
    main() 