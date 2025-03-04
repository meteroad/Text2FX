# Text2FX: 基于CLAP嵌入的文本引导音频效果

本项目是对论文"Text2FX: Harnessing CLAP Embeddings for Text-Guided Audio Effects"的复现，该方法利用CLAP（Contrastive Language-Audio Pretraining）模型的嵌入表示，通过梯度优化来调整音频效果参数，使处理后的音频在嵌入空间中接近目标文本描述。

## 功能特点

- 支持通过自然语言描述控制音频效果
- 无需针对新词汇或音频效果重新训练
- 支持多种音频效果，包括均衡器和混响
- 提供两种优化方法：余弦相似度和方向性优化

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8 或 12.0（如果使用GPU）
- 足够的磁盘空间用于存储预训练模型

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/meteroad/Text2FX.git
cd Text2FX
```

2. **创建并激活conda环境（推荐）**
```bash
# 创建新的conda环境
conda create -n text2fx python=3.10
conda activate text2fx

# 安装PyTorch（根据您的CUDA版本选择合适的命令）
# 对于CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 对于CUDA 12.0
conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

3. **下载预训练模型**
```bash
python download_clap_model.py
```

### 验证安装

运行测试脚本验证安装是否成功：
```bash
python test/test_clap.py
```

## 使用方法

### 基本用法

```python
import torch
import torchaudio
from text2fx import Text2FX

# 加载音频
audio, sr = torchaudio.load("audio/short_riff.wav")

# 初始化Text2FX
text2fx = Text2FX()

# 使用余弦相似度方法优化
target_text = "温暖的吉他声音，带有轻微的混响"
params, processed_audio = text2fx.optimize_cosine(
    audio, sr, target_text, 
    num_iterations=600,
    num_runs=3
)

# 保存处理后的音频
torchaudio.save("output.wav", processed_audio.squeeze(0).cpu(), sr)
```

### 命令行使用

```bash
python text2fx.py --audio_path audio/short_riff.wav --target_text "温暖的吉他声音，带有轻微的混响" --method cosine --output_path output.wav
```

对于方向性优化方法，需要提供对比文本：

```bash
python text2fx.py --audio_path audio/蛄蛹者_short.wav --target_text "温暖的吉他声音，带有轻微的混响" --contrast_text "冷酷的吉他声音，没有混响" --method directional --output_path output.wav
```

## 优化方法

### 余弦相似度优化 (Text2FX-cosine)

通过最小化处理后音频嵌入和目标文本提示嵌入之间的余弦距离来进行优化。

### 方向性优化 (Text2FX-directional)

利用对比文本提示与目标文本提示嵌入之间的差异来引导音频嵌入的移动，避免音频嵌入朝着与目标提示不相关的方向移动。

## 示例

运行示例脚本：

```bash
python example.py
```

这将生成两个输出文件：
- `output_cosine.wav`: 使用余弦相似度方法处理的音频
- `output_directional.wav`: 使用方向性方法处理的音频

## 常见问题

### CUDA相关错误

如果遇到CUDA相关错误，可以尝试以下解决方案：

1. 确保CUDA版本与PyTorch版本匹配
2. 检查CUDA环境变量是否正确设置
3. 如果不需要GPU加速，可以安装CPU版本的PyTorch

### 模型下载问题

如果无法从HuggingFace下载模型，可以：

1. 使用国内镜像源
2. 手动下载模型文件
3. 设置代理

## 项目结构

```
Text2FX/
├── audio/                    # 示例音频文件
│   ├── short_riff.wav       # 示例音频1
│   └── 蛄蛹者_short.wav     # 示例音频2
├── checkpoint/              # 预训练模型存放目录
├── test/                   # 测试脚本
│   └── test_clap.py        # CLAP模型测试
├── text2fx/                # 主要代码目录
│   ├── __init__.py
│   ├── clap.py            # CLAP模型相关代码
│   └── text2fx.py         # 主要实现代码
├── download_clap_model.py  # 模型下载脚本
├── example.py             # 示例脚本
├── requirements.txt       # 依赖包列表
└── text2fx.py            # 命令行入口
```

## 参考论文

"Text2FX: Harnessing CLAP Embeddings for Text-Guided Audio Effects"

## 依赖项目

- [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP): 提供音频和文本的联合嵌入表示
- [csteinmetz1/dasp-pytorch](https://github.com/csteinmetz1/dasp-pytorch): 提供可微分的音频信号处理器 
