# Text2FX: 基于CLAP嵌入的文本引导音频效果

本项目是对论文"Text2FX: Harnessing CLAP Embeddings for Text-Guided Audio Effects"的实现，该方法利用CLAP（Contrastive Language-Audio Pretraining）模型的嵌入表示，通过梯度优化来调整音频效果参数，使处理后的音频在嵌入空间中接近目标文本描述。

## 功能特点

- 支持通过自然语言描述控制音频效果
- 无需针对新词汇或音频效果重新训练
- 支持多种音频效果，包括均衡器和混响
- 提供两种优化方法：余弦相似度和方向性优化

## 安装依赖

```bash
pip install torch torchaudio numpy matplotlib tqdm laion-clap dasp-pytorch
```

## 使用方法

### 基本用法

```python
import torch
import torchaudio
from text2fx import Text2FX

# 加载音频
audio, sr = torchaudio.load("input.wav")

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
python text2fx.py --audio_path input.wav --target_text "温暖的吉他声音，带有轻微的混响" --method cosine --output_path output.wav
```

对于方向性优化方法，需要提供对比文本：

```bash
python text2fx.py --audio_path input.wav --target_text "温暖的吉他声音，带有轻微的混响" --contrast_text "冷酷的吉他声音，没有混响" --method directional --output_path output.wav
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

## 参考论文

"Text2FX: Harnessing CLAP Embeddings for Text-Guided Audio Effects"

## 依赖项目

- [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP): 提供音频和文本的联合嵌入表示
- [csteinmetz1/dasp-pytorch](https://github.com/csteinmetz1/dasp-pytorch): 提供可微分的音频信号处理器 