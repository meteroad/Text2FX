# Text2FX 安装指南

本文档提供了安装 Text2FX 及其依赖的详细步骤。

## 环境要求

- Python 3.10 或更高版本
- CUDA 11.8（如果需要 GPU 加速）

## 方法一：使用 pip 安装（推荐）

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/Text2FX.git
cd Text2FX
```

2. 创建并激活虚拟环境（可选但推荐）：

```bash
# 使用 conda
conda create -n text2fx python=3.10
conda activate text2fx

# 或使用 venv
python -m venv text2fx_env
source text2fx_env/bin/activate  # Linux/Mac
# 或
text2fx_env\Scripts\activate  # Windows
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 方法二：使用 setup.py 安装

```bash
git clone https://github.com/yourusername/Text2FX.git
cd Text2FX
pip install -e .
```

## 方法三：手动安装依赖

如果您遇到安装问题，可以尝试手动安装主要依赖：

```bash
# 安装 PyTorch 和 torchaudio（带 CUDA 支持）
pip install torch==2.0.0+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install laion_clap==1.1.6
pip install dasp-pytorch==0.0.1
pip install numpy==1.23.5 matplotlib==3.10.1 tqdm==4.67.1 librosa==0.10.2.post1
pip install transformers wget
```

## 下载预训练模型

安装完依赖后，您需要下载 CLAP 预训练模型：

```bash
python download_clap_model.py
```

## 验证安装

运行测试脚本验证安装是否成功：

```bash
python test/test_clap.py
python test/test_dasp.py
python test/test_fx_chain.py
```

## 常见问题

### 1. CUDA 相关错误

如果您遇到 CUDA 相关错误，可能是 PyTorch 版本与您的 CUDA 版本不兼容。请尝试安装与您的 CUDA 版本匹配的 PyTorch：

```bash
# 对于 CUDA 11.8
pip install torch==2.0.0+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 对于 CUDA 11.7
pip install torch==2.0.0+cu117 torchaudio==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 如果不需要 CUDA 支持
pip install torch==2.0.0 torchaudio==2.0.0
```

### 2. laion_clap 安装问题

如果 laion_clap 安装失败，可以尝试从源代码安装：

```bash
git clone https://github.com/LAION-AI/CLAP.git
cd CLAP
pip install -e .
```

### 3. dasp-pytorch 安装问题

如果 dasp-pytorch 安装失败，可以尝试从源代码安装：

```bash
git clone https://github.com/hugofloresgarcia/dasp-pytorch.git
cd dasp-pytorch
pip install -e .
``` 