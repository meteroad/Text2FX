#!/bin/bash

# 显示彩色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始安装 Text2FX 及其依赖...${NC}"

# 检查 Python 版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo -e "${YELLOW}检测到 Python 版本: ${python_version}${NC}"

# 检查是否有 CUDA
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${YELLOW}检测到 CUDA 版本: ${cuda_version}${NC}"
    has_cuda=true
else
    echo -e "${YELLOW}未检测到 CUDA，将安装 CPU 版本的 PyTorch${NC}"
    has_cuda=false
fi

# 创建虚拟环境（如果用户选择）
read -p "是否创建新的虚拟环境? (y/n): " create_env
if [[ $create_env == "y" || $create_env == "Y" ]]; then
    read -p "使用哪种虚拟环境? (conda/venv): " env_type
    if [[ $env_type == "conda" ]]; then
        echo -e "${YELLOW}创建 conda 环境...${NC}"
        conda create -n text2fx python=3.10 -y
        echo -e "${YELLOW}激活 conda 环境...${NC}"
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate text2fx
    elif [[ $env_type == "venv" ]]; then
        echo -e "${YELLOW}创建 venv 环境...${NC}"
        python -m venv text2fx_env
        echo -e "${YELLOW}激活 venv 环境...${NC}"
        source text2fx_env/bin/activate
    else
        echo -e "${RED}未知的虚拟环境类型，跳过环境创建${NC}"
    fi
fi

# 安装依赖
echo -e "${YELLOW}安装依赖...${NC}"
if [ "$has_cuda" = true ]; then
    # 根据 CUDA 版本选择合适的 PyTorch 版本
    if [[ $cuda_version == 12.0* ]]; then
        echo -e "${YELLOW}检测到 CUDA 12.0，安装兼容的 PyTorch...${NC}"
        echo -e "${YELLOW}注意：如果安装失败，建议降级到 CUDA 11.8 或使用 CPU 版本${NC}"
        pip install torch==2.0.0+cu120 torchaudio==2.0.0+cu120 --extra-index-url https://download.pytorch.org/whl/cu120
    elif [[ $cuda_version == 11.8* ]]; then
        echo -e "${YELLOW}安装 CUDA 11.8 兼容的 PyTorch...${NC}"
        pip install torch==2.0.0+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    elif [[ $cuda_version == 11.7* ]]; then
        echo -e "${YELLOW}安装 CUDA 11.7 兼容的 PyTorch...${NC}"
        pip install torch==2.0.0+cu117 torchaudio==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    elif [[ $cuda_version == 11.6* ]]; then
        echo -e "${YELLOW}安装 CUDA 11.6 兼容的 PyTorch...${NC}"
        pip install torch==2.0.0+cu116 torchaudio==2.0.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    else
        echo -e "${YELLOW}未找到完全匹配的 CUDA 版本，尝试安装 CUDA 11.8 兼容的 PyTorch...${NC}"
        echo -e "${YELLOW}如果安装失败，建议使用 CPU 版本的 PyTorch${NC}"
        pip install torch==2.0.0+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo -e "${YELLOW}安装 CPU 版本的 PyTorch...${NC}"
    pip install torch==2.0.0 torchaudio==2.0.0
fi

# 安装其他依赖
echo -e "${YELLOW}安装其他依赖...${NC}"
pip install -r requirements.txt

# 下载预训练模型
echo -e "${YELLOW}下载 CLAP 预训练模型...${NC}"
python download_clap_model.py

# 运行测试
echo -e "${YELLOW}运行测试脚本验证安装...${NC}"
python test/test_clap.py

echo -e "${GREEN}安装完成！${NC}"
echo -e "${GREEN}您可以通过运行以下命令来测试更多功能：${NC}"
echo -e "${YELLOW}python test/test_dasp.py${NC}"
echo -e "${YELLOW}python test/test_fx_chain.py${NC}"
echo -e "${YELLOW}python text2fx.py --audio_path audio/short_riff.wav --target_text \"明亮的吉他声音\" --output_path output.wav${NC}" 