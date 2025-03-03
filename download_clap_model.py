import os
import laion_clap
import subprocess
import sys

def download_file(url, destination):
    """使用wget下载文件，支持断点续传"""
    try:
        # 检查wget是否可用
        subprocess.run(["wget", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # 使用wget下载，支持断点续传
        cmd = ["wget", "-c", url, "-O", destination]
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            # 检查curl是否可用
            subprocess.run(["curl", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            # 使用curl下载，支持断点续传
            cmd = ["curl", "-C", "-", "-L", url, "-o", destination]
            print(f"执行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("错误: 系统中未找到wget或curl。请手动下载文件。")
            return False

def main():
    # 获取laion_clap包的位置
    package_dir = os.path.dirname(os.path.realpath(laion_clap.__file__))
    print(f"laion_clap包位置: {package_dir}")
    
    # 设置模型URL和目标路径
    model_url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt?download=true"
    model_path = os.path.join("checkpoint", "music_speech_epoch_15.pt")
    
    # 检查模型文件是否已存在
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"模型文件已存在: {model_path}")
        print(f"文件大小: {file_size / (1024*1024):.2f} MB")
        
        # 检查文件大小是否正确（约1.7GB）
        if file_size > 1800000000:  # 大约1.7GB
            print("文件大小看起来正确，可能已完整下载")
            return
        else:
            print(f"文件大小不正确，可能下载不完整。将尝试继续下载...")
    
    # 下载模型文件
    print(f"开始下载模型文件到: {model_path}")
    success = download_file(model_url, model_path)
    
    if success:
        print(f"模型文件下载完成: {model_path}")
        # 验证文件大小
        file_size = os.path.getsize(model_path)
        print(f"文件大小: {file_size / (1024*1024):.2f} MB")
        if file_size > 1800000000:  # 大约1.7GB
            print("文件大小看起来正确，下载可能已完成")
        else:
            print(f"警告: 文件大小({file_size / (1024*1024):.2f} MB)小于预期(~1.7GB)，下载可能不完整")
    else:
        print("下载失败，请尝试手动下载")
        print(f"下载链接: {model_url}")
        print(f"目标路径: {model_path}")

if __name__ == "__main__":
    main() 