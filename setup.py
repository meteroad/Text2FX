from setuptools import setup, find_packages

setup(
    name="text2fx",
    version="0.1.0",
    description="文本引导的音频效果控制",
    author="Text2FX Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "laion_clap>=1.1.6",
        "dasp-pytorch>=0.0.1",
        "numpy>=1.23.5",
        "matplotlib>=3.10.1",
        "tqdm>=4.67.1",
        "librosa>=0.10.2",
        "transformers>=4.0.0",
        "wget>=3.2",
        "argparse>=1.4.0",
    ],
    python_requires=">=3.10",
) 