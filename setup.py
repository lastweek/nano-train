from setuptools import setup, find_packages

setup(
    name="nano-train",
    version="0.0.1",
    description="A distributed LLM training framework",
    author="lastweek",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "numpy>=1.24.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.15.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
