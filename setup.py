"""
setup.py — MAC-CLIP package installer
"""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="mac_clip",
    version="1.0.0",
    author="Anonymous (BMVC 2026 Submission #1242)",
    description=(
        "MAC-CLIP: Morpho-Semantic Adaptive Contrastive Learning "
        "for Reliable Multi-Modality Medical Segmentation"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(exclude=["baselines", "scripts", "configs", "assets"]),
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "macclip-train=main:main",
            "macclip-ablation=scripts.ablation_study:main",
            "macclip-compare=scripts.compare_results:main",
            "macclip-repr=scripts.visualize_features:main",
            "macclip-baselines=baselines.baselines:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
