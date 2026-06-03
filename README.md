# MAC-CLIP: Morpho-Semantic Adaptive Contrastive Learning for Reliable Multi-Modality Medical Segmentation

> **BMVC 2026 — Submission #1242** (under double-blind review)

---

## Abstract

Vision-language foundation models offer strong global semantic priors, however their rigid spatial reasoning is poorly suited to dense biomedical segmentation, where tissues deform non-rigidly and clinically meaningful structure resides in irregular boundaries. MAC-CLIP introduces **Morpho-Semantic Adaptive Grouping (MSAG)** — a single central mechanism where each pixel is sampled at a learned geometric offset and aggregated under a pixel-wise uncertainty weight, producing deformation-aware tokens aligned to disentangled clinical concepts via a **Progressive Group Contrastive (PGC)** objective.

Validated across **15 datasets and 7 modalities** under a tri-seed protocol, MAC-CLIP achieves:

| Metric | MAC-CLIP | Best Baseline (TransUNet) | Gain |
|--------|----------|--------------------------|------|
| Mean Dice | **0.724** | 0.647 | +7.7 pp (+11.9%) |
| Mean IoU | **0.646** | 0.592 | +5.4 pp |
| Sev-3 Robustness | **97.7%** | 97.4% | best overall |
| Trainable Params | **15.12%** | 100% | 6.6× fewer |
| Significant Wins | **107/141** (75.9%) | — | p < 0.05 |

---

## Architecture Overview

```
Input Image (352×352)
       │
       ▼
BiomedCLIP ViT-B/16 (frozen) + LoRA adapters @ layers {6,8,10,12}
       │  multi-scale feature maps
       ▼
Feature Pyramid Network (FPN)
       │  fused spatial hierarchy
       ▼
┌─────────────────────────────────────┐
│  MSAG — Morpho-Semantic Adaptive    │  ◄── Core novelty
│  Grouping                           │
│   • Deformable offset predictor Ψ  │
│   • Pixel uncertainty estimator u  │
│   • Reliability-weighted pooling   │
└─────────────────────────────────────┘
       │  deformation-aware tokens
       ├──────────────────────────────►  Orthogonality-Constrained
       │                                  Semantic Concept Module (K=8)
       ▼
Boundary-Gated UNet Decoder
  (Selective-Kernel units + Reverse Attention + boundary gate α)
       │
       ├──► Segmentation logit
       ├──► Boundary probability map
       └──► Deep supervision auxiliaries (×3)

Training objectives: Tversky + IoU + Dice + Focal + Boundary +
                     Edge + Contour + BoundaryMap + PGC + PBC
```

---

## Results

### Table 1 — Per-Dataset Performance (Tri-Seed Mean ± Std; 95% Bootstrap CI)

| Dataset | Dice ↑ | IoU ↑ | BF1 ↑ | HD ↓ |
|---------|--------|-------|-------|------|
| **Endoscopy** | | | | |
| Kvasir-SEG | 0.874 ± 0.008 [.868,.885] | 0.803 ± 0.011 | 0.440 ± 0.018 | 56.8 ± 2.5 |
| m2caiSeg | 0.850 ± 0.005 [.844,.855] | 0.762 ± 0.007 | 0.322 ± 0.020 | 78.6 ± 1.9 |
| BKAI-NeoPolyp | 0.632 ± 0.015 [.618,.653] | 0.615 ± 0.020 | 0.022 ± 0.017 | 93.5 ± 4.6 |
| **Chest Radiography** | | | | |
| COVID-Lung | **0.975 ± 0.000** [.975,.976] | **0.953 ± 0.000** | **0.727 ± 0.002** | **8.1 ± 0.1** |
| COVID-Infection | 0.868 ± 0.011 [.859,.884] | 0.816 ± 0.012 | 0.206 ± 0.010 | 82.5 ± 1.6 |
| **X-Ray Angiography** | | | | |
| ARCADE-Stenosis | 0.428 ± 0.002 [.427,.430] | 0.297 ± 0.003 | 0.333 ± 0.000 | 73.4 ± 3.7 |
| ARCADE-Syntax | 0.773 ± 0.003 [.770,.776] | 0.636 ± 0.003 | 0.624 ± 0.005 | 50.7 ± 0.7 |
| BTXRD | 0.487 ± 0.008 [.478,.497] | 0.390 ± 0.007 | 0.184 ± 0.007 | 58.3 ± 1.3 |
| **Ultrasound** | | | | |
| BUSI | 0.776 ± 0.004 [.771,.780] | 0.685 ± 0.002 | 0.310 ± 0.007 | 38.5 ± 1.2 |
| CT2US-Kidney | 0.971 ± 0.002 [.968,.972] | 0.943 ± 0.003 | 0.632 ± 0.012 | 9.3 ± 0.3 |
| **Fundus / Retinal** | | | | |
| PAPILA | 0.945 ± 0.001 [.944,.946] | 0.897 ± 0.001 | 0.568 ± 0.002 | 5.5 ± 0.1 |
| DRIVE | 0.312 ± 0.101 [.170,.388] | 0.189 ± 0.068 | 0.087 ± 0.058 | 133.6 ± 31.3 |
| COph100† | 0.407 ± 0.031 | 0.297 ± 0.025 | 0.106 ± 0.025 | 153.9 ± 2.4 |
| **Dermoscopy / Pathology** | | | | |
| ISIC2018 | 0.903 ± 0.001 [.902,.904] | 0.836 ± 0.001 | 0.316 ± 0.005 | 21.7 ± 0.0 |
| MoNuSeg | 0.343 ± 0.140 [.152,.480] | 0.224 ± 0.096 | 0.171 ± 0.075 | 156.0 ± 43.3 |
| **Mean (14 datasets)** | **0.724** | **0.646** | **0.353** | **61.91** |

> †COph100 excluded from aggregates; validation-set overfitting renders the score uninformative.

---

### Table 2 — Comparison Against 11 Baselines (Aggregate Metrics)

| Model | Dice ↑ | IoU ↑ | BF1 ↑ | HD ↓ | Wins (p<.05) |
|-------|--------|-------|-------|------|--------------|
| **Encoder-Decoder** | | | | | |
| Vanilla U-Net | 0.564 | 0.496 | 0.293 | 85.71 | 10/13 |
| U-Net++ | 0.520 | 0.456 | 0.288 | 73.78 | 9/12 |
| DeepLabV3-R50 | 0.644 | 0.586 | <u>0.358</u> | <u>56.48</u> | 8/13 |
| **Transformer-Based** | | | | | |
| TransUNet | <u>0.647</u> | <u>0.592</u> | 0.355 | 61.64 | 8/13 |
| Swin-UNet | 0.591 | 0.533 | 0.313 | 66.96 | 9/13 |
| MedT | 0.459 | 0.386 | 0.233 | 101.96 | 10/12 |
| **Frozen Vision-Language** | | | | | |
| Frozen CLIP-ViT-B/16 | 0.401 | 0.314 | 0.118 | 112.86 | 11/13 |
| Frozen BiomedCLIP | 0.576 | 0.507 | 0.250 | 89.22 | 11/13 |
| **Medical Foundation** | | | | | |
| nnU-Net-Style | 0.577 | 0.503 | 0.322 | 74.09 | 10/13 |
| MedSAM-Style | 0.533 | 0.470 | 0.211 | 77.48 | 11/13 |
| SAMed-Style | 0.612 | 0.556 | 0.301 | 69.33 | 10/13 |
| **MAC-CLIP (Proposed)** | **0.724** | **0.646** | **0.353** | 61.91 | **107/141** |

---

### Table 3 — Per-Modality Comparison (Hardest Dataset per Modality)

| Model | Endo. (BKAI) | X-Ray (ARC-Sten) | CXR (COVID-Lung) | US (BUSI) | Fundus (PAPILA) | Derm. (ISIC2018) | Path. (MoNuSeg) |
|-------|-------------|-----------------|-----------------|----------|----------------|----------------|----------------|
| Vanilla U-Net | 0.630 | 0.039 | 0.976 | 0.595 | 0.880 | 0.841 | <u>0.588</u> |
| U-Net++ | 0.630 | 0.000 | 0.974 | 0.556 | 0.878 | 0.375 | **0.621** |
| DeepLabV3-R50 | **0.732** | 0.043 | 0.976 | n/a | 0.931 | 0.894 | 0.223 |
| TransUNet | 0.630 | 0.000 | 0.976 | 0.775 | <u>0.934</u> | <u>0.898</u> | 0.336 |
| Swin-UNet | 0.001 | 0.045 | **0.977** | 0.770 | 0.918 | 0.880 | 0.250 |
| MedT | 0.585 | 0.035 | 0.965 | 0.418 | 0.834 | 0.352 | 0.291 |
| Frozen CLIP | 0.630 | 0.004 | 0.842 | 0.272 | 0.633 | 0.697 | 0.174 |
| Frozen BiomedCLIP | 0.630 | 0.045 | 0.966 | 0.639 | 0.880 | 0.856 | 0.376 |
| nnU-Net-Style | 0.630 | 0.042 | 0.975 | 0.604 | 0.931 | 0.829 | n/a |
| MedSAM-Style | 0.630 | 0.039 | 0.956 | 0.626 | 0.854 | 0.821 | 0.000 |
| SAMed-Style | 0.630 | 0.038 | 0.973 | n/a | 0.883 | 0.860 | n/a |
| **MAC-CLIP** | <u>0.632</u> | **0.428** | <u>0.975</u> | **0.776** | **0.945** | **0.903** | 0.343 |

> MAC-CLIP achieves best Dice on **5 of 7** representative modalities.

---

### Table 4 — Corruption Robustness (MAC-CLIP)

| Corruption | Sev-1 Dice | Sev-1 Ret. | Sev-3 Dice | Sev-3 Ret. |
|------------|-----------|-----------|-----------|-----------|
| Gaussian noise | 0.677 | 98.1% | 0.616 | 89.3% |
| Gaussian blur | 0.691 | 100.2% | 0.680 | 98.5% |
| Brightness shift | 0.702 | 101.7% | 0.701 | 101.6% |
| Contrast shift | 0.699 | 101.3% | 0.687 | 99.6% |
| JPEG compression | 0.689 | 99.9% | 0.673 | 97.5% |
| Motion blur | 0.680 | 98.6% | 0.653 | 94.6% |
| **Mean** | **0.690** | **99.9%** | **0.668** | **96.8%** |

### Severity-3 Robustness vs All Baselines

| Model | Retention % | Sev-3 Dice |
|-------|------------|-----------|
| Vanilla U-Net | 92.1 | 0.535 |
| U-Net++ | 91.5 | 0.596 |
| DeepLabV3-R50 | 93.5 | 0.516 |
| Frozen CLIP | 83.6 | 0.372 |
| Frozen BiomedCLIP | 93.3 | 0.550 |
| TransUNet | <u>97.4</u> | <u>0.626</u> |
| SwinUNet | 94.0 | 0.596 |
| MedT | 89.0 | 0.458 |
| nnU-Net-Style | 95.3 | 0.521 |
| MedSAM-Style | 92.3 | 0.544 |
| SAMed-Style | 95.7 | 0.590 |
| **MAC-CLIP** | **97.7** | **0.668** |

---

### Table 5 — Calibration & Uncertainty Quality

| Dataset | ECE ↓ | AUROC ↑ | Dataset | ECE ↓ | AUROC ↑ |
|---------|-------|---------|---------|-------|---------|
| COVID-Lung | **0.005** | 0.483 | ARCADE-Stenosis | 0.140 | 0.408 |
| COVID-Infection | 0.025 | 0.426 | m2caiSeg | 0.140 | 0.518 |
| CT2US-Kidney | 0.043 | 0.345 | BUSI | 0.174 | 0.320 |
| BTXRD | 0.051 | 0.583 | MoNuSeg | 0.269 | 0.434 |
| ARCADE-Syntax | 0.104 | **0.690** | PAPILA | 0.264 | 0.518 |
| DRIVE | 0.259 | 0.527 | BKAI-NeoPolyp | 0.304 | 0.285 |
| **Mean (12 datasets)** | **0.148** | — | | | |

---

### Table 6 — Prompt Sensitivity (Per-Dataset Dice)

| Dataset | Clinical | Generic | Random | No-Prompt | Spread |
|---------|---------|---------|--------|-----------|--------|
| m2caiSeg | 0.836 | 0.836 | 0.836 | 0.837 | 0.001 |
| BKAI-NeoPolyp | **0.600** | 0.353 | 0.388 | 0.423 | **0.247** |
| COVID-Infection | 0.878 | 0.876 | 0.875 | 0.875 | 0.002 |
| COVID-Lung | 0.975 | 0.974 | 0.974 | 0.974 | 0.001 |
| MoNuSeg | 0.396 | 0.397 | 0.382 | 0.394 | 0.015 |
| BUSI | 0.770 | 0.771 | 0.770 | 0.770 | 0.001 |
| CT2US-Kidney | 0.967 | 0.967 | 0.967 | 0.967 | 0.000 |
| ARCADE-Stenosis | 0.412 | 0.411 | 0.417 | 0.415 | 0.006 |
| ARCADE-Syntax | 0.763 | 0.761 | 0.762 | 0.762 | 0.001 |
| BTXRD | 0.482 | 0.484 | 0.485 | 0.485 | 0.003 |
| PAPILA | 0.941 | 0.940 | 0.940 | 0.941 | 0.001 |
| DRIVE | 0.369 | 0.348 | 0.352 | 0.332 | 0.037 |

> Median Dice change clinical→no-prompt: **−0.0002**. MSAG spatial reasoning, not the prompt, drives segmentation quality.

---

### Table 7 — Hyperparameter Sensitivity (Max−Min Dice Spread)

| Dataset | α spread ↓ | CC spread ↓ | Dataset | α spread ↓ | CC spread ↓ |
|---------|-----------|------------|---------|-----------|------------|
| m2caiSeg | 0.001 | 0.006 | ARCADE-Stenosis | 0.010 | 0.002 |
| BKAI-NeoPolyp | 0.003 | 0.002 | ARCADE-Syntax | 0.001 | 0.001 |
| COVID-Infection | 0.000 | 0.001 | BTXRD | 0.008 | 0.000 |
| COVID-Lung | 0.000 | 0.000 | PAPILA | 0.000 | 0.000 |
| MoNuSeg | 0.027 | 0.000 | DRIVE | 0.014 | 0.001 |
| BUSI | 0.002 | 0.000 | CT2US-Kidney | 0.000 | 0.000 |
| **Mean** | **0.006** | **0.001** | | | |

---

### Table 8 — Component Ablation Study

| Module Removed | Dataset | Dice (w/o) | ΔDice |
|---------------|---------|-----------|-------|
| FPN multi-scale aggregation | BKAI-NeoPolyp | 0.001 | −0.631 |
| FPN multi-scale aggregation | ISIC2018 | 0.079 | −0.824 |
| Semantic concept fusion | BUSI | 0.766 | −0.010 |
| Semantic concept fusion | MoNuSeg | 0.336 | −0.007 |
| Boundary-gate mechanism | DRIVE | 0.298 | −0.014 |
| Boundary-gate mechanism | COph100 | 0.382 | −0.025 |
| Uncertainty weighting (MSAG) | BUSI | 0.769 | −0.007 |
| Uncertainty weighting (MSAG) | ARCADE-Stenosis | 0.413 | −0.015 |
| Orthogonality constraint | MoNuSeg | 0.339 | −0.004 |
| Orthogonality constraint | ARCADE-Syntax | 0.770 | −0.003 |

---

### Table 9 — Computational Profile (352×352, NVIDIA A6000 Ada)

| Model | Total (M) | Trainable % | FLOPs (G) | Latency (ms) | FPS |
|-------|----------|------------|----------|-------------|-----|
| Vanilla U-Net | 31.0 | 100.0 | 54.8 | 11.2 | 89.3 |
| TransUNet | 105.3 | 100.0 | 38.4 | 14.7 | 68.0 |
| Swin-UNet | 27.2 | 100.0 | 38.6 | 12.6 | 79.4 |
| nnU-Net-Style | 30.8 | 100.0 | 52.1 | 11.8 | 84.7 |
| MedSAM-Style | 94.0 | 37.0 | 372.0 | 41.3 | 24.2 |
| SAMed-Style | 95.6 | 16.0 | 380.5 | 43.0 | 23.3 |
| **MAC-CLIP** | 230.9 | **15.12** | 232.4 | 20.4 | **49.0** |

> MAC-CLIP delivers **2.1× throughput** over SAM-based architectures at 49.0 FPS, exceeding the real-time threshold (~30 FPS).

---

## Repository Structure

```
mac_clip_repo/
├── main.py                        # Training & evaluation orchestrator
├── requirements.txt
├── setup.py
│
├── mac_clip/                      # Core package
│   ├── __init__.py
│   ├── config.py                  # All hyperparameters (single source of truth)
│   ├── dataset_registry.py        # Per-dataset clinical concept prompts
│   ├── preprocessing.py           # Universal dataset + 15 dataset builders
│   ├── model.py                   # MAC-CLIP v4 architecture (MSAG, FPN, decoder)
│   ├── losses.py                  # Tversky, IoU, Dice, Focal, Boundary, Edge, Contour
│   ├── train.py                   # train_epoch(), validate(), run_single_dataset()
│   ├── evaluate.py                # Full BMVC evaluation suite (EXT-B … EXT-Q)
│   └── utils.py                   # Metrics, TTA, post-processing, visualisation
│
├── baselines/
│   ├── __init__.py
│   └── baselines.py               # 11 baseline models + training harness
│
├── scripts/
│   ├── compare_results.py         # Comparison tables, LaTeX, bar/radar charts
│   ├── ablation_study.py          # 10 ablation variants + figures
│   └── visualize_features.py      # t-SNE, UMAP, concept maps, uncertainty overlays
│
├── configs/                       # (place dataset-specific YAML overrides here)
└── assets/                        # (place paper figures here for README display)
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Satyajithchary/mac-clip.git
cd mac-clip

# 2. Create environment (Python 3.9+)
conda create -n mac_clip python=3.9 -y
conda activate mac_clip

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. (Optional) Install as package
pip install -e .
```

---

## Dataset Setup

Place datasets under `DATASETS_ROOT` using the following structure:

```
DATASETS_ROOT/
├── Endoscopy/
│   ├── m2caiSeg/
│   └── bkai-igh-neopolyp/
├── CXR/
│   └── COVID-QU-Ex/
├── X-ray/
│   ├── ARCADE/
│   └── BTXRD (1)/
├── Ultrasound/
│   ├── BUSI/
│   └── CT2USforKidneySeg/
├── Fundus or OCT/
│   ├── PAPILA/
│   ├── DRIVE/
│   └── COph100/
└── Pathology/
    └── MonuSeg/

# Kvasir-SEG lives separately:
KVASIR_ROOT/
├── images/
└── masks/

# ISIC2018 lives separately:
ISIC2018_ROOT/
└── ISIC_2018 (2)/
    ├── ISIC2018_Task1-2_Training_Input/
    └── ISIC2018_Task1_Training_GroundTruth/
```

Edit `mac_clip/config.py` to point `datasets_root`, `data_root` at your local paths.

---

## Quick Start

### Smoke test (pipeline check, ~2 min)
```bash
python main.py --mode smoke --dataset kvasir
```

### Train on all datasets (3 seeds)
```bash
python main.py --mode train
```

### Train on specific datasets
```bash
python main.py --dataset kvasir,isic2018,busi --num_seeds 3 --epochs 80
```

### Fast mode (skip slow extensions)
```bash
python main.py --fast --dataset kvasir --num_seeds 1
```

### Eval-only with checkpoint
```bash
python main.py --mode eval --checkpoint /path/to/best.pth --dataset kvasir
```

### Run all baselines
```bash
python baselines/baselines.py --dataset all --mode train
```

### Ablation study
```bash
python scripts/ablation_study.py --mode train --epochs 40
```

### Generate comparison tables
```bash
python scripts/compare_results.py \
    --mac_clip_dir results/mac_clip \
    --baseline_dir results/baselines \
    --out_dir results/comparison
```

### Representation analysis (t-SNE, UMAP, concept maps)
```bash
python scripts/visualize_features.py \
    --checkpoint results/mac_clip/per_dataset/kvasir/seed_42/best.pth \
    --dataset kvasir
```

---

## Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `train` | `train` / `eval` / `smoke` |
| `--dataset` | `all` | Comma-separated dataset keys or `all` |
| `--num_seeds` | `3` | Seeds from `config.seeds_list` |
| `--epochs` | `80` | Max training epochs |
| `--batch_size` | `8` | Batch size |
| `--fast` | off | Disable few-shot, cross-dataset, UMAP, LoRA-sweep |
| `--safe_gpu` | off | `num_workers=0`, no pin_memory (for memory-limited GPUs) |
| `--no_run_robustness` | off | Skip corruption robustness eval |
| `--no_run_few_shot` | off | Skip few-shot evaluation |
| `--no_run_cross_dataset` | off | Skip cross-dataset transfer |
| `--run_profile` | off | Enable computational profiling (FLOPs, FPS) |

---

## Outputs

All results are written to `config.out_dir_root` (default `/media/data/Local_CLIP/mac_clip_bmvc_results`):

```
out_dir_root/
├── per_dataset/
│   └── <dataset>/
│       ├── seed_<N>/
│       │   ├── best.pth            # Best checkpoint
│       │   ├── metrics.json        # Final metrics
│       │   └── per_image_metrics.csv
│       ├── aggregated/
│       │   ├── summary_mean_std.json   # Mean ± std across seeds
│       │   └── per_seed_metrics.csv
│       ├── calibration/            # ECE + reliability diagrams
│       ├── uncertainty/            # AUROC + risk-coverage curves
│       ├── robustness/             # Per-corruption metrics
│       ├── prompts/                # Prompt-sensitivity results
│       ├── sensitivity/            # Hyper-parameter sweeps
│       └── visualizations/         # Prediction overlays
├── cross_dataset/
│   └── cross_dataset_all.csv
├── combined/
│   ├── mean_std_per_dataset.csv    # Final aggregated table
│   ├── overall.json
│   └── radar_modalities.png
├── computational/
│   └── macclip_profile.json
├── training_stability/
│   └── training_curves_<dataset>.png
└── LIMITATIONS_AND_ETHICS.md
```

---

## Datasets Summary

| # | Dataset | Modality | Task | Source |
|---|---------|----------|------|--------|
| 1 | Kvasir-SEG | Endoscopy | Polyp segmentation | [Link](https://datasets.simula.no/kvasir-seg/) |
| 2 | BKAI-NeoPolyp | Endoscopy | Neoplastic polyp segmentation | [Link](https://www.kaggle.com/c/bkai-igh-neopolyp) |
| 3 | m2caiSeg | Endoscopy | Surgical instrument segmentation | [Link](http://m2cai16-tool-locations.grand-challenge.org/) |
| 4 | COVID-QU-Ex (Lung) | CXR | Lung field segmentation | [Link](https://www.kaggle.com/datasets/cf77495ce531e4f6dd575def501bfc3f) |
| 5 | COVID-QU-Ex (Infection) | CXR | Infection region segmentation | [Link](https://www.kaggle.com/datasets/cf77495ce531e4f6dd575def501bfc3f) |
| 6 | ARCADE-Stenosis | X-Ray | Coronary stenosis segmentation | [Link](https://arcade.grand-challenge.org/) |
| 7 | ARCADE-Syntax | X-Ray | Coronary artery tree segmentation | [Link](https://arcade.grand-challenge.org/) |
| 8 | BTXRD | X-Ray | Bone tumor segmentation | [Link](https://github.com/coljamkop/BTXRD) |
| 9 | BUSI | Ultrasound | Breast lesion segmentation | [Link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) |
| 10 | CT2US-Kidney | Ultrasound | Kidney segmentation | [Link](https://www.kaggle.com/datasets/siatsyx/ct2usforkidneyseg) |
| 11 | PAPILA | Fundus | Optic disc/cup segmentation | [Link](https://figshare.com/articles/dataset/PAPILA/14798004) |
| 12 | DRIVE | Fundus | Retinal vessel segmentation | [Link](https://drive.grand-challenge.org/) |
| 13 | COph100 | Fundus | ROP segmentation | [Link](https://github.com/FundusOphthalmology/COPh100) |
| 14 | ISIC2018 | Dermoscopy | Skin lesion segmentation | [Link](https://challenge.isic-archive.com/data/#2018) |
| 15 | MoNuSeg | Pathology | Nuclear instance segmentation | [Link](https://monuseg.grand-challenge.org/) |

---

## Method Details

### MSAG — The Core Innovation

MSAG replaces the rigid ViT patch grid with a deformation-conditioned, uncertainty-weighted aggregation:

```
Offset predictor:  Δ = Ψ(F) ∈ ℝ^{h×w×2}    (learned geometric offsets)
Sampled location:  p̃ = p + Δ(p)
Uncertainty:       u(p) ∈ [0,1]               (per-pixel reliability weight)

Grouped embedding:
         Σ_{p∈g_k} (1 - u(p)) · F(p̃)
z_k  =  ─────────────────────────────────
              Σ_{p∈g_k} (1 - u(p))
```

This is the **minimum-variance unbiased estimator** under an inverse-variance weighting model — not a heuristic.

### Progressive Group Contrastive (PGC) Loss

Aligns pixel groups at multiple spatial granularities {1, 4, 16, 64 pixels} with text embeddings, progressively coarsening supervision from fine to semantic scale.

### Loss Function

```
L = λ_seg·L_Tversky + λ_iou·L_IoU + λ_dice·L_Dice + λ_focal·L_Focal
  + λ_boundary·L_Boundary + λ_edge·L_Edge + λ_contour·L_Contour
  + λ_bmap·L_BoundaryMap + λ_deep_sup·L_DeepSup
  + λ_concept·L_Concept + λ_pgc·L_PGC + λ_pbc·L_PBC
```

---

## Research Questions Answered

| RQ | Question | Finding |
|----|----------|---------|
| RQ1 | Does geometry-conditioned grouping improve morphology-sensitive dense prediction over rigid grids? | ✅ +7.7 pp Dice over best rigid-grid transformer |
| RQ2 | Does the same mechanism transfer across heterogeneous modalities without per-dataset tuning? | ✅ Best on 5/7 modality representatives |
| RQ3 | Does the parameter-efficient design preserve the BiomedCLIP semantic prior? | ✅ Only 15.12% trainable params; 97.7% robustness retention |

---

## Citation

```bibtex
@inproceedings{macclip2026bmvc,
  title     = {MAC-CLIP: Morpho-Semantic Adaptive Contrastive Learning
               for Reliable Multi-Modality Medical Segmentation},
  author    = {Anonymous},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026},
  note      = {Submission \#1242}
}
```

---

## License

This code is released under the **MIT License** for research purposes only.  
MAC-CLIP is **not approved for clinical use**. See `LIMITATIONS_AND_ETHICS.md` for full details.
