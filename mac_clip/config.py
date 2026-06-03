"""
MAC-CLIP Configuration
======================
Central hyperparameter and path configuration for MAC-CLIP.
All other modules import from here; changing a value here propagates everywhere.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MACCLIPConfig:
    # ── Data paths ────────────────────────────────────────────────────────────
    datasets_root:  str = "/media/data/Segmentation_Datasets"
    checkpoint_dir: str = "/media/data/Local_CLIP/checkpoints/mac_clip_v4_multi"
    results_dir:    str = "/media/data/Local_CLIP/mac_clip_v4_multi_results"
    out_dir_root:   str = "/media/data/Local_CLIP/mac_clip_bmvc_results"

    # Legacy Kvasir-SEG path (kept for backward compatibility)
    data_root:  str = "/media/data/Local_CLIP/kvasir-seg/Kvasir-SEG/"
    images_dir: str = "images"
    masks_dir:  str = "masks"

    # ── Backbone ──────────────────────────────────────────────────────────────
    clip_model:      str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    clip_pretrained: str = ""
    image_size:      int = 352
    feature_dim:     int = 512

    # ── LoRA adapter ─────────────────────────────────────────────────────────
    lora_r:       int   = 16
    lora_alpha:   int   = 32
    lora_dropout: float = 0.1
    lora_layers:  List[int] = field(default_factory=lambda: [6, 8, 10, 12])

    # ── Morphological / MSAG module ───────────────────────────────────────────
    max_deformation:       float = 2.0
    uncertainty_threshold: float = 0.7

    # ── Clinical concept prompts (endoscopy default) ──────────────────────────
    clinical_concepts: List[str] = field(default_factory=lambda: [
        "irregular border lesion colonoscopy",
        "smooth rounded lesion colonoscopy",
        "granular mucosal surface endoscopy",
        "flat sessile lesion mucosa",
        "erythematous lesion colon mucosa",
        "pale lesion colon background",
        "small polyp focal lesion",
        "large polyp broad base",
    ])
    boundary_concept: str = "boundary edge lesion margin colonoscopy"
    interior_concept: str = "interior region polyp tissue colonoscopy"

    num_concepts: int = 8
    concept_dim:  int = 512

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:    int   = 8
    num_epochs:    int   = 80
    learning_rate: float = 2e-4
    weight_decay:  float = 1e-4

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda_seg:      float = 1.0
    lambda_iou:      float = 1.0
    lambda_dice:     float = 0.5
    lambda_focal:    float = 1.0
    lambda_boundary: float = 3.0
    lambda_edge:     float = 1.5
    lambda_contour:  float = 2.0
    lambda_bmap:     float = 3.0
    lambda_deep_sup: float = 0.4
    lambda_concept:  float = 0.2
    lambda_pgc:      float = 0.5
    lambda_pbc:      float = 0.3

    boundary_gate_alpha: float = 0.4
    tversky_alpha:       float = 0.3
    tversky_beta:        float = 0.7
    focal_gamma:         float = 2.0
    focal_alpha:         float = 0.75

    # Progressive Group Contrastive
    pgc_gamma:             float      = 0.5
    pgc_temperature:       float      = 0.07
    pgc_group_sizes:       List[int]  = field(default_factory=lambda: [1, 4, 16, 64])
    pgc_samples_per_size:  int        = 4

    # ── Post-processing ───────────────────────────────────────────────────────
    use_connected_components: bool  = True
    min_component_ratio:      float = 0.1

    # ── Evaluation ───────────────────────────────────────────────────────────
    n_visualizations: int  = 30
    seed:             int  = 42
    num_workers:      int  = 8
    mixed_precision:  bool = True
    use_tta:          bool = True

    # ── BMVC/CVPR-grade validation ────────────────────────────────────────────
    seeds_list:     List[int] = field(default_factory=lambda: [42, 3407, 1234, 999, 2026])
    bootstrap_iters: int      = 2000
    bootstrap_ci:    float    = 0.95

    cross_dataset_pairs: Dict[str, List[str]] = field(default_factory=lambda: {
        "kvasir":     ["bkai_neopolyp", "m2caiseg"],
        "drive":      ["coph100", "papila"],
        "busi":       ["ct2us_kidney"],
        "isic2018":   ["monuseg"],
        "covid_lung": ["covid_infection", "btxrd"],
    })

    corruption_severities: List[int] = field(default_factory=lambda: [1, 3])
    corruption_types:      List[str] = field(default_factory=lambda: [
        "gaussian_noise", "gaussian_blur", "brightness", "contrast",
        "jpeg_compression", "motion_blur",
    ])

    few_shot_fractions: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.20])
    few_shot_epochs:    int         = 20

    prompt_variants: List[str] = field(default_factory=lambda: [
        "clinical", "generic", "random", "no_prompt"
    ])

    # Hyper-parameter sensitivity sweeps
    sweep_lora_ranks:      List[int]   = field(default_factory=lambda: [4, 8, 16, 32])
    sweep_lambda_boundary: List[float] = field(default_factory=lambda: [0.5, 1.0, 3.0, 6.0])
    sweep_lambda_pgc:      List[float] = field(default_factory=lambda: [0.0, 0.2, 0.5, 1.0])
    sweep_group_sizes:     List        = field(default_factory=lambda: [
        [1, 4], [1, 4, 16], [1, 4, 16, 64]
    ])
    sweep_gate_alpha: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 1.0])
    sweep_cc_ratio:   List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])

    # Calibration
    ece_n_bins: int = 15

    # Representation analysis
    repr_max_pixels: int = 4000

    # Early stopping
    early_stop_patience:      int   = 25
    early_stop_min_delta:     float = 1e-4
    early_stop_warmup_epochs: int   = 30

    # Safe GPU mode (disables persistent workers + pinned memory)
    safe_gpu: bool = False


# Singleton — import this throughout the codebase
config = MACCLIPConfig()
