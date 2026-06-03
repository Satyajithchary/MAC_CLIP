"""
MAC-CLIP: Morpho-Semantic Adaptive Contrastive Learning
for Reliable Multi-Modality Medical Segmentation
=========================================================
BMVC 2026 — Submission #1242

Public API
----------
>>> from mac_clip import MACCLIPv4, MACCLIPConfig
>>> config = MACCLIPConfig()
>>> model  = MACCLIPv4(config).cuda()
"""

from mac_clip.config          import MACCLIPConfig, config
from mac_clip.model           import MACCLIPv4
from mac_clip.dataset_registry import DATASET_CONCEPTS, MODALITY_MAP
from mac_clip.preprocessing   import UniversalSegDataset, get_all_datasets
from mac_clip.utils           import set_seed, compute_metrics

__all__ = [
    "MACCLIPConfig",
    "MACCLIPv4",
    "config",
    "DATASET_CONCEPTS",
    "MODALITY_MAP",
    "UniversalSegDataset",
    "get_all_datasets",
    "set_seed",
    "compute_metrics",
]
