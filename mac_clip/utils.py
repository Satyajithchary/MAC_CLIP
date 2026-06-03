"""
Utilities
=========
Metric computation, Test-Time Augmentation (TTA), post-processing,
seed management, and visualisation helpers.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion, distance_transform_edt

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random-number generators for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Batch-level Dice, IoU, precision, recall (fast, GPU-friendly)."""
    pb = (preds > threshold).float()
    t  = targets.float()
    i  = (pb * t).sum()
    u  = pb.sum() + t.sum()
    dice = (2 * i + 1e-7) / (u + 1e-7)
    iou  = (i + 1e-7) / (u - i + 1e-7)
    tp   = (pb * t).sum()
    fp   = (pb * (1 - t)).sum()
    fn   = ((1 - pb) * t).sum()
    prec = (tp + 1e-7) / (tp + fp + 1e-7)
    rec  = (tp + 1e-7) / (tp + fn + 1e-7)
    return {
        "dice": dice.item(), "iou": iou.item(),
        "precision": prec.item(), "recall": rec.item(),
    }


def compute_boundary_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold_dist: int = 2,
) -> float:
    """Boundary F1 at 2-pixel tolerance."""
    pb  = cv2.Canny((pred * 255).astype(np.uint8), 50, 150) > 0
    gb  = cv2.Canny((gt   * 255).astype(np.uint8), 50, 150) > 0
    k   = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (threshold_dist * 2 + 1, threshold_dist * 2 + 1))
    gbd = cv2.dilate(gb.astype(np.uint8), k) > 0
    tp  = np.sum(pb & gbd)
    fp  = np.sum(pb & ~gbd)
    fn  = np.sum(~pb & gb)
    pr  = tp / (tp + fp + 1e-7)
    rc  = tp / (tp + fn + 1e-7)
    return 2 * pr * rc / (pr + rc + 1e-7)


def compute_hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Hausdorff distance in pixels."""
    try:
        pp = np.argwhere(pred > 0.5)
        gp = np.argwhere(gt   > 0.5)
        if len(pp) == 0 or len(gp) == 0:
            return 100.0
        return max(directed_hausdorff(pp, gp)[0], directed_hausdorff(gp, pp)[0])
    except Exception:
        return 100.0


def compute_assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Symmetric Surface Distance (pixels). Lower is better."""
    p = (pred > 0.5).astype(np.uint8)
    t = (gt   > 0.5).astype(np.uint8)
    if p.sum() == 0 or t.sum() == 0:
        return 100.0
    pb = p ^ binary_erosion(p)
    tb = t ^ binary_erosion(t)
    if pb.sum() == 0 or tb.sum() == 0:
        return 100.0
    dt_t = distance_transform_edt(~tb)
    dt_p = distance_transform_edt(~pb)
    return float(0.5 * (dt_t[pb.astype(bool)].mean() + dt_p[tb.astype(bool)].mean()))


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def connected_component_postprocess(
    pred_np: np.ndarray,
    min_ratio: float = 0.1,
) -> np.ndarray:
    """Keep only connected components whose area ≥ min_ratio × max_area."""
    binary = (pred_np > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return pred_np
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return pred_np
    areas    = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()
    valid    = np.zeros_like(binary, dtype=np.float32)
    for i, area in enumerate(areas, start=1):
        if area >= max_area * min_ratio:
            valid[labels == i] = 1.0
    return pred_np * valid


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

@torch.no_grad()
def tta_predict(model: torch.nn.Module, batch: Dict, config) -> torch.Tensor:
    """Average predictions over four flip augmentations."""
    images = batch["image"]
    masks  = batch["mask"]

    def _run(b: Dict) -> torch.Tensor:
        return model(b, compute_loss=False)["seg_preds"]

    preds = [_run(batch)]
    bh = {**batch, "image": torch.flip(images, [3]), "mask": torch.flip(masks, [2])}
    preds.append(torch.flip(_run(bh), [2]))
    bv = {**batch, "image": torch.flip(images, [2]), "mask": torch.flip(masks, [1])}
    preds.append(torch.flip(_run(bv), [1]))
    bhv = {**batch, "image": torch.flip(images, [2, 3]), "mask": torch.flip(masks, [1, 2])}
    preds.append(torch.flip(_run(bhv), [1, 2]))

    return torch.stack(preds).mean(0)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

_VIS_MEAN = np.array([0.481, 0.457, 0.408])
_VIS_STD  = np.array([0.268, 0.261, 0.275])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo CLIP normalisation; return H×W×3 float32 in [0,1]."""
    return (tensor.cpu().permute(1, 2, 0).numpy() * _VIS_STD + _VIS_MEAN).clip(0, 1)


def visualize_predictions(
    model: torch.nn.Module,
    loader,
    save_dir: str,
    n: int = 30,
    config=None,
    use_tta: bool = False,
    dataset_name: str = "dataset",
) -> None:
    """Save side-by-side prediction figures for up to *n* validation samples."""
    from mac_clip.dataset_registry import DATASET_CONCEPTS

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = 0
    concepts = DATASET_CONCEPTS.get(dataset_name, DATASET_CONCEPTS["kvasir"])["clinical_concepts"]

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out  = model(batch, compute_loss=False)
            sp   = tta_predict(model, batch, config) if use_tta else out["seg_preds"]
            bndp = out.get("bnd_preds")

            for i in range(sp.shape[0]):
                if count >= n:
                    return
                img  = denormalize(batch["image"][i])
                gt   = batch["mask"][i].cpu().numpy()
                pred = sp[i].cpu().numpy()
                if config and config.use_connected_components:
                    pred = connected_component_postprocess(pred, config.min_component_ratio)

                m = compute_metrics(torch.tensor(pred), torch.tensor(gt))
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                fig.suptitle(
                    f"[{dataset_name}] Sample {count+1} | "
                    f"Dice:{m['dice']:.4f}  IoU:{m['iou']:.4f}",
                    fontsize=12, fontweight="bold")

                axes[0, 0].imshow(img);              axes[0, 0].set_title("Original")
                axes[0, 1].imshow(gt, cmap="gray");  axes[0, 1].set_title("GT")
                axes[0, 2].imshow(pred, cmap="jet", vmin=0, vmax=1);   axes[0, 2].set_title("Raw Pred")
                axes[0, 3].imshow(pred, cmap="jet", vmin=0, vmax=1);   axes[0, 3].set_title("Post-proc")
                if bndp is not None:
                    axes[0, 4].imshow(bndp[i].cpu().numpy(), cmap="hot", vmin=0, vmax=1)
                    axes[0, 4].set_title("Boundary")
                else:
                    axes[0, 4].axis("off")

                cmaps = out.get("concept_maps")
                for j in range(3):
                    if cmaps is not None and j < cmaps.shape[1]:
                        axes[1, j].imshow(cmaps[i, j].cpu().numpy(), cmap="hot")
                        axes[1, j].set_title(concepts[j][:22])
                    axes[1, j].axis("off")

                overlay       = np.zeros((*pred.shape, 4))
                overlay[..., 1] = 1.0; overlay[..., 3] = pred * 0.5
                axes[1, 3].imshow(img); axes[1, 3].imshow(overlay); axes[1, 3].set_title("Overlay")
                axes[1, 4].imshow(np.abs(pred - gt), cmap="hot", vmin=0, vmax=1)
                axes[1, 4].set_title("Error Map")

                for ax in axes.flat:
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(save_dir / f"vis_{count:03d}.png", bbox_inches="tight", dpi=150)
                plt.close()
                count += 1

    print(f"  ✓ Saved {count} visualisations to {save_dir}")
