"""
scripts/visualize_features.py
==============================
Representation-analysis utilities for MAC-CLIP:

  1. t-SNE / UMAP of pixel embeddings (foreground vs background)
  2. Concept-embedding orthogonality heatmap
  3. Attention / concept-map overlays
  4. Feature distribution shift (source vs target domain)

Usage
-----
  python scripts/visualize_features.py \
      --checkpoint /path/to/best.pth \
      --dataset kvasir \
      --out_dir results/representation
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not found — t-SNE / UMAP disabled.")

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_clip.config       import MACCLIPConfig
from mac_clip.model        import MACCLIPv4
from mac_clip.preprocessing import get_all_datasets
from mac_clip.utils        import denormalize


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_pixel_features(
    model: MACCLIPv4,
    loader,
    max_pixels: int = 4000,
    device: torch.device = torch.device("cuda"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (features [N,D], labels [N])  where label=1=foreground, 0=background."""
    model.eval()
    feats_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []
    n_collected = 0

    for batch in loader:
        if n_collected >= max_pixels:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        gf, ms = model.encode_image_multiscale(batch["image"])
        fp, _  = model.fpn(ms)
        fused  = model.feat_proj(fp)                    # B, D, H, W
        deform, _ = model.morphological_module(fused)   # B, D, H, W

        B, D, H, W = deform.shape
        masks_dn = F.interpolate(
            batch["mask"].unsqueeze(1).float(), (H, W), mode="nearest").squeeze(1)

        for i in range(B):
            if n_collected >= max_pixels:
                break
            pf = F.normalize(deform[i], dim=0).permute(1, 2, 0).reshape(-1, D).cpu().numpy()
            lb = masks_dn[i].reshape(-1).cpu().numpy() > 0.5
            per_sample = min(max_pixels // max(len(loader) * B, 1) + 1, len(pf))
            fg_idx = np.where(lb)[0];  bg_idx = np.where(~lb)[0]
            n_fg   = min(per_sample // 2, len(fg_idx))
            n_bg   = min(per_sample // 2, len(bg_idx))
            if n_fg == 0 or n_bg == 0:
                continue
            rng    = np.random.default_rng(42 + n_collected)
            s_fg   = rng.choice(fg_idx, n_fg, replace=False)
            s_bg   = rng.choice(bg_idx, n_bg, replace=False)
            idx    = np.concatenate([s_fg, s_bg])
            feats_list.append(pf[idx])
            label_list.append(np.where(lb[idx], 1, 0))
            n_collected += len(idx)

    if not feats_list:
        return np.zeros((0, 512)), np.zeros(0)
    return np.concatenate(feats_list), np.concatenate(label_list)


# ---------------------------------------------------------------------------
# t-SNE / UMAP
# ---------------------------------------------------------------------------

def plot_tsne(feats: np.ndarray, labels: np.ndarray,
              save_path: Path, title: str = "t-SNE pixel features") -> None:
    if not HAS_SKLEARN or len(feats) == 0:
        print("  [skip] t-SNE — scikit-learn not available or empty features"); return

    print(f"  Running t-SNE on {len(feats)} points…")
    pca   = PCA(n_components=min(50, feats.shape[1]), random_state=42)
    feats_pca = pca.fit_transform(feats)
    emb   = TSNE(n_components=2, perplexity=30, n_iter=1000,
                 random_state=42, n_jobs=-1).fit_transform(feats_pca)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, colour, name in [(0, "#3498db", "Background"), (1, "#e74c3c", "Foreground")]:
        idx = labels == label
        ax.scatter(emb[idx, 0], emb[idx, 1], c=colour, s=4, alpha=0.5,
                   label=f"{name} (n={idx.sum()})", rasterized=True)
    ax.set_title(title); ax.legend(markerscale=3); ax.axis("off")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_umap(feats: np.ndarray, labels: np.ndarray,
              save_path: Path, title: str = "UMAP pixel features") -> None:
    if not HAS_UMAP or len(feats) == 0:
        print("  [skip] UMAP — umap-learn not available"); return

    print(f"  Running UMAP on {len(feats)} points…")
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15,
                             min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, colour, name in [(0, "#3498db", "Background"), (1, "#e74c3c", "Foreground")]:
        idx = labels == label
        ax.scatter(emb[idx, 0], emb[idx, 1], c=colour, s=4, alpha=0.5,
                   label=f"{name} (n={idx.sum()})", rasterized=True)
    ax.set_title(title); ax.legend(markerscale=3); ax.axis("off")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ Saved: {save_path}")


# ---------------------------------------------------------------------------
# Concept orthogonality
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_concept_orthogonality(
    model: MACCLIPv4,
    dataset_name: str,
    save_path: Path,
    device: torch.device = torch.device("cuda"),
) -> None:
    model.eval()
    concept_feats = model.encode_text_concepts(device)   # 1, K, D
    K = concept_feats.shape[1]
    cf = F.normalize(concept_feats[0], dim=-1).cpu().numpy()  # K, D
    sim = cf @ cf.T                                           # K, K

    labels = [c[:22] for c in model.config.clinical_concepts[:K]]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title(f"Concept embedding cosine similarity — {dataset_name}")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=5, color="black" if abs(sim[i,j]) < 0.5 else "white")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ Saved: {save_path}")

    mean_off_diag = float((np.abs(sim) - np.eye(K)).sum() / max(K * (K - 1), 1))
    print(f"  Mean |off-diag cosine sim| = {mean_off_diag:.4f}  "
          f"(lower → more orthogonal)")


# ---------------------------------------------------------------------------
# Concept map overlays
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_concept_maps(
    model: MACCLIPv4,
    loader,
    dataset_name: str,
    save_dir: Path,
    n_samples: int = 6,
    device: torch.device = torch.device("cuda"),
) -> None:
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    n_concepts = model.config.num_concepts
    concept_labels = [c[:22] for c in model.config.clinical_concepts[:n_concepts]]
    count = 0

    for batch in loader:
        if count >= n_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch, compute_loss=False)
        cmaps = out.get("concept_maps")   # B, K, H, W
        sp    = out["seg_preds"]

        for i in range(batch["image"].shape[0]):
            if count >= n_samples:
                break
            img  = denormalize(batch["image"][i])
            gt   = batch["mask"][i].cpu().numpy()
            pred = sp[i].cpu().numpy()

            n_cols = n_concepts + 3
            fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(4 * ((n_cols + 1) // 2), 8))
            axes = axes.flatten()

            axes[0].imshow(img);              axes[0].set_title("Image")
            axes[1].imshow(gt,   cmap="gray"); axes[1].set_title("GT")
            axes[2].imshow(pred, cmap="jet", vmin=0, vmax=1); axes[2].set_title("Prediction")

            if cmaps is not None:
                for k in range(min(n_concepts, len(axes) - 3)):
                    cmap_k = cmaps[i, k].cpu().numpy()
                    axes[k + 3].imshow(cmap_k, cmap="hot", vmin=-1, vmax=1)
                    axes[k + 3].set_title(concept_labels[k] if k < len(concept_labels) else f"C{k}",
                                          fontsize=7)

            for ax in axes:
                ax.axis("off")

            plt.suptitle(f"Concept maps — {dataset_name} — sample {count}", fontsize=10)
            plt.tight_layout()
            plt.savefig(save_dir / f"concept_maps_{count:03d}.png",
                        dpi=120, bbox_inches="tight")
            plt.close()
            count += 1

    print(f"  ✓ Saved {count} concept-map figures to {save_dir}")


# ---------------------------------------------------------------------------
# Uncertainty overlay
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_uncertainty(
    model: MACCLIPv4,
    loader,
    dataset_name: str,
    save_dir: Path,
    n_samples: int = 6,
    device: torch.device = torch.device("cuda"),
) -> None:
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for batch in loader:
        if count >= n_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch, compute_loss=False)
        sp  = out["seg_preds"]
        unc = out.get("uncertainty")      # B, H, W  or  None

        for i in range(batch["image"].shape[0]):
            if count >= n_samples:
                break
            img  = denormalize(batch["image"][i])
            pred = sp[i].cpu().numpy()
            gt   = batch["mask"][i].cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img);                    axes[0].set_title("Image")
            axes[1].imshow(gt,   cmap="gray");      axes[1].set_title("GT")
            axes[2].imshow(pred, cmap="jet", vmin=0, vmax=1); axes[2].set_title("Prediction")
            if unc is not None:
                u = unc[i].cpu().numpy()
                u_up = torch.nn.functional.interpolate(
                    torch.tensor(u[None, None]), pred.shape, mode="bilinear",
                    align_corners=False).squeeze().numpy()
                axes[3].imshow(u_up, cmap="hot", vmin=0, vmax=1)
                axes[3].set_title("Uncertainty")
            else:
                axes[3].axis("off")

            for ax in axes:
                ax.axis("off")
            plt.suptitle(f"Uncertainty — {dataset_name} — sample {count}", fontsize=10)
            plt.tight_layout()
            plt.savefig(save_dir / f"uncertainty_{count:03d}.png", dpi=120, bbox_inches="tight")
            plt.close()
            count += 1

    print(f"  ✓ Saved {count} uncertainty figures to {save_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAC-CLIP representation analysis")
    parser.add_argument("--checkpoint",    default=None,   help="Path to best.pth checkpoint")
    parser.add_argument("--dataset",       default="kvasir")
    parser.add_argument("--datasets_root", default="/media/data/Segmentation_Datasets")
    parser.add_argument("--kvasir_root",   default="/media/data/Local_CLIP/kvasir-seg/Kvasir-SEG/")
    parser.add_argument("--out_dir",       default="results/representation")
    parser.add_argument("--n_samples",     type=int, default=6)
    parser.add_argument("--max_pixels",    type=int, default=4000)
    parser.add_argument("--no_tsne",       action="store_true")
    parser.add_argument("--no_umap",       action="store_true")
    argv = [] if hasattr(sys, "ps1") or "ipykernel" in sys.modules else None
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = MACCLIPConfig()
    cfg.data_root     = args.kvasir_root
    cfg.datasets_root = args.datasets_root

    model = MACCLIPv4(cfg).to(device)
    model.set_dataset_concepts(args.dataset)

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  ✓ Loaded checkpoint: {args.checkpoint}")
    else:
        print("  [WARN] No checkpoint loaded — using random weights")

    # Load dataset
    builders = get_all_datasets(args.datasets_root, cfg)
    if args.dataset not in builders:
        print(f"  [ERROR] Unknown dataset: {args.dataset}"); return
    pair = builders[args.dataset]()
    if pair is None:
        print(f"  [ERROR] Dataset not found: {args.dataset}"); return
    _, val_ds = pair
    vl = DataLoader(val_ds, batch_size=min(cfg.batch_size, len(val_ds)),
                    shuffle=False, num_workers=4)

    # 1) t-SNE / UMAP
    if not args.no_tsne or not args.no_umap:
        print("\n[1] Extracting pixel features…")
        feats, labels = extract_pixel_features(model, vl, args.max_pixels, device)
        if len(feats) > 0:
            if not args.no_tsne:
                plot_tsne(feats, labels, out_dir / f"tsne_{args.dataset}.png",
                          title=f"t-SNE pixel features — {args.dataset}")
            if not args.no_umap:
                plot_umap(feats, labels, out_dir / f"umap_{args.dataset}.png",
                          title=f"UMAP pixel features — {args.dataset}")

    # 2) Concept orthogonality
    print("\n[2] Concept orthogonality…")
    plot_concept_orthogonality(model, args.dataset,
                                out_dir / f"concept_ortho_{args.dataset}.png", device)

    # 3) Concept maps
    print("\n[3] Concept map overlays…")
    visualize_concept_maps(model, vl, args.dataset,
                           out_dir / f"concept_maps_{args.dataset}",
                           n_samples=args.n_samples, device=device)

    # 4) Uncertainty
    print("\n[4] Uncertainty overlays…")
    visualize_uncertainty(model, vl, args.dataset,
                          out_dir / f"uncertainty_{args.dataset}",
                          n_samples=args.n_samples, device=device)

    print(f"\n✓ All representation figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
