"""
scripts/ablation_study.py
==========================
Trains and evaluates ablation variants of MAC-CLIP on Kvasir-SEG
(and optionally other datasets) to isolate the contribution of each
architectural component:

  A1  No LoRA            (frozen backbone, no LoRA adapters)
  A2  No MSAG            (skip deformable morphological module)
  A3  No concept module  (skip semantic concept scoring)
  A4  No PGC loss        (λ_pgc = 0)
  A5  No PBC loss        (λ_pbc = 0)
  A6  No boundary gate   (α = 0 in decoder gate)
  A7  No deep supervision (λ_deep_sup = 0)
  A8  No TTA             (test without flip augmentation)
  A9  No CC post-proc    (skip connected-component filtering)
  A10 Full MAC-CLIP      (all components enabled)

Outputs
-------
  results/ablation/ablation_table.csv
  results/ablation/ablation_latex.tex
  results/ablation/ablation_bar.png
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_clip.config       import MACCLIPConfig
from mac_clip.model        import MACCLIPv4
from mac_clip.preprocessing import build_kvasir_dataset
from mac_clip.train        import train_epoch, validate
from mac_clip.utils        import set_seed


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

ABLATION_VARIANTS: List[Dict] = [
    {
        "name":  "A1 – No LoRA",
        "key":   "no_lora",
        "patch": {"lora_layers": []},
    },
    {
        "name":  "A2 – No MSAG",
        "key":   "no_msag",
        "patch": {},
        "model_patch": lambda m: setattr(
            m, "morphological_module",
            _IdentityMorpho(m.config.feature_dim)),
    },
    {
        "name":  "A3 – No Concept Module",
        "key":   "no_concept",
        "patch": {"lambda_concept": 0.0},
        "model_patch": lambda m: setattr(
            m, "semantic_module",
            _ZeroConceptModule()),
    },
    {
        "name":  "A4 – No PGC Loss",
        "key":   "no_pgc",
        "patch": {"lambda_pgc": 0.0},
    },
    {
        "name":  "A5 – No PBC Loss",
        "key":   "no_pbc",
        "patch": {"lambda_pbc": 0.0},
    },
    {
        "name":  "A6 – No Boundary Gate",
        "key":   "no_bgate",
        "patch": {"boundary_gate_alpha": 0.0},
        "post_init": lambda m: m.decoder.gate_alpha.data.fill_(0.0),
    },
    {
        "name":  "A7 – No Deep Supervision",
        "key":   "no_deepsup",
        "patch": {"lambda_deep_sup": 0.0},
    },
    {
        "name":  "A8 – No TTA",
        "key":   "no_tta",
        "patch": {"use_tta": False},
    },
    {
        "name":  "A9 – No CC Post-proc",
        "key":   "no_cc",
        "patch": {"use_connected_components": False},
    },
    {
        "name":  "A10 – Full MAC-CLIP",
        "key":   "full",
        "patch": {},
    },
]


# ---------------------------------------------------------------------------
# Stub modules for ablations
# ---------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


class _IdentityMorpho(nn.Module):
    """Replaces MSAG with an identity pass (no deformation, uniform uncertainty)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, features, seg_logits=None):
        ones = torch.ones(
            features.shape[0], features.shape[2], features.shape[3],
            device=features.device, dtype=features.dtype)
        return features, ones


class _ZeroConceptModule(nn.Module):
    """Replaces SemanticModule — returns zero concept maps + zero seg contribution."""

    def forward(self, features, text_embeddings):
        B, D, H, W = features.shape
        n = text_embeddings.shape[1]
        maps = torch.zeros(B, n, H, W, device=features.device, dtype=features.dtype)
        seg  = torch.zeros(B, 1, H, W, device=features.device, dtype=features.dtype)
        return maps, seg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patched_config(base_cfg: MACCLIPConfig, patch: Dict) -> MACCLIPConfig:
    cfg = copy.deepcopy(base_cfg)
    for k, v in patch.items():
        setattr(cfg, k, v)
    return cfg


def run_ablation_variant(
    variant: Dict,
    base_config: MACCLIPConfig,
    train_ds,
    val_ds,
    num_epochs: int,
    smoke: bool,
    seed: int,
) -> Dict:
    set_seed(seed)
    cfg   = _patched_config(base_config, variant.get("patch", {}))
    model = MACCLIPv4(cfg).cuda()
    model.set_dataset_concepts("kvasir")

    if "model_patch" in variant:
        variant["model_patch"](model)
    if "post_init" in variant:
        variant["post_init"](model)

    eff_bs = min(cfg.batch_size, len(train_ds))
    tl = DataLoader(train_ds, batch_size=eff_bs, shuffle=True,
                    num_workers=0 if smoke else 4, drop_last=False)
    vl = DataLoader(val_ds,   batch_size=eff_bs, shuffle=False,
                    num_workers=0 if smoke else 4)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7)
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    n_ep        = 1 if smoke else num_epochs
    max_batches = 2 if smoke else None
    best_dice   = 0.0
    best_state  = None

    for ep in range(n_ep):
        train_epoch(model, tl, optimizer, scaler, ep, cfg, max_batches=max_batches)
        v = validate(model, vl, cfg, use_tta=False)
        scheduler.step()
        if v["dice"] > best_dice:
            best_dice  = v["dice"]
            best_state = {k: val.clone() for k, val in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    use_tta = cfg.use_tta and not smoke
    final   = validate(model, vl, cfg, compute_all_metrics=True, use_tta=use_tta)
    final.update({"variant": variant["name"], "key": variant["key"]})
    print(f"  [{variant['key']:<14}] Dice={final['dice']:.4f}  "
          f"IoU={final['iou']:.4f}  BF1={final.get('boundary_f1',0):.4f}")

    del model; torch.cuda.empty_cache()
    return final


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_ablation_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["variant", "key", "dice", "iou", "boundary_f1", "hausdorff"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  ✓ Saved: {path}")


def save_ablation_latex(rows: List[Dict], path: Path) -> None:
    full_row = next((r for r in rows if r["key"] == "full"), None)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Ablation study on Kvasir-SEG. "
        "Each row removes one component from the full MAC-CLIP model.}",
        "  \\label{tab:ablation}",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    \\textbf{Variant} & \\textbf{Dice} & \\textbf{IoU} "
        "& \\textbf{BF1} & \\textbf{HD} \\\\",
        "    \\midrule",
    ]
    for r in rows:
        dice = float(r.get("dice", 0) or 0)
        iou  = float(r.get("iou",  0) or 0)
        bf1  = float(r.get("boundary_f1", 0) or 0)
        hd   = float(r.get("hausdorff",   0) or 0)
        is_full = (r["key"] == "full")
        d_str = f"\\textbf{{{dice:.4f}}}" if is_full else f"{dice:.4f}"
        lines.append(
            f"    {r['variant']} & {d_str} & {iou:.4f} & {bf1:.4f} & {hd:.2f} \\\\"
        )
        if is_full:
            lines.insert(-1, "    \\midrule")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  ✓ Saved: {path}")


def plot_ablation_bar(rows: List[Dict], path: Path) -> None:
    names  = [r["variant"].split("–")[-1].strip() for r in rows]
    dices  = [float(r.get("dice", 0) or 0)  for r in rows]
    ious   = [float(r.get("iou",  0) or 0)  for r in rows]
    colours = ["#2ecc71" if r["key"] == "full" else "#3498db" for r in rows]

    x = np.arange(len(names)); width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, dices, width, label="Dice", color=colours, alpha=0.85,
           edgecolor="black")
    ax.bar(x + width / 2, ious,  width, label="IoU",  color=colours, alpha=0.55,
           edgecolor="black", hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("MAC-CLIP Ablation Study (Kvasir-SEG)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAC-CLIP ablation study")
    parser.add_argument("--datasets_root", default="/media/data/Segmentation_Datasets")
    parser.add_argument("--kvasir_root",
                        default="/media/data/Local_CLIP/kvasir-seg/Kvasir-SEG/")
    parser.add_argument("--out_dir",  default="results/ablation")
    parser.add_argument("--epochs",   type=int, default=40)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--mode",     default="train", choices=["train", "smoke"])
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant keys or 'all'")
    argv = [] if hasattr(sys, "ps1") or "ipykernel" in sys.modules else None
    args = parser.parse_args(argv)

    smoke   = (args.mode == "smoke")
    out_dir = Path(args.out_dir)

    cfg = MACCLIPConfig()
    cfg.data_root = args.kvasir_root

    print("\n" + "=" * 60)
    print("  MAC-CLIP Ablation Study")
    print("=" * 60)

    train_ds, val_ds = build_kvasir_dataset(args.kvasir_root, cfg)
    print(f"  Train: {len(train_ds)}   Val: {len(val_ds)}")

    variants_to_run = (
        ABLATION_VARIANTS if args.variants == "all"
        else [v for v in ABLATION_VARIANTS
              if v["key"] in [k.strip() for k in args.variants.split(",")]]
    )

    rows: List[Dict] = []
    for variant in variants_to_run:
        print(f"\n  Running variant: {variant['name']}")
        result = run_ablation_variant(
            variant, cfg, train_ds, val_ds,
            num_epochs=args.epochs, smoke=smoke, seed=args.seed)
        rows.append(result)

        # Save intermediate JSON
        (out_dir / "per_variant").mkdir(parents=True, exist_ok=True)
        with open(out_dir / "per_variant" / f"{variant['key']}.json", "w") as f:
            json.dump(result, f, indent=2)

    save_ablation_csv(rows, out_dir / "ablation_table.csv")
    save_ablation_latex(rows, out_dir / "ablation_latex.tex")
    plot_ablation_bar(rows,  out_dir / "ablation_bar.png")

    # Print summary
    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY")
    print(f"  {'Variant':<30} {'Dice':>8} {'IoU':>8} {'BF1':>8}")
    print("  " + "-" * 56)
    for r in rows:
        print(f"  {r['variant']:<30} "
              f"{float(r.get('dice',0) or 0):>8.4f} "
              f"{float(r.get('iou', 0) or 0):>8.4f} "
              f"{float(r.get('boundary_f1',0) or 0):>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
