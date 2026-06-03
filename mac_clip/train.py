"""
Training & Validation
======================
train_epoch()  — one training pass over a DataLoader
validate()     — full validation with optional TTA and extended metrics
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from mac_clip.config import MACCLIPConfig
from mac_clip.utils import (
    compute_boundary_f1,
    compute_hausdorff_distance,
    compute_metrics,
    connected_component_postprocess,
    tta_predict,
)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    config: MACCLIPConfig,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """One full training epoch.

    Parameters
    ----------
    max_batches : int, optional
        Stop after this many batches (used in smoke-test / debug mode).
    """
    model.train()
    totals  = {"loss": 0.0, "dice": 0.0}
    n_steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", leave=False)
    for batch in pbar:
        if max_batches is not None and n_steps >= max_batches:
            break

        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            out  = model(batch, epoch=epoch, compute_loss=True)
            loss = out["loss"]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += loss.item()
        totals["dice"] += out["dice"]
        n_steps += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{out['dice']:.4f}")

    n_steps = max(n_steps, 1)
    return {k: v / n_steps for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validate(
    model: torch.nn.Module,
    loader,
    config: MACCLIPConfig,
    compute_all_metrics: bool = False,
    use_tta: bool = False,
) -> Dict[str, float]:
    """Run inference on the validation set and return mean metrics.

    Parameters
    ----------
    compute_all_metrics : bool
        If True, also compute Boundary F1 and Hausdorff distance.
    use_tta : bool
        If True, average predictions over four flip augmentations.
    """
    model.eval()
    metrics: Dict[str, List[float]] = {"dice": [], "iou": []}
    if compute_all_metrics:
        metrics.update({"boundary_f1": [], "hausdorff": []})

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            sp = (tta_predict(model, batch, config)
                  if use_tta and config.use_tta
                  else model(batch, compute_loss=False)["seg_preds"])

            for i in range(sp.shape[0]):
                pred = sp[i].cpu().numpy()
                gt   = batch["mask"][i].cpu().numpy()

                if config.use_connected_components:
                    pred = connected_component_postprocess(pred, config.min_component_ratio)

                pred_bin = (pred > 0.5).astype(float)
                m = compute_metrics(torch.tensor(pred), torch.tensor(gt))
                metrics["dice"].append(m["dice"])
                metrics["iou"].append(m["iou"])

                if compute_all_metrics:
                    metrics["boundary_f1"].append(compute_boundary_f1(pred_bin, gt))
                    metrics["hausdorff"].append(compute_hausdorff_distance(pred_bin, gt))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Single-dataset training pipeline
# ---------------------------------------------------------------------------

def run_single_dataset(
    model: torch.nn.Module,
    dataset_name: str,
    train_ds,
    val_ds,
    config: MACCLIPConfig,
    args,
    all_results: Dict,
) -> Dict:
    """Train + evaluate MAC-CLIP on one dataset.

    Modes
    -----
    train : full training (num_epochs, all features on)
    eval  : load checkpoint, evaluate only
    smoke : 2 train batches + subset val; verifies the pipeline runs end-to-end
    """
    import json
    from pathlib import Path
    from torch.utils.data import DataLoader

    from mac_clip.utils import visualize_predictions

    smoke = (args.mode == "smoke")

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name.upper()}" + ("  [SMOKE]" if smoke else ""))
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print("=" * 70)

    model.set_dataset_concepts(dataset_name)
    model._concept_cache = {}

    ckpt_dir   = Path(config.checkpoint_dir) / dataset_name
    result_dir = Path(config.results_dir)    / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    eff_bs   = min(config.batch_size, len(train_ds))
    eff_bs_v = min(config.batch_size, len(val_ds))
    do_drop  = len(train_ds) >= eff_bs * 2

    safe = getattr(config, "safe_gpu", False)
    nw_t = 0 if safe else config.num_workers
    nw_v = 0 if safe else min(2, config.num_workers)
    pin  = not safe

    if smoke:
        from torch.utils.data import Subset
        train_loader = DataLoader(
            Subset(train_ds, list(range(min(eff_bs * 2, len(train_ds))))),
            batch_size=eff_bs, shuffle=True, num_workers=2, drop_last=False)
        val_loader = DataLoader(
            Subset(val_ds, list(range(min(eff_bs_v, len(val_ds))))),
            batch_size=eff_bs_v, shuffle=False, num_workers=2)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=eff_bs, shuffle=True,
            num_workers=nw_t, drop_last=do_drop, pin_memory=pin)
        val_loader = DataLoader(
            val_ds, batch_size=eff_bs_v, shuffle=False,
            num_workers=nw_v, pin_memory=pin)

    # ── Eval-only mode ──────────────────────────────────────────────────────
    if args.mode == "eval" and args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  ✓ Loaded checkpoint: {args.checkpoint}")

    # ── Training mode ───────────────────────────────────────────────────────
    elif args.mode in ("train", "smoke"):
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=5e-7)
        scaler    = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

        best_dice    = 0.0
        patience_ctr = 0
        n_epochs     = 1 if smoke else config.num_epochs
        max_batches  = 2 if smoke else None

        for epoch in range(n_epochs):
            tr = train_epoch(model, train_loader, optimizer, scaler, epoch, config,
                             max_batches=max_batches)
            use_tta_now = (not smoke and config.use_tta and
                           ((epoch + 1) % 5 == 0 or epoch == n_epochs - 1))
            val = validate(model, val_loader, config, use_tta=use_tta_now)
            scheduler.step()

            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{n_epochs} | LR={lr_now:.7f} | "
                  f"Train Dice={tr['dice']:.4f} | Val Dice={val['dice']:.4f} "
                  f"IoU={val['iou']:.4f}" + (" [TTA]" if use_tta_now else ""))

            if val["dice"] > best_dice:
                best_dice    = val["dice"]
                patience_ctr = 0
                if not smoke:
                    torch.save({
                        "epoch":                epoch,
                        "model_state_dict":     model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_dice":            best_dice,
                        "dataset":              dataset_name,
                    }, str(ckpt_dir / "best.pth"))
                    print(f"    ✓ NEW BEST Dice={best_dice:.4f}")
            else:
                patience_ctr += 1
                if not smoke and patience_ctr >= config.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        if not smoke and (ckpt_dir / "best.pth").exists():
            ckpt = torch.load(str(ckpt_dir / "best.pth"), weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  ✓ Loaded best checkpoint (Dice={best_dice:.4f})")

    # ── Final evaluation ────────────────────────────────────────────────────
    use_tta_eval = (not smoke) and config.use_tta
    print(f"\n  Final evaluation [{dataset_name}]"
          + (" (smoke: subset, no TTA)" if smoke else " with TTA + CC post-processing…"))
    val_m = validate(model, val_loader, config, compute_all_metrics=True, use_tta=use_tta_eval)

    print(f"  {'Dice':>10}: {val_m['dice']:.4f}")
    print(f"  {'IoU':>10}: {val_m['iou']:.4f}")
    print(f"  {'BF1':>10}: {val_m['boundary_f1']:.4f}")
    print(f"  {'Hausdorff':>10}: {val_m['hausdorff']:.2f} px")

    val_m["dataset"] = dataset_name
    fname = "smoke_metrics.json" if smoke else "metrics.json"
    with open(str(result_dir / fname), "w") as f:
        json.dump(val_m, f, indent=2)

    n_vis   = 3 if smoke else config.n_visualizations
    vis_dir = result_dir / "visualizations"
    visualize_predictions(
        model, val_loader, str(vis_dir),
        n=n_vis, config=config,
        use_tta=use_tta_eval,
        dataset_name=dataset_name,
    )

    all_results[dataset_name] = val_m
    return val_m
