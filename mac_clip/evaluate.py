"""
Evaluation Extensions  (BMVC/CVPR-grade validation suite)
==========================================================
EXT-B  Advanced per-image metrics  (clDice, ASSD, AUROC)
EXT-C  Statistical testing         (Wilcoxon, Cohen's d, 95% CI)
EXT-D  Calibration                 (ECE + reliability diagram)
EXT-E  Uncertainty quality         (AUROC + risk-coverage curve)
EXT-F  Corruption robustness       (6 corruption types × 2 severities)
EXT-G  Computational profiling     (params, FLOPs, FPS, peak-mem)
EXT-H  Few-shot / low-data eval
EXT-I  Prompt-sensitivity ablation
EXT-J  Cross-dataset transfer
EXT-K  Hyper-parameter sensitivity sweeps
EXT-L  Representation analysis     (t-SNE / UMAP / orthogonality)
EXT-M  Failure-case visualisation
EXT-N  Training-stability tracking
EXT-O  Multi-seed orchestrator
EXT-P  Aggregation + radar charts
EXT-Q  Limitations & Ethics report
"""

import csv
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

matplotlib.use("Agg")

# Optional imports
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY_STATS = True
except Exception:
    HAS_SCIPY_STATS = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except Exception:
    HAS_FVCORE = False

try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

from mac_clip.config import MACCLIPConfig
from mac_clip.train import train_epoch, validate
from mac_clip.utils import (
    compute_metrics, connected_component_postprocess,
    set_seed, tta_predict, visualize_predictions,
)


# ---------------------------------------------------------------------------
# EXT-A — Output folder helpers
# ---------------------------------------------------------------------------

def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _exp_root(config: MACCLIPConfig) -> Path:
    return _ensure(Path(config.out_dir_root))


def _dir_per_dataset(c, ds):     return _ensure(_exp_root(c) / "per_dataset" / ds)
def _dir_seed(c, ds, seed):      return _ensure(_dir_per_dataset(c, ds) / f"seed_{seed}")
def _dir_aggregated(c, ds):      return _ensure(_dir_per_dataset(c, ds) / "aggregated")
def _dir_robustness(c, ds):      return _ensure(_dir_per_dataset(c, ds) / "robustness")
def _dir_calibration(c, ds):     return _ensure(_dir_per_dataset(c, ds) / "calibration")
def _dir_uncertainty(c, ds):     return _ensure(_dir_per_dataset(c, ds) / "uncertainty")
def _dir_few_shot(c, ds):        return _ensure(_dir_per_dataset(c, ds) / "few_shot")
def _dir_prompts(c, ds):         return _ensure(_dir_per_dataset(c, ds) / "prompts")
def _dir_sensitivity(c, ds):     return _ensure(_dir_per_dataset(c, ds) / "sensitivity")
def _dir_visualizations(c, ds):  return _ensure(_dir_per_dataset(c, ds) / "visualizations")
def _dir_ablations(c, ds):       return _ensure(_dir_per_dataset(c, ds) / "ablations")
def _dir_cross_dataset(c):       return _ensure(_exp_root(c) / "cross_dataset")
def _dir_representation(c):      return _ensure(_exp_root(c) / "representation")
def _dir_computational(c):       return _ensure(_exp_root(c) / "computational")
def _dir_combined(c):            return _ensure(_exp_root(c) / "combined")
def _dir_training_stability(c):  return _ensure(_exp_root(c) / "training_stability")


def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def append_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None):
    path = Path(path); new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# EXT-B — Advanced per-image metrics
# ---------------------------------------------------------------------------

def _binary(arr: np.ndarray, t: float = 0.5) -> np.ndarray:
    return (arr > t).astype(np.uint8)


def per_image_dice(pred, gt):
    p = _binary(pred); t = _binary(gt)
    i = (p & t).sum()
    return float((2.0 * i + 1e-7) / (p.sum() + t.sum() + 1e-7))


def per_image_iou(pred, gt):
    p = _binary(pred); t = _binary(gt)
    return float(((p & t).sum() + 1e-7) / ((p | t).sum() + 1e-7))


def per_image_bf1(pred, gt, tol: int = 2):
    p = _binary(pred); t = _binary(gt)
    pb = cv2.Canny((p * 255).astype(np.uint8), 50, 150) > 0
    tb = cv2.Canny((t * 255).astype(np.uint8), 50, 150) > 0
    if pb.sum() == 0 and tb.sum() == 0: return 1.0
    if pb.sum() == 0 or  tb.sum() == 0: return 0.0
    dt_t = cv2.distanceTransform((~tb).astype(np.uint8), cv2.DIST_L2, 5)
    dt_p = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 5)
    prec = (pb & (dt_t < tol)).sum() / (pb.sum() + 1e-8)
    rec  = (tb & (dt_p < tol)).sum() / (tb.sum() + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))


def per_image_hausdorff(pred, gt):
    from scipy.spatial.distance import directed_hausdorff
    p = _binary(pred); t = _binary(gt)
    if p.sum() == 0 and t.sum() == 0: return 0.0
    if p.sum() == 0 or  t.sum() == 0: return 100.0
    pp = np.argwhere(p > 0); tp = np.argwhere(t > 0)
    return float(max(directed_hausdorff(pp, tp)[0], directed_hausdorff(tp, pp)[0]))


def per_image_clDice(pred, gt):
    """Centreline-Dice for tubular structures."""
    if not HAS_SKIMAGE:
        return per_image_dice(pred, gt)
    p = _binary(pred); t = _binary(gt)
    if p.sum() == 0 and t.sum() == 0: return 1.0
    if p.sum() == 0 or  t.sum() == 0: return 0.0
    sp = skeletonize(p > 0); st = skeletonize(t > 0)
    tprec = (sp & (t > 0)).sum() / (sp.sum() + 1e-8)
    tsens = (st & (p > 0)).sum() / (st.sum() + 1e-8)
    return float(2 * tprec * tsens / (tprec + tsens + 1e-8))


def per_image_all_metrics(pred, gt) -> Dict:
    return {
        "dice":      per_image_dice(pred, gt),
        "iou":       per_image_iou(pred, gt),
        "bf1":       per_image_bf1(pred, gt),
        "hausdorff": per_image_hausdorff(pred, gt),
        "clDice":    per_image_clDice(pred, gt),
    }


@torch.no_grad()
def collect_per_image_metrics(model, loader, config: MACCLIPConfig,
                               use_tta: bool = False, tag: str = "default") -> List[Dict]:
    model.eval()
    rows: List[Dict] = []
    idx = 0
    for batch in loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        sp = (tta_predict(model, batch, config) if use_tta
              else model(batch, compute_loss=False)["seg_preds"])
        for i in range(sp.shape[0]):
            pred = sp[i].cpu().numpy()
            gt   = batch["mask"][i].cpu().numpy()
            if config.use_connected_components:
                pred = connected_component_postprocess(pred, config.min_component_ratio)
            m = per_image_all_metrics(pred, gt)
            m["idx"] = idx; m["tag"] = tag
            rows.append(m); idx += 1
    return rows


# ---------------------------------------------------------------------------
# EXT-C — Statistical tests
# ---------------------------------------------------------------------------

def bootstrap_ci(values: List[float], n_iter: int = 2000,
                 ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng   = np.random.default_rng(seed)
    n     = len(arr)
    means = np.array([rng.choice(arr, n, replace=True).mean() for _ in range(n_iter)])
    lo    = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi    = float(np.percentile(means, 100 * (1 + ci) / 2))
    return float(arr.mean()), lo, hi


def cohen_d(x: List[float], y: List[float]) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0
    nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) /
                   max(nx + ny - 2, 1))
    return float((x.mean() - y.mean()) / sp) if sp > 1e-12 else 0.0


def paired_significance(metric_ours: List[float],
                        metric_baseline: List[float]) -> Dict:
    n = min(len(metric_ours), len(metric_baseline))
    if n < 2:
        return {"n": n, "wilcoxon_p": float("nan"),
                "ttest_p": float("nan"), "cohen_d": float("nan")}
    a = np.asarray(metric_ours[:n], dtype=float)
    b = np.asarray(metric_baseline[:n], dtype=float)
    if not HAS_SCIPY_STATS:
        return {"n": n, "wilcoxon_p": float("nan"),
                "ttest_p": float("nan"), "cohen_d": cohen_d(a, b)}
    try:
        diffs = a - b
        _, w_p = (scipy_stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                  if not np.allclose(diffs, 0) else (None, 1.0))
    except Exception:
        w_p = float("nan")
    try:
        _, t_p = scipy_stats.ttest_rel(a, b, nan_policy="omit")
    except Exception:
        t_p = float("nan")
    return {"n": n, "wilcoxon_p": float(w_p),
            "ttest_p": float(t_p), "cohen_d": cohen_d(a, b)}


# ---------------------------------------------------------------------------
# EXT-D — Calibration (ECE)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_calibration(model, loader, config: MACCLIPConfig,
                    ds_name: str, save_dir: Path) -> Dict:
    model.eval()
    probs_buf, gts_buf = [], []
    n_seen = 0
    for batch in loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        sp = model(batch, compute_loss=False)["seg_preds"].cpu().numpy()
        gt = batch["mask"].cpu().numpy()
        for i in range(sp.shape[0]):
            take = min(10_000, sp[i].size)
            idx  = np.random.choice(sp[i].size, take, replace=False)
            probs_buf.append(sp[i].reshape(-1)[idx])
            gts_buf.append((gt[i].reshape(-1) > 0.5).astype(np.float32)[idx])
            n_seen += take
        if n_seen >= 4_000_000:
            break

    if not probs_buf:
        return {"ece": float("nan")}

    probs = np.concatenate(probs_buf); gts = np.concatenate(gts_buf)
    bins  = np.linspace(0, 1, config.ece_n_bins + 1)
    ece   = 0.0; bin_rows = []
    for i in range(config.ece_n_bins):
        mask  = (probs >= bins[i]) & (probs <= bins[i + 1] if i == config.ece_n_bins - 1
                                      else probs < bins[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        conf = float(probs[mask].mean()); acc = float(gts[mask].mean())
        w    = n_bin / probs.size
        ece += w * abs(acc - conf)
        bin_rows.append({"bin_lo": bins[i], "bin_hi": bins[i + 1],
                         "n": int(n_bin), "conf": conf, "acc": acc, "weight": w})

    fig, ax = plt.subplots(figsize=(5, 5))
    confs = [r["conf"] for r in bin_rows]; accs = [r["acc"] for r in bin_rows]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.bar(confs, accs, width=1.0 / max(len(bin_rows), 1),
           edgecolor="black", alpha=0.7, label="Empirical")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability diagram — {ds_name}\nECE = {ece:.4f}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"reliability_{ds_name}.png", dpi=140); plt.close()

    write_csv(save_dir / f"calibration_bins_{ds_name}.csv", bin_rows)
    import json
    with open(save_dir / f"ece_{ds_name}.json", "w") as f:
        json.dump({"dataset": ds_name, "ece": ece, "n_pixels": int(probs.size)}, f, indent=2)
    return {"ece": ece, "n_pixels": int(probs.size)}


# ---------------------------------------------------------------------------
# EXT-E — Uncertainty quality
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_uncertainty_quality(model, loader, config: MACCLIPConfig,
                             ds_name: str, save_dir: Path) -> Dict:
    model.eval()
    unc_all, err_all = [], []
    for batch in loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out   = model(batch, compute_loss=False)
        sp    = out["seg_preds"].cpu().numpy()
        gt    = batch["mask"].cpu().numpy()

        gf, ms = model.encode_image_multiscale(batch["image"])
        fp, _  = model.fpn(ms)
        fused  = model.feat_proj(fp)
        _, unc = model.morphological_module(fused)
        unc_up = F.interpolate(unc.unsqueeze(1), gt.shape[-2:],
                               mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
        err    = np.abs(sp - gt)

        for i in range(sp.shape[0]):
            take = min(10_000, unc_up[i].size)
            idx  = np.random.choice(unc_up[i].size, take, replace=False)
            unc_all.append(unc_up[i].reshape(-1)[idx])
            err_all.append(err[i].reshape(-1)[idx])

    if not unc_all:
        return {"auroc": float("nan")}

    unc = np.concatenate(unc_all); err = np.concatenate(err_all)
    err_bin = (err > 0.5).astype(np.uint8)
    auroc   = float("nan")
    if HAS_SKLEARN and 0 < err_bin.sum() < err_bin.size:
        try:
            auroc = float(roc_auc_score(err_bin, unc))
        except Exception:
            pass

    order = np.argsort(unc); err_sorted = err_bin[order]; cum = np.cumsum(err_sorted)
    coverages = np.linspace(0.05, 1.0, 20)
    risk      = [float(cum[max(1, int(c * len(order))) - 1] / max(1, int(c * len(order))))
                 for c in coverages]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(coverages, risk, "o-", lw=2)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Error rate")
    ax.set_title(f"Risk-coverage — {ds_name}\nAUROC={auroc:.3f}")
    ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_dir / f"risk_coverage_{ds_name}.png", dpi=140); plt.close()

    write_csv(save_dir / f"risk_coverage_{ds_name}.csv",
              [{"coverage": float(c), "risk": float(r)} for c, r in zip(coverages, risk)])
    import json
    with open(save_dir / f"uncertainty_{ds_name}.json", "w") as f:
        json.dump({"dataset": ds_name, "auroc_unc_error": auroc,
                   "n_pixels": int(unc.size)}, f, indent=2)
    return {"auroc": auroc, "n_pixels": int(unc.size)}


# ---------------------------------------------------------------------------
# EXT-F — Corruption robustness
# ---------------------------------------------------------------------------

class CorruptDataset(Dataset):
    """Wraps an existing dataset and applies a corruption at runtime."""

    def __init__(self, base_ds, corruption: str, severity: int = 1):
        self.base = base_ds; self.corruption = corruption; self.severity = severity

    def __len__(self): return len(self.base)

    def _apply(self, img: np.ndarray) -> np.ndarray:
        c = self.corruption; s = self.severity
        lut = lambda lst: lst[min(s - 1, len(lst) - 1)]
        if c == "gaussian_noise":
            return np.clip(img.astype(np.float32) +
                           np.random.normal(0, lut([10, 20, 30, 40, 50]), img.shape), 0, 255).astype(np.uint8)
        if c == "gaussian_blur":
            k = lut([3, 5, 7, 9, 11]); return cv2.GaussianBlur(img, (k, k), 0)
        if c == "brightness":
            return np.clip(img.astype(np.int32) + lut([10, 30, 50, 70, 90]), 0, 255).astype(np.uint8)
        if c == "contrast":
            f = lut([0.85, 0.7, 0.55, 0.4, 0.25])
            return np.clip((img.astype(np.float32) - img.mean()) * f + img.mean(), 0, 255).astype(np.uint8)
        if c == "jpeg_compression":
            q = lut([70, 50, 30, 20, 10])
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q])
            return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else img
        if c == "motion_blur":
            k = lut([5, 9, 13, 17, 21])
            kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1.0 / k
            return cv2.filter2D(img, -1, kern)
        return img

    def __getitem__(self, idx):
        img_path  = self.base.image_paths[idx]
        mask_path = self.base.mask_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            sz  = self.base.config.image_size
            img = np.zeros((sz, sz, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.base._load_mask(mask_path)
        if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
            h, w = img.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img = self._apply(img)
        aug = self.base.transform(image=img, mask=mask)
        return {"image": aug["image"], "mask": aug["mask"],
                "text": random.choice(self.base.clinical_concepts),
                "image_path": img_path}


def run_robustness(model, val_ds, config: MACCLIPConfig,
                   ds_name: str, save_dir: Path) -> List[Dict]:
    print(f"\n  [Robustness] {ds_name}")
    rows: List[Dict] = []
    eff_bs = min(config.batch_size, len(val_ds))
    for c in config.corruption_types:
        for s in config.corruption_severities:
            loader = DataLoader(CorruptDataset(val_ds, c, s),
                                batch_size=eff_bs, shuffle=False,
                                num_workers=min(4, config.num_workers), pin_memory=True)
            m = validate(model, loader, config, compute_all_metrics=True, use_tta=False)
            rows.append({"dataset": ds_name, "corruption": c, "severity": s,
                         "dice": m["dice"], "iou": m["iou"],
                         "bf1": m["boundary_f1"], "hausdorff": m["hausdorff"]})
            print(f"    {c}/sev{s}: Dice={m['dice']:.4f}  IoU={m['iou']:.4f}")
    write_csv(save_dir / f"robustness_{ds_name}.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# EXT-G — Computational profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def profile_model(model, image_size: int = 352,
                  n_warmup: int = 5, n_runs: int = 20) -> Dict:
    model.eval()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = {"params_total": total, "params_trainable": trainable,
           "trainable_pct": 100.0 * trainable / max(total, 1)}

    device = next(model.parameters()).device
    x      = torch.randn(1, 3, image_size, image_size, device=device)
    dummy  = {"image": x, "mask": torch.zeros(1, image_size, image_size, device=device)}

    if HAS_FVCORE:
        try:
            class _W(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    return self.m({"image": x, "mask": torch.zeros(
                        x.shape[0], x.shape[-2], x.shape[-1], device=x.device)},
                        compute_loss=False)["seg_preds"]
            out["flops_g"] = float(FlopCountAnalysis(_W(model).to(device), x).total() / 1e9)
        except Exception:
            out["flops_g"] = float("nan")
    else:
        out["flops_g"] = float("nan")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        for _ in range(n_warmup):
            model(dummy, compute_loss=False)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            model(dummy, compute_loss=False)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / n_runs
        out.update({
            "latency_ms":   float(elapsed * 1000),
            "fps":          float(1.0 / max(elapsed, 1e-6)),
            "peak_mem_mb":  float(torch.cuda.max_memory_allocated() / 1024 ** 2),
        })
    else:
        out.update({"latency_ms": float("nan"), "fps": float("nan"), "peak_mem_mb": float("nan")})
    return out


# ---------------------------------------------------------------------------
# EXT-H — Few-shot evaluation
# ---------------------------------------------------------------------------

def run_few_shot(model_ctor: Callable, dataset_name: str, train_ds, val_ds,
                 config: MACCLIPConfig, save_dir: Path,
                 fractions: Optional[List[float]] = None) -> List[Dict]:
    fractions = fractions or config.few_shot_fractions
    rows: List[Dict] = []
    eff_bs_v = min(config.batch_size, len(val_ds))
    vl = DataLoader(val_ds, batch_size=eff_bs_v, shuffle=False,
                    num_workers=min(2, config.num_workers), pin_memory=True)

    for frac in fractions:
        n_sub = max(1, int(round(frac * len(train_ds))))
        idx   = list(range(len(train_ds))); random.seed(config.seed); random.shuffle(idx)
        sub   = Subset(train_ds, idx[:n_sub])
        eff_bs = min(config.batch_size, n_sub)
        tl = DataLoader(sub, batch_size=eff_bs, shuffle=True,
                        num_workers=min(4, config.num_workers), pin_memory=True)

        model = model_ctor().cuda()
        model.set_dataset_concepts(dataset_name)
        opt   = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate, weight_decay=config.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

        best  = 0.0; best_state = None; pctr = 0
        n_ep  = min(config.few_shot_epochs, max(config.num_epochs // 2, 10))
        for ep in range(n_ep):
            train_epoch(model, tl, opt, scaler, ep, config)
            v = validate(model, vl, config, use_tta=False)
            if v["dice"] > best + 1e-3:
                best = v["dice"]; pctr = 0
                best_state = {k: v_.detach().cpu().clone()
                              for k, v_ in model.state_dict().items()}
            else:
                pctr += 1
                if pctr >= max(5, n_ep // 4):
                    break
        if best_state:
            model.load_state_dict(best_state)

        v = validate(model, vl, config, compute_all_metrics=True, use_tta=False)
        rows.append({"dataset": dataset_name, "fraction": float(frac),
                     "n_train": int(n_sub), "dice": float(v["dice"]),
                     "iou": float(v["iou"]), "bf1": float(v.get("boundary_f1", float("nan"))),
                     "hausdorff": float(v.get("hausdorff", float("nan"))), "epochs": n_ep})
        print(f"    [few-shot {frac:.2%}|n={n_sub}] Dice={v['dice']:.4f}")
        del model; torch.cuda.empty_cache()

    write_csv(save_dir / f"few_shot_{dataset_name}.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# EXT-I — Prompt sensitivity
# ---------------------------------------------------------------------------

GENERIC_CONCEPTS = [
    "object foreground region", "object background region", "smooth shape region",
    "rough shape region", "elongated region", "compact region", "bright region", "dark region",
]


@torch.no_grad()
def run_prompt_sensitivity(model, val_ds, dataset_name: str,
                           config: MACCLIPConfig, save_dir: Path) -> List[Dict]:
    eff_bs = min(config.batch_size, len(val_ds))
    vl     = DataLoader(val_ds, batch_size=eff_bs, shuffle=False,
                        num_workers=min(4, config.num_workers), pin_memory=True)
    orig_c, orig_b, orig_i = (list(config.clinical_concepts),
                               config.boundary_concept, config.interior_concept)

    vocab  = ["alpha", "beta", "gamma", "delta", "kappa", "omega", "lyra",
              "node", "ridge", "facet", "patch", "field", "halo", "core"]
    rng    = random.Random(config.seed)
    rand_p = [" ".join(rng.sample(vocab, 6)) for _ in range(len(orig_c))]

    variants = {
        "clinical":  (orig_c, orig_b, orig_i),
        "generic":   (GENERIC_CONCEPTS, "edge boundary region", "interior body region"),
        "random":    (rand_p, "kappa edge tau halo", "omega core ridge"),
        "no_prompt": ([""] * len(orig_c), "", ""),
    }
    rows: List[Dict] = []
    for vname, (plist, bnd, int_) in variants.items():
        config.clinical_concepts = plist; config.boundary_concept = bnd; config.interior_concept = int_
        model.config.clinical_concepts = plist; model.config.boundary_concept = bnd
        model.config.interior_concept  = int_; model._concept_cache = {}
        m = validate(model, vl, config, compute_all_metrics=True, use_tta=False)
        rows.append({"dataset": dataset_name, "prompt_variant": vname,
                     "dice": m["dice"], "iou": m["iou"],
                     "bf1": m.get("boundary_f1", float("nan")),
                     "hausdorff": m.get("hausdorff", float("nan"))})
        print(f"    prompt='{vname:<10}': Dice={m['dice']:.4f}")

    config.clinical_concepts = orig_c; config.boundary_concept = orig_b
    config.interior_concept  = orig_i; model._concept_cache = {}
    model.config.clinical_concepts = orig_c; model.config.boundary_concept = orig_b
    model.config.interior_concept  = orig_i
    write_csv(save_dir / f"prompts_{dataset_name}.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# EXT-J — Cross-dataset transfer
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_cross_dataset(model, source_ds: str, target_ds: str,
                      target_val_ds, config: MACCLIPConfig) -> Dict:
    model.set_dataset_concepts(target_ds); model._concept_cache = {}
    eff_bs = min(config.batch_size, len(target_val_ds))
    loader = DataLoader(target_val_ds, batch_size=eff_bs, shuffle=False,
                        num_workers=min(4, config.num_workers), pin_memory=True)
    m = validate(model, loader, config, compute_all_metrics=True, use_tta=False)
    return {"source": source_ds, "target": target_ds,
            "dice": m["dice"], "iou": m["iou"],
            "bf1": m.get("boundary_f1", float("nan")),
            "hausdorff": m.get("hausdorff", float("nan"))}


# ---------------------------------------------------------------------------
# EXT-K — Hyper-parameter sensitivity (inference-time)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sensitivity_inference(model, val_ds, dataset_name: str,
                               config: MACCLIPConfig, save_dir: Path) -> List[Dict]:
    eff_bs = min(config.batch_size, len(val_ds))
    loader = DataLoader(val_ds, batch_size=eff_bs, shuffle=False,
                        num_workers=min(2, config.num_workers), pin_memory=True)
    rows: List[Dict] = []

    orig_alpha = model.decoder.gate_alpha.data.clone()
    for a in getattr(config, "sweep_gate_alpha", [0.0, 0.3, 0.6, 1.0]):
        model.decoder.gate_alpha.data.fill_(a)
        m = validate(model, loader, config, compute_all_metrics=True, use_tta=False)
        rows.append({"dataset": dataset_name, "param": "boundary_gate_alpha", "value": a,
                     "dice": m["dice"], "iou": m["iou"], "bf1": m.get("boundary_f1", float("nan"))})
    model.decoder.gate_alpha.data.copy_(orig_alpha)

    orig_ratio = config.min_component_ratio
    for r in getattr(config, "sweep_cc_ratio", [0.0, 0.1, 0.3]):
        config.min_component_ratio = r
        m = validate(model, loader, config, compute_all_metrics=True, use_tta=False)
        rows.append({"dataset": dataset_name, "param": "cc_min_ratio", "value": r,
                     "dice": m["dice"], "iou": m["iou"], "bf1": m.get("boundary_f1", float("nan"))})
    config.min_component_ratio = orig_ratio

    write_csv(save_dir / f"sensitivity_{dataset_name}.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# EXT-N — Training stability
# ---------------------------------------------------------------------------

def save_training_curves(history: Dict[int, List[Dict]],
                         save_dir: Path, dataset_name: str) -> None:
    if not history:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for seed, rows in history.items():
        ep = [r["epoch"] for r in rows]
        ax[0].plot(ep, [r["train_loss"] for r in rows], label=f"seed={seed}")
        ax[1].plot(ep, [r["val_dice"]   for r in rows], label=f"seed={seed}")
    ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].set_title("Val Dice");      ax[1].set_xlabel("epoch")
    for a in ax: a.legend(); a.grid(alpha=0.3)
    fig.suptitle(f"Training stability — {dataset_name}")
    plt.tight_layout()
    plt.savefig(save_dir / f"training_curves_{dataset_name}.png", dpi=140)
    plt.close()

    rows_out = [dict(r, seed=seed) for seed, hist in history.items() for r in hist]
    write_csv(save_dir / f"training_history_{dataset_name}.csv", rows_out)


# ---------------------------------------------------------------------------
# EXT-O — Multi-seed orchestrator
# ---------------------------------------------------------------------------

def train_one_seed_full(model_ctor: Callable, dataset_name: str,
                        train_ds, val_ds, config: MACCLIPConfig,
                        args, seed: int):
    """Train one seed end-to-end. Returns (final_metrics, per_image_rows, history, model)."""
    import json

    set_seed(seed)
    model = model_ctor().cuda()
    model.set_dataset_concepts(dataset_name)

    smoke       = (args.mode == "smoke")
    eff_bs      = min(config.batch_size, len(train_ds))
    eff_bs_v    = min(config.batch_size, len(val_ds))
    do_drop     = len(train_ds) >= eff_bs * 2
    safe        = getattr(config, "safe_gpu", False)
    nw_t        = 0 if safe else min(config.num_workers, 6)
    nw_v        = 0 if safe else min(2, config.num_workers)
    pin         = not safe
    persistent  = (not safe)

    train_loader = DataLoader(train_ds, batch_size=eff_bs, shuffle=True,
                              num_workers=nw_t, drop_last=do_drop, pin_memory=pin,
                              persistent_workers=(persistent and nw_t > 0))
    val_loader   = DataLoader(val_ds, batch_size=eff_bs_v, shuffle=False,
                              num_workers=nw_v, pin_memory=pin,
                              persistent_workers=(persistent and nw_v > 0))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=5e-7)
    scaler    = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    seed_dir    = _dir_seed(config, dataset_name, seed)
    history     = []
    best_dice   = 0.0
    patience    = config.early_stop_patience
    min_delta   = config.early_stop_min_delta
    warmup      = config.early_stop_warmup_epochs
    pctr        = 0
    n_epochs    = 1 if smoke else config.num_epochs
    max_batches = 2 if smoke else None

    for ep in range(n_epochs):
        tr = train_epoch(model, train_loader, optimizer, scaler, ep, config,
                         max_batches=max_batches)
        use_tta_now = (not smoke and config.use_tta and
                       ((ep + 1) % 5 == 0 or ep == n_epochs - 1))
        val = validate(model, val_loader, config, use_tta=use_tta_now)
        scheduler.step()
        history.append({"epoch": ep + 1, "train_loss": float(tr["loss"]),
                        "train_dice": float(tr["dice"]), "val_dice": float(val["dice"]),
                        "val_iou": float(val["iou"])})
        print(f"    [seed={seed}] ep {ep+1}/{n_epochs} | "
              f"loss={tr['loss']:.4f} | val Dice={val['dice']:.4f}"
              + (" [TTA]" if use_tta_now else ""))

        if val["dice"] > best_dice + min_delta:
            best_dice = val["dice"]; pctr = 0
            if not smoke:
                torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                            "best_dice": best_dice, "seed": seed,
                            "dataset": dataset_name},
                           str(seed_dir / "best.pth"))
        else:
            pctr += 1
            if (not smoke and ep + 1 >= warmup
                    and best_dice > 0.05 and pctr >= patience):
                print(f"    [seed={seed}] Early stopping at epoch {ep + 1}")
                break

    if not smoke and (seed_dir / "best.pth").exists():
        ckpt = torch.load(str(seed_dir / "best.pth"), weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    final = validate(model, val_loader, config, compute_all_metrics=True,
                     use_tta=(not smoke) and config.use_tta)
    final["seed"] = seed; final["dataset"] = dataset_name
    with open(seed_dir / "metrics.json", "w") as f:
        json.dump(final, f, indent=2)

    per_img = collect_per_image_metrics(model, val_loader, config,
                                        use_tta=False, tag=f"seed_{seed}")
    for r in per_img:
        r["seed"] = seed; r["dataset"] = dataset_name
    write_csv(seed_dir / "per_image_metrics.csv", per_img)

    # Release dataloader workers
    for ldr in [train_loader, val_loader]:
        try:
            if hasattr(ldr, "_iterator") and ldr._iterator:
                ldr._iterator._shutdown_workers()
        except Exception:
            pass
    del train_loader, val_loader

    return final, per_img, history, model


def run_multi_seed_dataset(model_ctor: Callable, dataset_name: str,
                            train_ds, val_ds, config: MACCLIPConfig,
                            args, seeds: List[int]) -> Dict:
    """Train across all seeds; aggregate stats."""
    print("\n" + "=" * 70)
    print(f"  MULTI-SEED — {dataset_name.upper()}  ({len(seeds)} seeds)")
    print("=" * 70)

    per_seed, per_img_pool, history_by_seed = [], [], {}
    last_model = None

    for seed in seeds:
        final, per_img, hist, model = train_one_seed_full(
            model_ctor, dataset_name, train_ds, val_ds, config, args, seed)
        per_seed.append(final); per_img_pool.extend(per_img)
        history_by_seed[seed] = hist; last_model = model

    agg_dir = _dir_aggregated(config, dataset_name)
    write_csv(agg_dir / "per_seed_metrics.csv", per_seed)
    write_csv(agg_dir / "per_image_all_seeds.csv", per_img_pool)

    summary = {}
    for metric in ["dice", "iou", "boundary_f1", "hausdorff"]:
        vals = [s.get(metric, float("nan")) for s in per_seed
                if not math.isnan(s.get(metric, float("nan")))]
        if not vals:
            continue
        mu, lo, hi = bootstrap_ci(vals, config.bootstrap_iters, config.bootstrap_ci, config.seed)
        summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                           "min": float(np.min(vals)),   "max": float(np.max(vals)),
                           "ci95_lo": lo, "ci95_hi": hi, "n_seeds": len(vals)}

    import json
    with open(agg_dir / "summary_mean_std.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(agg_dir / "summary_mean_std.csv",
              [{"metric": k, **v} for k, v in summary.items()])

    save_training_curves(history_by_seed, _dir_training_stability(config), dataset_name)

    print(f"\n  ✓ Multi-seed summary — {dataset_name}:")
    for k, v in summary.items():
        print(f"      {k:<14}: {v['mean']:.4f} ± {v['std']:.4f}  "
              f"[{v['ci95_lo']:.4f}, {v['ci95_hi']:.4f}]")

    return {"summary": summary, "per_seed_metrics": per_seed,
            "per_image_pool": per_img_pool, "last_model": last_model}


# ---------------------------------------------------------------------------
# EXT-P — Global aggregation + radar chart
# ---------------------------------------------------------------------------

def build_radar_chart(per_dataset_summary: Dict, save_path: Path) -> None:
    from mac_clip.dataset_registry import MODALITY_MAP
    groups: Dict[str, List[float]] = {}
    for ds, summ in per_dataset_summary.items():
        m = summ.get("summary", {}).get("dice", {}).get("mean")
        if m is None:
            continue
        grp = MODALITY_MAP.get(ds, "Other")
        groups.setdefault(grp, []).append(m)
    if not groups:
        return
    labels  = list(groups.keys())
    values  = [float(np.mean(groups[k])) for k in labels]
    angles  = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]
    values += values[:1]

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="polar")
    ax.plot(angles, values, "o-", linewidth=2, label="MAC-CLIP")
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_title("Mean Dice across modalities", pad=20)
    plt.tight_layout(); plt.savefig(save_path, dpi=140); plt.close()


def aggregate_all_datasets(per_ds_summary: Dict, config: MACCLIPConfig) -> List[Dict]:
    combined_dir = _dir_combined(config)
    rows = []
    for ds, summ in per_ds_summary.items():
        s = summ.get("summary", {})
        rows.append({
            "dataset":    ds,
            "dice_mean":  s.get("dice", {}).get("mean",     float("nan")),
            "dice_std":   s.get("dice", {}).get("std",      float("nan")),
            "dice_ci_lo": s.get("dice", {}).get("ci95_lo",  float("nan")),
            "dice_ci_hi": s.get("dice", {}).get("ci95_hi",  float("nan")),
            "iou_mean":   s.get("iou",  {}).get("mean",     float("nan")),
            "iou_std":    s.get("iou",  {}).get("std",      float("nan")),
            "bf1_mean":   s.get("boundary_f1", {}).get("mean", float("nan")),
            "bf1_std":    s.get("boundary_f1", {}).get("std",  float("nan")),
            "hd_mean":    s.get("hausdorff", {}).get("mean",   float("nan")),
            "hd_std":     s.get("hausdorff", {}).get("std",    float("nan")),
            "n_seeds":    s.get("dice", {}).get("n_seeds", 0),
        })
    write_csv(combined_dir / "mean_std_per_dataset.csv", rows)
    if rows:
        import json
        overall = {
            "dice_mean": float(np.nanmean([r["dice_mean"] for r in rows])),
            "iou_mean":  float(np.nanmean([r["iou_mean"]  for r in rows])),
            "bf1_mean":  float(np.nanmean([r["bf1_mean"]  for r in rows])),
            "hd_mean":   float(np.nanmean([r["hd_mean"]   for r in rows])),
            "n_datasets": len(rows),
        }
        with open(combined_dir / "overall.json", "w") as f:
            json.dump(overall, f, indent=2)
    build_radar_chart(per_ds_summary, combined_dir / "radar_modalities.png")
    return rows


# ---------------------------------------------------------------------------
# EXT-Q — Limitations & Ethics auto-report
# ---------------------------------------------------------------------------

ETHICS_REPORT = """\
# MAC-CLIP — Limitations and Ethical Considerations

1. **Annotation bias.** Several datasets include manual annotations by a small number of experts.
   Inter-annotator variability is real; metrics are relative-ranking signals, not absolute clinical thresholds.

2. **Modality imbalance.** The 15-dataset benchmark is dominated by endoscopy, X-ray and fundus imagery.
   Ultrasound and OCT are under-represented; modality-specific performance should be read in this light.

3. **Domain shift.** Distribution-shift robustness is validated through synthetic corruptions only.
   Real-world acquisition variation (scanner vendors, populations, protocols) is not fully covered.

4. **Uncertainty estimates.** The uncertainty maps are not calibrated for clinical-decision use.
   They are an analytic signal, not a calibrated probability of disease.

5. **Clinical deployment.** MAC-CLIP is a research artefact and is **not approved for clinical use**.
   Any downstream clinical application is subject to liability and patient-safety constraints.

6. **Prompt sensitivity.** Performance depends, sometimes meaningfully, on the exact wording of
   clinical prompts (see prompts_all.csv). Non-clinical or random prompts can degrade performance.

7. **Compute footprint.** Training requires a GPU with ≥ 16 GB memory at the published resolution.
"""


def write_limitations(config: MACCLIPConfig) -> None:
    p = _exp_root(config) / "LIMITATIONS_AND_ETHICS.md"
    p.write_text(ETHICS_REPORT)
    print(f"\n  ✓ Limitations & Ethics report saved to {p}")
