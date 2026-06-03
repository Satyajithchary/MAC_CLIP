"""
main.py — MAC-CLIP Training & Evaluation Orchestrator
=======================================================
Usage examples
--------------
# Full training on all datasets (3 seeds each):
  python main.py

# Single dataset, fast mode:
  python main.py --dataset kvasir --fast

# Smoke test (pipeline check, no meaningful metrics):
  python main.py --mode smoke

# Eval-only with existing checkpoint:
  python main.py --mode eval --checkpoint /path/to/best.pth --dataset kvasir

# Specific datasets, 1 seed, 40 epochs:
  python main.py --dataset kvasir,isic2018,busi --num_seeds 1 --epochs 40

See python main.py --help for all flags.
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

from mac_clip.config       import MACCLIPConfig
from mac_clip.evaluate     import (
    _dir_computational, _exp_root, aggregate_all_datasets,
    profile_model, run_calibration, run_cross_dataset,
    run_few_shot, run_multi_seed_dataset, run_prompt_sensitivity,
    run_robustness, run_sensitivity_inference, run_uncertainty_quality,
    train_one_seed_full, write_csv, write_limitations,
    _dir_calibration, _dir_uncertainty, _dir_robustness,
    _dir_prompts, _dir_sensitivity, _dir_visualizations,
    _dir_cross_dataset, collect_per_image_metrics,
)
from mac_clip.model        import MACCLIPv4
from mac_clip.preprocessing import get_all_datasets
from mac_clip.train        import run_single_dataset
from mac_clip.utils        import set_seed, visualize_predictions


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MAC-CLIP v4 — BMVC/CVPR-grade evaluation suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Core ──────────────────────────────────────────────────────────────
    p.add_argument("--mode", default="train",
                   choices=["train", "eval", "smoke"],
                   help="train | eval | smoke")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint path (eval mode only)")
    p.add_argument("--dataset", default="all",
                   help="Comma-separated dataset keys, or 'all'")
    p.add_argument("--datasets_root", default=None,
                   help="Override datasets root directory")
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--out_dir_root", default=None,
                   help="Root output directory for all results")
    p.add_argument("--num_seeds", type=int, default=3,
                   help="Number of seeds from config.seeds_list (max 5)")
    p.add_argument("--patience", type=int, default=None,
                   help="Early-stop patience (default: config value = 25)")

    # ── Extension flags ───────────────────────────────────────────────────
    def _bool(name, dest, default=True):
        p.add_argument(f"--{name}",    dest=dest, action="store_true")
        p.add_argument(f"--no_{name}", dest=dest, action="store_false")
        p.set_defaults(**{dest: default})

    _bool("run_multi_seed",     "run_multi_seed")
    _bool("run_robustness",     "run_robustness")
    _bool("run_cross_dataset",  "run_cross_dataset")
    _bool("run_few_shot",       "run_few_shot")
    _bool("run_calibration",    "run_calibration")
    _bool("run_uncertainty",    "run_uncertainty")
    _bool("run_prompts",        "run_prompts")
    _bool("run_sensitivity",    "run_sensitivity")
    _bool("run_representation", "run_representation")
    _bool("run_visualizations", "run_visualizations")
    _bool("run_ablation",       "run_ablation")
    _bool("run_significance",   "run_significance")
    _bool("run_profile",        "run_profile")
    p.add_argument("--run_lora_sweep", action="store_true", default=False,
                   help="LoRA rank sweep — expensive (retrains model per rank)")

    # ── Convenience speed flags ───────────────────────────────────────────
    p.add_argument("--fast", action="store_true",
                   help="Disable slow extensions (few-shot, cross-dataset, representation, lora-sweep)")
    p.add_argument("--safe_gpu", action="store_true",
                   help="num_workers=0, no persistent workers / pin_memory")
    p.add_argument("--ablation", action="store_true",
                   help="Alias for --run_ablation (backward compat)")

    return p


# ---------------------------------------------------------------------------
# GPU clean-up helper
# ---------------------------------------------------------------------------

def _hard_gpu_cleanup(deep: bool = False) -> None:
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        for fn in [torch.cuda.synchronize, torch.cuda.empty_cache,
                   torch.cuda.ipc_collect, torch.cuda.reset_peak_memory_stats]:
            try: fn()
            except Exception: pass
    if deep:
        time.sleep(8.0)
        gc.collect()
        if torch.cuda.is_available():
            try: torch.cuda.empty_cache()
            except Exception: pass


def _is_nvml_error(e: Exception) -> bool:
    s = str(e)
    return any(t in s for t in ("NVML", "nvmlInit", "CUDA error", "cublas"))


# ---------------------------------------------------------------------------
# Extension runner for one dataset
# ---------------------------------------------------------------------------

def run_extensions_for_dataset(
    model_ctor, dataset_name: str, train_ds, val_ds,
    config: MACCLIPConfig, args, seeds: List[int], builders: Dict,
) -> Dict:
    """Run the full BMVC/CVPR evaluation suite for one dataset."""
    result: Dict = {"dataset": dataset_name}
    t0 = time.time()

    # 1) Multi-seed training
    if args.run_multi_seed:
        ms = run_multi_seed_dataset(model_ctor, dataset_name, train_ds, val_ds,
                                    config, args, seeds)
        result.update({k: ms[k] for k in ("summary", "per_seed_metrics", "per_image_pool")})
        model = ms["last_model"]
    else:
        set_seed(seeds[0])
        model = model_ctor().cuda()
        model.set_dataset_concepts(dataset_name)
        all_tmp: Dict = {}
        run_single_dataset(model, dataset_name, train_ds, val_ds, config, args, all_tmp)
        dice = all_tmp[dataset_name]["dice"]
        result["summary"] = {"dice": {"mean": dice, "std": 0.0, "n_seeds": 1}}
        from torch.utils.data import DataLoader
        vl = DataLoader(val_ds, batch_size=min(config.batch_size, len(val_ds)), shuffle=False)
        result["per_image_pool"] = collect_per_image_metrics(
            model, vl, config, use_tta=False, tag="single")

    eff_bs = min(config.batch_size, len(val_ds))
    from torch.utils.data import DataLoader
    vl_eval = DataLoader(val_ds, batch_size=eff_bs, shuffle=False,
                         num_workers=min(2, config.num_workers))

    # 2) Calibration + uncertainty
    if args.run_calibration:
        result["calibration"] = run_calibration(
            model, vl_eval, config, dataset_name,
            _dir_calibration(config, dataset_name))
    if args.run_uncertainty:
        result["uncertainty"] = run_uncertainty_quality(
            model, vl_eval, config, dataset_name,
            _dir_uncertainty(config, dataset_name))

    # 3) Robustness
    if args.run_robustness:
        result["robustness_rows"] = run_robustness(
            model, val_ds, config, dataset_name,
            _dir_robustness(config, dataset_name))

    # 4) Prompt sensitivity
    if args.run_prompts:
        result["prompt_rows"] = run_prompt_sensitivity(
            model, val_ds, dataset_name, config,
            _dir_prompts(config, dataset_name))

    # 5) Hyper-parameter sensitivity
    if args.run_sensitivity:
        result["sensitivity_rows"] = run_sensitivity_inference(
            model, val_ds, dataset_name, config,
            _dir_sensitivity(config, dataset_name))

    # 6) Failure-case + visualisations
    if args.run_visualizations:
        viz_dir = _dir_visualizations(config, dataset_name)
        visualize_predictions(model, vl_eval, str(viz_dir),
                              n=config.n_visualizations, config=config,
                              use_tta=False, dataset_name=dataset_name)

    # 7) Cross-dataset transfer
    if args.run_cross_dataset:
        targets = config.cross_dataset_pairs.get(dataset_name, [])
        cd_rows = []
        for tgt in targets:
            if tgt not in builders:
                continue
            try:
                tgt_pair = builders[tgt]()
                if tgt_pair is None or tgt_pair[0] is None or len(tgt_pair[1]) == 0:
                    continue
                row = run_cross_dataset(model, dataset_name, tgt, tgt_pair[1], config)
                cd_rows.append(row)
                print(f"    [cross] {dataset_name} → {tgt}: "
                      f"Dice={row['dice']:.4f}  IoU={row['iou']:.4f}")
            except Exception as e:
                print(f"    [cross-skip] {dataset_name} → {tgt}: {e}")
        write_csv(_dir_cross_dataset(config) / f"{dataset_name}_cross.csv", cd_rows)
        result["cross_dataset_rows"] = cd_rows
        model.set_dataset_concepts(dataset_name)
        model._concept_cache = {}

    # 8) Few-shot
    if args.run_few_shot:
        result["few_shot_rows"] = run_few_shot(
            model_ctor, dataset_name, train_ds, val_ds, config,
            _dir_few_shot(config, dataset_name) if hasattr(config, "out_dir_root")
            else Path(config.results_dir) / dataset_name / "few_shot")

    print(f"\n  ⏱  {dataset_name} extensions: {(time.time()-t0)/60:.1f} min")
    result["model"] = model
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    # Works in both script and notebook environments
    argv = [] if hasattr(sys, "ps1") or "ipykernel" in sys.modules else None
    args = parser.parse_args(argv)
    if args.ablation:
        args.run_ablation = True

    # ── Build config ─────────────────────────────────────────────────────
    config = MACCLIPConfig()
    if args.datasets_root:
        config.datasets_root = args.datasets_root
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.out_dir_root:
        config.out_dir_root = args.out_dir_root
    if args.patience:
        config.early_stop_patience = args.patience
    if args.safe_gpu:
        config.safe_gpu = True
        print("\n  [safe_gpu] num_workers=0, no persistent workers, no pin_memory")

    seeds = list(config.seeds_list)[: max(1, args.num_seeds)]

    # ── Apply --fast shortcut ─────────────────────────────────────────────
    if args.fast:
        print("\n  [fast] Disabling few-shot, cross-dataset, representation, lora-sweep")
        args.run_few_shot       = False
        args.run_cross_dataset  = False
        args.run_representation = False
        args.run_lora_sweep     = False

    # ── Smoke mode ────────────────────────────────────────────────────────
    if args.mode == "smoke":
        print("\n" + "!" * 70)
        print("  SMOKE-TEST MODE — pipeline check only; metrics not meaningful")
        print("!" * 70)
        seeds = seeds[:1]
        for attr in ("run_lora_sweep", "run_representation", "run_few_shot",
                     "run_cross_dataset", "run_robustness", "run_significance"):
            setattr(args, attr, False)

    # ── Print configuration ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MAC-CLIP — BMVC/CVPR Validation Suite")
    print("=" * 70)
    print(f"  Output root   : {config.out_dir_root}")
    print(f"  Seeds         : {seeds}")
    print(f"  Mode          : {args.mode}")
    print(f"  Epochs (max)  : {config.num_epochs}")
    print(f"  Early-stop    : patience={config.early_stop_patience}  "
          f"min_delta={config.early_stop_min_delta}")
    for attr in sorted(a for a in dir(args) if a.startswith("run_")):
        print(f"  {attr:<22}: {getattr(args, attr)}")
    print("=" * 70)

    set_seed(seeds[0])
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    # ── Dataset registry ──────────────────────────────────────────────────
    builders = get_all_datasets(config.datasets_root, config)
    datasets_to_run = (
        list(builders.keys()) if args.dataset == "all"
        else [d.strip() for d in args.dataset.split(",")]
    )

    # ── Computational profile (one-time) ──────────────────────────────────
    if args.run_profile:
        print("\n[Profiling] Computational cost…")
        try:
            prof_model = MACCLIPv4(config).cuda()
            prof = profile_model(prof_model, image_size=config.image_size)
            print(f"  Params (train/total): {prof['params_trainable']:,} / "
                  f"{prof['params_total']:,}  ({prof['trainable_pct']:.2f}%)")
            if not math.isnan(prof.get("flops_g", float("nan"))):
                print(f"  FLOPs : {prof['flops_g']:.2f} G")
            print(f"  Latency: {prof['latency_ms']:.1f} ms  |  FPS: {prof['fps']:.1f}  |  "
                  f"Peak VRAM: {prof['peak_mem_mb']:.0f} MB")
            comp_dir = _dir_computational(config)
            with open(comp_dir / "macclip_profile.json", "w") as f:
                json.dump(prof, f, indent=2)
            write_csv(comp_dir / "macclip_profile.csv", [prof])
            del prof_model; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [profile-skip] {e}")

    # ── Per-dataset loop ──────────────────────────────────────────────────
    def _model_ctor():
        return MACCLIPv4(config)

    per_ds_summary: Dict = {}
    cross_global:   List = []
    t0_all = time.time()

    for ds_key in datasets_to_run:
        if ds_key not in builders:
            print(f"\n[SKIP] Unknown dataset: {ds_key}")
            continue

        print(f"\n{'='*70}\nBuilding dataset: {ds_key}\n{'='*70}")
        try:
            pair = builders[ds_key]()
        except Exception as e:
            print(f"  [ERROR] Building {ds_key}: {e}")
            _hard_gpu_cleanup(); continue
        if pair is None or pair[0] is None:
            print(f"  [SKIP] {ds_key}: builder returned None"); continue
        train_ds, val_ds = pair
        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"  [SKIP] {ds_key}: empty split"); continue

        last_err = None
        for attempt in range(3):
            try:
                ds_result = run_extensions_for_dataset(
                    _model_ctor, ds_key, train_ds, val_ds,
                    config, args, seeds, builders)
                per_ds_summary[ds_key] = ds_result
                cross_global.extend(ds_result.get("cross_dataset_rows", []))
                ds_result.pop("model", None)
                last_err = None
                break
            except Exception as e:
                import traceback
                last_err = e
                print(f"\n  [ERROR attempt {attempt+1}/3] {ds_key}: {e}")
                if _is_nvml_error(e) and attempt < 2:
                    print("  → NVML error; deep cleanup + retry…")
                    _hard_gpu_cleanup(deep=True)
                else:
                    traceback.print_exc()
                    break

        try:
            del train_ds, val_ds, pair
        except Exception:
            pass
        _hard_gpu_cleanup(deep=(last_err is not None and _is_nvml_error(last_err)))

    # ── Global aggregation ────────────────────────────────────────────────
    if per_ds_summary:
        print("\n\n" + "=" * 70)
        print("  GLOBAL AGGREGATION")
        print("=" * 70)
        aggregate_all_datasets(per_ds_summary, config)

        # Cross-dataset combined CSV
        write_csv(_dir_cross_dataset(config) / "cross_dataset_all.csv", cross_global)

        # Limitations report
        write_limitations(config)

        # Pretty-print final summary
        print("\n" + "=" * 70)
        print("  FINAL SUMMARY  (mean ± std across seeds)")
        print("=" * 70)
        fmt = f"{'Dataset':<22} {'Dice':>16} {'IoU':>16} {'BF1':>16} {'HD':>10}"
        print(fmt); print("-" * 70)
        for ds in sorted(per_ds_summary.keys()):
            s  = per_ds_summary[ds].get("summary", {})
            d  = s.get("dice", {}); i = s.get("iou", {})
            b  = s.get("boundary_f1", {}); h = s.get("hausdorff", {})
            def _f(stat): return (f"{stat['mean']:.4f}±{stat['std']:.4f}"
                                  if stat else "N/A")
            print(f"{ds:<22} {_f(d):>16} {_f(i):>16} {_f(b):>16} "
                  f"{h.get('mean', float('nan')):>10.2f}")
        print("=" * 70)
        print(f"\n✓ Done.  Total elapsed: {(time.time()-t0_all)/60:.1f} min")
        print(f"  Results saved to: {config.out_dir_root}")
    else:
        print("\nNo datasets completed successfully — nothing to aggregate.")


if __name__ == "__main__":
    main()
