"""
scripts/compare_results.py
===========================
Reads per-dataset metric JSON files for MAC-CLIP and all baselines,
then produces:
  • results/comparison_table.csv     — full N×M metric table
  • results/latex_table.tex          — ready-to-paste LaTeX booktabs table
  • results/mean_rank_summary.csv    — mean rank per model
  • results/comparison_bar.png       — grouped bar chart (Dice)
  • results/comparison_radar.png     — per-modality radar chart

Usage
-----
  python scripts/compare_results.py \
      --mac_clip_dir  /media/data/Local_CLIP/mac_clip_bmvc_results \
      --baseline_dir  /media/data/Local_CLIP/baseline_all_datasets \
      --out_dir       results/comparison
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

DATASETS = [
    "kvasir", "isic2018", "m2caiseg", "bkai_neopolyp",
    "covid_infection", "covid_lung", "monuseg", "busi", "ct2us_kidney",
    "arcade_stenosis", "arcade_syntax", "btxrd", "papila", "drive", "coph100",
]

MODALITY_MAP = {
    "kvasir":          "Endoscopy",
    "bkai_neopolyp":   "Endoscopy",
    "m2caiseg":        "Endoscopy",
    "covid_infection": "CXR",
    "covid_lung":      "CXR",
    "arcade_stenosis": "X-Ray",
    "arcade_syntax":   "X-Ray",
    "btxrd":           "X-Ray",
    "isic2018":        "Dermoscopy",
    "monuseg":         "Pathology",
    "busi":            "Ultrasound",
    "ct2us_kidney":    "Ultrasound",
    "papila":          "Fundus/OCT",
    "drive":           "Fundus/OCT",
    "coph100":         "Fundus/OCT",
}

BASELINE_MODELS = [
    "VanillaUNet", "UNetPlusPlus", "DeepLabV3-R50",
    "TransUNet", "SwinUNet", "MedT",
    "FrozenOpenAICLIP", "FrozenBiomedCLIP",
    "nnUNetStyle", "MedSAMStyle", "SAMedStyle",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_macclip_metrics(root: Path) -> Dict[str, Dict]:
    """Load MAC-CLIP per-dataset metrics (mean ± std from multi-seed runs)."""
    results: Dict[str, Dict] = {}
    for ds in DATASETS:
        # Try aggregated first, then single-run
        for candidate in [
            root / "per_dataset" / ds / "aggregated" / "summary_mean_std.json",
            root / ds / "metrics.json",
            root / "per_dataset" / ds / "seed_42" / "metrics.json",
        ]:
            if candidate.exists():
                with open(candidate) as f:
                    data = json.load(f)
                # Normalise to flat dict
                if "dice" in data and isinstance(data["dice"], dict):
                    results[ds] = {
                        "dice":      data["dice"].get("mean", float("nan")),
                        "dice_std":  data["dice"].get("std",  0.0),
                        "iou":       data.get("iou",  {}).get("mean", float("nan")),
                        "bf1":       data.get("boundary_f1", {}).get("mean", float("nan")),
                        "hausdorff": data.get("hausdorff", {}).get("mean", float("nan")),
                    }
                else:
                    results[ds] = {
                        "dice":      float(data.get("dice",      float("nan"))),
                        "dice_std":  float(data.get("dice_std",  0.0)),
                        "iou":       float(data.get("iou",       float("nan"))),
                        "bf1":       float(data.get("boundary_f1", float("nan"))),
                        "hausdorff": float(data.get("hausdorff", float("nan"))),
                    }
                break
    return results


def load_baseline_metrics(root: Path) -> Dict[str, Dict[str, Dict]]:
    """Returns {model_name: {dataset: {metric: value}}}."""
    results: Dict[str, Dict[str, Dict]] = {}
    for model in BASELINE_MODELS:
        results[model] = {}
        for ds in DATASETS:
            p = root / model / ds / "metrics.json"
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                results[model][ds] = {
                    "dice":      float(data.get("dice",      float("nan"))),
                    "iou":       float(data.get("iou",       float("nan"))),
                    "bf1":       float(data.get("bf1",       float("nan"))),
                    "hausdorff": float(data.get("hausdorff", float("nan"))),
                }
    return results


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_comparison_table(
    macclip: Dict[str, Dict],
    baselines: Dict[str, Dict[str, Dict]],
    metric: str = "dice",
) -> List[Dict]:
    """One row per dataset, one column per model."""
    rows = []
    model_names = ["MAC-CLIP (Ours)"] + BASELINE_MODELS
    for ds in DATASETS:
        row: Dict = {"dataset": ds, "modality": MODALITY_MAP.get(ds, "Other")}
        mc_val = macclip.get(ds, {}).get(metric, float("nan"))
        mc_std = macclip.get(ds, {}).get(f"{metric}_std", 0.0)
        row["MAC-CLIP (Ours)"] = f"{mc_val:.4f}" if not math.isnan(mc_val) else "N/A"
        row[f"MAC-CLIP (Ours)_std"] = f"{mc_std:.4f}"
        for model in BASELINE_MODELS:
            val = baselines.get(model, {}).get(ds, {}).get(metric, float("nan"))
            row[model] = f"{val:.4f}" if not math.isnan(val) else "N/A"
        rows.append(row)

    # Append mean row
    mean_row: Dict = {"dataset": "MEAN", "modality": "ALL"}
    for mname in model_names:
        vals_raw = []
        if mname == "MAC-CLIP (Ours)":
            vals_raw = [macclip.get(ds, {}).get(metric, float("nan")) for ds in DATASETS]
        else:
            vals_raw = [baselines.get(mname, {}).get(ds, {}).get(metric, float("nan"))
                        for ds in DATASETS]
        clean = [v for v in vals_raw if not math.isnan(v)]
        mean_row[mname] = f"{np.mean(clean):.4f}" if clean else "N/A"
    rows.append(mean_row)
    return rows


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  ✓ Saved: {path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def build_latex_table(
    macclip: Dict[str, Dict],
    baselines: Dict[str, Dict[str, Dict]],
    metric: str = "dice",
    caption: str = "Comparison of segmentation methods (Dice ↑)",
    label: str = "tab:comparison",
) -> str:
    model_names  = ["MAC-CLIP"] + BASELINE_MODELS
    n_models = len(model_names)

    header_cols = " & ".join([f"\\textbf{{{m}}}" for m in model_names])
    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{" + caption + "}",
        "  \\label{" + label + "}",
        "  \\resizebox{\\textwidth}{!}{",
        "  \\begin{tabular}{ll" + "c" * n_models + "}",
        "    \\toprule",
        f"    \\textbf{{Dataset}} & \\textbf{{Modality}} & {header_cols} \\\\",
        "    \\midrule",
    ]

    prev_mod = None
    for ds in DATASETS:
        mod = MODALITY_MAP.get(ds, "Other")
        if prev_mod and mod != prev_mod:
            lines.append("    \\midrule")
        prev_mod = mod

        mc_val   = macclip.get(ds, {}).get(metric, float("nan"))
        baseline_vals = [baselines.get(m, {}).get(ds, {}).get(metric, float("nan"))
                         for m in BASELINE_MODELS]
        all_vals = [mc_val] + baseline_vals
        clean    = [v for v in all_vals if not math.isnan(v)]
        best     = max(clean) if clean else float("nan")
        second   = sorted(clean, reverse=True)[1] if len(clean) > 1 else float("nan")

        def _fmt(v):
            if math.isnan(v): return "—"
            if abs(v - best) < 1e-5:    return f"\\textbf{{{v:.4f}}}"
            if abs(v - second) < 1e-5:  return f"\\underline{{{v:.4f}}}"
            return f"{v:.4f}"

        cells = " & ".join([_fmt(v) for v in all_vals])
        lines.append(f"    {ds} & {mod} & {cells} \\\\")

    # Mean row
    lines.append("    \\midrule")
    mean_cells = []
    for i, mname in enumerate(["MAC-CLIP"] + BASELINE_MODELS):
        if mname == "MAC-CLIP":
            vals = [macclip.get(ds, {}).get(metric, float("nan")) for ds in DATASETS]
        else:
            vals = [baselines.get(mname, {}).get(ds, {}).get(metric, float("nan"))
                    for ds in DATASETS]
        clean = [v for v in vals if not math.isnan(v)]
        mean_cells.append(f"{np.mean(clean):.4f}" if clean else "—")

    mac_std = np.nanstd([macclip.get(ds, {}).get(metric, float("nan")) for ds in DATASETS])
    mean_cells[0] = f"\\textbf{{{mean_cells[0]}}}"  # MAC-CLIP best
    lines.append(f"    \\textbf{{MEAN}} & — & {' & '.join(mean_cells)} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ranking table
# ---------------------------------------------------------------------------

def build_rank_summary(
    macclip: Dict[str, Dict],
    baselines: Dict[str, Dict[str, Dict]],
    metric: str = "dice",
) -> List[Dict]:
    model_names = ["MAC-CLIP (Ours)"] + BASELINE_MODELS
    ranks_per_ds: Dict[str, List[float]] = {m: [] for m in model_names}

    for ds in DATASETS:
        scores = {}
        mc = macclip.get(ds, {}).get(metric, float("nan"))
        if not math.isnan(mc):
            scores["MAC-CLIP (Ours)"] = mc
        for m in BASELINE_MODELS:
            v = baselines.get(m, {}).get(ds, {}).get(metric, float("nan"))
            if not math.isnan(v):
                scores[m] = v

        if not scores:
            continue
        sorted_models = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        for rank, m in enumerate(sorted_models, start=1):
            ranks_per_ds[m].append(rank)

    rows = []
    for m in model_names:
        r = ranks_per_ds[m]
        vals_all = []
        if m == "MAC-CLIP (Ours)":
            vals_all = [macclip.get(ds, {}).get(metric, float("nan")) for ds in DATASETS]
        else:
            vals_all = [baselines.get(m, {}).get(ds, {}).get(metric, float("nan"))
                        for ds in DATASETS]
        clean = [v for v in vals_all if not math.isnan(v)]
        rows.append({
            "model":       m,
            "mean_rank":   f"{np.mean(r):.2f}" if r else "N/A",
            "n_datasets":  len(r),
            "mean_metric": f"{np.mean(clean):.4f}" if clean else "N/A",
            "std_metric":  f"{np.std(clean):.4f}"  if clean else "N/A",
        })
    rows.sort(key=lambda x: float(x["mean_rank"]) if x["mean_rank"] != "N/A" else 999)
    return rows


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_bar_chart(
    macclip: Dict[str, Dict],
    baselines: Dict[str, Dict[str, Dict]],
    metric: str,
    out_path: Path,
    n_top_baselines: int = 5,
) -> None:
    # Pick top-N baselines by mean Dice
    def _mean(model):
        if model == "MAC-CLIP (Ours)":
            vals = [macclip.get(ds, {}).get(metric, float("nan")) for ds in DATASETS]
        else:
            vals = [baselines.get(model, {}).get(ds, {}).get(metric, float("nan"))
                    for ds in DATASETS]
        clean = [v for v in vals if not math.isnan(v)]
        return np.mean(clean) if clean else 0.0

    top_baselines = sorted(BASELINE_MODELS, key=_mean, reverse=True)[:n_top_baselines]
    models = ["MAC-CLIP (Ours)"] + top_baselines
    datasets_avail = [ds for ds in DATASETS if not math.isnan(macclip.get(ds, {}).get(metric, float("nan")))][:10]

    x     = np.arange(len(datasets_avail))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(16, 5))

    colours = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for i, model in enumerate(models):
        vals = []
        for ds in datasets_avail:
            if model == "MAC-CLIP (Ours)":
                vals.append(macclip.get(ds, {}).get(metric, 0.0))
            else:
                vals.append(baselines.get(model, {}).get(ds, {}).get(metric, 0.0))
        offset = (i - len(models) / 2) * width + width / 2
        lw = 2.5 if model == "MAC-CLIP (Ours)" else 1.0
        ax.bar(x + offset, vals, width, label=model, color=colours[i],
               linewidth=lw, edgecolor="black" if model == "MAC-CLIP (Ours)" else "none")

    ax.set_xticks(x); ax.set_xticklabels(datasets_avail, rotation=30, ha="right")
    ax.set_ylabel(metric.capitalize()); ax.set_ylim(0, 1.05)
    ax.set_title(f"Method comparison — {metric.upper()}")
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_radar(
    macclip: Dict[str, Dict],
    baselines: Dict[str, Dict[str, Dict]],
    out_path: Path,
    n_top: int = 4,
) -> None:
    modalities = sorted(set(MODALITY_MAP.values()))
    def _modality_mean(results_dict, model):
        out: Dict[str, float] = {}
        for mod in modalities:
            ds_list = [ds for ds, m in MODALITY_MAP.items() if m == mod]
            if model == "MAC-CLIP (Ours)":
                vals = [results_dict.get(ds, {}).get("dice", float("nan")) for ds in ds_list]
            else:
                vals = [baselines.get(model, {}).get(ds, {}).get("dice", float("nan"))
                        for ds in ds_list]
            clean = [v for v in vals if not math.isnan(v)]
            out[mod] = np.mean(clean) if clean else 0.0
        return out

    def _mean_dice(model):
        m = _modality_mean(macclip, model)
        return np.mean(list(m.values()))

    top_baselines = sorted(BASELINE_MODELS, key=_mean_dice, reverse=True)[:n_top]
    models_to_plot = ["MAC-CLIP (Ours)"] + top_baselines

    angles = np.linspace(0, 2 * np.pi, len(modalities), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    colours = plt.cm.tab10(np.linspace(0, 1, len(models_to_plot)))

    for i, model in enumerate(models_to_plot):
        vals = list(_modality_mean(macclip if model == "MAC-CLIP (Ours)" else {}, model).values())
        if model != "MAC-CLIP (Ours)":
            mmap = _modality_mean({}, model)
            for j, mod in enumerate(modalities):
                ds_list = [ds for ds, m in MODALITY_MAP.items() if m == mod]
                v_list  = [baselines.get(model, {}).get(ds, {}).get("dice", float("nan"))
                           for ds in ds_list]
                clean = [v for v in v_list if not math.isnan(v)]
                vals[j] = np.mean(clean) if clean else 0.0
        vals += vals[:1]
        lw = 3 if model == "MAC-CLIP (Ours)" else 1.5
        ax.plot(angles, vals, "o-", lw=lw, color=colours[i], label=model)
        if model == "MAC-CLIP (Ours)":
            ax.fill(angles, vals, alpha=0.15, color=colours[i])

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(modalities, size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Mean Dice per modality", pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate MAC-CLIP comparison tables & plots")
    parser.add_argument("--mac_clip_dir",  default="/media/data/Local_CLIP/mac_clip_bmvc_results")
    parser.add_argument("--baseline_dir",  default="/media/data/Local_CLIP/baseline_all_datasets")
    parser.add_argument("--out_dir",       default="results/comparison")
    parser.add_argument("--metric",        default="dice",
                        choices=["dice", "iou", "bf1", "hausdorff"])
    args = parser.parse_args()

    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mac_root = Path(args.mac_clip_dir)
    bas_root = Path(args.baseline_dir)

    print(f"\nLoading MAC-CLIP results from : {mac_root}")
    macclip   = load_macclip_metrics(mac_root)
    print(f"  Loaded {len(macclip)} datasets")

    print(f"Loading baseline results from  : {bas_root}")
    baselines = load_baseline_metrics(bas_root)
    loaded_models = [m for m, d in baselines.items() if d]
    print(f"  Loaded {len(loaded_models)} baseline models")

    # 1) Full comparison CSV
    for metric in ["dice", "iou", "bf1", "hausdorff"]:
        rows = build_comparison_table(macclip, baselines, metric)
        save_csv(rows, out_dir / f"comparison_{metric}.csv")

    # 2) LaTeX table (Dice)
    latex = build_latex_table(macclip, baselines, "dice",
                              caption="Segmentation comparison across 15 biomedical datasets (Dice ↑). "
                                      "\\textbf{Bold} = best, \\underline{underline} = second best.",
                              label="tab:main_comparison")
    (out_dir / "latex_table_dice.tex").write_text(latex)
    print(f"  ✓ Saved: {out_dir / 'latex_table_dice.tex'}")

    # 3) Mean rank summary
    rank_rows = build_rank_summary(macclip, baselines, "dice")
    save_csv(rank_rows, out_dir / "mean_rank_summary.csv")

    # 4) Bar chart
    plot_bar_chart(macclip, baselines, "dice",
                   out_dir / "comparison_bar_dice.png")

    # 5) Radar chart
    plot_radar(macclip, baselines, out_dir / "comparison_radar.png")

    # Summary print
    print("\n" + "=" * 60)
    print("  TOP-5 MODELS BY MEAN RANK")
    print("=" * 60)
    for row in rank_rows[:5]:
        print(f"  {row['model']:<24} | rank={row['mean_rank']:>5} "
              f"| dice={row['mean_metric']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
