"""
Preprocessing & Dataset Builders
=================================
Universal dataset class and per-dataset builder functions for all 15
biomedical segmentation datasets used in MAC-CLIP.

Supported datasets
------------------
  Endoscopy   : Kvasir-SEG, m2caiSeg, BKAI-IGH NeoPolyp
  CXR         : COVID-QU-Ex (infection + lung)
  X-Ray       : ARCADE (stenosis + syntax), BTXRD
  Ultrasound  : BUSI, CT2US-Kidney
  Fundus/OCT  : DRIVE, PAPILA, COph100
  Dermoscopy  : ISIC 2018
  Pathology   : MoNuSeg
"""

import random
import json
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from mac_clip.config import MACCLIPConfig
from mac_clip.dataset_registry import DATASET_CONCEPTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_paths(
    image_paths: List[str],
    mask_paths: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Reproducible train / val split when no official split exists."""
    n = len(image_paths)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    n_train = int(n * train_ratio)
    tr, va = idx[:n_train], idx[n_train:]
    return (
        [image_paths[i] for i in tr], [mask_paths[i] for i in tr],
        [image_paths[i] for i in va], [mask_paths[i] for i in va],
    )


# ---------------------------------------------------------------------------
# Universal Dataset
# ---------------------------------------------------------------------------

class UniversalSegDataset(Dataset):
    """Loads any (image, mask) pair for all supported datasets.

    Handles:
      * Adaptive mask binarisation ({0,1} vs {0,255} conventions)
      * NIfTI masks (COph100)
      * Spatial size mismatches between image and mask
      * Per-dataset clinical concept prompts
    """

    # CLIP normalisation constants
    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        split: str = "train",
        config: Optional[MACCLIPConfig] = None,
        dataset_name: str = "generic",
    ):
        self.image_paths  = image_paths
        self.mask_paths   = mask_paths
        self.split        = split
        self.config       = config or MACCLIPConfig()
        self.dataset_name = dataset_name

        concepts_cfg           = DATASET_CONCEPTS.get(dataset_name, DATASET_CONCEPTS["kvasir"])
        self.clinical_concepts = concepts_cfg["clinical_concepts"]

        sz = self.config.image_size
        mn, sd = self._MEAN, self._STD

        if split == "train":
            self.transform = A.Compose([
                A.Resize(sz, sz),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.4),
                A.PadIfNeeded(sz, sz, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                A.CenterCrop(sz, sz),
                A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
                A.HueSaturationValue(10, 30, 20, p=0.4),
                A.ElasticTransform(alpha=120, sigma=6, alpha_affine=4, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.CoarseDropout(max_holes=4, max_height=32, max_width=32, fill_value=0, p=0.2),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.Normalize(mean=mn, std=sd),
                ToTensorV2(),
            ], is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(sz, sz),
                A.Normalize(mean=mn, std=sd),
                ToTensorV2(),
            ], is_check_shapes=False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask, handling {0,1} / {0,255} / NIfTI conventions."""
        sz = self.config.image_size
        if path is None or not Path(path).exists():
            return np.zeros((sz, sz), dtype=np.float32)

        ext = Path(path).suffix.lower()

        # NIfTI (COph100)
        if ext in (".gz", ".nii"):
            try:
                import nibabel as nib
                vol = nib.load(path).get_fdata()
                if vol.ndim == 3:
                    vol = vol[:, :, 0]
                return (vol > 0).astype(np.float32)
            except Exception:
                return np.zeros((sz, sz), dtype=np.float32)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.zeros((sz, sz), dtype=np.float32)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        m = img.max()
        if m <= 1:
            return (img > 0).astype(np.float32)
        return (img > m / 2).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        img_path  = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path)
        if img is None:
            sz = self.config.image_size
            img = np.zeros((sz, sz, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self._load_mask(mask_path)

        if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
            h, w = img.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        aug = self.transform(image=img, mask=mask)
        return {
            "image":      aug["image"],
            "mask":       aug["mask"],
            "text":       random.choice(self.clinical_concepts),
            "image_path": img_path,
        }


# ---------------------------------------------------------------------------
# Dataset Builders
# ---------------------------------------------------------------------------

def build_kvasir_dataset(root: str, config: MACCLIPConfig):
    """Kvasir-SEG polyp segmentation."""
    ip = Path(root) / "images"
    mp = Path(root) / "masks"
    imgs = sorted(ip.glob("*.jpg"))
    n_tr = int(0.8 * len(imgs))
    tr_i = [str(f) for f in imgs[:n_tr]]
    va_i = [str(f) for f in imgs[n_tr:]]
    tr_m = [str(mp / (f.stem + ".jpg")) for f in imgs[:n_tr]]
    va_m = [str(mp / (f.stem + ".jpg")) for f in imgs[n_tr:]]
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "kvasir"),
        UniversalSegDataset(va_i, va_m, "val",   config, "kvasir"),
    )


def build_isic2018_dataset(root: str, config: MACCLIPConfig):
    """ISIC 2018 skin-lesion segmentation (Task 1 binary masks preferred)."""
    root_candidates = [
        Path(root) / "ISIC_2018 (2)",
        Path(root) / "ISIC_2018",
        Path(root) / "ISIC2018",
    ]
    seg_root = next(
        (c for c in root_candidates if (c / "ISIC2018_Task1-2_Training_Input").exists()),
        None,
    )
    if seg_root is None:
        print(f"  [ISIC2018] root not found under {root}")
        return None, None

    img_dir  = seg_root / "ISIC2018_Task1-2_Training_Input"
    task1    = seg_root / "ISIC2018_Task1_Training_GroundTruth"
    task2    = seg_root / "ISIC2018_Task2_Training_GroundTruth_v3"
    msk_dir  = task1 if task1.exists() else (task2 if task2.exists() else None)
    if msk_dir is None:
        return None, None

    use_task1 = (task1 == msk_dir)
    image_paths, mask_paths = [], []
    for img_f in sorted(img_dir.glob("*.jpg")):
        stem = img_f.stem
        mp = (msk_dir / f"{stem}_segmentation.png") if use_task1 else None
        if mp is None:
            candidates = list(msk_dir.glob(f"{stem}*.png"))
            mp = candidates[0] if candidates else None
        if mp and mp.exists():
            image_paths.append(str(img_f))
            mask_paths.append(str(mp))

    if not image_paths:
        return None, None
    tr_i, tr_m, va_i, va_m = _split_paths(image_paths, mask_paths)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "isic2018"),
        UniversalSegDataset(va_i, va_m, "val",   config, "isic2018"),
    )


def build_m2caiseg_dataset(root: str, config: MACCLIPConfig):
    """m2caiSeg surgical instrument segmentation."""
    base = Path(root) / "Endoscopy" / "m2caiSeg" / "m2caiSeg dataset"

    def _collect(img_dir: Path, gt_dir: Path):
        ips, mps = [], []
        if not img_dir.exists():
            return ips, mps
        for f in sorted(img_dir.glob("*.jpg")):
            gf = gt_dir / (f.stem + "_gt.png")
            if gf.exists():
                ips.append(str(f))
                mps.append(str(gf))
        return ips, mps

    tr_i, tr_m = _collect(base / "train" / "images", base / "train" / "groundtruth")
    va_i, va_m = _collect(base / "test"  / "images", base / "test"  / "groundtruth")

    if not tr_i:
        all_i, all_m = _collect(base / "trainval" / "images", base / "trainval" / "groundtruth")
        if not all_i:
            return None, None
        tr_i, tr_m, va_i, va_m = _split_paths(all_i, all_m)

    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "m2caiseg"),
        UniversalSegDataset(va_i, va_m, "val",   config, "m2caiseg"),
    )


def build_bkai_neopolyp_dataset(root: str, config: MACCLIPConfig):
    """BKAI-IGH NeoPolyp segmentation."""
    base   = Path(root) / "Endoscopy" / "bkai-igh-neopolyp"
    tr_img = base / "train" / "train"
    tr_gt  = base / "train_gt" / "train_gt"

    ips, mps = [], []
    for ext in ["*.jpeg", "*.jpg", "*.png"]:
        for f in sorted(tr_img.glob(ext)):
            for gext in [".jpeg", ".jpg", ".png"]:
                gf = tr_gt / (f.stem + gext)
                if gf.exists():
                    ips.append(str(f))
                    mps.append(str(gf))
                    break

    if not ips:
        return None, None
    tr_i, tr_m, va_i, va_m = _split_paths(ips, mps)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "bkai_neopolyp"),
        UniversalSegDataset(va_i, va_m, "val",   config, "bkai_neopolyp"),
    )


def _collect_covid_quex(split_dir: Path, category: str, seg_type: str):
    img_dir = split_dir / category / "images"
    msk_dir = split_dir / category / ("infection masks" if seg_type == "infection" else "lung masks")
    ips, mps = [], []
    if img_dir.exists():
        for f in sorted(img_dir.glob("*.png")):
            mf = msk_dir / f.name
            if mf.exists():
                ips.append(str(f))
                mps.append(str(mf))
    return ips, mps


def build_covid_infection_dataset(root: str, config: MACCLIPConfig):
    """COVID-QU-Ex infection segmentation."""
    base = Path(root) / "CXR" / "COVID-QU-Ex" / "Infection Segmentation Data" / "Infection Segmentation Data"
    tr_i, tr_m, va_i, va_m = [], [], [], []
    for cat in ["COVID-19", "Non-COVID", "Normal"]:
        a, b = _collect_covid_quex(base / "Train", cat, "infection")
        c, d = _collect_covid_quex(base / "Val",   cat, "infection")
        tr_i += a; tr_m += b; va_i += c; va_m += d
    if not tr_i:
        return None, None
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "covid_infection"),
        UniversalSegDataset(va_i, va_m, "val",   config, "covid_infection"),
    )


def build_covid_lung_dataset(root: str, config: MACCLIPConfig):
    """COVID-QU-Ex lung segmentation."""
    base = Path(root) / "CXR" / "COVID-QU-Ex" / "Lung Segmentation Data" / "Lung Segmentation Data"
    tr_i, tr_m, va_i, va_m = [], [], [], []
    for cat in ["COVID-19", "Non-COVID", "Normal"]:
        a, b = _collect_covid_quex(base / "Train", cat, "lung")
        c, d = _collect_covid_quex(base / "Val",   cat, "lung")
        tr_i += a; tr_m += b; va_i += c; va_m += d
    if not tr_i:
        return None, None
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "covid_lung"),
        UniversalSegDataset(va_i, va_m, "val",   config, "covid_lung"),
    )


def build_monuseg_dataset(root: str, config: MACCLIPConfig):
    """MoNuSeg nuclear instance segmentation."""
    base = Path(root) / "Pathology" / "MonuSeg"

    def _collect(idir: Path, mdir: Path):
        ips, mps = [], []
        if not idir.exists():
            return ips, mps
        for ext in ["*.tif", "*.png", "*.jpg"]:
            for f in sorted(idir.glob(ext)):
                for mext in [".png", " .png"]:
                    mf = mdir / (f.stem + mext)
                    if mf.exists():
                        ips.append(str(f)); mps.append(str(mf)); break
                else:
                    cands = list(mdir.glob(f.stem + "*.png"))
                    if cands:
                        ips.append(str(f)); mps.append(str(cands[0]))
        return ips, mps

    tr_i, tr_m = _collect(
        base / "kmms_training" / "kmms_training" / "images",
        base / "kmms_training" / "kmms_training" / "masks",
    )
    va_i, va_m = _collect(
        base / "kmms_test" / "kmms_test" / "images",
        base / "kmms_test" / "kmms_test" / "masks",
    )
    if not tr_i:
        return None, None
    if not va_i:
        tr_i, tr_m, va_i, va_m = _split_paths(tr_i, tr_m)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "monuseg"),
        UniversalSegDataset(va_i, va_m, "val",   config, "monuseg"),
    )


def build_busi_dataset(root: str, config: MACCLIPConfig):
    """BUSI breast ultrasound segmentation."""
    base = Path(root) / "Ultrasound" / "BUSI" / "Dataset_BUSI_with_GT"
    ips, mps = [], []
    for cat in ["benign", "malignant"]:
        cat_dir = base / cat
        for ext in ["*.png", "*.jpg"]:
            for f in sorted((cat_dir).glob(ext)):
                if "_mask" not in f.stem:
                    mf = cat_dir / (f.stem + "_mask.png")
                    if mf.exists():
                        ips.append(str(f)); mps.append(str(mf))
    if not ips:
        return None, None
    tr_i, tr_m, va_i, va_m = _split_paths(ips, mps)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "busi"),
        UniversalSegDataset(va_i, va_m, "val",   config, "busi"),
    )


def build_ct2us_kidney_dataset(root: str, config: MACCLIPConfig):
    """CT2US Kidney segmentation."""
    img_dir = Path(root) / "Ultrasound" / "CT2USforKidneySeg" / "slice" / "slice"
    msk_dir = Path(root) / "Ultrasound" / "CT2USforKidneySeg" / "mask" / "mask"
    ips, mps = [], []
    if img_dir.exists() and msk_dir.exists():
        for f in sorted(img_dir.glob("*.png")):
            mf = msk_dir / f.name
            if mf.exists():
                ips.append(str(f)); mps.append(str(mf))
    if not ips:
        return None, None
    tr_i, tr_m, va_i, va_m = _split_paths(ips, mps)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "ct2us_kidney"),
        UniversalSegDataset(va_i, va_m, "val",   config, "ct2us_kidney"),
    )


def build_arcade_dataset(root: str, task: str, config: MACCLIPConfig):
    """ARCADE coronary artery dataset (task = 'stenosis' | 'syntax')."""
    base   = Path(root) / "X-ray" / "ARCADE" / "arcade" / task
    ds_key = f"arcade_{task}"
    tr_i, tr_m, va_i, va_m = [], [], [], []

    for split_name in ["train", "val"]:
        idir     = base / split_name / "images"
        ann_file = base / split_name / "annotations" / f"{split_name}.json"
        if not ann_file.exists():
            continue
        with open(ann_file) as f:
            ann = json.load(f)

        mask_cache = base / f"mask_cache_{task}_{split_name}"
        mask_cache.mkdir(exist_ok=True)

        from collections import defaultdict
        ann_map = defaultdict(list)
        for a in ann.get("annotations", []):
            ann_map[a["image_id"]].append(a)

        for img_info in ann.get("images", []):
            img_file  = idir / img_info["file_name"]
            if not img_file.exists():
                continue
            mask_file = mask_cache / (Path(img_info["file_name"]).stem + ".png")
            if not mask_file.exists():
                h, w = img_info["height"], img_info["width"]
                mask = np.zeros((h, w), dtype=np.uint8)
                for a in ann_map.get(img_info["id"], []):
                    seg = a.get("segmentation", [])
                    if isinstance(seg, list):
                        for poly in seg:
                            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                cv2.imwrite(str(mask_file), mask)

            dest = (tr_i, tr_m) if split_name == "train" else (va_i, va_m)
            dest[0].append(str(img_file))
            dest[1].append(str(mask_file))

    if not tr_i:
        return None, None
    if not va_i:
        tr_i, tr_m, va_i, va_m = _split_paths(tr_i, tr_m)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, ds_key),
        UniversalSegDataset(va_i, va_m, "val",   config, ds_key),
    )


def build_btxrd_dataset(root: str, config: MACCLIPConfig):
    """BTXRD bone-tumour X-ray dataset."""
    base      = Path(root) / "X-ray" / "BTXRD (1)" / "BTXRD"
    img_dir   = base / "images"
    ann_dir   = base / "Annotations"
    mask_cache = base / "mask_cache"
    mask_cache.mkdir(exist_ok=True)

    if not img_dir.exists() or not ann_dir.exists():
        return None, None

    ips, mps = [], []
    for ann_file in sorted(ann_dir.glob("*.json")):
        stem     = ann_file.stem
        img_file = next(
            (img_dir / (stem + ext) for ext in [".jpeg", ".jpg", ".png"]
             if (img_dir / (stem + ext)).exists()), None
        )
        if img_file is None:
            continue
        mask_file = mask_cache / (stem + ".png")
        if not mask_file.exists():
            try:
                with open(ann_file) as f:
                    ann = json.load(f)
                img_cv = cv2.imread(str(img_file))
                if img_cv is None:
                    continue
                h, w   = img_cv.shape[:2]
                mask   = np.zeros((h, w), dtype=np.uint8)
                shapes = ann.get("shapes", ann.get("annotations", []))
                for shape in shapes if isinstance(shapes, list) else []:
                    pts = shape.get("points", shape.get("segmentation", []))
                    if pts and isinstance(pts[0], list):
                        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
                cv2.imwrite(str(mask_file), mask)
            except Exception:
                continue
        if mask_file.exists():
            ips.append(str(img_file))
            mps.append(str(mask_file))

    if not ips:
        return None, None
    tr_i, tr_m, va_i, va_m = _split_paths(ips, mps)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "btxrd"),
        UniversalSegDataset(va_i, va_m, "val",   config, "btxrd"),
    )


def build_papila_dataset(root: str, config: MACCLIPConfig):
    """PAPILA optic-disc / cup segmentation."""
    base      = Path(root) / "Fundus or OCT" / "PAPILA"
    papila_db = next(base.glob("PapilaDB-*"), None)
    if papila_db is None:
        return None, None

    img_dir    = papila_db / "FundusImages"
    cont_dir   = papila_db / "ExpertsSegmentations" / "Contours"
    mask_cache = papila_db / "mask_cache"
    mask_cache.mkdir(exist_ok=True)

    ips, mps = [], []
    for img_f in sorted(img_dir.glob("*.jpg")):
        stem       = img_f.stem
        mask_file  = mask_cache / (stem + ".png")
        cont_disc  = cont_dir / f"{stem}_disc_exp1.txt"
        if not mask_file.exists() and cont_disc.exists():
            img_cv = cv2.imread(str(img_f))
            if img_cv is None:
                continue
            h, w = img_cv.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for cf in [cont_disc, cont_dir / f"{stem}_cup_exp1.txt"]:
                if cf.exists():
                    try:
                        pts = np.loadtxt(str(cf)).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    except Exception:
                        pass
            cv2.imwrite(str(mask_file), mask)
        if mask_file.exists():
            ips.append(str(img_f))
            mps.append(str(mask_file))

    if not ips:
        return None, None
    tr_i, tr_m, va_i, va_m = _split_paths(ips, mps)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "papila"),
        UniversalSegDataset(va_i, va_m, "val",   config, "papila"),
    )


def build_drive_dataset(root: str, config: MACCLIPConfig):
    """DRIVE retinal vessel segmentation."""
    base = Path(root) / "Fundus or OCT" / "DRIVE"
    for zp, dest in [(base / "training.zip", base / "training"),
                     (base / "test.zip",     base / "test")]:
        if not dest.exists() and zp.exists():
            with zipfile.ZipFile(str(zp)) as z:
                z.extractall(str(dest.parent))

    def _collect(d: Path):
        ips, mps = [], []
        idir = d / "images"
        mdir = d / "1st_manual"
        if not idir.exists():
            return ips, mps
        for f in sorted(idir.glob("*.tif")):
            for ext in ["_manual1.gif", "_manual1.png"]:
                mf = mdir / (f.stem.split("_")[0] + ext)
                if mf.exists():
                    ips.append(str(f)); mps.append(str(mf)); break
        return ips, mps

    tr_i, tr_m = _collect(base / "training")
    va_i, va_m = _collect(base / "test")
    if not tr_i:
        return None, None
    if not va_i:
        tr_i, tr_m, va_i, va_m = _split_paths(tr_i, tr_m)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "drive"),
        UniversalSegDataset(va_i, va_m, "val",   config, "drive"),
    )


def build_coph100_dataset(root: str, config: MACCLIPConfig):
    """COph100 retinopathy-of-prematurity fundus segmentation.

    Tries six different directory layouts automatically.
    """
    # Locate base directory
    base = None
    fundus_root = Path(root) / "Fundus or OCT"
    if fundus_root.exists():
        for d in sorted(fundus_root.iterdir()):
            if d.is_dir() and "coph100" in d.name.lower():
                for inner in [d, *d.iterdir()]:
                    if inner.is_dir():
                        if any(True for _ in inner.rglob("*_mask.png")):
                            base = inner
                            break
                if base:
                    break

    if base is None:
        hits = list(Path(root).rglob("COph100"))
        base = hits[0] if hits else None

    if base is None:
        return None, None

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    NON_IMG  = ("_mask", "_overlay", "_label", "_jpg_label")

    def _is_img(p: Path):
        return p.suffix.lower() in IMG_EXTS and not any(t in p.stem.lower() for t in NON_IMG)

    def _try_add(img_f: Path, mask_f: Path, ips, mps) -> bool:
        if img_f.exists() and mask_f.exists():
            ips.append(str(img_f)); mps.append(str(mask_f))
            return True
        return False

    ips, mps = [], []

    # Layout 1 — patient/<img> + <stem>_mask.png
    if not ips:
        for pd in sorted(base.iterdir()):
            if not pd.is_dir(): continue
            for img_f in sorted(pd.iterdir()):
                if not _is_img(img_f): continue
                for mn in [img_f.stem + "_mask.png", img_f.stem + ".png"]:
                    if _try_add(img_f, pd / mn, ips, mps): break

    # Layout 2 — patient/images + patient/masks
    if not ips:
        for pd in sorted(base.iterdir()):
            if not pd.is_dir(): continue
            for img_f in sorted((pd / "images").glob("*")):
                if not _is_img(img_f): continue
                for ext in [".png", ".jpg"]:
                    if _try_add(img_f, pd / "masks" / (img_f.stem + ext), ips, mps): break

    # Layout 3 — flat base/images + base/masks
    if not ips:
        for img_f in sorted((base / "images").glob("*")):
            if _is_img(img_f):
                for ext in [".png", ".jpg"]:
                    if _try_add(img_f, base / "masks" / (img_f.stem + ext), ips, mps): break

    # Layout 4 — fully flat
    if not ips:
        for img_f in sorted(base.iterdir()):
            if img_f.is_file() and _is_img(img_f):
                if _try_add(img_f, base / (img_f.stem + "_mask.png"), ips, mps): continue

    # Layout 6 — nuclear rglob
    if not ips:
        for mf in sorted(base.rglob("*_mask.png")):
            stem = mf.stem.replace("_mask", "")
            for ext in IMG_EXTS:
                img_f = mf.parent / (stem + ext)
                if img_f.exists():
                    ips.append(str(img_f)); mps.append(str(mf)); break

    if not ips:
        return None, None

    tr_i, tr_m, va_i, va_m = _split_paths(ips, mps)
    return (
        UniversalSegDataset(tr_i, tr_m, "train", config, "coph100"),
        UniversalSegDataset(va_i, va_m, "val",   config, "coph100"),
    )


# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------

def get_all_datasets(datasets_root: str, config: MACCLIPConfig) -> Dict[str, Callable]:
    """Returns {dataset_key: builder_lambda}. Builders are lazy (not called yet)."""
    kvasir_root = config.data_root
    return {
        "kvasir":          lambda: build_kvasir_dataset(kvasir_root, config),
        "isic2018":        lambda: build_isic2018_dataset(datasets_root, config),
        "m2caiseg":        lambda: build_m2caiseg_dataset(datasets_root, config),
        "bkai_neopolyp":   lambda: build_bkai_neopolyp_dataset(datasets_root, config),
        "covid_infection": lambda: build_covid_infection_dataset(datasets_root, config),
        "covid_lung":      lambda: build_covid_lung_dataset(datasets_root, config),
        "monuseg":         lambda: build_monuseg_dataset(datasets_root, config),
        "busi":            lambda: build_busi_dataset(datasets_root, config),
        "ct2us_kidney":    lambda: build_ct2us_kidney_dataset(datasets_root, config),
        "arcade_stenosis": lambda: build_arcade_dataset(datasets_root, "stenosis", config),
        "arcade_syntax":   lambda: build_arcade_dataset(datasets_root, "syntax",   config),
        "btxrd":           lambda: build_btxrd_dataset(datasets_root, config),
        "papila":          lambda: build_papila_dataset(datasets_root, config),
        "drive":           lambda: build_drive_dataset(datasets_root, config),
        "coph100":         lambda: build_coph100_dataset(datasets_root, config),
    }
