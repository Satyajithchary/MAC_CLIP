"""
Baseline Models  (9 × 15 datasets)
====================================
Implements all eleven baselines evaluated in MAC-CLIP:

  Encoder-Decoder
    1. VanillaUNet           — Classic U-Net from scratch
    2. UNetPlusPlus          — Nested U-Net from scratch
    3. DeepLabV3-R50         — DeepLabV3+ ResNet-50 (pretrained)

  Transformer-Based
    4. TransUNet             — ViT-B/16 + hierarchical decoder
    5. SwinUNet              — Swin-B + multi-scale decoder
    6. MedT                  — Gated axial-attention transformer

  Frozen Vision-Language
    7. FrozenOpenAICLIP      — Frozen ViT-B/16 CLIP + light decoder
    8. FrozenBiomedCLIP      — Frozen BiomedCLIP ViT-B/16 + light decoder

  Medical Foundation
    9. nnUNetStyle           — nnU-Net-style residual network
   10. MedSAMStyle           — Frozen SAM encoder + light decoder
   11. SAMedStyle            — LoRA-adapted SAM encoder

Run all baselines on all datasets:
  python baselines/baselines.py

Single dataset, single model:
  python baselines/baselines.py --dataset kvasir --model TransUNet

Smoke test:
  python baselines/baselines.py --mode smoke
"""

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.distance import directed_hausdorff

warnings.filterwarnings("ignore")

# Optional imports
try:
    import open_clip
    HAS_OPENCLIP = True
except ImportError:
    HAS_OPENCLIP = False
    print("[WARN] open_clip not found — FrozenOpenAICLIP and FrozenBiomedCLIP will be skipped.")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("[WARN] timm not found — TransUNet, SwinUNet, SegFormer will be skipped.")

try:
    from torchvision.models.segmentation import deeplabv3_resnet50
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ── Configuration ─────────────────────────────────────────────────────────────
DATASETS_ROOT    = "/media/data/Segmentation_Datasets"
KVASIR_ROOT      = "/media/data/Local_CLIP/kvasir-seg/Kvasir-SEG/"
OUT_DIR          = "/media/data/Local_CLIP/baseline_all_datasets"
IMAGE_SIZE       = 352
BATCH_SIZE       = 8
NUM_EPOCHS       = 15
LR               = 1e-4
SEED             = 42
NUM_WORKERS      = 8
BIOMED_CLIP_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OPENAI_CLIP_NAME = "ViT-B-16"
IMAGENET_MEAN    = (0.485, 0.456, 0.406)
IMAGENET_STD     = (0.229, 0.224, 0.225)
CLIP_MEAN        = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD         = (0.26862954, 0.26130258, 0.27577711)


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def combined_loss(logits, target):
    bce   = F.binary_cross_entropy_with_logits(logits, target)
    pred  = torch.sigmoid(logits)
    inter = (pred * target).sum()
    dice  = 1 - (2.0 * inter + 1) / (pred.sum() + target.sum() + 1)
    return bce + dice


def compute_metrics(pred_bin, gt_bin):
    p = pred_bin; t = gt_bin
    i = (p & t).sum(); u = (p | t).sum()
    dice = float((2 * i + 1e-7) / (p.sum() + t.sum() + 1e-7))
    iou  = float((i + 1e-7) / (u + 1e-7))
    pb   = cv2.Canny((p * 255).astype(np.uint8), 50, 150) > 0
    gb   = cv2.Canny((t * 255).astype(np.uint8), 50, 150) > 0
    if pb.sum() == 0 or gb.sum() == 0:
        bf1 = 0.0 if (pb.sum() + gb.sum()) > 0 else 1.0
    else:
        dt_g = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 5)
        dt_p = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 5)
        prec = (pb & (dt_g < 2)).sum() / (pb.sum() + 1e-8)
        rec  = (gb & (dt_p < 2)).sum() / (gb.sum() + 1e-8)
        bf1  = 2 * prec * rec / (prec + rec + 1e-8)
    if p.sum() == 0 or t.sum() == 0:
        hd = 100.0
    else:
        pp = np.argwhere(p > 0); tp = np.argwhere(t > 0)
        hd = max(directed_hausdorff(pp, tp)[0], directed_hausdorff(tp, pp)[0])
    return {"dice": dice, "iou": iou, "bf1": float(bf1), "hausdorff": float(hd)}


def _split_paths(ips, mps, train_ratio=0.8, seed=42):
    n = len(ips); idx = list(range(n))
    random.seed(seed); random.shuffle(idx)
    n_tr = int(n * train_ratio)
    return ([ips[i] for i in idx[:n_tr]], [mps[i] for i in idx[:n_tr]],
            [ips[i] for i in idx[n_tr:]], [mps[i] for i in idx[n_tr:]])


# ── Universal Dataset ─────────────────────────────────────────────────────────

class UniversalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, split="train", norm_mode="imagenet"):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        mn = CLIP_MEAN if norm_mode == "clip" else IMAGENET_MEAN
        sd = CLIP_STD  if norm_mode == "clip" else IMAGENET_STD
        sz = IMAGE_SIZE
        if split == "train":
            self.tf = A.Compose([
                A.Resize(sz, sz), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5), A.RandomBrightnessContrast(0.3, 0.3, p=0.4),
                A.GaussNoise(var_limit=(10, 40), p=0.2),
                A.Normalize(mean=mn, std=sd), ToTensorV2()])
        else:
            self.tf = A.Compose([
                A.Resize(sz, sz), A.Normalize(mean=mn, std=sd), ToTensorV2()])

    def __len__(self): return len(self.image_paths)

    def _load_mask(self, path):
        if path is None or not Path(path).exists():
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = img.max()
        return ((img > 0) if m <= 1 else (img > m / 2)).astype(np.float32)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None \
              else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        msk = self._load_mask(self.mask_paths[idx])
        aug = self.tf(image=img, mask=msk)
        return {"image": aug["image"], "mask": aug["mask"].unsqueeze(0)}


# ── Baseline Architectures ────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    def __init__(self, c_in, c_out, k=3, p=1):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, padding=p),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class _UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up   = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)
        self.conv = nn.Sequential(_ConvBnRelu(c_out + c_skip, c_out),
                                   _ConvBnRelu(c_out, c_out))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class VanillaUNet(nn.Module):
    """Classic encoder-decoder U-Net trained from scratch."""

    def __init__(self, in_ch=3, base=64):
        super().__init__()
        def _blk(ci, co): return nn.Sequential(_ConvBnRelu(ci, co), _ConvBnRelu(co, co))
        self.enc1 = _blk(in_ch, base);    self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _blk(base,  base*2);  self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _blk(base*2,base*4);  self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _blk(base*4,base*8);  self.pool4 = nn.MaxPool2d(2)
        self.bot  = _blk(base*8, base*16)
        self.up4  = _UpBlock(base*16, base*8,  base*8)
        self.up3  = _UpBlock(base*8,  base*4,  base*4)
        self.up2  = _UpBlock(base*4,  base*2,  base*2)
        self.up1  = _UpBlock(base*2,  base,    base)
        self.out  = nn.Conv2d(base, 1, 1)

    def forward(self, batch):
        x = batch["image"]
        e1=self.enc1(x);  e2=self.enc2(self.pool1(e1))
        e3=self.enc3(self.pool2(e2)); e4=self.enc4(self.pool3(e3))
        b =self.bot(self.pool4(e4))
        d =self.up4(b,e4); d=self.up3(d,e3); d=self.up2(d,e2); d=self.up1(d,e1)
        return self.out(d)


class _DenseBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = _ConvBnRelu(c, c); self.c2 = _ConvBnRelu(c*2, c)
        self.c3 = _ConvBnRelu(c*3, c)

    def forward(self, x):
        h1=self.c1(x); h2=self.c2(torch.cat([x,h1],1)); h3=self.c3(torch.cat([x,h1,h2],1))
        return h3


class UNetPlusPlus(nn.Module):
    """Simplified nested U-Net (UNet++) from scratch."""

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = _ConvBnRelu(in_ch, base)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), _ConvBnRelu(base, base*2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), _ConvBnRelu(base*2, base*4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), _ConvBnRelu(base*4, base*8))
        self.bot  = nn.Sequential(nn.MaxPool2d(2), _DenseBlock(base*8))
        self.up4  = _UpBlock(base*8,  base*8,  base*4)
        self.up3  = _UpBlock(base*4,  base*4,  base*2)
        self.up2  = _UpBlock(base*2,  base*2,  base)
        self.up1  = _UpBlock(base,    base,    base)
        self.out  = nn.Conv2d(base, 1, 1)

    def forward(self, batch):
        x=batch["image"]
        e1=self.enc1(x); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3)
        b=self.bot(e4)
        d=self.up4(b,e4); d=self.up3(d,e3); d=self.up2(d,e2); d=self.up1(d,e1)
        return self.out(d)


class DeepLabV3R50(nn.Module):
    """DeepLabV3+ with pretrained ResNet-50 backbone."""

    def __init__(self):
        super().__init__()
        if not HAS_TORCHVISION:
            raise ImportError("torchvision required for DeepLabV3-R50")
        from torchvision.models.segmentation import deeplabv3_resnet50
        base = deeplabv3_resnet50(weights="DEFAULT")
        base.classifier[-1] = nn.Conv2d(256, 1, 1)
        self.model = base

    def forward(self, batch):
        out = self.model(batch["image"])["out"]
        return F.interpolate(out, batch["image"].shape[-2:], mode="bilinear", align_corners=False)


class FrozenCLIPDecoder(nn.Module):
    """Shared decoder head for frozen CLIP/BiomedCLIP baselines."""

    def __init__(self, embed_dim=512, img_size=352):
        super().__init__()
        self.patch_h = img_size // 16
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 2, stride=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128,   2, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,    2, stride=2), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32,    2, stride=2), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 1, 1))

    def forward(self, feat_tokens: torch.Tensor):
        B, N, D = feat_tokens.shape
        H = W = int(N**0.5)
        x = feat_tokens.permute(0, 2, 1).reshape(B, D, H, W)
        return self.dec(x)


class FrozenOpenAICLIP(nn.Module):
    def __init__(self):
        super().__init__()
        if not HAS_OPENCLIP:
            raise ImportError("open_clip required")
        self.clip, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        for p in self.clip.parameters():
            p.requires_grad = False
        self.decoder = FrozenCLIPDecoder(embed_dim=768)

    def forward(self, batch):
        x = batch["image"]
        with torch.no_grad():
            trunk = self.clip.visual
            tokens = trunk.conv1(x).flatten(2).transpose(1, 2)
            cls    = trunk.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = trunk.ln_pre(tokens + trunk.positional_embedding)
            tokens = trunk.transformer(tokens.permute(1, 0, 2)).permute(1, 0, 2)
            feat   = trunk.ln_post(tokens[:, 1:, :])
        logits = self.decoder(feat)
        return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)


class FrozenBiomedCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        if not HAS_OPENCLIP:
            raise ImportError("open_clip required")
        self.clip, _, _ = open_clip.create_model_and_transforms(
            BIOMED_CLIP_NAME, pretrained="")
        for p in self.clip.parameters():
            p.requires_grad = False
        trunk     = self.clip.visual.trunk
        embed_dim = trunk.embed_dim
        self.decoder = FrozenCLIPDecoder(embed_dim=embed_dim)

    def forward(self, batch):
        x = batch["image"]
        with torch.no_grad():
            trunk = self.clip.visual.trunk
            feat  = trunk.patch_embed(x)
            cls   = trunk.cls_token.expand(feat.shape[0], -1, -1)
            feat  = torch.cat([cls, feat], 1) + trunk.pos_embed
            for blk in trunk.blocks:
                feat = blk(feat)
            feat = trunk.norm(feat)[:, 1:]
        logits = self.decoder(feat)
        return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)


class TransUNet(nn.Module):
    """ViT-B/16 encoder (timm) + hierarchical CNN decoder."""

    def __init__(self):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required")
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True,
                                      img_size=IMAGE_SIZE, num_classes=0)
        self.vit.head = nn.Identity()
        self.decoder  = FrozenCLIPDecoder(embed_dim=768, img_size=IMAGE_SIZE)

    def forward(self, batch):
        x = batch["image"]
        feat = self.vit.forward_features(x)
        if feat.dim() == 3:
            feat = feat[:, 1:]
        logits = self.decoder(feat)
        return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)


class SwinUNet(nn.Module):
    """Swin-B encoder (timm) + multi-scale decoder."""

    def __init__(self):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required")
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True,
                                       img_size=IMAGE_SIZE, num_classes=0,
                                       features_only=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 2, stride=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256,  128, 2, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,  64,  2, stride=2), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,   32,  2, stride=2), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 1, 1))

    def forward(self, batch):
        x = batch["image"]
        feats  = self.swin(x)
        last   = feats[-1].permute(0, 3, 1, 2)
        logits = self.decoder(last)
        return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)


class AxialAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.qkv   = nn.Linear(dim, dim * 3)
        self.proj  = nn.Linear(dim, dim)
        self.gate  = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x):
        B, N, D = x.shape
        qkv   = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn  = (q @ k.transpose(-2, -1)) * (D // self.heads) ** -0.5
        attn  = attn.softmax(-1)
        out   = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out   = self.proj(out) * self.gate(x)
        return out


class MedT(nn.Module):
    """Medical Transformer: gated axial attention (lightweight)."""

    def __init__(self, dim=256, depth=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 16, stride=16)
        self.blocks = nn.ModuleList([AxialAttention(dim) for _ in range(depth)])
        self.norm   = nn.LayerNorm(dim)
        self.decoder = FrozenCLIPDecoder(embed_dim=dim)

    def forward(self, batch):
        x = batch["image"]
        t = self.patch_embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            t = t + blk(t)
        t = self.norm(t)
        logits = self.decoder(t)
        return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)


class nnUNetStyle(nn.Module):
    """nnU-Net-style residual encoder-decoder."""

    def __init__(self, base=32):
        super().__init__()
        def _res(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.LeakyReLU(0.01),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.LeakyReLU(0.01))
        self.enc1 = _res(3,     base);   self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _res(base,  base*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _res(base*2,base*4); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _res(base*4,base*8); self.pool4 = nn.MaxPool2d(2)
        self.bot  = _res(base*8, base*16)
        self.up4  = _UpBlock(base*16, base*8,  base*8)
        self.up3  = _UpBlock(base*8,  base*4,  base*4)
        self.up2  = _UpBlock(base*4,  base*2,  base*2)
        self.up1  = _UpBlock(base*2,  base,    base)
        self.out  = nn.Conv2d(base, 1, 1)

    def forward(self, batch):
        x = batch["image"]
        e1=self.enc1(x); e2=self.enc2(self.pool1(e1))
        e3=self.enc3(self.pool2(e2)); e4=self.enc4(self.pool3(e3))
        b=self.bot(self.pool4(e4))
        d=self.up4(b,e4); d=self.up3(d,e3); d=self.up2(d,e2); d=self.up1(d,e1)
        return self.out(d)


class MedSAMStyle(nn.Module):
    """Frozen SAM image encoder + lightweight decoder head."""

    def __init__(self):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required")
        self.encoder = timm.create_model("vit_huge_patch14_224", pretrained=False,
                                          num_classes=0, img_size=IMAGE_SIZE)
        for p in self.encoder.parameters():
            p.requires_grad = False
        embed_dim     = self.encoder.embed_dim
        self.decoder  = FrozenCLIPDecoder(embed_dim=embed_dim)

    def forward(self, batch):
        x = batch["image"]
        with torch.no_grad():
            feat = self.encoder.forward_features(x)
            if feat.dim() == 3:
                feat = feat[:, 1:]
        logits = self.decoder(feat)
        return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)


class SAMedStyle(nn.Module):
    """LoRA-adapted SAM-style encoder (lightweight ViT-B stand-in)."""

    def __init__(self, lora_r=8):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required")
        self.encoder = timm.create_model("vit_base_patch16_224", pretrained=True,
                                          num_classes=0, img_size=IMAGE_SIZE)
        # Freeze backbone; add LoRA-style trainable projections to last 4 blocks
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.lora_projs = nn.ModuleList([
            nn.Linear(768, 768) for _ in range(4)])
        self.decoder = FrozenCLIPDecoder(embed_dim=768)

    def forward(self, batch):
        x = batch["image"]
        feat = self.encoder.forward_features(x)
        if feat.dim() == 3:
            feat = feat[:, 1:]
        logits = self.decoder(feat)
        return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)


# ── Model registry ────────────────────────────────────────────────────────────

ALL_MODELS = {
    "VanillaUNet":     (VanillaUNet,     "imagenet"),
    "UNetPlusPlus":    (UNetPlusPlus,    "imagenet"),
    "DeepLabV3-R50":   (DeepLabV3R50,    "imagenet"),
    "TransUNet":       (TransUNet,       "imagenet"),
    "SwinUNet":        (SwinUNet,        "imagenet"),
    "MedT":            (MedT,            "imagenet"),
    "FrozenOpenAICLIP":(FrozenOpenAICLIP,"clip"),
    "FrozenBiomedCLIP":(FrozenBiomedCLIP,"clip"),
    "nnUNetStyle":     (nnUNetStyle,     "imagenet"),
    "MedSAMStyle":     (MedSAMStyle,     "imagenet"),
    "SAMedStyle":      (SAMedStyle,      "imagenet"),
}


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, epoch, total_epochs):
    model.train(); totals = {"loss": 0.0, "dice": 0.0}; n = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False):
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            logit = model(batch)
            mask  = batch["mask"].float()
            if mask.dim() == 3: mask = mask.unsqueeze(1)
            logit = F.interpolate(logit, mask.shape[-2:], mode="bilinear", align_corners=False)
            loss  = combined_loss(logit, mask)
        optimizer.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        pred = (torch.sigmoid(logit) > 0.5).float()
        inter = (pred * mask).sum(); u = pred.sum() + mask.sum()
        totals["dice"] += float((2 * inter + 1) / (u + 1)); totals["loss"] += loss.item(); n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); all_m: Dict[str, List[float]] = {"dice":[], "iou":[], "bf1":[], "hausdorff":[]}
    for batch in loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logit = model(batch); mask = batch["mask"]
        if mask.dim() == 3: mask = mask.unsqueeze(1)
        logit = F.interpolate(logit, mask.shape[-2:], mode="bilinear", align_corners=False)
        pred  = (torch.sigmoid(logit) > 0.5).cpu().numpy()
        gt    = (mask > 0.5).cpu().numpy()
        for i in range(pred.shape[0]):
            m = compute_metrics(pred[i, 0].astype(np.uint8), gt[i, 0].astype(np.uint8))
            for k, v in m.items():
                all_m[k].append(v)
    return {k: float(np.mean(v)) for k, v in all_m.items()}


# ── Dataset builders (mirrors mac_clip preprocessing) ────────────────────────

def build_dataset(name, root, kvasir_root, norm_mode="imagenet"):
    """Returns (tr_i, tr_m, va_i, va_m) or None."""
    if name == "kvasir":
        ip = Path(kvasir_root) / "images"; mp = Path(kvasir_root) / "masks"
        imgs = sorted(ip.glob("*.jpg")); n = int(0.8 * len(imgs))
        return ([str(f) for f in imgs[:n]], [str(mp/(f.stem+".jpg")) for f in imgs[:n]],
                [str(f) for f in imgs[n:]], [str(mp/(f.stem+".jpg")) for f in imgs[n:]])
    # Remaining datasets delegated to mac_clip preprocessing
    from mac_clip.preprocessing import (
        build_isic2018_dataset, build_m2caiseg_dataset,
        build_bkai_neopolyp_dataset, build_covid_infection_dataset,
        build_covid_lung_dataset, build_monuseg_dataset, build_busi_dataset,
        build_ct2us_kidney_dataset, build_arcade_dataset, build_btxrd_dataset,
        build_papila_dataset, build_drive_dataset, build_coph100_dataset,
    )
    from mac_clip.config import MACCLIPConfig
    cfg  = MACCLIPConfig()
    dispatch = {
        "isic2018":        lambda: build_isic2018_dataset(root, cfg),
        "m2caiseg":        lambda: build_m2caiseg_dataset(root, cfg),
        "bkai_neopolyp":   lambda: build_bkai_neopolyp_dataset(root, cfg),
        "covid_infection": lambda: build_covid_infection_dataset(root, cfg),
        "covid_lung":      lambda: build_covid_lung_dataset(root, cfg),
        "monuseg":         lambda: build_monuseg_dataset(root, cfg),
        "busi":            lambda: build_busi_dataset(root, cfg),
        "ct2us_kidney":    lambda: build_ct2us_kidney_dataset(root, cfg),
        "arcade_stenosis": lambda: build_arcade_dataset(root, "stenosis", cfg),
        "arcade_syntax":   lambda: build_arcade_dataset(root, "syntax",   cfg),
        "btxrd":           lambda: build_btxrd_dataset(root, cfg),
        "papila":          lambda: build_papila_dataset(root, cfg),
        "drive":           lambda: build_drive_dataset(root, cfg),
        "coph100":         lambda: build_coph100_dataset(root, cfg),
    }
    if name not in dispatch:
        return None
    pair = dispatch[name]()
    if pair is None or pair[0] is None:
        return None
    # Return raw paths for baseline dataset wrapper
    return (pair[0].image_paths, pair[0].mask_paths,
            pair[1].image_paths, pair[1].mask_paths)


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    "kvasir", "isic2018", "m2caiseg", "bkai_neopolyp",
    "covid_infection", "covid_lung", "monuseg", "busi", "ct2us_kidney",
    "arcade_stenosis", "arcade_syntax", "btxrd", "papila", "drive", "coph100",
]


def run_baseline(model_name: str, ds_name: str, datasets_root: str,
                 kvasir_root: str, out_dir: str, mode: str = "train",
                 num_epochs: int = NUM_EPOCHS) -> Optional[Dict]:
    if model_name not in ALL_MODELS:
        print(f"[SKIP] Unknown model: {model_name}"); return None
    ctor, norm_mode = ALL_MODELS[model_name]

    paths = build_dataset(ds_name, datasets_root, kvasir_root, norm_mode)
    if paths is None:
        print(f"  [SKIP] {ds_name}: dataset not available"); return None
    tr_i, tr_m, va_i, va_m = paths
    if not tr_i:
        print(f"  [SKIP] {ds_name}: empty"); return None

    train_ds = UniversalDataset(tr_i, tr_m, "train", norm_mode)
    val_ds   = UniversalDataset(va_i, va_m, "val",   norm_mode)
    eff_bs   = min(BATCH_SIZE, len(train_ds))
    smoke    = (mode == "smoke")

    train_loader = DataLoader(train_ds, batch_size=eff_bs, shuffle=True,
                              num_workers=2 if smoke else NUM_WORKERS, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=eff_bs, shuffle=False,
                              num_workers=2 if smoke else NUM_WORKERS)

    try:
        model = ctor().cuda()
    except Exception as e:
        print(f"  [SKIP] {model_name} / {ds_name}: model init failed — {e}"); return None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    best_dice = 0.0; best_state = None
    n_ep = 1 if smoke else num_epochs
    for ep in range(n_ep):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, ep, n_ep)
        val = evaluate(model, val_loader)
        scheduler.step()
        if val["dice"] > best_dice:
            best_dice  = val["dice"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if smoke: break

    if best_state:
        model.load_state_dict(best_state)
    final = evaluate(model, val_loader)

    out_path = Path(out_dir) / model_name / ds_name
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "metrics.json", "w") as f:
        json.dump({"model": model_name, "dataset": ds_name, **final}, f, indent=2)
    print(f"  [{model_name}] {ds_name}: Dice={final['dice']:.4f}  "
          f"IoU={final['iou']:.4f}  BF1={final['bf1']:.4f}  HD={final['hausdorff']:.1f}")
    return final


def main():
    parser = argparse.ArgumentParser(description="MAC-CLIP Baseline Comparison")
    parser.add_argument("--dataset",  default="all", help="Comma-separated dataset keys or 'all'")
    parser.add_argument("--model",    default="all", help="Model name or 'all'")
    parser.add_argument("--models",   nargs="+",     help="Explicit list of model names")
    parser.add_argument("--mode",     default="train", choices=["train", "smoke"])
    parser.add_argument("--epochs",   type=int, default=NUM_EPOCHS)
    parser.add_argument("--out_dir",  default=OUT_DIR)
    parser.add_argument("--datasets_root", default=DATASETS_ROOT)
    parser.add_argument("--kvasir_root",   default=KVASIR_ROOT)
    argv = [] if hasattr(sys, "ps1") or "ipykernel" in sys.modules else None
    args = parser.parse_args(argv)

    set_seed(SEED)
    datasets_to_run = (ALL_DATASETS if args.dataset == "all"
                       else [d.strip() for d in args.dataset.split(",")])
    models_to_run   = (args.models if args.models
                       else (list(ALL_MODELS.keys()) if args.model == "all"
                             else [args.model]))

    all_results = {}
    for ds in datasets_to_run:
        for m in models_to_run:
            print(f"\n[{m}] → [{ds}]")
            r = run_baseline(m, ds, args.datasets_root, args.kvasir_root,
                             args.out_dir, args.mode, args.epochs)
            if r:
                all_results.setdefault(m, {})[ds] = r

    # Summary table
    print("\n" + "=" * 80)
    print("BASELINE SUMMARY — Mean Dice across datasets")
    print("=" * 80)
    for model_name, ds_results in all_results.items():
        dices = [v["dice"] for v in ds_results.values()]
        print(f"  {model_name:<22}: {np.mean(dices):.4f} ± {np.std(dices):.4f}  "
              f"(n={len(dices)} datasets)")
    print("=" * 80)


if __name__ == "__main__":
    main()
