"""
MAC-CLIP Model Architecture
============================
Implements the MAC-CLIP v4 segmentation network:

  1. BiomedCLIP backbone (frozen) with LoRA adapters at layers {6,8,10,12}
  2. Feature Pyramid Network (FPN) for multi-scale aggregation
  3. Morpho-Semantic Adaptive Grouping (MSAG) module — the core novelty
  4. Orthogonality-constrained Semantic Concept Module
  5. Boundary-Gated UNet Decoder with Selective-Kernel and Reverse-Attention units
  6. Progressive Group Contrastive (PGC) + Pixel Boundary Contrastive (PBC) losses
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mac_clip.config import MACCLIPConfig
from mac_clip.dataset_registry import DATASET_CONCEPTS
from mac_clip.losses import (
    BoundaryAwareLoss, BoundaryMapLoss, ContourLoss, DiceLoss,
    EdgeAwareLoss, FocalLoss, IoULoss, TverskyLoss,
)


# ---------------------------------------------------------------------------
# LoRA Adapter
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """Low-rank adaptation of a linear projection."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A  = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B  = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor, original_weight: torch.Tensor) -> torch.Tensor:
        return F.linear(x, original_weight) + (x @ self.lora_A @ self.lora_B) * self.scaling


# ---------------------------------------------------------------------------
# Progressive Group Contrastive Loss
# ---------------------------------------------------------------------------

class ProgressiveGroupContrastiveLoss(nn.Module):
    """PGC loss: aligns pixel groups at multiple spatial granularities with text."""

    def __init__(self, config: MACCLIPConfig):
        super().__init__()
        self.temperature = config.pgc_temperature
        self.gamma       = config.pgc_gamma
        self.group_sizes = config.pgc_group_sizes
        self.n_samples   = config.pgc_samples_per_size

    def _weight(self, k: int, K: int) -> float:
        return (min(k, K) / K) ** self.gamma if K > 0 else 1.0

    def _pool_group(self, feat_flat: torch.Tensor, mask_flat: torch.Tensor, k: int):
        pos_idx = mask_flat.nonzero(as_tuple=True)[0]
        if len(pos_idx) < k:
            return None
        chosen = pos_idx[torch.randperm(len(pos_idx), device=feat_flat.device)[:k]]
        return feat_flat[chosen].mean(0)

    def forward(self, pixel_feats: torch.Tensor, text_feats: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        B, D, H, W = pixel_feats.shape
        device = pixel_feats.device
        loss_total = torch.tensor(0.0, device=device)
        count = 0

        masks_small = F.interpolate(masks.unsqueeze(1).float(), (H, W), mode="nearest").squeeze(1)
        feats_flat  = pixel_feats.permute(0, 2, 3, 1).reshape(B, H * W, D)
        masks_flat  = masks_small.reshape(B, H * W)
        tf_norm     = F.normalize(text_feats, dim=-1)

        for b in range(B):
            ff = feats_flat[b]
            mf = masks_flat[b]
            K  = int(mf.sum().item())
            if K == 0:
                continue
            for k in self.group_sizes:
                w = self._weight(k, K)
                group_vecs = []
                for _ in range(self.n_samples):
                    gv = self._pool_group(ff, mf, k)
                    if gv is not None:
                        group_vecs.append(F.normalize(gv, dim=-1))
                if not group_vecs:
                    continue
                sims   = torch.stack(group_vecs) @ tf_norm.T / self.temperature
                labels = torch.full((len(group_vecs),), b, dtype=torch.long, device=device)
                loss_total = loss_total + w * F.cross_entropy(sims, labels)
                count += 1

        return loss_total / count if count > 0 else loss_total


class PixelBoundaryContrastiveLoss(nn.Module):
    """PBC loss: pulls boundary pixels toward boundary text, interior → interior text."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, pixel_feats: torch.Tensor, boundary_text: torch.Tensor,
                interior_text: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, D, H, W = pixel_feats.shape
        device = pixel_feats.device

        m  = masks.unsqueeze(1).float()
        sx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=m.dtype, device=device)
        sy = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=m.dtype, device=device)
        bnd_map = (torch.sqrt(F.conv2d(m, sx, padding=1)**2 + F.conv2d(m, sy, padding=1)**2 + 1e-6) > 0.3).float()
        bnd_map = F.max_pool2d(bnd_map, 3, 1, 1)

        bnd_small = F.interpolate(bnd_map, (H, W), mode="nearest").squeeze(1)
        int_small = (F.interpolate(m, (H, W), mode="nearest").squeeze(1) * (1 - bnd_small))

        bt = F.normalize(boundary_text, dim=-1)
        it = F.normalize(interior_text,  dim=-1)
        loss_total = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            feats_b = F.normalize(pixel_feats[b].permute(1, 2, 0).reshape(-1, D), dim=-1)
            bnd_idx = bnd_small[b].reshape(-1).nonzero(as_tuple=True)[0]
            int_idx = int_small[b].reshape(-1).nonzero(as_tuple=True)[0]
            if len(bnd_idx) < 4 or len(int_idx) < 4:
                continue
            bnd_idx = bnd_idx[torch.randperm(len(bnd_idx), device=device)[:64]]
            int_idx = int_idx[torch.randperm(len(int_idx), device=device)[:64]]

            sim_bnd = (feats_b[bnd_idx] @ torch.stack([bt, it], 1)) / self.temperature
            sim_int = (feats_b[int_idx] @ torch.stack([bt, it], 1)) / self.temperature
            loss_total += F.cross_entropy(sim_bnd, torch.zeros(len(bnd_idx), dtype=torch.long, device=device))
            loss_total += F.cross_entropy(sim_int, torch.ones( len(int_idx), dtype=torch.long, device=device))
            count += 2

        return loss_total / count if count > 0 else loss_total


# ---------------------------------------------------------------------------
# MSAG — Morpho-Semantic Adaptive Grouping module
# ---------------------------------------------------------------------------

class DeformableConv2d(nn.Module):
    """Deformable convolution with learned spatial offsets (MSAG offset branch)."""

    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, max_def: float = 2.0):
        super().__init__()
        self.max_def  = max_def
        self.off_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.GELU(),
            nn.Conv2d(in_ch, 2 * ks * ks, 1),
        )
        self.reg_conv = nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.tanh(self.off_conv(x)) * self.max_def  # offset field (used externally)
        return self.reg_conv(x)


class UncertaintyEstimator(nn.Module):
    """Predicts per-pixel reliability weight u(p) ∈ [0,1]."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.est = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1), nn.BatchNorm2d(in_ch // 2), nn.GELU(),
            nn.Conv2d(in_ch // 2, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.est(x).squeeze(1)


class MorphologicalModule(nn.Module):
    """MSAG: offset-conditioned deformable feature aggregation + uncertainty estimation."""

    def __init__(self, feature_dim: int, config: MACCLIPConfig):
        super().__init__()
        self.deform = DeformableConv2d(feature_dim, feature_dim, 3, config.max_deformation)
        self.uncert = UncertaintyEstimator(feature_dim)

    def forward(self, features: torch.Tensor, seg_logits=None):
        return self.deform(features), self.uncert(features)


# ---------------------------------------------------------------------------
# Orthogonality-Constrained Concept Heads
# ---------------------------------------------------------------------------

class ConceptProjector(nn.Module):
    def __init__(self, input_dim: int, concept_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, concept_dim, 1), nn.BatchNorm2d(concept_dim), nn.GELU(),
            nn.Conv2d(concept_dim, concept_dim, 3, padding=1), nn.BatchNorm2d(concept_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SemanticModule(nn.Module):
    """Concept-aligned pixel scoring with orthogonality-constrained heads."""

    def __init__(self, feature_dim: int, num_concepts: int, concept_dim: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.projectors   = nn.ModuleList([
            ConceptProjector(feature_dim, concept_dim) for _ in range(num_concepts)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(num_concepts, 64, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, features: torch.Tensor, text_embeddings: torch.Tensor):
        maps = []
        for k in range(self.num_concepts):
            Vk  = F.normalize(self.projectors[k](features), dim=1)
            Tk  = F.normalize(text_embeddings[:, k, :], dim=1)
            sim = (Vk * Tk.unsqueeze(2).unsqueeze(3)).sum(1, keepdim=True)
            maps.append(sim)
        maps = torch.cat(maps, dim=1)
        return maps, self.fusion(maps)


# ---------------------------------------------------------------------------
# Decoder components
# ---------------------------------------------------------------------------

class SelectiveKernelUnit(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.GELU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.GELU(),
        )
        mid = max(channels // reduction, 16)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv3(x); b = self.conv5(x)
        w = self.fc2(F.gelu(self.fc1((a + b).mean(dim=[2, 3]))))
        w = torch.softmax(w.view(x.shape[0], 2, x.shape[1]), dim=1)
        return a * w[:, 0, :, None, None] + b * w[:, 1, :, None, None]


class ReverseAttention(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward(self, features: torch.Tensor, coarse_seg: torch.Tensor) -> torch.Tensor:
        seg_up  = F.interpolate(coarse_seg, size=features.shape[-2:], mode="bilinear", align_corners=False)
        ra_mask = 1 - torch.sigmoid(seg_up)
        return self.conv(features * ra_mask)


class FPNFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, num_levels: int = 4):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                          nn.BatchNorm2d(out_channels), nn.GELU())
            for _ in range(num_levels)
        ])
        self.smooths = nn.ModuleList([
            nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1),
                          nn.BatchNorm2d(out_channels), nn.GELU())
            for _ in range(num_levels)
        ])
        self.output = nn.Sequential(
            nn.Conv2d(out_channels * num_levels, out_channels * 2, 1),
            nn.BatchNorm2d(out_channels * 2), nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )

    def forward(self, features: List[torch.Tensor]):
        N  = len(features)
        ls = [self.laterals[i](features[i]) for i in range(N)]
        for i in range(N - 2, -1, -1):
            ls[i] = ls[i] + F.interpolate(ls[i + 1], size=ls[i].shape[-2:], mode="nearest")
        sm = [self.smooths[i](ls[i]) for i in range(N)]
        ts = sm[0].shape[-2:]
        aligned = [F.interpolate(s, ts, mode="bilinear", align_corners=False)
                   if s.shape[-2:] != ts else s for s in sm]
        return self.output(torch.cat(aligned, dim=1)), sm


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x   = self.dwconv(x).permute(0, 2, 3, 1)
        x   = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        return x.permute(0, 3, 1, 2) + res


class SkipFusion(nn.Module):
    def __init__(self, up_dim: int, skip_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(up_dim + skip_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim), nn.GELU(),
            ConvNeXtBlock(out_dim),
        )

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([x_up, x_skip], dim=1))


class UNetDecoderV4(nn.Module):
    """Boundary-gated UNet decoder with Selective-Kernel + Reverse-Attention."""

    def __init__(self, feature_dim: int = 512, skip_dim: int = 256,
                 boundary_gate_alpha: float = 0.4):
        super().__init__()
        self.up1   = nn.Sequential(nn.ConvTranspose2d(feature_dim, 256, 2, stride=2),
                                   nn.BatchNorm2d(256), nn.GELU())
        self.sk1   = SelectiveKernelUnit(256)
        self.fuse1 = SkipFusion(256, skip_dim, 256)

        self.up2   = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2),
                                   nn.BatchNorm2d(128), nn.GELU())
        self.sk2   = SelectiveKernelUnit(128)
        self.ra2   = ReverseAttention(128, 128)
        self.fuse2 = SkipFusion(128, 128, 128)

        self.up3   = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2),
                                   nn.BatchNorm2d(64), nn.GELU())
        self.sk3   = SelectiveKernelUnit(64)
        self.ra3   = ReverseAttention(64, 64)
        self.fuse3 = SkipFusion(64, 64, 64)

        self.up4   = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
        )

        self.skip_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(skip_dim, 256, 1), nn.BatchNorm2d(256), nn.GELU()),
            nn.Sequential(nn.Conv2d(skip_dim, 128, 1), nn.BatchNorm2d(128), nn.GELU()),
            nn.Sequential(nn.Conv2d(skip_dim,  64, 1), nn.BatchNorm2d(64),  nn.GELU()),
        ])

        self.aux1     = nn.Conv2d(256, 1, 1)
        self.aux2     = nn.Conv2d(128, 1, 1)
        self.aux3     = nn.Conv2d(64,  1, 1)
        self.seg_head = nn.Conv2d(32,  1, 1)

        self.boundary_branch = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        self.gate_alpha = nn.Parameter(torch.tensor(boundary_gate_alpha))

    def forward(self, x: torch.Tensor, fpn_smoothed: List[torch.Tensor],
                target_size: Tuple[int, int]):
        def _up(t):
            return F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)

        x1 = self.sk1(self.up1(x))
        s1 = F.interpolate(self.skip_proj[0](fpn_smoothed[-1]), x1.shape[-2:],
                           mode="bilinear", align_corners=False)
        x1 = self.fuse1(x1, s1); a1 = self.aux1(x1)

        x2 = self.sk2(self.up2(x1)); x2 = self.ra2(x2, a1)
        s2 = F.interpolate(self.skip_proj[1](fpn_smoothed[-2]), x2.shape[-2:],
                           mode="bilinear", align_corners=False)
        x2 = self.fuse2(x2, s2); a2 = self.aux2(x2)

        x3 = self.sk3(self.up3(x2)); x3 = self.ra3(x3, a2)
        s3 = F.interpolate(self.skip_proj[2](fpn_smoothed[-3]), x3.shape[-2:],
                           mode="bilinear", align_corners=False)
        x3 = self.fuse3(x3, s3)

        bnd_logit = self.boundary_branch(x3)
        seg_logit = self.seg_head(self.up4(x3))

        alpha_clamped = torch.clamp(self.gate_alpha, 0.0, 1.0)
        refined_seg   = seg_logit * (1.0 + alpha_clamped * torch.sigmoid(bnd_logit))

        return _up(refined_seg), _up(bnd_logit), [_up(a1), _up(a2), _up(self.aux3(x3))]


# ---------------------------------------------------------------------------
# MAC-CLIP v4 Main Model
# ---------------------------------------------------------------------------

class MACCLIPv4(nn.Module):
    """
    MAC-CLIP: Morpho-Semantic Adaptive Contrastive Learning for
    Multi-Modality Medical Segmentation.

    Architecture:
        BiomedCLIP (frozen) + LoRA → FPN → MSAG → SemanticModule → BoundaryGatedDecoder
    """

    def __init__(self, config: MACCLIPConfig):
        super().__init__()
        self.config = config
        import open_clip

        print("\n" + "=" * 70)
        print("MAC-CLIP v4: Morpho-Semantic Adaptive Contrastive Learning")
        print("=" * 70)

        # Backbone
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            config.clip_model, pretrained=config.clip_pretrained)

        if "hf-hub" in config.clip_model:
            from transformers import AutoTokenizer
            mn = config.clip_model.replace("hf-hub:", "")
            self.hf_tokenizer   = AutoTokenizer.from_pretrained(mn)
            self.tokenizer_type = "hf"
            self.context_length = 256
        else:
            self.tokenizer      = open_clip.get_tokenizer(config.clip_model)
            self.tokenizer_type = "open_clip"
            self.context_length = 77

        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.is_timm = hasattr(self.clip_model.visual, "trunk")
        if self.is_timm:
            self.visual_trunk = self.clip_model.visual.trunk
            embed_dim = self.visual_trunk.embed_dim
            if hasattr(self.visual_trunk, "patch_embed"):
                self.visual_trunk.patch_embed.strict_img_size = False
        else:
            self.visual_trunk = self.clip_model.visual
            embed_dim = self.clip_model.visual.transformer.width
        self.clip_hidden_dim = embed_dim

        self.feature_layers = config.lora_layers
        self.lora_adapters  = self._create_lora_adapters()

        fpn_out = 256
        self.fpn       = FPNFusion(embed_dim, fpn_out, len(config.lora_layers))
        self.feat_proj = nn.Sequential(
            nn.Conv2d(fpn_out, config.feature_dim, 1),
            nn.BatchNorm2d(config.feature_dim), nn.GELU(),
        )

        self.morphological_module = MorphologicalModule(config.feature_dim, config)
        self.semantic_module      = SemanticModule(config.feature_dim, config.num_concepts,
                                                   config.concept_dim)
        self.decoder              = UNetDecoderV4(config.feature_dim, fpn_out,
                                                  config.boundary_gate_alpha)

        self.edge_head = nn.Sequential(
            nn.Conv2d(config.feature_dim, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        self.concept_embeddings = nn.Parameter(
            torch.randn(1, config.num_concepts, 512) * 0.01)

        self._concept_cache: Dict[str, torch.Tensor] = {}

        # Losses
        self.tversky_loss  = TverskyLoss(config.tversky_alpha, config.tversky_beta)
        self.iou_loss      = IoULoss()
        self.dice_loss     = DiceLoss()
        self.focal_loss    = FocalLoss(config.focal_alpha, config.focal_gamma)
        self.boundary_loss = BoundaryAwareLoss(sigma=8.0)
        self.edge_loss     = EdgeAwareLoss()
        self.contour_loss  = ContourLoss()
        self.bmap_loss     = BoundaryMapLoss()
        self.pgc_loss      = ProgressiveGroupContrastiveLoss(config)
        self.pbc_loss      = PixelBoundaryContrastiveLoss(temperature=config.pgc_temperature)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params    : {total:,}")
        print(f"  Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Dataset concept hot-swap
    # ------------------------------------------------------------------

    def set_dataset_concepts(self, dataset_name: str) -> None:
        """Replace clinical prompts when switching datasets."""
        self._concept_cache = {}
        cfg = DATASET_CONCEPTS.get(dataset_name, DATASET_CONCEPTS["kvasir"])
        self.config.clinical_concepts = cfg["clinical_concepts"]
        self.config.boundary_concept  = cfg["boundary_concept"]
        self.config.interior_concept  = cfg["interior_concept"]
        print(f"  ✓ Concepts set for: {dataset_name}")

    # ------------------------------------------------------------------
    # LoRA creation
    # ------------------------------------------------------------------

    def _create_lora_adapters(self) -> nn.ModuleDict:
        adapters = nn.ModuleDict()
        blocks   = (self.visual_trunk.blocks if self.is_timm
                    else self.visual_trunk.transformer.resblocks)
        for idx in self.config.lora_layers:
            if idx > len(blocks):
                continue
            blk = blocks[idx - 1]
            tgt = (getattr(blk.attn, "proj", None) or getattr(blk.attn, "qkv", None)
                   if self.is_timm else getattr(blk.attn, "out_proj", None))
            if tgt is None:
                continue
            adapters[f"layer_{idx}"] = LoRALayer(
                tgt.in_features, tgt.out_features, self.config.lora_r, self.config.lora_alpha)
        return adapters

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def encode_image_multiscale(self, images: torch.Tensor):
        return self._encode_timm(images) if self.is_timm else self._encode_openai(images)

    def _encode_timm(self, images: torch.Tensor):
        trunk = self.visual_trunk
        x     = trunk.patch_embed(images)
        cls   = trunk.cls_token.expand(x.shape[0], -1, -1)
        x     = torch.cat((cls, x), dim=1)
        pe    = trunk.pos_embed
        if x.shape[1] != pe.shape[1]:
            cp, sp = pe[:, :1], pe[:, 1:]
            N0 = sp.shape[1]; H0 = int(math.sqrt(N0))
            N1 = x.shape[1] - 1; H1 = int(math.sqrt(N1))
            sp = F.interpolate(sp.reshape(1, H0, H0, -1).permute(0, 3, 1, 2),
                               (H1, H1), mode="bicubic", align_corners=False)
            pe = torch.cat((cp, sp.permute(0, 2, 3, 1).reshape(1, N1, -1)), dim=1)
        x = trunk.pos_drop(x + pe)
        if hasattr(trunk, "norm_pre"):
            x = trunk.norm_pre(x)
        feats = []
        for i, blk in enumerate(trunk.blocks):
            x = blk(x)
            if (i + 1) in self.feature_layers:
                f = x[:, 1:]
                B, N, C = f.shape; H = int(math.sqrt(N))
                feats.append(f.permute(0, 2, 1).reshape(B, C, H, H))
        x  = trunk.norm(x); gf = x[:, 0]
        if hasattr(self.clip_model.visual, "head"):
            gf = self.clip_model.visual.head(gf)
        return gf, feats

    def _encode_openai(self, images: torch.Tensor):
        vis = self.visual_trunk
        x   = vis.conv1(images).reshape(images.shape[0], -1,
                                         images.shape[0]).permute(0, 2, 1)
        cls = vis.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x   = torch.cat([cls, x], dim=1)
        pe  = vis.positional_embedding.to(x.dtype)
        if x.shape[1] != pe.shape[0]:
            cp, sp = pe[:1], pe[1:]
            N0 = sp.shape[0]; H0 = int(math.sqrt(N0))
            N1 = x.shape[1] - 1; H1 = int(math.sqrt(N1))
            sp = F.interpolate(sp.reshape(1, H0, H0, -1).permute(0, 3, 1, 2),
                               (H1, H1), mode="bicubic", align_corners=False)
            pe = torch.cat([cp, sp.permute(0, 2, 3, 1).reshape(N1, -1)], dim=0)
        x = vis.ln_pre(x + pe).permute(1, 0, 2)
        feats = []
        for i, blk in enumerate(vis.transformer.resblocks):
            x = blk(x)
            if (i + 1) in self.feature_layers:
                f = x.permute(1, 0, 2)[:, 1:]
                B, N, C = f.shape; H = int(math.sqrt(N))
                feats.append(f.permute(0, 2, 1).reshape(B, C, H, H))
        x  = vis.ln_post(x.permute(1, 0, 2))
        gf = x[:, 0]
        if vis.proj is not None:
            gf = gf @ vis.proj
        return gf, feats

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text_concepts(self, device: torch.device) -> torch.Tensor:
        cache_key = "|".join(self.config.clinical_concepts)
        if cache_key in self._concept_cache:
            return self._concept_cache[cache_key]
        with torch.no_grad():
            tokens = self._tokenize(self.config.clinical_concepts, device)
            tf = F.normalize(self.clip_model.encode_text(tokens), dim=-1)
        result = tf.unsqueeze(0) + self.concept_embeddings
        self._concept_cache[cache_key] = result
        return result

    def encode_single_text(self, text: str, device: torch.device) -> torch.Tensor:
        cache_key = f"__single__{text}"
        if cache_key in self._concept_cache:
            return self._concept_cache[cache_key].to(device)
        with torch.no_grad():
            tokens = self._tokenize([text], device)
            result = F.normalize(self.clip_model.encode_text(tokens)[0], dim=-1)
        self._concept_cache[cache_key] = result.cpu()
        return result

    def _tokenize(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if self.tokenizer_type == "hf":
            enc = self.hf_tokenizer(
                texts, padding="max_length", truncation=True,
                max_length=self.context_length, return_tensors="pt",
            )
            return enc["input_ids"].to(device)
        import open_clip
        return self.tokenizer(texts, context_length=self.context_length).to(device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: Dict, epoch: int = 0, compute_loss: bool = True) -> Dict:
        images = batch["image"]
        masks  = batch["mask"]
        device = images.device
        B      = images.shape[0]

        global_feat, multi_scale_feats = self.encode_image_multiscale(images)
        fused_fpn, fpn_smoothed = self.fpn(multi_scale_feats)
        fused = self.feat_proj(fused_fpn)

        concept_feats                = self.encode_text_concepts(device).expand(B, -1, -1)
        deformed, uncertainty        = self.morphological_module(fused)
        concept_maps, concept_seg    = self.semantic_module(deformed, concept_feats)
        concept_seg = F.interpolate(concept_seg, masks.shape[-2:], mode="bilinear", align_corners=False)

        seg_logit, bnd_logit, aux_outputs = self.decoder(deformed, fpn_smoothed, masks.shape[-2:])
        edge_logit = F.interpolate(self.edge_head(deformed), masks.shape[-2:],
                                   mode="bilinear", align_corners=False)

        combined_seg = 0.5 * seg_logit + 0.3 * concept_seg
        seg_probs    = torch.sigmoid(combined_seg.squeeze(1))
        bnd_probs    = torch.sigmoid(bnd_logit.squeeze(1))
        edge_probs   = torch.sigmoid(edge_logit.squeeze(1))

        inter      = (seg_probs * masks).sum()
        batch_dice = (2 * inter + 1) / (seg_probs.sum() + masks.sum() + 1)

        if not compute_loss:
            return {"loss": torch.tensor(0.0), "dice": batch_dice.item(),
                    "seg_preds": seg_probs, "bnd_preds": bnd_probs,
                    "concept_maps": concept_maps, "uncertainty": uncertainty}

        cfg = self.config
        tv_l      = self.tversky_loss(seg_probs, masks)
        iou_l     = self.iou_loss(seg_probs, masks)
        dice_l    = self.dice_loss(seg_probs, masks)
        focal_l   = self.focal_loss(seg_probs, masks)
        bnd_l     = self.boundary_loss(seg_probs, masks)
        edge_l    = self.edge_loss(seg_probs, masks)
        contour_l = self.contour_loss(seg_probs, masks)
        bmap_l    = self.bmap_loss(bnd_probs, masks)
        aux_l     = sum(self.tversky_loss(torch.sigmoid(a.squeeze(1)), masks) * w
                        for a, w in zip(aux_outputs, [0.2, 0.3, 0.5]))
        concept_l = self.dice_loss(torch.sigmoid(concept_seg.squeeze(1)), masks)

        text_for_pgc = F.normalize(concept_feats.mean(dim=1), dim=-1)
        pgc_l        = self.pgc_loss(deformed, text_for_pgc, masks)

        bnd_text = self.encode_single_text(cfg.boundary_concept, device)
        int_text = self.encode_single_text(cfg.interior_concept, device)
        pbc_l    = self.pbc_loss(deformed, bnd_text, int_text, masks)

        total_loss = (
            cfg.lambda_seg      * tv_l      +
            cfg.lambda_iou      * iou_l     +
            cfg.lambda_dice     * dice_l    +
            cfg.lambda_focal    * focal_l   +
            cfg.lambda_boundary * bnd_l     +
            cfg.lambda_edge     * edge_l    +
            cfg.lambda_contour  * contour_l +
            cfg.lambda_bmap     * bmap_l    +
            cfg.lambda_deep_sup * aux_l     +
            cfg.lambda_concept  * concept_l +
            cfg.lambda_pgc      * pgc_l     +
            cfg.lambda_pbc      * pbc_l
        )

        return {"loss": total_loss, "dice": batch_dice.item(),
                "seg_preds": seg_probs, "bnd_preds": bnd_probs,
                "edge_preds": edge_probs, "concept_maps": concept_maps,
                "uncertainty": uncertainty}
