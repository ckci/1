"""
Partial implementation of the improved YOLOv8s model described in:
"Lightweight apple surface defect detection using an improved YOLOv8s model".

Code availability and scope
---------------------------
This file contains the principal network components and a compact training and
inference pipeline associated with the manuscript. It is not the complete
internal project repository. To protect project confidentiality,
project-specific datasets, deployment interfaces, experiment records, private
file paths, internal engineering utilities, and other confidential components
are not included.

The manuscript specifies the main modules but does not disclose every
layer-level configuration or internal training detail. Where a detail is not
specified, this file uses a documented, technically reasonable implementation
choice. Exact numerical results also depend on the dataset, preprocessing,
software environment, hardware, and undistributed project components.

Implemented components
----------------------
1. DualConv: parallel 3x3 grouped and 1x1 pointwise convolution, summed.
2. HWD: fixed Haar transform -> LL/LH/HL/HH -> Conv-BN-ReLU.
3. C2PSA: split channels, apply PSA blocks (MHA + FFN) to one branch,
   concatenate and fuse.
4. AFPN: progressive C3/C4 fusion followed by adaptive spatial fusion of
   C3/C4/C5 into P3/P4/P5.
5. Anchor-free DFL detection heads at strides 8, 16, and 32.

Paper settings used by the training entry point
------------------------------------------------
Python 3.8.10, PyTorch 2.0.0, CUDA 11.8, 640x640 input, 100 epochs,
batch size 16, Adam, lr=0.001, weight decay=0.0005, momentum=0.937.
The paper reports two classes: defective apple and non-defective apple.

Expected YOLO-format dataset layout
-----------------------------------
dataset_root/
    images/train/*.jpg
    labels/train/*.txt
    images/val/*.jpg
    labels/val/*.txt

Each label row is: class_id center_x center_y width height (normalized).

Examples
--------
    python improved_yolov8s.py --mode smoke
    python improved_yolov8s.py --mode train --data D:/apple_dataset
    python improved_yolov8s.py --mode predict \
        --weights runs_paper/best.pt --image test.jpg --output prediction.jpg

Dependencies
------------
    pip install torch pillow numpy
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ("defective apple", "non-defective apple")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autopad(kernel_size: int, dilation: int = 1) -> int:
    effective = dilation * (kernel_size - 1) + 1
    return effective // 2


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


def valid_groups(c1: int, c2: int, requested: int) -> int:
    """Largest group count <= requested that divides both channel counts."""
    return max(g for g in range(min(requested, c1, c2), 0, -1) if c1 % g == 0 and c2 % g == 0)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride,
            autopad(kernel_size),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        if activation == "relu":
            self.act: nn.Module = nn.ReLU(inplace=True)
        elif activation == "identity":
            self.act = nn.Identity()
        else:
            self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DualConv(nn.Module):
    """Paper-style DualConv: grouped 3x3 branch + pointwise 1x1 branch."""

    def __init__(
        self,
        c1: int,
        c2: int,
        stride: int = 1,
        groups: int = 4,
        activation: bool = True,
    ) -> None:
        super().__init__()
        g = valid_groups(c1, c2, groups)
        self.grouped = nn.Conv2d(c1, c2, 3, stride, 1, groups=g, bias=False)
        self.pointwise = nn.Conv2d(c1, c2, 1, stride, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.grouped(x) + self.pointwise(x)))


class DualBottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True, groups: int = 4) -> None:
        super().__init__()
        hidden = make_divisible(channels * 0.5)
        self.cv1 = DualConv(channels, hidden, groups=groups)
        self.cv2 = DualConv(hidden, channels, groups=groups, activation=False)
        self.use_shortcut = shortcut
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.cv2(self.cv1(x))
        return self.act(x + y) if self.use_shortcut else self.act(y)


class C2fDual(nn.Module):
    """YOLOv8-like C2f block whose internal 3x3 operations use DualConv."""

    def __init__(self, c1: int, c2: int, repeats: int = 1, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = make_divisible(c2 * expansion)
        self.hidden = hidden
        self.cv1 = ConvBNAct(c1, 2 * hidden, 1)
        self.blocks = nn.ModuleList(DualBottleneck(hidden) for _ in range(repeats))
        self.cv2 = ConvBNAct((2 + repeats) * hidden, c2, 1)

    def forward(self, x: Tensor) -> Tensor:
        chunks = list(self.cv1(x).split((self.hidden, self.hidden), dim=1))
        for block in self.blocks:
            chunks.append(block(chunks[-1]))
        return self.cv2(torch.cat(chunks, dim=1))


class HaarDownsample(nn.Module):
    """Lossless Haar subband encoding followed by Conv-BN-ReLU."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.project = ConvBNAct(4 * c1, c2, 1, activation="relu")

    @staticmethod
    def haar(x: Tensor) -> Tensor:
        # Pad odd dimensions so every sample has a complete 2x2 Haar cell.
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return torch.cat((ll, lh, hl, hh), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.project(self.haar(x))


class Attention2D(nn.Module):
    """Multi-head self-attention on flattened spatial positions."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        heads = max(1, min(heads, channels))
        while channels % heads:
            heads -= 1
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).reshape(b, 3, self.heads, self.head_dim, n)
        q, k, v = qkv.unbind(dim=1)
        weights = torch.matmul(q.transpose(-2, -1), k) * self.scale
        weights = weights.softmax(dim=-1)
        out = torch.matmul(v, weights.transpose(-2, -1))
        out = out.reshape(b, c, h, w)
        return self.proj(out)


class PSABlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = make_divisible(channels * expansion)
        self.attn = Attention2D(channels, heads)
        self.ffn = nn.Sequential(
            ConvBNAct(channels, hidden, 1),
            ConvBNAct(hidden, channels, 1, activation="identity"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    """C2PSA reconstructed from Fig. 8: split, PSA stack, concat, fuse."""

    def __init__(self, channels: int, repeats: int = 1, heads: int = 4) -> None:
        super().__init__()
        hidden = channels // 2
        self.pre = ConvBNAct(channels, 2 * hidden, 1)
        self.blocks = nn.Sequential(*(PSABlock(hidden, heads) for _ in range(repeats)))
        self.post = ConvBNAct(2 * hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        bypass, attended = self.pre(x).chunk(2, dim=1)
        attended = self.blocks(attended)
        return self.post(torch.cat((bypass, attended), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden = c1 // 2
        self.cv1 = ConvBNAct(c1, hidden, 1)
        self.cv2 = ConvBNAct(4 * hidden, c2, 1)
        self.pool = nn.MaxPool2d(kernel_size, 1, kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class AdaptiveSpatialFusion(nn.Module):
    """Align multiple levels and learn a location-dependent softmax weight."""

    def __init__(self, input_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(ConvBNAct(c, out_channels, 1) for c in input_channels)
        compressed = max(8, out_channels // 16)
        self.weight_features = nn.ModuleList(
            ConvBNAct(out_channels, compressed, 1) for _ in input_channels
        )
        self.weight_logits = nn.Conv2d(compressed * len(input_channels), len(input_channels), 1)
        self.refine = ConvBNAct(out_channels, out_channels, 3)

    def forward(self, features: Sequence[Tensor], target_size: Tuple[int, int]) -> Tensor:
        aligned: List[Tensor] = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(feature)

        logits = self.weight_logits(torch.cat([
            encoder(feature) for encoder, feature in zip(self.weight_features, aligned)
        ], dim=1))
        weights = logits.softmax(dim=1)
        fused = sum(feature * weights[:, i : i + 1] for i, feature in enumerate(aligned))
        return self.refine(fused)


class AFPN(nn.Module):
    """Progressive two-level then three-level adaptive feature fusion."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        c3, c4, c5 = channels
        self.c3_proj = ConvBNAct(c3, c3, 1)
        self.c4_proj = ConvBNAct(c4, c4, 1)
        self.c5_proj = ConvBNAct(c5, c5, 1)

        self.stage2_p3 = AdaptiveSpatialFusion((c3, c4), c3)
        self.stage2_p4 = AdaptiveSpatialFusion((c3, c4), c4)

        self.stage3_p3 = AdaptiveSpatialFusion((c3, c4, c5), c3)
        self.stage3_p4 = AdaptiveSpatialFusion((c3, c4, c5), c4)
        self.stage3_p5 = AdaptiveSpatialFusion((c3, c4, c5), c5)

    def forward(self, inputs: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        c3, c4, c5 = inputs
        c3 = self.c3_proj(c3)
        c4 = self.c4_proj(c4)
        c5 = self.c5_proj(c5)

        p3_2 = self.stage2_p3((c3, c4), c3.shape[-2:])
        p4_2 = self.stage2_p4((c3, c4), c4.shape[-2:])

        p3 = self.stage3_p3((p3_2, p4_2, c5), p3_2.shape[-2:])
        p4 = self.stage3_p4((p3_2, p4_2, c5), p4_2.shape[-2:])
        p5 = self.stage3_p5((p3_2, p4_2, c5), c5.shape[-2:])
        return p3, p4, p5


class ImprovedBackbone(nn.Module):
    """YOLOv8s-width backbone reconstructed from Fig. 4."""

    def __init__(self, width: float = 0.5, depth: float = 0.33) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = [make_divisible(v * width) for v in (64, 128, 256, 512, 1024)]
        n1 = max(1, round(3 * depth))
        n2 = max(1, round(6 * depth))
        n3 = max(1, round(6 * depth))
        n4 = max(1, round(3 * depth))

        self.channels = (c3, c4, c5)
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.stage2 = nn.Sequential(
            DualConv(c1, c2, stride=2),
            C2fDual(c2, c2, n1),
        )
        self.stage3 = nn.Sequential(
            HaarDownsample(c2, c3),
            C2fDual(c3, c3, n2),
        )
        self.stage4 = nn.Sequential(
            HaarDownsample(c3, c4),
            C2fDual(c4, c4, n3),
        )
        self.stage5 = nn.Sequential(
            HaarDownsample(c4, c5),
            C2fDual(c5, c5, n4),
            C2PSA(c5, repeats=1, heads=4),
            SPPF(c5, c5),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class DFLDetectHead(nn.Module):
    """YOLOv8-like anchor-free head producing 4*reg_max + nc channels."""

    def __init__(self, channels: Sequence[int], num_classes: int = 2, reg_max: int = 16) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        output_channels = 4 * reg_max + num_classes
        self.heads = nn.ModuleList()
        for c in channels:
            hidden = max(64, c // 2)
            self.heads.append(nn.Sequential(
                ConvBNAct(c, hidden, 3),
                ConvBNAct(hidden, hidden, 3),
                nn.Conv2d(hidden, output_channels, 1),
            ))

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        return [head(feature) for head, feature in zip(self.heads, features)]


class ImprovedYOLOv8s(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        width: float = 0.5,
        depth: float = 0.33,
        reg_max: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = (8, 16, 32)
        self.backbone = ImprovedBackbone(width=width, depth=depth)
        self.neck = AFPN(self.backbone.channels)
        self.detect = DFLDetectHead(self.backbone.channels, num_classes, reg_max)

    def forward(self, x: Tensor, decode: bool = False, conf: float = 0.25) -> object:
        features = self.neck(self.backbone(x))
        raw = self.detect(features)
        if decode and not self.training:
            return self.decode(raw, conf_threshold=conf)
        return raw

    def decode(
        self,
        outputs: Sequence[Tensor],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ) -> List[Dict[str, Tensor]]:
        all_boxes: List[Tensor] = []
        all_scores: List[Tensor] = []
        all_labels: List[Tensor] = []
        bins = torch.arange(self.reg_max, device=outputs[0].device, dtype=outputs[0].dtype)

        for prediction, stride in zip(outputs, self.strides):
            b, _, h, w = prediction.shape
            regression = prediction[:, : 4 * self.reg_max]
            classes = prediction[:, 4 * self.reg_max :].sigmoid()
            regression = regression.view(b, 4, self.reg_max, h, w).softmax(dim=2)
            distances = (regression * bins.view(1, 1, -1, 1, 1)).sum(dim=2)

            yy, xx = torch.meshgrid(
                torch.arange(h, device=prediction.device, dtype=prediction.dtype),
                torch.arange(w, device=prediction.device, dtype=prediction.dtype),
                indexing="ij",
            )
            cx = (xx + 0.5) * stride
            cy = (yy + 0.5) * stride
            boxes = torch.stack((
                cx[None] - distances[:, 0] * stride,
                cy[None] - distances[:, 1] * stride,
                cx[None] + distances[:, 2] * stride,
                cy[None] + distances[:, 3] * stride,
            ), dim=-1).view(b, -1, 4)
            scores, labels = classes.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes).max(dim=-1)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        boxes = torch.cat(all_boxes, dim=1)
        scores = torch.cat(all_scores, dim=1)
        labels = torch.cat(all_labels, dim=1)
        results: List[Dict[str, Tensor]] = []
        for batch_index in range(boxes.shape[0]):
            mask = scores[batch_index] >= conf_threshold
            b_boxes = boxes[batch_index][mask]
            b_scores = scores[batch_index][mask]
            b_labels = labels[batch_index][mask]
            keep_all: List[Tensor] = []
            for class_id in b_labels.unique():
                class_mask = b_labels == class_id
                class_indices = class_mask.nonzero(as_tuple=False).squeeze(1)
                kept = nms(b_boxes[class_mask], b_scores[class_mask], iou_threshold)
                keep_all.append(class_indices[kept])
            if keep_all:
                keep = torch.cat(keep_all)
                keep = keep[b_scores[keep].argsort(descending=True)[:max_detections]]
            else:
                keep = torch.empty(0, dtype=torch.long, device=boxes.device)
            results.append({
                "boxes": b_boxes[keep],
                "scores": b_scores[keep],
                "labels": b_labels[keep],
            })
        return results


def box_iou_one_to_many(box: Tensor, boxes: Tensor) -> Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    inter = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    area1 = (box[2:] - box[:2]).clamp(min=0).prod()
    area2 = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    return inter / (area1 + area2 - inter + 1e-7)


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    kept: List[Tensor] = []
    while order.numel() > 0:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou_one_to_many(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]
    return torch.stack(kept)


def dfl_cross_entropy(logits: Tensor, target: Tensor, reg_max: int) -> Tensor:
    target = target.clamp(0, reg_max - 1 - 1e-3)
    left = target.floor().long()
    right = (left + 1).clamp(max=reg_max - 1)
    weight_right = target - left.float()
    weight_left = 1.0 - weight_right
    return (
        F.cross_entropy(logits, left, reduction="none") * weight_left
        + F.cross_entropy(logits, right, reduction="none") * weight_right
    ).mean()


def pairwise_iou_loss(pred_box: Tensor, target_box: Tensor) -> Tensor:
    top_left = torch.maximum(pred_box[:2], target_box[:2])
    bottom_right = torch.minimum(pred_box[2:], target_box[2:])
    inter = (bottom_right - top_left).clamp(min=0).prod()
    area_p = (pred_box[2:] - pred_box[:2]).clamp(min=0).prod()
    area_t = (target_box[2:] - target_box[:2]).clamp(min=0).prod()
    iou = inter / (area_p + area_t - inter + 1e-7)
    return 1.0 - iou


class ReferenceDetectionLoss(nn.Module):
    """
    Compact center-cell assignment loss for training experiments.

    The manuscript does not specify a modified YOLOv8 loss. This implementation
    follows the anchor-free DFL idea but is intentionally simpler than the
    official Ultralytics task-aligned assigner.
    """

    def __init__(self, num_classes: int = 2, reg_max: int = 16) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = (8, 16, 32)

    @staticmethod
    def choose_level(width_px: float, height_px: float) -> int:
        scale = math.sqrt(max(width_px * height_px, 1.0))
        return 0 if scale < 64 else (1 if scale < 160 else 2)

    def forward(self, outputs: Sequence[Tensor], targets: Tensor, image_size: int) -> Dict[str, Tensor]:
        device = outputs[0].device
        cls_loss = torch.zeros((), device=device)
        box_loss = torch.zeros((), device=device)
        dfl_loss = torch.zeros((), device=device)
        positives = 0

        cls_targets = [torch.zeros_like(p[:, 4 * self.reg_max :]) for p in outputs]
        assignments: List[Tuple[int, int, int, int, Tensor]] = []

        for row in targets:
            batch_id = int(row[0].item())
            class_id = int(row[1].item())
            cx, cy, bw, bh = row[2:].tolist()
            level = self.choose_level(bw * image_size, bh * image_size)
            _, _, h, w = outputs[level].shape
            gx = min(w - 1, max(0, int(cx * w)))
            gy = min(h - 1, max(0, int(cy * h)))
            cls_targets[level][batch_id, class_id, gy, gx] = 1.0
            assignments.append((level, batch_id, gy, gx, row))

        for prediction, target_map in zip(outputs, cls_targets):
            logits = prediction[:, 4 * self.reg_max :]
            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(logits, target_map, reduction="mean")

        bins = torch.arange(self.reg_max, device=device, dtype=outputs[0].dtype)
        for level, batch_id, gy, gx, row in assignments:
            prediction = outputs[level]
            _, _, h, w = prediction.shape
            cx, cy, bw, bh = row[2:].to(device)
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            center_x = torch.tensor(gx + 0.5, device=device, dtype=prediction.dtype)
            center_y = torch.tensor(gy + 0.5, device=device, dtype=prediction.dtype)
            target_distances = torch.stack((center_x - x1, center_y - y1, x2 - center_x, y2 - center_y))

            reg_logits = prediction[batch_id, : 4 * self.reg_max, gy, gx].view(4, self.reg_max)
            dfl_loss = dfl_loss + dfl_cross_entropy(reg_logits, target_distances, self.reg_max)
            pred_distances = (reg_logits.softmax(dim=1) * bins[None]).sum(dim=1)
            pred_box = torch.stack((
                center_x - pred_distances[0],
                center_y - pred_distances[1],
                center_x + pred_distances[2],
                center_y + pred_distances[3],
            ))
            target_box = torch.stack((x1, y1, x2, y2))
            box_loss = box_loss + pairwise_iou_loss(pred_box, target_box)
            positives += 1

        normalizer = max(positives, 1)
        box_loss = box_loss / normalizer
        dfl_loss = dfl_loss / normalizer
        total = 7.5 * box_loss + 0.5 * cls_loss + 1.5 * dfl_loss
        return {"total": total, "box": box_loss, "cls": cls_loss, "dfl": dfl_loss}


class YoloTxtDataset(Dataset):
    def __init__(self, root: Path, split: str = "train", image_size: int = 640, augment: bool = False) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.image_size = image_size
        self.augment = augment
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")
        self.images = sorted(p for p in self.image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.images:
            raise FileNotFoundError(f"No images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _augment(self, image: Image.Image) -> Image.Image:
        # Matches the manuscript's reported inversion, blur, and brightness changes.
        if random.random() < 0.15:
            image = ImageOps.invert(image)
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
        if random.random() < 0.30:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
        return image

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        if self.augment:
            image = self._augment(image)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

        relative = image_path.relative_to(self.image_dir).with_suffix(".txt")
        label_path = self.label_dir / relative
        labels: List[List[float]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(v) for v in parts])
        label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        return image_tensor, label_tensor, str(image_path)


def collate_yolo(batch: Sequence[Tuple[Tensor, Tensor, str]]) -> Tuple[Tensor, Tensor, List[str]]:
    images, labels, paths = zip(*batch)
    merged: List[Tensor] = []
    for batch_index, label in enumerate(labels):
        if label.numel():
            ids = torch.full((label.shape[0], 1), batch_index, dtype=label.dtype)
            merged.append(torch.cat((ids, label), dim=1))
    targets = torch.cat(merged, dim=0) if merged else torch.zeros((0, 6), dtype=torch.float32)
    return torch.stack(images), targets, list(paths)


@torch.no_grad()
def evaluate_loss(
    model: ImprovedYOLOv8s,
    loader: DataLoader,
    criterion: ReferenceDetectionLoss,
    device: torch.device,
    image_size: int,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        losses = criterion(model(images), targets, image_size)
        total += float(losses["total"].item())
        batches += 1
    return total / max(batches, 1)


def train_reference(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_set = YoloTxtDataset(Path(args.data), "train", args.imgsz, augment=True)
    val_set = YoloTxtDataset(Path(args.data), "val", args.imgsz, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_yolo,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_yolo,
    )

    model = ImprovedYOLOv8s(num_classes=args.classes).to(device)
    criterion = ReferenceDetectionLoss(args.classes, model.reg_max)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.momentum, 0.999),
    )
    output_dir = Path(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (images, targets, _) in enumerate(train_loader, start=1):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(images), targets, args.imgsz)
            losses["total"].backward()
            optimizer.step()
            running += float(losses["total"].item())
            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch:03d}/{args.epochs} step {step:04d}/{len(train_loader)} "
                    f"loss={running / step:.4f} box={losses['box'].item():.4f} "
                    f"cls={losses['cls'].item():.4f} dfl={losses['dfl'].item():.4f}"
                )

        val_loss = evaluate_loss(model, val_loader, criterion, device, args.imgsz)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "class_names": CLASS_NAMES[: args.classes],
            "partial_code_release": True,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, output_dir / "best.pt")
        print(f"epoch {epoch:03d}: validation loss={val_loss:.4f}, best={best_loss:.4f}")


def load_model(weights: str, classes: int, device: torch.device) -> ImprovedYOLOv8s:
    model = ImprovedYOLOv8s(num_classes=classes).to(device)
    checkpoint = torch.load(weights, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_image(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.weights, args.classes, device)
    original = Image.open(args.image).convert("RGB")
    original_w, original_h = original.size
    resized = original.resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    detections = model(tensor, decode=True, conf=args.conf)[0]

    draw = ImageDraw.Draw(original)
    scale_x = original_w / args.imgsz
    scale_y = original_h / args.imgsz
    palette = ((220, 45, 45), (45, 170, 70), (60, 100, 220))
    for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
        x1, y1, x2, y2 = box.detach().cpu().tolist()
        coords = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        class_id = int(label.item())
        color = palette[class_id % len(palette)]
        draw.rectangle(coords, outline=color, width=3)
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        draw.text((coords[0] + 3, max(0, coords[1] - 14)), f"{name} {score.item():.2f}", fill=color)
    original.save(args.output)
    print(f"Saved prediction to {args.output}")


def smoke_test(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ImprovedYOLOv8s(num_classes=args.classes).to(device).eval()
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        outputs = model(x)
    parameters = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Parameters: {parameters / 1e6:.3f} M")
    for stride, output in zip(model.strides, outputs):
        print(f"stride {stride}: {tuple(output.shape)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Partial implementation of the improved YOLOv8s model")
    parser.add_argument("--mode", choices=("smoke", "train", "predict"), default="smoke")
    parser.add_argument("--data", type=str, default="", help="YOLO-format dataset root")
    parser.add_argument("--image", type=str, default="", help="Input image for prediction")
    parser.add_argument("--weights", type=str, default="", help="Checkpoint path")
    parser.add_argument("--output", type=str, default="prediction.jpg")
    parser.add_argument("--run-dir", type=str, default="runs_paper")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        if not args.data:
            raise SystemExit("--data is required in train mode")
        train_reference(args)
    elif args.mode == "predict":
        if not args.image or not args.weights:
            raise SystemExit("--image and --weights are required in predict mode")
        predict_image(args)
    else:
        smoke_test(args)


if __name__ == "__main__":
    main()
