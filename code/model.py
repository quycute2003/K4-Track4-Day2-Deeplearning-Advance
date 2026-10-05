"""timm backbones, frozen features, optimizer groups and fvcore compute counts."""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50", "resnext50": "resnext50_32x4d",
    # timm 1.0.30 defaults ConvNeXt-Tiny to ImageNet-12k pretraining.
    # Use the explicit ImageNet-1k tag for the baseline backbone comparison.
    "convnext_tiny": "convnext_tiny.fb_in1k", "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224", "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init="finetune",
                *, img_size=224):
    import timm
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"Unknown init: {init}")
    if init == "frozen" and not pretrained:
        raise ValueError("frozen initialization requires pretrained weights")
    name = SUGGESTED_BACKBONES.get(name, name)
    kwargs = {"img_size": img_size} if name.startswith(("vit_", "deit_", "swin_")) else {}
    model = timm.create_model(name, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate, **kwargs)
    if init == "frozen":
        freeze_backbone(model)
    return model


def _frozen_train(self, mode=True):
    """Keep features (including functional dropout) in eval after model.train()."""
    nn.Module.train(self, False)
    self.get_classifier().train(mode)
    return self


def freeze_backbone(model):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in head_ids)
    model.train = MethodType(_frozen_train, model)
    model.train(True)


def param_groups(model, lr_backbone, lr_head, weight_decay):
    """Three conceptual groups: decayed backbone, norm/bias, new head.

    Split the head into two optimizer groups to also exclude head bias/norm from
    decay, as required by README. Every trainable parameter occurs exactly once.
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    norm_types = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm)
    norm_ids = {id(p) for m in model.modules() if isinstance(m, norm_types)
                for p in m.parameters(recurse=False)}
    no_decay_names = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = id(p) in head_ids
        no_decay = p.ndim <= 1 or id(p) in norm_ids or name in no_decay_names
        key = (head, no_decay)
        groups.setdefault(key, {"params": [], "lr": lr_head if head else lr_backbone,
                                "weight_decay": 0.0 if no_decay else weight_decay,
                                "name": ("head" if head else "backbone") +
                                        ("_no_decay" if no_decay else "_decay")})["params"].append(p)
    if not groups:
        raise ValueError("No trainable parameters")
    return list(groups.values())


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size=224):
    """fvcore counts one fused multiply-add as one operation (MAC convention).

    Record unsupported operations on model.gmac_report; do not hide them.
    Temporarily disable fused attention so matmul attention is traced by fvcore.
    """
    from fvcore.nn import FlopCountAnalysis
    modes = {m: m.training for m in model.modules()}
    fused = {m: m.fused_attn for m in model.modules() if hasattr(m, "fused_attn")}
    try:
        model.eval()
        for m in fused:
            m.fused_attn = False
        p = next(model.parameters())
        x = torch.zeros(1, 3, img_size, img_size, device=p.device, dtype=p.dtype)
        with torch.no_grad():
            analysis = FlopCountAnalysis(model, x)
            analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            total = analysis.total()
        model.gmac_report = {"tool": "fvcore (1 FMA = 1 operation)",
                             "unsupported_ops": dict(analysis.unsupported_ops())}
        return total / 1e9
    finally:
        for m, value in fused.items():
            m.fused_attn = value
        for m, mode in modes.items():
            m.training = mode
