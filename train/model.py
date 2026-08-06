"""LDNet with 56 outputs + helpers to expand pretrained 55-landmark weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from train.shgnet_base import LDNet
from train.config import NUM_LANDMARKS_55, NUM_LANDMARKS_56, PRETRAINED_55


def build_ldnet56(
    nstack: int = 2,
    layer: int = 4,
    in_channel: int = 256,
) -> LDNet:
    return LDNet(
        nstack=nstack,
        layer=layer,
        in_channel=in_channel,
        out_channel=NUM_LANDMARKS_56,
    )


def _expand_conv_weight(
    old_w: torch.Tensor, old_b: Optional[torch.Tensor], new_out: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy first old_out channels; init new channel(s) from mean of existing."""
    old_out = old_w.shape[0]
    new_w = old_w.new_zeros((new_out, *old_w.shape[1:]))
    new_b = old_w.new_zeros(new_out)
    new_w[:old_out] = old_w
    if old_b is not None:
        new_b[:old_out] = old_b
    # init piercing channel from mean of landmark channels
    new_w[old_out:] = old_w.mean(dim=0, keepdim=True)
    if old_b is not None:
        new_b[old_out:] = old_b.mean()
    return new_w, new_b


def load_pretrained_expand_to_56(
    checkpoint: Path | str = PRETRAINED_55,
    device: str | torch.device = "cpu",
) -> tuple[LDNet, Dict[str, Any]]:
    """
    Load 55-LM checkpoint, build 56-LM LDNet, copy shared weights,
    expand final heatmap heads (outs + merge_pred) from 55 → 56.
    """
    path = Path(checkpoint)
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(f"Invalid checkpoint (missing model_state_dict): {path}")

    arch = ckpt.get("arch") or {}
    nstack = int(arch.get("nstack", 2))
    layer = int(arch.get("layer", 4))
    in_channel = int(arch.get("in_channel", 256))
    old_out = int(arch.get("out_channel", NUM_LANDMARKS_55))

    model = build_ldnet56(nstack=nstack, layer=layer, in_channel=in_channel)
    state = ckpt["model_state_dict"]
    new_state = model.state_dict()

    for key, tensor in state.items():
        if key not in new_state:
            continue
        target = new_state[key]
        if tensor.shape == target.shape:
            new_state[key] = tensor
            continue
        # Expand outs.*.conv.weight/bias and merge_pred.*.conv.weight
        if "outs." in key and key.endswith(".conv.weight") and tensor.ndim == 4:
            bias_key = key.replace(".weight", ".bias")
            old_b = state.get(bias_key)
            w, b = _expand_conv_weight(tensor, old_b, NUM_LANDMARKS_56)
            new_state[key] = w
            if bias_key in new_state:
                new_state[bias_key] = b
            continue
        if "outs." in key and key.endswith(".conv.bias"):
            # handled with weight
            continue
        if "merge_pred." in key and key.endswith(".conv.weight") and tensor.ndim == 4:
            # merge_pred: Conv(out_channel → in_channel), so in_channels expand
            # weight shape: [in_channel, out_channel, 1, 1]
            oc_old = tensor.shape[1]
            if oc_old == old_out and target.shape[1] == NUM_LANDMARKS_56:
                w = target.clone()
                w[:, :oc_old] = tensor
                w[:, oc_old:] = tensor.mean(dim=1, keepdim=True)
                new_state[key] = w
            continue

    model.load_state_dict(new_state, strict=True)
    model.to(device)

    meta = {
        "arch": {
            "nstack": nstack,
            "layer": layer,
            "in_channel": in_channel,
            "out_channel": NUM_LANDMARKS_56,
        },
        "source_checkpoint": str(path),
        "old_out_channel": old_out,
    }
    return model, meta


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_final_layer(model: LDNet) -> None:
    """Stage 1: train only heatmap output heads."""
    freeze_all(model)
    for m in model.outs:
        for p in m.parameters():
            p.requires_grad = True


def unfreeze_last_hourglass(model: LDNet) -> None:
    """Stage 2: last hourglass + feature + outs (+ merge if present)."""
    freeze_all(model)
    last = model.nstack - 1
    for p in model.hourglass[last].parameters():
        p.requires_grad = True
    for p in model.feature[last].parameters():
        p.requires_grad = True
    for p in model.outs[last].parameters():
        p.requires_grad = True
    # also keep earlier outs trainable lightly for stack consistency
    for m in model.outs:
        for p in m.parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
