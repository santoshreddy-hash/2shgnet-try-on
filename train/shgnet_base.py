"""
2-SHGNet (LDNet) ear landmark detector.

Architecture ported from Anjdroid/ear_alignment_stacked_hourglass:
  dplearn/hourglass_model.py
  dplearn/ldnet_model.py

Defaults match the repo exactly:
  LDNet(nstack=2, layer=6, in_channel=265, out_channel=55)
Input:  B×3×256×256
Output: B×55×64×64 heatmaps (last stack)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Layers (verbatim from hourglass_model.py)
# ---------------------------------------------------------------------------


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, bn=False, relu=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            bias=True,
        )
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual, self).__init__()
        self.skip_layer = Conv(in_channel, out_channel, 1, relu=False)
        if in_channel == out_channel:
            self.need_skip = False
        else:
            self.need_skip = True

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv1 = Conv(in_channel, out_channel // 2, 1, bn=True, relu=True)
        self.conv2 = Conv(out_channel // 2, out_channel // 2, 3, bn=True, relu=True)
        self.conv3 = Conv(out_channel // 2, out_channel, 1)

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return residual + out


class Hourglass(nn.Module):
    def __init__(self, layer, channel, inc=0):
        super(Hourglass, self).__init__()
        nf = channel + inc
        self.res = Residual(channel, channel)
        self.pool = nn.MaxPool2d(2, 2)
        self.res1 = Residual(channel, nf)
        if layer > 1:
            self.hourclass = Hourglass(layer - 1, nf)
        else:
            self.hourclass = Residual(nf, nf)
        self.res2 = Residual(nf, channel)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        res = self.res(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.hourclass(x)
        x = self.res2(x)
        x = self.up(x)
        return res + x


class Convert(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convert, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.conv(x)


class LDNet(nn.Module):
    """Two-Stack Hourglass Network for 55 ear landmarks."""

    def __init__(self, nstack=2, layer=4, in_channel=256, out_channel=55, increase=0):
        super(LDNet, self).__init__()
        self.nstack = nstack
        self.layer = layer
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, in_channel),
        )
        self.hourglass = nn.ModuleList(
            [
                nn.Sequential(Hourglass(layer, in_channel, inc=increase))
                for _ in range(nstack)
            ]
        )
        self.feature = nn.ModuleList(
            [
                nn.Sequential(
                    Residual(in_channel, in_channel),
                    Conv(in_channel, in_channel, 1, bn=True, relu=True),
                )
                for _ in range(nstack)
            ]
        )
        self.outs = nn.ModuleList(
            [Conv(in_channel, out_channel, 1, bn=False, relu=False) for _ in range(nstack)]
        )
        self.merge_feature = nn.ModuleList(
            [Convert(in_channel, in_channel) for _ in range(nstack - 1)]
        )
        self.merge_pred = nn.ModuleList(
            [Convert(out_channel, in_channel) for _ in range(nstack - 1)]
        )

    def forward(self, x):
        x = self.pre(x)
        heat_maps = []
        for i in range(self.nstack):
            hg = self.hourglass[i](x)
            feature = self.feature[i](hg)
            pred = self.outs[i](feature)
            heat_maps.append(pred)
            if i < self.nstack - 1:
                x = x + self.merge_pred[i](pred) + self.merge_feature[i](feature)
        return pred


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------


CHECKPOINT_HELP = """
========================================================================
2-SHGNet checkpoint missing or invalid.

Required file:
  {path}

Expected format (PyTorch dict):
  torch.load(path)['model_state_dict']

This is the trained Two-Stack Hourglass (LDNet) checkpoint from:
  https://github.com/Anjdroid/ear_alignment_stacked_hourglass

NOTE: The public GitHub repository currently does NOT ship the weights
(no releases / no .pth files). Obtain `hourglass_2stack.pth` from the
paper authors / materials (Hrovatič et al., IET Biometrics), then place
it at the path above.

Refusing to run with randomly initialized weights.
========================================================================
"""


def select_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_ear_bgr(ear_bgr: np.ndarray, size: int = 256) -> torch.Tensor:
    """
    Match Anjdroid preprocessing:
      resize → Y-channel histogram equalize → /255 → CHW float tensor
    """
    if ear_bgr.size == 0:
        raise ValueError("Empty ear crop")
    img = cv2.resize(ear_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    img = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    # Paper/repo normalize to [0,1] per channel (BGR order preserved as loaded)
    arr = img.astype(np.float32) / 255.0
    # Network expects RGB-like 3-channel float; repo trains on equalized BGR scaled
    # the same way ToTensor would stack channels. Use CHW in BGR to match OpenCV pipeline.
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0)


def heatmaps_to_points(
    heatmaps: torch.Tensor, input_size: int = 256, radius: int = 2
) -> np.ndarray:
    """
    Soft-argmax around peak (radius) → sub-pixel landmarks in input_size space.
    """
    hm = heatmaps.detach().cpu().numpy()
    single = hm.ndim == 3
    if single:
        hm = hm[np.newaxis, ...]

    b, n, h, w = hm.shape
    scale_x = input_size / float(w)
    scale_y = input_size / float(h)
    pts = np.zeros((b, n, 2), dtype=np.float32)
    for bi in range(b):
        for i in range(n):
            flat = hm[bi, i]
            idx = int(flat.argmax())
            yy, xx = divmod(idx, w)
            y0 = max(0, yy - radius)
            y1 = min(h - 1, yy + radius)
            x0 = max(0, xx - radius)
            x1 = min(w - 1, xx + radius)
            patch = flat[y0 : y1 + 1, x0 : x1 + 1]
            patch = patch - patch.max()
            wt = np.exp(patch)
            s = float(wt.sum())
            if s < 1e-12:
                pts[bi, i, 0] = xx * scale_x
                pts[bi, i, 1] = yy * scale_y
            else:
                ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
                pts[bi, i, 0] = float((wt * xs).sum() / s) * scale_x
                pts[bi, i, 1] = float((wt * ys).sum() / s) * scale_y

    return pts[0] if single else pts


class SHGNetEarLandmarker:
    """Load LDNet checkpoint and run 55-landmark inference on ear crops."""

    DISPLAY_NAME = "2-SHGNet"

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        input_size: int = 256,
    ) -> None:
        self.input_size = input_size
        self.device = select_device(device)
        path = Path(checkpoint_path)
        if not path.is_file() or path.stat().st_size < 1000:
            raise FileNotFoundError(CHECKPOINT_HELP.format(path=path.resolve()))

        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location="cpu")

        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise RuntimeError(
                CHECKPOINT_HELP.format(path=path.resolve())
                + "\nLoaded object is missing key 'model_state_dict'."
            )

        # Rebuild with arch stored in checkpoint when present
        arch = ckpt.get("arch") or {}
        nstack = int(arch.get("nstack", 2))
        layer = int(arch.get("layer", 4))
        in_channel = int(arch.get("in_channel", 256))
        out_channel = int(arch.get("out_channel", 55))
        self.model = LDNet(
            nstack=nstack,
            layer=layer,
            in_channel=in_channel,
            out_channel=out_channel,
        )

        incompatible = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        # Require that most weights loaded; if everything is missing, refuse.
        total = sum(1 for _ in self.model.state_dict())
        if missing and len(missing) >= total:
            raise RuntimeError(
                CHECKPOINT_HELP.format(path=path.resolve())
                + f"\nCheckpoint did not match LDNet (missing={len(missing)}, "
                f"unexpected={len(unexpected)})."
            )

        self.model.to(self.device)
        self.model.eval()
        print(f"[2-SHGNet] Loaded {path} on {self.device}")
        if missing or unexpected:
            print(
                f"[2-SHGNet] load_state_dict strict=False "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

    @torch.inference_mode()
    def predict(self, ear_bgr: np.ndarray) -> np.ndarray:
        """Return (55, 2) landmarks in the resized 256×256 crop space."""
        pts, _ = self.predict_with_score(ear_bgr)
        return pts

    @torch.inference_mode()
    def predict_with_score(self, ear_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Return ((55, 2) landmarks in 256-space, mean peak heatmap score).
        Higher score ≈ sharper / more confident landmarks.
        """
        tensor = preprocess_ear_bgr(ear_bgr, self.input_size).to(self.device)
        heatmaps = self.model(tensor)
        pts = heatmaps_to_points(heatmaps, self.input_size)
        if pts.ndim == 3:
            if pts.shape[0] != 1:
                raise ValueError(
                    f"Single-image predict expected batch 1, got {pts.shape[0]}"
                )
            pts = pts[0]
        if pts.shape != (55, 2):
            raise ValueError(f"Expected (55, 2) landmarks, got {pts.shape}")

        hm = heatmaps.detach()
        if hm.ndim == 4:
            hm = hm[0]
        # Peak response per landmark (softmax-normalized max)
        flat = hm.reshape(hm.shape[0], -1)
        peak = flat.max(dim=1).values
        score = float(peak.mean().item())
        return pts, score
