"""Model factory. Networks return logits of shape (B, 1, H, W)."""

import torch.nn as nn

from experiments.models import TransUNet


def build_model(model_cfg: dict, image_size: int) -> nn.Module:
    name = str(model_cfg.get("name", "transunet")).lower()
    if name != "transunet":
        raise ValueError(f"Unknown model '{name}'. Available: transunet")
    out_channels = int(model_cfg.get("out_channels", 128))
    patch_dim = int(model_cfg.get("patch_dim", 16))
    if out_channels != 128:
        raise ValueError(
            "TransUNet in this repository hardcodes a 512-channel bottleneck, "
            "which matches out_channels=128 only."
        )
    if patch_dim != 16 or int(image_size) % 16 != 0:
        raise ValueError(
            "This TransUNet downsamples four times by 2, so patch_dim must be 16 "
            "and image_size must be a multiple of 16."
        )
    return TransUNet(
        img_dim=int(image_size),
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=out_channels,
        head_num=int(model_cfg.get("head_num", 4)),
        mlp_dim=int(model_cfg.get("mlp_dim", 256)),
        block_num=int(model_cfg.get("block_num", 4)),
        patch_dim=patch_dim,
        class_num=int(model_cfg.get("class_num", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
