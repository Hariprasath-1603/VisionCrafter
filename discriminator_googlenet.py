"""GoogLeNet-based discriminator producing logits (no sigmoid).

Pretrained weights (ImageNet-1K) are pulled automatically by torchvision on first
use and cached under ~/.cache/torch/hub/checkpoints. No manual downloads needed.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision import models


def _apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Recursively wrap Conv/Linear layers with spectral norm for stability."""
    for name, layer in module.named_children():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            setattr(module, name, spectral_norm(layer))
        else:
            _apply_spectral_norm(layer)
    return module


class GoogLeNetDiscriminator(nn.Module):
    """Wrap torchvision GoogLeNet to output a single logit for real/fake.

    Example usage:
        from torchvision.models import googlenet, GoogLeNet_Weights
        model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)

    Pretrained weights auto-download to ~/.cache/torch/hub/checkpoints.
    """

    def __init__(self, pretrained: bool = True, use_spectral_norm: bool = False, freeze_backbone: bool = False):
        super().__init__()
        # Auto-download ImageNet weights if pretrained=True
        # Note: pretrained GoogLeNet requires aux_logits=True during loading, then we disable them
        if not pretrained:
            print("[WARN] pretrained=False: discriminator will train from scratch (no ImageNet weights).")
            self.backbone = models.googlenet(weights=None, aux_logits=False, transform_input=False)
        else:
            # Load with aux_logits=True (required by pretrained weights), then disable
            self.backbone = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=True, transform_input=False)
            self.backbone.aux_logits = False
            self.backbone.aux1 = None
            self.backbone.aux2 = None

        # Optionally freeze all backbone layers except final head
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final FC layer with single-logit head for real/fake classification
        in_feats = self.backbone.fc.in_features
        fc = nn.Linear(in_feats, 1)
        if use_spectral_norm:
            fc = spectral_norm(fc)
        self.backbone.fc = fc

        # Ensure head is always trainable even if backbone is frozen
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape: (N, 3, H, W); returns logits shaped [N, 1]
        out = self.backbone(x)
        return out.view(-1, 1)
