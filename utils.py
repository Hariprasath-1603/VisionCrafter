"""Utility helpers for saving images, normalization, checkpoints, EMA, and metadata."""
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch
from torchvision.utils import save_image

# ImageNet normalization stats (used by torchvision backbones). Pretrained weights
# are downloaded automatically to ~/.cache/torch/hub/checkpoints on first use.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_samples(generator: torch.nn.Module, noise: torch.Tensor, epoch: int, out_dir: Path, nrow: int = 8) -> None:
    """Save a grid of generated samples for inspection."""
    generator.eval()
    with torch.no_grad():
        fake = generator(noise.to(next(generator.parameters()).device))
        # Rescale from [-1,1] to [0,1] for saving
        fake = (fake + 1) / 2
        save_image(fake, out_dir / f"samples_epoch_{epoch:03d}.png", nrow=nrow)
    generator.train()


def denormalize_imagenet(t: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized tensor back to [0,1] range for visualization."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(1, 3, 1, 1)
    return (t * std + mean).clamp(0.0, 1.0)


def gen_to_imagenet_norm(gen_tanh: torch.Tensor) -> torch.Tensor:
    """Map generator output in [-1,1] (tanh) -> ImageNet-normalized tensor."""
    mean = torch.tensor(IMAGENET_MEAN, device=gen_tanh.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=gen_tanh.device).view(1, 3, 1, 1)
    gen_01 = (gen_tanh + 1) / 2  # [-1,1] -> [0,1]
    return (gen_01 - mean) / std


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: Path, filename: str = "checkpoint.pt") -> None:
    ensure_dirs(checkpoint_dir)
    torch.save(state, checkpoint_dir / filename)


def load_checkpoint(checkpoint_path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location)


def load_checkpoint_robust(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    map_location: torch.device = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Robust checkpoint loader that handles multiple common checkpoint formats:
      - {'model': state_dict, ...}
      - {'model_state_dict': state_dict, ...}
      - {'state_dict': state_dict, ...}
      - raw state_dict (OrderedDict)
      - full checkpoint with optimizer/scheduler states

    Returns: {'loaded': bool, 'messages': list[str], 'epoch': int or None, ...}
    """
    result = {"loaded": False, "messages": [], "epoch": None}

    if map_location is None:
        map_location = torch.device("cpu")

    if not Path(path).is_file():
        result["messages"].append(f"Checkpoint not found: {path}")
        return result

    try:
        ckpt = torch.load(path, map_location=map_location)
    except Exception as e:
        result["messages"].append(f"Failed to load checkpoint: {e}")
        return result

    # Extract model state_dict from various formats
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("model", "model_state_dict", "state_dict", "generator", "discriminator"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                result["messages"].append(f"Found model weights under key '{key}'")
                break
        if state_dict is None and "model" not in ckpt:
            # Assume entire dict is the state_dict
            state_dict = ckpt
            result["messages"].append("Loaded checkpoint as raw state_dict")
    else:
        state_dict = ckpt
        result["messages"].append("Loaded checkpoint as raw state_dict")

    # Load model weights
    if state_dict is not None:
        try:
            incompatible = model.load_state_dict(state_dict, strict=strict)
            if incompatible.missing_keys:
                result["messages"].append(f"Missing keys: {incompatible.missing_keys[:5]}{'...' if len(incompatible.missing_keys) > 5 else ''}")
            if incompatible.unexpected_keys:
                result["messages"].append(f"Unexpected keys: {incompatible.unexpected_keys[:5]}{'...' if len(incompatible.unexpected_keys) > 5 else ''}")
            result["loaded"] = True
            result["messages"].append("Model weights loaded successfully")
        except Exception as e:
            result["messages"].append(f"Error loading model state_dict: {e}")
            # Try non-strict load
            try:
                model.load_state_dict(state_dict, strict=False)
                result["loaded"] = True
                result["messages"].append("Model weights loaded with strict=False (some keys may not match)")
            except Exception as e2:
                result["messages"].append(f"Non-strict load also failed: {e2}")

    # Load optimizer state if available
    if optimizer is not None and isinstance(ckpt, dict):
        for key in ("optimizer", "optimizer_state_dict", "opt_g", "opt_d"):
            if key in ckpt:
                try:
                    optimizer.load_state_dict(ckpt[key])
                    result["messages"].append(f"Optimizer state loaded from '{key}'")
                except Exception as e:
                    result["messages"].append(f"Failed to load optimizer state: {e}")
                break

    # Load scheduler state if available
    if scheduler is not None and isinstance(ckpt, dict):
        for key in ("scheduler", "scheduler_state_dict", "lr_scheduler"):
            if key in ckpt:
                try:
                    scheduler.load_state_dict(ckpt[key])
                    result["messages"].append(f"Scheduler state loaded from '{key}'")
                except Exception as e:
                    result["messages"].append(f"Failed to load scheduler state: {e}")
                break

    # Extract epoch if available
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        result["epoch"] = ckpt["epoch"]
        result["messages"].append(f"Resumed from epoch {ckpt['epoch']}")

    return result


def load_state_to_model(
    path: Path,
    model: torch.nn.Module,
    map_location: torch.device = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Simpler helper to load weights into a model. Returns status dict.
    Useful for Flask/Gradio apps where you just need to load and go.
    """
    return load_checkpoint_robust(path, model, map_location=map_location, strict=strict)


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def write_metadata(path: Path, data: Dict[str, Any]) -> None:
    ensure_dirs(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
