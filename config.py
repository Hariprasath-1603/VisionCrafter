"""Centralized hyperparameters and paths for the GAN project.

Defaults are defined here and can be overridden by passing a YAML config path
to ``train.py --config configs/default.yaml``. We keep this file in Python so
code can safely import constants and the YAML acts as a user-facing override.
"""

import yaml
import torch
from pathlib import Path
from typing import Any, Dict


def _default_values() -> Dict[str, Any]:
	return {
		"EPOCHS": 200,
		"BATCH_SIZE": 128,
		"LEARNING_RATE": 2e-4,
		"BETA1": 0.5,
		"BETA2": 0.999,
		"NOISE_DIM": 100,
		"IMAGE_SIZE": 64,
		"IMAGE_CHANNELS": 3,
		"DATASET_ROOT": "./data",
		"NUM_WORKERS": 4,
		"OUTPUT_DIR": "./outputs",
		"SAVE_IMAGE_EVERY": 10,
	}


def load_yaml(path: Path) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def resolve_config(yaml_path: Path | None = None) -> Dict[str, Any]:
	cfg = _default_values()
	if yaml_path is not None and yaml_path.is_file():
		cfg.update({k.upper(): v for k, v in load_yaml(yaml_path).items()})
	# Normalize paths as Path objects
	cfg["DATASET_ROOT"] = Path(cfg["DATASET_ROOT"])
	cfg["OUTPUT_DIR"] = Path(cfg["OUTPUT_DIR"])
	cfg["SAMPLES_DIR"] = cfg["OUTPUT_DIR"] / "samples"
	cfg["CHECKPOINT_DIR"] = cfg["OUTPUT_DIR"] / "checkpoints"
	cfg["LOG_DIR"] = cfg["OUTPUT_DIR"] / "logs"
	cfg["DEVICE"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return cfg


# Default (when no YAML is provided)
_CFG = resolve_config()

EPOCHS = _CFG["EPOCHS"]
BATCH_SIZE = _CFG["BATCH_SIZE"]
LEARNING_RATE = _CFG["LEARNING_RATE"]
BETA1 = _CFG["BETA1"]
BETA2 = _CFG["BETA2"]
NOISE_DIM = _CFG["NOISE_DIM"]
IMAGE_SIZE = _CFG["IMAGE_SIZE"]
IMAGE_CHANNELS = _CFG["IMAGE_CHANNELS"]
DATASET_ROOT = _CFG["DATASET_ROOT"]
NUM_WORKERS = _CFG["NUM_WORKERS"]
OUTPUT_DIR = _CFG["OUTPUT_DIR"]
SAMPLES_DIR = _CFG["SAMPLES_DIR"]
CHECKPOINT_DIR = _CFG["CHECKPOINT_DIR"]
LOG_DIR = _CFG["LOG_DIR"]
SAVE_IMAGE_EVERY = _CFG["SAVE_IMAGE_EVERY"]
DEVICE = _CFG["DEVICE"]
