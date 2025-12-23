# GAN Image Generation (PyTorch)

DCGAN-style Generator trained adversarially against a GoogLeNet (Inception v1) Discriminator to synthesize 64×64 RGB images. The codebase focuses on image generation with a stable training setup, TensorBoard logging, checkpointing, optional EMA, optional FID, and a lightweight Gradio sampler.

This README documents only the GAN-based image generation parts of the project.

## Architecture
```
z ~ N(0, I) (dim=100)
      │
      ▼
+---------------------+
|     Generator       |
| ConvT: 1→4→8→16→32  |
| Upsample to 64×64   |
| Tanh → [-1, 1]      |
+---------------------+
      │ (mapped to ImageNet norm for D)
      ▼
+-----------------------------+
|   GoogLeNet Discriminator   |
| Inception backbone (torchvision)
| FC head → 1 logit            |
| Optional spectral norm       |
+-----------------------------+
```

Core implementations:
- Generator: see generator in [generator.py](generator.py).
- Discriminator wrapper: see [discriminator_googlenet.py](discriminator_googlenet.py).
- Training loop and config wiring: see [train.py](train.py) and [config.py](config.py).
- Utilities (checkpoints, normalization, EMA): see [utils.py](utils.py).

## Quickstart
1) Install dependencies (PyTorch + extras):
```bash
pip install torch torchvision tensorboard gradio pyyaml pytorch-fid
```

2) Train on CIFAR-10 (auto-downloads to ./data):
```bash
python train.py \
  --dataset cifar10 \
  --dataset-root ./data \
  --epochs 50 \
  --batch-size 128 \
  --lr 2e-4 \
  --noise-dim 100 \
  --pretrained-d \
  --output-dir ./outputs
```
Notes:
- Disable pretrained discriminator: add `--no-pretrained-d`.
- Save interval for sample grids is controlled by `--save-every` (default 10).
- Resume from a checkpoint: `--checkpoint outputs/checkpoints/gan_epoch_010.pt`.

3) Sample with Gradio (from a trained generator):
```bash
python app.py --checkpoint outputs/checkpoints/best_generator.pt --noise-dim 100
```
This opens a small UI to render an image grid sampled from the generator.

## Datasets
- `cifar10`: handled automatically by torchvision; use `--dataset cifar10 --dataset-root ./data`.
- `imagefolder`: point `--data-root` to a root containing class subfolders (`root/class_x/*.jpg`).
- `imagenet`: pass the training split path via `--data-root /path/to/imagenet/train` if you have access.

All images are resized/cropped to 64×64. The discriminator expects ImageNet-style normalization; the generator’s outputs (in [-1, 1]) are mapped into that normalization internally during training.

## Configuration
Defaults are centralized in [config.py](config.py). You can override via YAML:
```bash
python train.py --config configs/default.yaml
```
CLI flags take precedence over YAML. Key options include `--epochs`, `--batch-size`, `--lr`, `--noise-dim`, `--pretrained-d`, `--use-spectral-norm`, `--freeze-backbone`, `--ema-decay`, `--r1-gamma`, and dataset settings.

## Logging and Outputs
- TensorBoard logs: [outputs/logs](outputs/logs)
  - Launch with:
    ```bash
    tensorboard --logdir ./outputs/logs
    ```
- Sample grids during training: [outputs/samples](outputs/samples) as `samples_epoch_XXX.png`.
- Checkpoints: [outputs/checkpoints](outputs/checkpoints)
  - Per-epoch `gan_epoch_XXX.pt`, best generator `best_generator.pt`, optional `ema_generator.pt`.
- Training metadata: [outputs/metadata.json](outputs/metadata.json)

## Evaluate (optional FID)
Enable periodic FID during training:
```bash
python train.py --eval-fid-every 10 --fid-num-images 1000
```
Or compute FID for an existing folder of generated images:
```bash
python evaluate.py --model-path outputs/checkpoints/gan_epoch_050.pt --gen-folder outputs/samples
```
Requires `pytorch-fid`.

## Minimal MNIST Example (vanilla GAN)
For a compact reference implementation on MNIST (28×28, single channel), see [vanilla_gan_mnist.py](vanilla_gan_mnist.py). Run it directly to train and emit sample grids under a local `samples/` directory.

## Implementation Notes
- Pretrained ImageNet weights for the GoogLeNet backbone are auto-downloaded by torchvision on first use and cached under `~/.cache/torch/hub/checkpoints`.
- When `--freeze-backbone` is set, only the discriminator head trains; the backbone remains fixed.
- Generator outputs are Tanh-scaled; utilities in [utils.py](utils.py) map between Tanh space and ImageNet normalization for correctness.

## Project Structure (GAN parts)
- [generator.py](generator.py): DCGAN-style generator.
- [discriminator_googlenet.py](discriminator_googlenet.py): GoogLeNet-based discriminator wrapper.
- [train.py](train.py): training loop, checkpoints, optional FID/EMA.
- [utils.py](utils.py): checkpoints, normalization, EMA, metadata, sample saving.
- [app.py](app.py): Gradio-based sampling UI from a trained generator.
- [configs/default.yaml](configs/default.yaml): example YAML overrides.

---
This repository section intentionally focuses on GAN-based image generation. Components unrelated to GAN generation (e.g., alternative diffusion pipelines or unrelated Flask UIs) are out of scope for this README.
