# Image Generation using GAN with GoogLeNet Discriminator in PyTorch

## Overview
GAN setup combining a DCGAN-style generator with a GoogLeNet (Inception v1) discriminator to synthesize 64×64 RGB images. Features include TensorBoard logging, YAML/CLI configuration, checkpointing with best/EMA snapshots, optional FID logging, Gradio sampling app, and support for CIFAR-10 or custom ImageFolder datasets.

## Architecture Diagram
```
Noise z ~ N(0, I) (100-d)
      |
      v
+---------------------+
|     Generator       |
| ConvT 4x4 -> 4x4    |
| ConvT 4x4 -> 8x8    |
| ConvT 4x4 -> 16x16  |
| ConvT 4x4 -> 32x32  |
| ConvT 4x4 -> 64x64  |
| Tanh -> Fake Image  |
+---------------------+
      |
      v
+-----------------------------+
|   GoogLeNet Discriminator   |
| Inception blocks            |
| FC -> Sigmoid (real/fake)   |
| Optional spectral norm      |
+-----------------------------+
```

## Dataset
- Built-in support: `cifar10`, `imagenet`, or generic `imagefolder` (class-subfolders required).
- ImageNet (ILSVRC2012) must be downloaded separately (requires registration) and arranged as:
      - `/path/to/imagenet/train/<class>/*.JPEG`
      - `/path/to/imagenet/val/<class>/*.JPEG`
- Images are resized/cropped to 64×64 and normalized with ImageNet mean/std; generator outputs are tanh -> mapped to ImageNet normalization before being scored by D.

## Installation
```bash
pip install torch torchvision tensorboard pytorch-fid gradio pyyaml
```

## Configuration
- Python defaults live in `config.py`.
- YAML overrides in `configs/default.yaml` and can be passed via `--config path/to.yaml`.

## Training
ImageNet 64×64 example (pretrained discriminator, spectral norm head):
```bash
python train.py \
      --dataset imagenet \
      --data-root /path/to/imagenet/train \
      --image-size 64 \
      --batch-size 128 \
      --epochs 50 \
      --lr 2e-4 \
      --noise-dim 100 \
      --pretrained-d \
      --use-spectral-norm \
      --output-dir ./outputs
```
Other notes:
- Resume: `--checkpoint outputs/checkpoints/gan_epoch_010.pt`.
- Disable pretrained weights: `--no-pretrained-d` (boolean optional flag).
- CIFAR-10: `--dataset cifar10 --dataset-root ./data`.

## TensorBoard
```bash
tensorboard --logdir ./outputs/logs
```
Logs: generator/discriminator losses, LR, optional R1/FID, and sample images every save interval.

## Pretrained weights (auto-download)
Torchvision will download ImageNet weights on first use (cached at `~/.cache/torch/hub/checkpoints`). Examples:
```bash
python - << 'EOF'
import torchvision.models as models
m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
print("Downloaded automatically!")
EOF

python - << 'EOF'
import torchvision.models as models
m = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
print("Downloaded automatically!")
EOF
```

Alternative options:
```bash
pip install timm
python - << 'EOF'
import timm
m = timm.create_model("resnet50", pretrained=True)
print("timm auto-downloaded!")
EOF

python - << 'EOF'
from tensorflow.keras.applications import ResNet50
m = ResNet50(weights='imagenet')
print("Keras automatically downloaded .h5!")
EOF
```

## Sampling (Gradio UI)
```bash
python app.py --checkpoint outputs/checkpoints/best_generator.pt --noise-dim 100
```
Opens a simple web UI to draw grids of generated samples.

## Outputs and Registry
- Samples: `outputs/samples/samples_epoch_XXX.png` every N epochs.
- Checkpoints: per-epoch plus `best_generator.pt` (lowest avg G loss) and `ema_generator.pt` (if EMA on).
- Metadata: `outputs/metadata.json` tracks latest epoch, best/EMA paths, and last FID (when enabled).

## Evaluation (FID)
```bash
python evaluate.py --model-path outputs/checkpoints/gan_epoch_200.pt --gen-folder outputs/samples
```
or enable periodic FID during training with `--eval-fid-every N --fid-num-images 1000` (requires `pytorch-fid`).

## Future Scope
- Swap GoogLeNet for other backbones; add DiffAugment or WGAN-GP; tune R1/EMA; push to higher resolutions with deeper G.
- Integrate W&B/MLflow for experiment tracking and richer dashboards.

## License
- MIT (adapt as needed for your project).
