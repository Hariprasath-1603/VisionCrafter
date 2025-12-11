"""Compute FID score between generated images and CIFAR-10 real images."""
import argparse
import tempfile
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FID between generated images and CIFAR-10")
    parser.add_argument("--model-path", type=str, required=True, help="Checkpoint path (unused for FID only, kept for compatibility)")
    parser.add_argument("--gen-folder", type=str, required=True, help="Folder containing generated images")
    parser.add_argument("--dataset-root", type=str, default=str(config.DATASET_ROOT), help="CIFAR-10 root")
    parser.add_argument("--num-real", type=int, default=5000, help="Number of real images to sample")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for FID computation")
    return parser.parse_args()


def export_real_images(dataset_root: Path, num_images: int, out_dir: Path) -> None:
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(num_images, len(dataset))):
        img, _ = dataset[idx]
        save_image(img, out_dir / f"real_{idx:05d}.png")


def main():
    args = parse_args()
    gen_folder = Path(args.gen_folder)
    if not gen_folder.exists():
        raise FileNotFoundError(f"Generated images folder not found: {gen_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = Path(tmpdir) / "real"
        export_real_images(Path(args.dataset_root), args.num_real, real_dir)

        fid_value = calculate_fid_given_paths(
            [str(gen_folder), str(real_dir)],
            batch_size=args.batch_size,
            device=device,
            dims=2048,
        )
        print(f"FID: {fid_value:.4f}")


if __name__ == "__main__":
    main()
