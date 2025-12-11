"""Train a GAN with a Generator + pretrained discriminator on configurable datasets.

Pretrained ImageNet weights for the discriminator download automatically via
torchvision on first use (cached under ~/.cache/torch/hub/checkpoints).
"""
import argparse
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image

import config
from generator import Generator
from discriminator_googlenet import GoogLeNetDiscriminator
from utils import (
    ensure_dirs,
    save_samples,
    save_checkpoint,
    load_checkpoint,
    load_state_to_model,
    EMA,
    AverageMeter,
    write_metadata,
    IMAGENET_MEAN,
    IMAGENET_STD,
    gen_to_imagenet_norm,
    denormalize_imagenet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAN with pretrained discriminator")
    parser.add_argument("--config", type=str, default="", help="YAML config path (overrides defaults)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate for Adam")
    parser.add_argument("--noise-dim", type=int, default=config.NOISE_DIM, help="Latent noise dimension")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to resume checkpoint")
    parser.add_argument("--output-dir", type=str, default=str(config.OUTPUT_DIR), help="Base output directory")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "imagefolder", "imagenet"], default="cifar10", help="Dataset type")
    parser.add_argument("--data-root", type=str, default="", help="Path to dataset root (train split)")
    parser.add_argument("--dataset-root", type=str, default=str(config.DATASET_ROOT), help="Alias for data-root (backward compat)")
    parser.add_argument("--image-size", type=int, default=64, help="Resize/crop size (defaults to 64)")
    parser.add_argument("--save-every", type=int, default=config.SAVE_IMAGE_EVERY, help="Epoch interval to save images")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--pretrained-d", action=argparse.BooleanOptionalAction, default=True, help="Use pretrained ImageNet weights for discriminator (auto-download)")
    parser.add_argument("--use-spectral-norm", action="store_true", help="Apply spectral normalization to discriminator head")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze discriminator backbone layers (only train head)")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay for generator weights (0 to disable)")
    parser.add_argument("--r1-gamma", type=float, default=0.0, help="R1 gradient penalty strength (0 to disable)")
    parser.add_argument("--eval-fid-every", type=int, default=0, help="Compute FID every N epochs (0 to skip)")
    parser.add_argument("--fid-num-images", type=int, default=1000, help="Number of fake images for FID when enabled")
    parser.add_argument("--load-weights", type=str, default="", help="Path to .pt file to load generator weights (for fine-tuning or eval)")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only (skip training), requires --load-weights")
    return parser.parse_args()


def make_dataloader(name: str, root: Path, batch_size: int, num_workers: int, image_size: int) -> DataLoader:
    """Create dataloader with ImageNet-style normalization."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def prepare_models(noise_dim: int, lr: float, beta1: float, beta2: float, pretrained_d: bool, use_spectral_norm: bool, freeze_backbone: bool, img_channels: int) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.Optimizer]:
    generator = Generator(noise_dim=noise_dim, img_channels=img_channels)
    discriminator = GoogLeNetDiscriminator(pretrained=pretrained_d, use_spectral_norm=use_spectral_norm, freeze_backbone=freeze_backbone)

    generator.to(config.DEVICE)
    discriminator.to(config.DEVICE)

    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    # Only optimize D params that require gradients (handles frozen backbone)
    opt_d = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr, betas=(beta1, beta2))
    return generator, discriminator, opt_g, opt_d


def maybe_resume(
    checkpoint_path: Path,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
) -> Tuple[int, int]:
    if checkpoint_path and checkpoint_path.is_file():
        state = load_checkpoint(checkpoint_path, map_location=config.DEVICE)
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("global_step", 0)
        print(f"Resumed from checkpoint '{checkpoint_path}' at epoch {start_epoch}")
        return start_epoch, global_step
    return 1, 0


def train():
    args = parse_args()
    
    # Validate --eval-only requires --load-weights
    if args.eval_only and not args.load_weights:
        raise ValueError("--eval-only requires --load-weights to be specified")
    
    cfg_from_yaml = config.resolve_config(Path(args.config)) if args.config else config._CFG
    # Allow CLI to override YAML where passed
    epochs = args.epochs or cfg_from_yaml["EPOCHS"]
    batch_size = args.batch_size or cfg_from_yaml["BATCH_SIZE"]
    lr = args.lr or cfg_from_yaml["LEARNING_RATE"]
    noise_dim = args.noise_dim or cfg_from_yaml["NOISE_DIM"]
    dataset_root = Path(args.data_root or args.dataset_root or cfg_from_yaml["DATASET_ROOT"])
    output_dir = Path(args.output_dir or cfg_from_yaml["OUTPUT_DIR"])
    save_every = args.save_every or cfg_from_yaml["SAVE_IMAGE_EVERY"]
    num_workers = args.num_workers or cfg_from_yaml["NUM_WORKERS"]
    image_size = args.image_size or cfg_from_yaml.get("IMAGE_SIZE", config.IMAGE_SIZE)
    img_channels = cfg_from_yaml.get("IMAGE_CHANNELS", config.IMAGE_CHANNELS)

    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    ensure_dirs(output_dir, samples_dir, checkpoints_dir, logs_dir)

    dataloader = make_dataloader(args.dataset, dataset_root, batch_size, num_workers, image_size)
    generator, discriminator, opt_g, opt_d = prepare_models(
        noise_dim=noise_dim,
        lr=lr,
        beta1=config.BETA1,
        beta2=config.BETA2,
        pretrained_d=args.pretrained_d,
        use_spectral_norm=args.use_spectral_norm,
        freeze_backbone=args.freeze_backbone,
        img_channels=img_channels,
    )

    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, noise_dim, device=config.DEVICE)
    writer = SummaryWriter(log_dir=logs_dir)

    ema = EMA(generator, decay=args.ema_decay) if args.ema_decay > 0 else None
    gen_loss_meter = AverageMeter()

    start_epoch, global_step = maybe_resume(
        Path(args.checkpoint) if args.checkpoint else Path(),
        generator,
        discriminator,
        opt_g,
        opt_d,
    )

    # Load weights from external checkpoint (for fine-tuning or eval)
    if args.load_weights:
        weights_path = Path(args.load_weights)
        if not weights_path.is_file():
            raise FileNotFoundError(f"--load-weights file not found: {weights_path}")
        print(f"Loading generator weights from: {weights_path}")
        load_state_to_model(weights_path, generator, map_location=config.DEVICE, strict=False)
        print(f"Successfully loaded weights from: {weights_path}")

    best_gen_loss = float("inf")

    def maybe_r1_penalty(real_imgs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        if args.r1_gamma <= 0:
            return torch.tensor(0.0, device=real_imgs.device)
        real_imgs.requires_grad_(True)
        grad_real = torch.autograd.grad(
            outputs.sum(), real_imgs, create_graph=True, retain_graph=True
        )[0]
        grad_penalty = grad_real.pow(2).reshape(real_imgs.size(0), -1).sum(dim=1).mean()
        return 0.5 * args.r1_gamma * grad_penalty

    def maybe_eval_fid(epoch_idx: int):
        if args.eval_fid_every <= 0 or epoch_idx % args.eval_fid_every != 0:
            return None
        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths
        except ImportError:
            print("pytorch-fid not installed; skipping FID")
            return None
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_dir = Path(tmpdir) / "fake"
            real_dir = Path(tmpdir) / "real"
            fake_dir.mkdir(parents=True, exist_ok=True)
            real_dir.mkdir(parents=True, exist_ok=True)

            # Export fake images
            generator.eval()
            with torch.no_grad():
                n_to_gen = args.fid_num_images
                bs = min(batch_size, 256)
                saved = 0
                while saved < n_to_gen:
                    cur = min(bs, n_to_gen - saved)
                    noise = torch.randn(cur, noise_dim, device=config.DEVICE)
                    fake = generator(noise)
                    fake = (fake + 1) / 2
                    for i in range(cur):
                        save_image(fake[i], fake_dir / f"fake_{saved + i:05d}.png")
                    saved += cur
            generator.train()

            # Export real images without normalization
            real_ds = (
                datasets.CIFAR10(root=dataset_root, train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.CenterCrop(image_size),
                                     transforms.ToTensor(),
                                 ]))
                if args.dataset == "cifar10"
                else datasets.ImageFolder(root=dataset_root, transform=transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                    ]))
            )
            export_count = min(args.fid_num_images, len(real_ds))
            for idx in range(export_count):
                img, _ = real_ds[idx]
                transforms.ToPILImage()(img).save(real_dir / f"real_{idx:05d}.png")

            fid = calculate_fid_given_paths(
                [str(fake_dir), str(real_dir)],
                batch_size=min(batch_size, 256),
                device=config.DEVICE,
                dims=2048,
            )
            writer.add_scalar("Metrics/FID", fid, epoch_idx)
            print(f"Epoch {epoch_idx}: FID {fid:.4f}")
            return fid

    # Evaluation-only mode: generate samples and compute FID, then exit
    if args.eval_only:
        print("=== Evaluation-only mode ===")
        generator.eval()
        with torch.no_grad():
            fake_samples = generator(fixed_noise)
            fake_samples = (fake_samples + 1) / 2  # denormalize from [-1,1] to [0,1]
            eval_samples_dir = output_dir / "eval_samples"
            eval_samples_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(fake_samples):
                save_image(img, eval_samples_dir / f"sample_{i:03d}.png")
            print(f"Saved {len(fake_samples)} samples to {eval_samples_dir}")
        
        # Compute FID if enabled
        if args.eval_fid_every > 0:
            print("Computing FID...")
            fid_score = maybe_eval_fid(1)  # force eval
            if fid_score is not None:
                print(f"Final FID: {fid_score:.4f}")
        
        writer.close()
        print("Evaluation complete.")
        return

    for epoch in range(start_epoch, epochs + 1):
        gen_loss_meter.reset()
        for batch_idx, (real, _) in enumerate(dataloader, start=1):
            real = real.to(config.DEVICE)
            cur_batch_size = real.size(0)

            # Train Discriminator
            discriminator.zero_grad()
            real_labels = torch.ones(cur_batch_size, 1, device=config.DEVICE)
            fake_labels = torch.zeros(cur_batch_size, 1, device=config.DEVICE)

            output_real = discriminator(real)
            loss_real = criterion(output_real, real_labels)

            noise = torch.randn(cur_batch_size, noise_dim, device=config.DEVICE)
            fake_images = generator(noise)
            fake_for_d = gen_to_imagenet_norm(fake_images.detach())
            output_fake = discriminator(fake_for_d)
            loss_fake = criterion(output_fake, fake_labels)

            r1_penalty = maybe_r1_penalty(real, output_real)
            loss_d = loss_real + loss_fake + r1_penalty
            loss_d.backward()
            opt_d.step()

            # Train Generator
            generator.zero_grad()
            fake_for_g = gen_to_imagenet_norm(fake_images)
            output = discriminator(fake_for_g)
            loss_g = criterion(output, real_labels)
            loss_g.backward()
            opt_g.step()

            if ema is not None:
                ema.update(generator)

            gen_loss_meter.update(loss_g.item(), cur_batch_size)

            # Logging
            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss D: {loss_d.item():.4f} Loss G: {loss_g.item():.4f}"
                )
            writer.add_scalar("Loss/Discriminator", loss_d.item(), global_step)
            writer.add_scalar("Loss/Generator", loss_g.item(), global_step)
            writer.add_scalar("LR", args.lr, global_step)
            if args.r1_gamma > 0:
                writer.add_scalar("Reg/R1", r1_penalty.item(), global_step)
            global_step += 1

        # Save samples periodically
        if epoch % save_every == 0 or epoch == epochs:
            save_samples(generator, fixed_noise, epoch, samples_dir, nrow=8)
            writer.add_images("Samples", ((generator(fixed_noise) + 1) / 2).clamp(0, 1), epoch)
            writer.add_images("Real", denormalize_imagenet(real[:16]), epoch)

        fid_score = maybe_eval_fid(epoch)

        # Track best generator by avg generator loss
        if gen_loss_meter.avg < best_gen_loss:
            best_gen_loss = gen_loss_meter.avg
            best_path = checkpoints_dir / "best_generator.pt"
            save_checkpoint({"generator": generator.state_dict()}, checkpoints_dir, filename="best_generator.pt")
            print(f"New best generator saved (loss={best_gen_loss:.4f}) -> {best_path}")

        # Save EMA snapshot
        if ema is not None:
            ema_path = checkpoints_dir / "ema_generator.pt"
            save_checkpoint({"generator": ema.shadow}, checkpoints_dir, filename="ema_generator.pt")

        # Save checkpoint every epoch
        save_checkpoint(
            {
                "epoch": epoch,
                "global_step": global_step,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "args": vars(args),
                "cfg": cfg_from_yaml,
                "fid": fid_score,
                "best_gen_loss": best_gen_loss,
            },
            checkpoints_dir,
            filename=f"gan_epoch_{epoch:03d}.pt",
        )
        print(f"Epoch {epoch} complete. Checkpoint saved.")

        write_metadata(
            output_dir / "metadata.json",
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_gen_loss": best_gen_loss,
                "last_checkpoint": str(checkpoints_dir / f"gan_epoch_{epoch:03d}.pt"),
                "best_generator": str(checkpoints_dir / "best_generator.pt"),
                "ema_generator": str(checkpoints_dir / "ema_generator.pt") if ema is not None else None,
                "fid": fid_score,
            },
        )

    writer.close()


if __name__ == "__main__":
    train()
