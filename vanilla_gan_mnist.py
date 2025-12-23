import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# ---------------------------
# Configuration
# ---------------------------
LATENT_DIM = 100  # Size of input noise vector z
IMG_CHANNELS = 1   # MNIST is grayscale
IMG_SIZE = 28      # Output image size (H=W=28)
BATCH_SIZE = 128
EPOCHS = 10
LR = 2e-4
BETAS = (0.5, 0.999)  # Adam betas commonly used for GANs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_DIR = "samples"
PRINT_EVERY = 200  # Iterations between logging

os.makedirs(SAMPLE_DIR, exist_ok=True)


# ---------------------------
# Model definitions
# ---------------------------
class Generator(nn.Module):
    """Upsamples latent noise into an image using transposed convolutions."""

    def __init__(self, latent_dim: int, img_channels: int):
        super().__init__()
        # Project and reshape noise into a small feature map, then upsample.
        self.net = nn.Sequential(
            # Input: (latent_dim) -> feature map of size 256 x 7 x 7
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            # Upsample to 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Upsample to 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Final conv to get 1 channel output in [-1, 1] via Tanh
            nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """CNN that outputs probability of input image being real."""

    def __init__(self, img_channels: int):
        super().__init__()
        # Downsample image to a single logit.
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------
# Utility helpers
# ---------------------------
def save_samples(generator: Generator, epoch: int, n_grid: int = 8) -> None:
    """Generate a grid of sample images and save to disk."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_grid * n_grid, LATENT_DIM, device=DEVICE)
        fake = generator(z)
        fake = (fake + 1) / 2  # map from [-1,1] to [0,1] for visualization
        grid = utils.make_grid(fake, nrow=n_grid)
        utils.save_image(grid, os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}.png"))
    generator.train()


def make_dataloaders(batch_size: int) -> DataLoader:
    """Create the MNIST dataloader with resizing and normalization to [-1,1]."""
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return loader


# ---------------------------
# Training loop
# ---------------------------
def train() -> None:
    dataloader = make_dataloaders(BATCH_SIZE)

    generator = Generator(LATENT_DIM, IMG_CHANNELS).to(DEVICE)
    discriminator = Discriminator(IMG_CHANNELS).to(DEVICE)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

    for epoch in range(1, EPOCHS + 1):
        for i, (real_imgs, _) in enumerate(dataloader, start=1):
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # -------------------------------------------------
            # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # -------------------------------------------------
            discriminator.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            # Real images loss
            outputs_real = discriminator(real_imgs)
            loss_real = criterion(outputs_real, real_labels)

            # Fake images loss
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_imgs = generator(z).detach()  # detach so gradients do not flow into G when updating D
            outputs_fake = discriminator(fake_imgs)
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # -------------------------------------------------
            # Train Generator: maximize log(D(G(z)))
            # -------------------------------------------------
            generator.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            generated_imgs = generator(z)
            outputs = discriminator(generated_imgs)
            # Use real labels to flip targets so G is rewarded when D believes fakes are real
            loss_g = criterion(outputs, real_labels)
            loss_g.backward()
            opt_g.step()

            if i % PRINT_EVERY == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS}] Step [{i}/{len(dataloader)}] "
                    f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}"
                )

        save_samples(generator, epoch)
        print(f"Saved samples for epoch {epoch} to '{SAMPLE_DIR}'.")

    # Final sample after training completes
    save_samples(generator, EPOCHS)
    print("Training complete.")


if __name__ == "__main__":
    train()


