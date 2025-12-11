"""GAN Generator producing 64x64 RGB images from noise."""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN-style generator using ConvTranspose2d blocks."""

    def __init__(self, noise_dim: int = 100, img_channels: int = 3, feature_maps: int = 64):
        super().__init__()
        # Blocks: (noise_dim) -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.net = nn.Sequential(
            # Input Z: (N, noise_dim, 1, 1)
            nn.ConvTranspose2d(noise_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # 4x4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 8x8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 32x32
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # 64x64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Expect z shape: (N, noise_dim). Reshape to (N, noise_dim, 1, 1)
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)
