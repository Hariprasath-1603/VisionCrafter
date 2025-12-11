"""Simple Gradio app for GAN image sampling."""
import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid
import gradio as gr

from generator import Generator


def load_generator(checkpoint: Path, noise_dim: int, device: torch.device) -> Generator:
    model = Generator(noise_dim=noise_dim, img_channels=3)
    state = torch.load(checkpoint, map_location=device)
    # Accept checkpoints that store nested dict
    state_dict = state.get("generator", state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def sample_images(model: Generator, noise_dim: int, device: torch.device, num: int = 16) -> torch.Tensor:
    with torch.no_grad():
        z = torch.randn(num, noise_dim, device=device)
        imgs = model(z)
        imgs = (imgs + 1) / 2
        grid = make_grid(imgs, nrow=int(num ** 0.5))
        return grid.cpu().permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio sampler for GAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator or full checkpoint")
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_generator(Path(args.checkpoint), args.noise_dim, device)

    def infer(num_images: int, seed: int):
        torch.manual_seed(seed)
        img = sample_images(model, args.noise_dim, device, num=num_images)
        return img

    demo = gr.Interface(
        fn=infer,
        inputs=[
            gr.Slider(4, 64, value=16, step=1, label="Number of samples"),
            gr.Number(value=0, precision=0, label="Seed"),
        ],
        outputs=gr.Image(type="numpy", label="Generated grid"),
        title="GAN Sampler (GoogLeNet D)",
        description="Generate 64x64 RGB samples from the trained generator.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
