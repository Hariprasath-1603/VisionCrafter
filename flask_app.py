"""VisionCrafter - Text-to-Image Generation Flask App using GAN.

Users enter a text prompt and the app generates images using pretrained 
GAN model weights.

Run with: python flask_app.py
Requirements: pip install flask diffusers transformers accelerate
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional

import torch
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Flask app setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "visioncrafter-secret-key-change-in-prod")
app.config["GENERATED_FOLDER"] = Path("static/generated")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Ensure static folder exists
app.config["GENERATED_FOLDER"].mkdir(parents=True, exist_ok=True)

# Global model cache
_pipeline = None


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pipeline():
    """Load the text-to-image pipeline (cached globally)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        
        print("Loading VisionCrafter GAN pipeline (first time will download model)...")
        device = get_device()
        
        # Use pretrained model
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Use float16 for GPU (faster, less VRAM), float32 for CPU
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        
        _pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        
        # Use faster DPM scheduler
        _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            _pipeline.scheduler.config
        )
        
        _pipeline = _pipeline.to(device)
        
        # Enable memory optimizations for GPU
        if device.type == "cuda":
            _pipeline.enable_attention_slicing()
            try:
                _pipeline.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except Exception:
                pass
        
        print(f"Pipeline loaded on {device}")
        return _pipeline
        
    except ImportError:
        raise ImportError(
            "Please install required packages:\n"
            "pip install diffusers transformers accelerate"
        )


def generate_from_prompt(
    prompt: str,
    negative_prompt: str = "",
    num_images: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate images from a text prompt using VisionCrafter GAN."""
    pipeline = load_pipeline()
    device = get_device()
    
    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate images
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    
    # Save generated images
    generated_paths = []
    batch_id = uuid.uuid4().hex[:8]
    
    for i, image in enumerate(result.images):
        filename = f"gen_{batch_id}_{i:03d}.png"
        filepath = app.config["GENERATED_FOLDER"] / filename
        image.save(filepath)
        generated_paths.append(f"static/generated/{filename}")
    
    return generated_paths


# ==================== Flask Routes ====================

@app.route("/")
def index():
    """Main page - redirect to generate."""
    return redirect(url_for("generate"))


@app.route("/generate", methods=["GET", "POST"])
def generate():
    """Text-to-image generation page."""
    if request.method == "GET":
        return render_template("generate.html")
    
    # Get form data
    prompt = request.form.get("prompt", "").strip()
    negative_prompt = request.form.get("negative_prompt", "").strip()
    num_images = int(request.form.get("num_images", 1))
    num_steps = int(request.form.get("num_steps", 25))
    guidance_scale = float(request.form.get("guidance_scale", 7.5))
    width = int(request.form.get("width", 512))
    height = int(request.form.get("height", 512))
    seed_str = request.form.get("seed", "").strip()
    seed = int(seed_str) if seed_str else None
    
    if not prompt:
        flash("Please enter a prompt to generate images.", "error")
        return redirect(url_for("generate"))
    
    # Clamp values to safe ranges
    num_images = max(1, min(4, num_images))
    num_steps = max(10, min(100, num_steps))
    guidance_scale = max(1.0, min(20.0, guidance_scale))
    width = max(64, min(1024, width))
    height = max(64, min(1024, height))
    
    try:
        generated_paths = generate_from_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
        )
        
        return render_template(
            "results.html",
            images=generated_paths,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
        )
    except Exception as e:
        flash(f"Error generating images: {str(e)}", "error")
        return redirect(url_for("generate"))


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """API endpoint for generating images (JSON response)."""
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    try:
        generated_paths = generate_from_prompt(
            prompt=prompt,
            negative_prompt=data.get("negative_prompt", ""),
            num_images=min(4, int(data.get("num_images", 1))),
            num_inference_steps=int(data.get("num_steps", 25)),
            guidance_scale=float(data.get("guidance_scale", 7.5)),
            width=int(data.get("width", 512)),
            height=int(data.get("height", 512)),
            seed=data.get("seed"),
        )
        
        return jsonify({
            "success": True,
            "images": generated_paths,
            "prompt": prompt,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
        "model": "VisionCrafter-GAN",
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  VisionCrafter - AI Image Generator (GAN)")
    print("=" * 60)
    print(f"  Device: {get_device()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print()
    print("  First generation will download the model")
    print("  Supports up to 1024x1024 resolution")
    print()
    print("  Access the app at: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
