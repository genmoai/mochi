#! /usr/bin/env python
import json
import os
import time

import click
import numpy as np
import torch

try:
    from mps_conv3d import patch_conv3d, is_available as conv3d_available
    if conv3d_available():
        patch_conv3d()
        print("✓ MPS Conv3D patched")
except ImportError:
    pass

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiMultiGPUPipeline,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    get_device,
    linear_quadratic_schedule,
)

pipeline = None
model_dir_path = None
lora_path = None
quantize_nf4 = False
attention_mode = None
# Check for available GPUs - works with CUDA, falls back to MPS/CPU
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
elif torch.backends.mps.is_available():
    num_gpus = 1  # MPS is single-device
    print("Using MPS (Apple Silicon)")
else:
    num_gpus = 0  # CPU mode
    print("No GPU detected, using CPU")
cpu_offload = False


def configure_model(model_dir_path_, lora_path_, cpu_offload_, quantize_nf4_=False, attention_mode_=None):
    global model_dir_path, lora_path, cpu_offload, quantize_nf4, attention_mode
    model_dir_path = model_dir_path_
    lora_path = lora_path_
    cpu_offload = cpu_offload_
    quantize_nf4 = quantize_nf4_
    attention_mode = attention_mode_


def load_model():
    global num_gpus, pipeline, model_dir_path, lora_path, quantize_nf4, attention_mode
    if pipeline is None:
        MOCHI_DIR = model_dir_path
        device = get_device()
        if device.type == "cuda":
            print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
        elif device.type == "mps":
            print("Launching on MPS (Apple Silicon)")
            if quantize_nf4:
                print("NF4 quantization enabled (~5GB model size)")
        else:
            print("Launching on CPU")
        # Multi-GPU only supported on CUDA
        klass = MochiSingleGPUPipeline if (num_gpus <= 1 or device.type != "cuda") else MochiMultiGPUPipeline
        # Check for local T5 weights, else use HuggingFace
        t5_local = f"{MOCHI_DIR}/../t5"
        t5_path = t5_local if os.path.exists(t5_local) else None
        if t5_path:
            print(f"Using local T5: {t5_path}")

        kwargs = dict(
            text_encoder_factory=T5ModelFactory(model_dir=t5_path),
            dit_factory=DitModelFactory(
                model_path=f"{MOCHI_DIR}/dit.safetensors",
                lora_path=lora_path,
                model_dtype="bf16",
                quantize_nf4=quantize_nf4,
                attention_mode=attention_mode,
            ),
            decoder_factory=DecoderModelFactory(
                model_path=f"{MOCHI_DIR}/decoder.safetensors",
            ),
        )
        if num_gpus > 1:
            assert not lora_path, f"Lora not supported in multi-GPU mode"
            assert not cpu_offload, "CPU offload not supported in multi-GPU mode"
            kwargs["world_size"] = num_gpus
        else:
            kwargs["cpu_offload"] = cpu_offload
            kwargs["decode_type"] = "tiled_spatial"
            kwargs["fast_init"] = not lora_path
            kwargs["strict_load"] = not lora_path
            kwargs["decode_args"] = dict(overlap=8)
        pipeline = klass(**kwargs)


def generate_video(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_inference_steps,
    threshold_noise=0.025,
    linear_steps=None,
    output_dir="outputs",
):
    load_model()

    # Fast mode parameters: threshold_noise=0.1, linear_steps=6, cfg_scale=1.5, num_inference_steps=8
    sigma_schedule = linear_quadratic_schedule(num_inference_steps, threshold_noise, linear_steps)

    # cfg_schedule should be a list of floats of length num_inference_steps.
    # For simplicity, we just use the same cfg scale at all timesteps,
    # but more optimal schedules may use varying cfg, e.g:
    # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
    cfg_schedule = [cfg_scale] * num_inference_steps

    args = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": num_inference_steps,
        # Batched CFG requires flash attention (B=2 not supported by SDPA path)
        "batch_cfg": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }

    with progress_bar(type="tqdm"):
        final_frames = pipeline(**args)

        final_frames = final_frames[0]

        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"output_{int(time.time())}.mp4")

        save_video(final_frames, output_path)
        json_path = os.path.splitext(output_path)[0] + ".json"
        json.dump(args, open(json_path, "w"), indent=4)

        return output_path


from textwrap import dedent

DEFAULT_PROMPT = dedent("""
A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl 
filled with lemons and sprigs of mint against a peach-colored background. 
The hand gently tosses the lemon up and catches it, showcasing its smooth texture. 
A beige string bag sits beside the bowl, adding a rustic touch to the scene. 
Additional lemons, one halved, are scattered around the base of the bowl. 
The even lighting enhances the vibrant colors and creates a fresh, 
inviting atmosphere.
""")


@click.command()
@click.option("--prompt", default=DEFAULT_PROMPT, help="Prompt for video generation.")
@click.option("--sweep-file", help="JSONL file containing one config per line.")
@click.option("--negative_prompt", default="", help="Negative prompt for video generation.")
@click.option("--width", default=848, type=int, help="Width of the video.")
@click.option("--height", default=480, type=int, help="Height of the video.")
@click.option("--num_frames", default=163, type=int, help="Number of frames.")
@click.option("--seed", default=1710977262, type=int, help="Random seed.")
@click.option("--cfg_scale", default=6.0, type=float, help="CFG Scale.")
@click.option("--num_steps", default=64, type=int, help="Number of inference steps.")
@click.option("--model_dir", required=True, help="Path to the model directory.")
@click.option("--lora_path", required=False, help="Path to the lora file.")
@click.option("--cpu_offload", is_flag=True, help="Whether to offload model to CPU")
@click.option("--quantize", is_flag=True, help="Use NF4 quantization (4-bit) for MPS - reduces memory from ~20GB to ~5GB")
@click.option("--attention-mode", type=click.Choice(["flash", "mps_flash", "sdpa", "sage"]), default=None, help="Attention mode (auto-detected if not specified)")
@click.option("--out_dir", default="outputs", help="Output directory for generated videos")
@click.option("--threshold-noise", default=0.025, help="threshold noise")
@click.option("--linear-steps", default=None, type=int, help="linear steps")
def generate_cli(
    prompt, sweep_file, negative_prompt, width, height, num_frames, seed, cfg_scale, num_steps,
    model_dir, lora_path, cpu_offload, quantize, attention_mode, out_dir, threshold_noise, linear_steps
):
    configure_model(model_dir, lora_path, cpu_offload, quantize, attention_mode)

    if sweep_file:
        with open(sweep_file, 'r') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                config = json.loads(line)
                current_prompt = config.get('prompt', prompt)
                current_cfg_scale = config.get('cfg_scale', cfg_scale)
                current_num_steps = config.get('num_steps', num_steps)
                current_threshold_noise = config.get('threshold_noise', threshold_noise)
                current_linear_steps = config.get('linear_steps', linear_steps)
                current_seed = config.get('seed', seed)
                current_width = config.get('width', width)
                current_height = config.get('height', height)
                current_num_frames = config.get('num_frames', num_frames)

                output_path = generate_video(
                    current_prompt,
                    negative_prompt,
                    current_width,
                    current_height,
                    current_num_frames,
                    current_seed,
                    current_cfg_scale,
                    current_num_steps,
                    threshold_noise=current_threshold_noise,
                    linear_steps=current_linear_steps,
                    output_dir=out_dir,
                )
                click.echo(f"Video {i+1} generated at: {output_path}")
    else:
        output_path = generate_video(
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            seed,
            cfg_scale,
            num_steps,
            threshold_noise=threshold_noise,
            linear_steps=linear_steps,
            output_dir=out_dir,
        )
        click.echo(f"Video generated at: {output_path}")


if __name__ == "__main__":
    generate_cli()
