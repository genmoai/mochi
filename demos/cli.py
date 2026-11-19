#! /usr/bin/env python
import json
import os
import time
from uuid import uuid4

import click
import numpy as np
import torch

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiMultiGPUPipeline,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)

pipeline = None
model_dir_path = None
lora_path = None
num_gpus = torch.cuda.device_count()
cpu_offload = False


def configure_model(model_dir_path_, lora_path_, cpu_offload_, fast_model_=False):
    global model_dir_path, lora_path, cpu_offload
    model_dir_path = model_dir_path_
    lora_path = lora_path_
    cpu_offload = cpu_offload_


def load_model():
    global num_gpus, pipeline, model_dir_path, lora_path
    if pipeline is None:
        MOCHI_DIR = model_dir_path
        print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
        klass = MochiSingleGPUPipeline if num_gpus == 1 else MochiMultiGPUPipeline
        kwargs = dict(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(
                model_path=f"{MOCHI_DIR}/dit.safetensors",
                lora_path=lora_path,
                model_dtype="bf16",
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
        # We *need* flash attention to batch cfg
        # and it's only worth doing in a high-memory regime (assume multiple GPUs)
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
        output_path = os.path.join(output_dir, f"output_{int(time.time())}_{uuid4().hex}.mp4")

        save_video(final_frames, output_path)
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(args, f, indent=4)

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
@click.option("--out_dir", default="outputs", help="Output directory for generated videos")
@click.option("--threshold-noise", default=0.025, help="threshold noise")
@click.option("--linear-steps", default=None, type=int, help="linear steps")
def generate_cli(
    prompt, sweep_file, negative_prompt, width, height, num_frames, seed, cfg_scale, num_steps, 
    model_dir, lora_path, cpu_offload, out_dir, threshold_noise, linear_steps
):
    configure_model(model_dir, lora_path, cpu_offload)

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
