#! /usr/bin/env python
import sys
from pathlib import Path
from textwrap import dedent

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)

MOCHI_DIR = sys.argv[1]
assert Path(MOCHI_DIR).exists(), f"Model directory {MOCHI_DIR} does not exist."
pipeline = MochiSingleGPUPipeline(
    text_encoder_factory=T5ModelFactory(),
    dit_factory=DitModelFactory(model_path=f"{MOCHI_DIR}/dit.safetensors", model_dtype="bf16"),
    decoder_factory=DecoderModelFactory(
        model_path=f"{MOCHI_DIR}/vae.safetensors",
        model_stats_path=f"{MOCHI_DIR}/vae_stats.json",
    ),
    cpu_offload=True,
    decode_type="tiled_full",
)

PROMPT = dedent("""
A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl 
filled with lemons and sprigs of mint against a peach-colored background. 
The hand gently tosses the lemon up and catches it, showcasing its smooth texture. 
A beige string bag sits beside the bowl, adding a rustic touch to the scene. 
Additional lemons, one halved, are scattered around the base of the bowl. 
The even lighting enhances the vibrant colors and creates a fresh, 
inviting atmosphere.
""")

video = pipeline(
    height=480,
    width=848,
    num_frames=31,
    num_inference_steps=64,
    sigma_schedule=linear_quadratic_schedule(64, 0.025),
    cfg_schedule=[4.5] * 64,
    batch_cfg=False,
    prompt=PROMPT,
    negative_prompt="",
    seed=12345,
)

with progress_bar(type="tqdm"):
    save_video(video[0], "video.mp4")
