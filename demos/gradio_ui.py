#! /usr/bin/env python


import sys

import click
import gradio as gr

sys.path.append("..")
from cli import configure_model, generate_video

with gr.Blocks() as demo:
    gr.Markdown("Video Generator")
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            value="A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere.",
        )
        negative_prompt = gr.Textbox(label="Negative Prompt", value="")
        seed = gr.Number(label="Seed", value=1710977262, precision=0)
    with gr.Row():
        width = gr.Number(label="Width", value=848, precision=0)
        height = gr.Number(label="Height", value=480, precision=0)
        num_frames = gr.Number(label="Number of Frames", value=163, precision=0)
    with gr.Row():
        cfg_scale = gr.Number(label="CFG Scale", value=6.0)
        num_inference_steps = gr.Number(label="Number of Inference Steps", value=100, precision=0)
    btn = gr.Button("Generate Video")
    output = gr.Video()

    btn.click(
        generate_video,
        inputs=[
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            seed,
            cfg_scale,
            num_inference_steps,
        ],
        outputs=output,
    )


@click.command()
@click.option("--model_dir", required=True, help="Path to the model directory.")
@click.option("--lora_path", required=False, help="Path to the lora file.")
@click.option("--cpu_offload", is_flag=True, help="Whether to offload model to CPU")
def launch(model_dir, lora_path, cpu_offload):
    configure_model(model_dir, lora_path, cpu_offload)
    demo.launch()


if __name__ == "__main__":
    launch()
