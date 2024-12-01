#! /usr/bin/env python

import sys
import click
import gradio as gr

sys.path.append("..")
from cli import configure_model, generate_video

# Enhanced Gradio App
with gr.Blocks(css=".gradio-container {font-family: 'Arial', sans-serif; background-color: #f9f9f9;}") as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center; color: #4CAF50;">🎥 Mochi Video Generator</h1>
        <p style="text-align: center;">Generate stunning videos with ease!</p>
        """,
        elem_id="header",
    )
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            value="A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere.",
            lines=3,
            placeholder="Enter your prompt here...",
        )
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            value="",
            lines=2,
            placeholder="Enter negative prompt here...",
        )
    with gr.Row():
        seed = gr.Number(label="Seed", value=1710977262, precision=0)
        cfg_scale = gr.Number(label="CFG Scale", value=4.5)
    with gr.Row():
        width = gr.Number(label="Width", value=848, precision=0)
        height = gr.Number(label="Height", value=480, precision=0)
    with gr.Row():
        num_frames = gr.Number(label="Number of Frames", value=163, precision=0)
        num_inference_steps = gr.Number(label="Number of Inference Steps", value=200, precision=0)
    btn = gr.Button("🎬 Generate Video", elem_id="generate-btn")
    output = gr.Video(label="Generated Video")
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

# Command-line interface with GPU, host, and port options
@click.command()
@click.option("--model_dir", required=True, help="Path to the model directory.")
@click.option("--lora_path", required=False, help="Path to the lora file.")
@click.option("--cpu_offload", is_flag=True, help="Whether to offload model to CPU.")
@click.option("--gpu", default=None, help="Specify the GPU to use (e.g.: 0).")
@click.option("--host", default="127.0.0.1", help="Host address for the Gradio app.")
@click.option("--port", default=7860, help="Port number for the Gradio app.")
def launch(model_dir, lora_path, cpu_offload, gpu, host, port):
    # Configure the model
    configure_model(model_dir, lora_path, cpu_offload, gpu=gpu)
    
    # Launch the Gradio app
    demo.launch(server_name=host, server_port=port)



if __name__ == "__main__":
    launch()
