#! /usr/bin/env python3
import click
import os

@click.command()
@click.argument('repo_id', required=True)
@click.argument('output_dir', required=True)
def download_weights(repo_id, output_dir):
    model = "dit.safetensors"
    vae = "vae.safetensors"
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    model_download_path = os.path.join(output_dir)
    if not os.path.exists(model_download_path):
        print(f"Downloading mochi model to: {model_download_path}")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"*{model}*"],
            local_dir=model_download_path,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"Dit already exists in: {model_download_path}")

    # VAE
    vae_download_path = os.path.join(output_dir)
    vae_path = os.path.join(vae_download_path, vae)

    if not os.path.exists(vae_path):
        print(f"Downloading mochi VAE to: {vae_path}")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"*{vae}*"],
            local_dir=vae_download_path,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"Decoder already exists in: {vae_path}")

if __name__ == "__main__":
    download_weights()
