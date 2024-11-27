#! /usr/bin/env python3
import os

import click
from huggingface_hub import snapshot_download


# Based off of Kijai's script
@click.command()
@click.argument('output_dir', required=True)
def download_weights(output_dir):
    repo_id = "genmo/mochi-1-preview"
    model = "dit.safetensors"
    decoder = "decoder.safetensors"
    encoder = "encoder.safetensors"

    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    def download_file(repo_id, output_dir, filename, description):
        file_path = os.path.join(output_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading mochi {description} to: {file_path}")
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{filename}*"],
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
        else:
            print(f"{description} already exists in: {file_path}")
        assert os.path.exists(file_path)

    download_file(repo_id, output_dir, decoder, "decoder")
    download_file(repo_id, output_dir, encoder, "encoder")
    download_file(repo_id, output_dir, model, "model")


if __name__ == "__main__":
    download_weights()
