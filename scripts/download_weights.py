#! /usr/bin/env python3
import os
import tempfile

import click
from huggingface_hub import hf_hub_download, snapshot_download
import shutil

BASE_MODEL_FILES = [
    # (repo_id, remote_file_path, local_file_path)
    ("genmo/mochi-1-preview", "decoder.safetensors", "decoder.safetensors"),
    ("genmo/mochi-1-preview", "encoder.safetensors", "encoder.safetensors"),
    ("genmo/mochi-1-preview", "dit.safetensors", "dit.safetensors"),
]

FAST_MODEL_FILE = ("FastVideo/FastMochi", "dit.safetensors", "dit.fast.safetensors")


@click.command()
@click.argument('output_dir', required=True)
@click.option('--fast_model', is_flag=True, help='Download FastMochi model instead of standard model')
@click.option('--hf_transfer', is_flag=True, help='Enable faster downloads using hf_transfer (requires: pip install "huggingface_hub[hf_transfer]")')
def download_weights(output_dir, fast_model, hf_transfer):
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    if hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("Using hf_transfer for faster downloads (requires: pip install 'huggingface_hub[hf_transfer]')")

    model_files = BASE_MODEL_FILES
    if fast_model:
        # Replace the standard DIT model with the fast model
        model_files = [f for f in model_files if not f[2].startswith("dit.")]
        model_files.append(FAST_MODEL_FILE)

    for repo_id, remote_path, local_path in model_files:
        local_file_path = os.path.join(output_dir, local_path)
        if not os.path.exists(local_file_path):
            if hf_transfer:
                # I don't know if `hf_transfer` works with `snapshot_download`
                print(f"Downloading {local_path} from {repo_id} to: {local_file_path}")
                out_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    local_dir=output_dir,
                )
                print(f"Copying {out_path} to {local_file_path}")
                # copy instead of mv to avoid destroying huggingface cache
                shutil.copy2(out_path, local_file_path)
            else:
                with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
                    snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=[f"*{remote_path}*"],
                        local_dir=tmp_dir,
                        local_dir_use_symlinks=False,
                    )
                    shutil.move(os.path.join(tmp_dir, remote_path), local_file_path)
        else:
            print(f"{local_path} already exists in: {local_file_path}")
        assert os.path.exists(local_file_path), f"File {local_file_path} does not exist"

if __name__ == "__main__":
    download_weights()
