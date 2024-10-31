#! /usr/bin/env python3
from pathlib import Path

import click
import torch
from safetensors.torch import save_file


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
def convert_to_safetensors(input_path):
    model = torch.load(input_path)
    model = {
        k: v.contiguous() for k, v in model.items()
    }
    assert 'vae_ema' not in model
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".safetensors")
    save_file(model, str(output_path))
    click.echo(f"Converted {input_path} to {output_path}")


if __name__ == "__main__":
    convert_to_safetensors()
