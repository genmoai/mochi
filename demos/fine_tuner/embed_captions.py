#! /usr/bin/env python3
from pathlib import Path

import click
import torch
from tqdm import tqdm
from transformers import T5Tokenizer

from genmo.mochi_preview.pipelines import T5_MODEL, T5ModelFactory, get_conditioning_for_prompts


@click.command()
@click.argument("captions_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--device_id", default=0, help="GPU device ID to use")
@click.option("--overwrite", "-ow", is_flag=True, help="Overwrite existing embeddings")
def process_captions(captions_dir: Path, device_id: int, overwrite=True) -> None:
    """Process all text files in a directory using T5 encoder.

    Args:
        captions_dir: Directory containing input text files
        device_id: GPU device ID to use
    """

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Get all text file paths
    text_paths = list(captions_dir.glob("**/*.txt"))
    if not text_paths:
        print(f"No text files found in {captions_dir}")
        return

    # Initialize model and tokenizer
    model_factory = T5ModelFactory()
    device = f"cuda:{device_id}"
    model = model_factory.get_model(local_rank=0, device_id=device_id, world_size=1)
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL, legacy=False)

    with tqdm(total=len(text_paths)) as pbar:
        for text_path in text_paths:
            embed_path = text_path.with_suffix(".embed.pt")
            if embed_path.exists() and not overwrite:
                pbar.write(f"Skipping {text_path} - embeddings already exist")
                continue

            pbar.write(f"Processing {text_path}")
            try:
                with open(text_path) as f:
                    text = f.read().strip()

                with torch.inference_mode():
                    conditioning = get_conditioning_for_prompts(tokenizer, model, device, [text])

                torch.save(conditioning, embed_path)

            except Exception as e:
                import traceback

                traceback.print_exc()
                pbar.write(f"Error processing {text_path}: {str(e)}")

            pbar.update(1)


if __name__ == "__main__":
    process_captions()
