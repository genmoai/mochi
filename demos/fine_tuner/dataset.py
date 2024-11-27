from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader, Dataset


def load_to_cpu(x):
    return torch.load(x, map_location=torch.device("cpu"), weights_only=True)


class LatentEmbedDataset(Dataset):
    def __init__(self, file_paths, repeat=1):
        self.items = [
            (Path(p).with_suffix(".latent.pt"), Path(p).with_suffix(".embed.pt"))
            for p in file_paths
            if Path(p).with_suffix(".latent.pt").is_file() and Path(p).with_suffix(".embed.pt").is_file()
        ]
        self.items = self.items * repeat
        print(f"Loaded {len(self.items)}/{len(file_paths)} valid file pairs.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        latent_path, embed_path = self.items[idx]
        return load_to_cpu(latent_path), load_to_cpu(embed_path)


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def process_videos(directory):
    dir_path = Path(directory)
    mp4_files = [str(f) for f in dir_path.glob("**/*.mp4") if not f.name.endswith(".recon.mp4")]
    assert mp4_files, f"No mp4 files found"

    dataset = LatentEmbedDataset(mp4_files)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for latents, embeds in dataloader:
        print([(k, v.shape) for k, v in latents.items()])


if __name__ == "__main__":
    process_videos()
