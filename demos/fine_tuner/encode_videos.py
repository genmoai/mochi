#! /usr/bin/env python3
import os
from pathlib import Path
import traceback
from typing import Optional

import click
import ray
import torch
import torchvision
from einops import rearrange

import genmo.mochi_preview.dit.joint_model.context_parallel as cp
import genmo.mochi_preview.vae.cp_conv as cp_conv
from genmo.lib.progress import get_new_progress_bar, progress_bar
from genmo.lib.utils import Timer, save_video
from genmo.mochi_preview.pipelines import DecoderModelFactory, EncoderModelFactory
from genmo.mochi_preview.vae.models import add_fourier_features, decode_latents


class GPUContext:
    def __init__(
        self,
        *,
        encoder_factory: Optional[EncoderModelFactory] = None,
        decoder_factory: Optional[DecoderModelFactory] = None,
    ):
        t = Timer()
        self.device = torch.device(f"cuda")
        if encoder_factory is not None:
            with t("load_encoder"):
                self.encoder = encoder_factory.get_model()
        if decoder_factory is not None:
            with t("load_decoder"):
                self.decoder = decoder_factory.get_model()
        t.print_stats()


def preprocess(ctx: GPUContext, vid_path: Path, shape: str, reconstruct: bool):
    T, H, W = [int(s) for s in shape.split("x")]
    assert (T - 1) % 6 == 0, "Expected T to be 1 mod 6"
    video, _, metadata = torchvision.io.read_video(
        str(vid_path), output_format="THWC", pts_unit="secs")
    fps = metadata["video_fps"]
    video = rearrange(video, "t h w c -> c t h w")
    og_shape = video.shape
    assert video.shape[2] == H, f"Expected {vid_path} to have height {H}, got {video.shape}"
    assert video.shape[3] == W, f"Expected {vid_path} to have width {W}, got {video.shape}"
    assert video.shape[1] >= T, f"Expected {vid_path} to have at least {T} frames, got {video.shape}"
    if video.shape[1] > T:
        video = video[:, :T]
        print(f"Trimmed video from {og_shape[1]} to first {T} frames")
    video = video.unsqueeze(0)
    video = video.float() / 127.5 - 1.0
    video = video.to(ctx.device)
    video = add_fourier_features(video)

    assert video.ndim == 5
    video = cp.local_shard(video, dim=2)  # split along time dimension

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ldist = ctx.encoder(video)

        print(f"{og_shape} -> {ldist.mean.shape}")
        torch.save(
            dict(mean=ldist.mean, logvar=ldist.logvar),
            vid_path.with_suffix(".latent.pt"),
        )

        if reconstruct:
            latents = ldist.sample()
            frames = decode_latents(ctx.decoder, latents)
            frames = frames.cpu().numpy()
            save_video(frames[0], str(vid_path.with_suffix(".recon.mp4")), fps=fps)


@click.command()
@click.argument("videos_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option(
    "--model_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to folder containing Mochi's VAE encoder and decoder weights. Download from Hugging Face: https://huggingface.co/genmo/mochi-1-preview/blob/main/encoder.safetensors and https://huggingface.co/genmo/mochi-1-preview/blob/main/decoder.safetensors",
    default="weights/",
)
@click.option("--num_gpus", default=1, help="Number of GPUs to split the encoder over")
@click.option(
    "--recon_interval", default=10, help="Reconstruct one out of every N videos (0 to disable reconstruction)"
)
@click.option("--shape", default="163x480x848", help="Shape of the video to encode")
@click.option("--overwrite", "-ow", is_flag=True, help="Overwrite existing latents")
def batch_process(
    videos_dir: Path, model_dir: Path, num_gpus: int, recon_interval: int, shape: str, overwrite: bool
) -> None:
    """Process all videos in a directory using multiple GPUs.

    Args:
        videos_dir: Directory containing input videos
        encoder_path: Path to encoder model weights
        decoder_path: Path to decoder model weights
        num_gpus: Number of GPUs to use for parallel processing
        recon_interval: Frequency of video reconstructions (0 to disable)
    """

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Get all video paths
    video_paths = list(videos_dir.glob("**/*.mp4"))
    if not video_paths:
        print(f"No MP4 files found in {videos_dir}")
        return

    preproc = GPUContext(
        encoder_factory=EncoderModelFactory(model_path=os.path.join(model_dir, "encoder.safetensors")),
        decoder_factory=DecoderModelFactory(model_path=os.path.join(model_dir, "decoder.safetensors")),
    )
    with progress_bar(type="ray_tqdm"):
        for idx, video_path in get_new_progress_bar((list(enumerate(sorted(video_paths))))):
            if str(video_path).endswith(".recon.mp4"):
                print(f"Skipping {video_path} b/c it is a reconstruction")
                continue

            print(f"Processing {video_path}")
            try:
                if video_path.with_suffix(".latent.pt").exists() and not overwrite:
                    print(f"Skipping {video_path}")
                    continue

                preprocess(
                    ctx=preproc,
                    vid_path=video_path,
                    shape=shape,
                    reconstruct=recon_interval != 0 and idx % recon_interval == 0,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {video_path}: {str(e)}")


if __name__ == "__main__":
    batch_process()
