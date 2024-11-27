import time

import click
import torch
import torchvision
from einops import rearrange
from safetensors.torch import load_file

from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines import DecoderModelFactory, decode_latents_tiled_spatial
from genmo.mochi_preview.vae.models import Encoder, add_fourier_features


@click.command()
@click.argument("mochi_dir", type=str)
@click.argument("video_path", type=click.Path(exists=True))
def reconstruct(mochi_dir, video_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    decoder_factory = DecoderModelFactory(
        model_path=f"{mochi_dir}/decoder.safetensors",
    )
    decoder = decoder_factory.get_model(world_size=1, device_id=0, local_rank=0)

    config = dict(
        prune_bottlenecks=[False, False, False, False, False],
        has_attentions=[False, True, True, True, True],
        affine=True,
        bias=True,
        input_is_conv_1x1=True,
        padding_mode="replicate",
    )

    # Create VAE encoder
    encoder = Encoder(
        in_channels=15,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 6],
        num_res_blocks=[3, 3, 4, 6, 3],
        latent_dim=12,
        temporal_reductions=[1, 2, 3],
        spatial_reductions=[2, 2, 2],
        **config,
    )
    device = torch.device("cuda:0")
    encoder = encoder.to(device, memory_format=torch.channels_last_3d)
    encoder.load_state_dict(load_file(f"{mochi_dir}/encoder.safetensors"))
    encoder.eval()

    video, _, metadata = torchvision.io.read_video(video_path, output_format="THWC")
    fps = metadata["video_fps"]
    video = rearrange(video, "t h w c -> c t h w")
    video = video.unsqueeze(0)
    assert video.dtype == torch.uint8
    # Convert to float in [-1, 1] range.
    video = video.float() / 127.5 - 1.0
    video = video.to(device)
    video = add_fourier_features(video)
    torch.cuda.synchronize()

    # Encode video to latent
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t0 = time.time()
            ldist = encoder(video)
            torch.cuda.synchronize()
            print(f"Time to encode: {time.time() - t0:.2f}s")
            t0 = time.time()
            frames = decode_latents_tiled_spatial(decoder, ldist.sample(), num_tiles_w=2, num_tiles_h=2)
            torch.cuda.synchronize()
            print(f"Time to decode: {time.time() - t0:.2f}s")
    t0 = time.time()
    save_video(frames.cpu().numpy()[0], f"{video_path}.recon.mp4", fps=fps)
    print(f"Time to save: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    reconstruct()
