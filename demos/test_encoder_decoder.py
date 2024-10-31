import time
import click
import torch
import subprocess
import numpy as np
import cv2
from einops import rearrange

from genmo.mochi_preview.vae.models import Encoder, Decoder, add_fourier_features

@click.command()
@click.argument('mochi_dir', type=str)
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('model_size', type=str, default="small")
def reconstruct(mochi_dir, video_path, model_size):
    def load_video(video_path: str, num_frames: int, fps: float, width: int, height: int):
        """Load video using ffmpeg with basic resizing."""
        ffmpeg_command = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps},scale={width}:{height}",
            "-frames:v", str(num_frames),
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-"
        ]

        process = subprocess.Popen(
            ffmpeg_command, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        raw_output, err = process.communicate()

        if process.returncode != 0:
            return None

        # Convert to numpy array and reshape
        frame_size = height * width * 3
        video = np.frombuffer(raw_output, np.uint8).reshape(-1, height, width, 3)

        # Ensure we have exactly num_frames
        if len(video) < num_frames:
            return None
        video = video[:num_frames]

        return torch.from_numpy(video.copy())


    from genmo.lib.utils import save_video
    from genmo.mochi_preview.pipelines import DecoderModelFactory, decode_latents_tiled_spatial
    from safetensors.torch import load_file

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    decoder_factory = DecoderModelFactory(
        model_path=f"{mochi_dir}/decoder.safetensors",
        model_stats_path=f"{mochi_dir}/decoder_stats.json"
    )
    decoder = decoder_factory.get_model(
        world_size=1,
        device_id=0,
        local_rank=0
    )

    config = dict(
        prune_bottlenecks=[True, True, True, False, False],
        has_attentions=[False, False, False, False, False],
        affine=False,
        bias=False,
        padding_mode="zeros"
    ) if model_size == "small" else dict(
        prune_bottlenecks=[False, False, False, False, False],
        has_attentions=[False, True, True, True, True],
        affine=True,
        bias=True,
        input_is_conv_1x1=True,
        padding_mode="replicate"
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
    encoder.load_state_dict(load_file(f"{mochi_dir}/encoder.{model_size}.safetensors"))
    encoder.eval()

    # get fps and numframes in video
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video = load_video(video_path, num_frames, fps, width, height)
    video = rearrange(video, 't h w c -> c t h w')
    video = video.unsqueeze(0)
    assert video.dtype == torch.uint8
    assert video.shape == (1, 3, num_frames, height, width)
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
    save_video(frames.cpu().numpy()[0], f"{video_path}.{model_size}.recon.mp4", fps=fps)

if __name__ == "__main__":
    reconstruct()