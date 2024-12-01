import modal
from pathlib import Path

# Creating our Modal App
app = modal.App("mochi-finetune")

# Creating volumes for data, intermediate data, and produced weights
videos_volume = modal.Volume.from_name("mochi-tune-videos", create_if_missing=True)
videos_prepared_volume = modal.Volume.from_name("mochi-tune-videos-prepared", create_if_missing=True)
weights_volume = modal.Volume.from_name("mochi-tune-weights", create_if_missing=True)
finetunes_volume = modal.Volume.from_name("mochi-tune-finetunes", create_if_missing=True)
outputs_volume = modal.Volume.from_name("mochi-tune-outputs", create_if_missing=True)

USERNAME = "genmoai"
REPOSITORY = "mochi"
CLONE_CMD = f"git clone https://github.com/{USERNAME}/{REPOSITORY}.git"

# Building our container image
base_img = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg", "bc", "zlib1g-dev", "libjpeg-dev", "wget")
    .run_commands(CLONE_CMD)
    .workdir(REPOSITORY)
    .pip_install("gdown", "setuptools", "wheel")
    .run_commands('pip install -e . --no-build-isolation')
)

MINUTES = 60
HOURS = 60 * MINUTES

# Remote function for downloading a labeled video dataset from Google Drive
# Run it with:
#   modal run main::download_videos
@app.function(image=base_img,
    volumes={
        "/videos": videos_volume,
    }
)
def download_videos():
    '''Downloads videos from google drive into our volume'''
    import gdown
    import zipfile

    name = "dissolve"
    url = "https://drive.google.com/uc?id=1ldoBppcsv5Ueoikh0zCmNviojRCrGXQN"
    output = f"{name}.zip"
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall("/videos")

# Remote function for downloading the model weights from Hugging Face
# Run it with:
#   modal run main::download_weights
@app.function(image=base_img, 
    volumes={
        "/weights": weights_volume,
    },
    timeout=1*HOURS,
)
def download_weights():
    # HF-transfer and snapshot download tend to hang on the large model, so we download it manually with wget
    import subprocess
    print("🍡 Downloading weights from Hugging Face. This may take 30 minutes.")
    # ~30 min
    subprocess.run(["wget", "https://huggingface.co/genmo/mochi-1-preview/resolve/main/dit.safetensors", "-O", "/weights/dit.safetensors"])
    # ~1 min
    subprocess.run(["wget", "https://huggingface.co/genmo/mochi-1-preview/resolve/main/decoder.safetensors", "-O", "/weights/decoder.safetensors"])
    # ~20 sec
    subprocess.run(["wget", "https://huggingface.co/genmo/mochi-1-preview/resolve/main/encoder.safetensors", "-O", "/weights/encoder.safetensors"])

# Remote function for preprocessing the video dataset
# Run it with:
#   modal run main::preprocess
@app.function(
    image=base_img, 
    volumes={
        "/videos": videos_volume,
        "/videos_prepared": videos_prepared_volume,
        "/weights": weights_volume,
    },
    timeout=30*MINUTES,
    gpu="H100"
)
def preprocess():
    import subprocess
    print("🍡 Preprocessing videos. This may take 2-3 minutes.")
    video_dir = "videos_dissolve"
    subprocess.run([
        "bash", "demos/fine_tuner/preprocess.bash", 
        "-v", f"/videos/{video_dir}/",
        "-o", "/videos_prepared/", 
        "-w", "/weights/", 
        "-n", "37"
    ])

# Remote function for finetuning the model using the prepared dataset
# Configure the run in lora.yaml
# Run it with:
#   modal run main::finetune
@app.function(
    image=base_img, 
    volumes={
        "/videos": videos_volume,
        "/videos_prepared": videos_prepared_volume,
        "/weights": weights_volume,
        "/finetunes": finetunes_volume,
    },
    mounts=[modal.Mount.from_local_file("lora.yaml", remote_path=f"{REPOSITORY}/lora.yaml")],
    timeout=4*HOURS,
    gpu="H100"
)
def finetune():
    import subprocess
    print("🍡 Finetuning Mochi. This may take 3 hours.")
    print("🍡 See your mochi-tune-finetunes volume for intermediate checkpoints and samples.")
    subprocess.run([
        "bash", "demos/fine_tuner/run.bash", 
        "-c", "lora.yaml", # from our locally mounted yaml file
        "-n", "1", 
    ])

# Remote function (Modal @cls) for running inference on one or multiple videos
# Run it with the @local_entrypoint below
@app.cls(
    image = base_img,
    volumes={
        "/weights": weights_volume,
        "/finetunes": finetunes_volume,
        "/outputs": outputs_volume,
    },
    timeout=30*MINUTES,
    gpu="H100"
)
class MochiLora():
    def __init__(self, model_dir: str = "/weights", lora_path: str = None, cpu_offload: bool = False):
        self.model_dir = model_dir
        self.lora_path = lora_path
        self.cpu_offload = cpu_offload

    @modal.enter()
    def start(self):
        from genmo.mochi_preview.pipelines import (
            DecoderModelFactory,
            DitModelFactory,
            MochiMultiGPUPipeline,
            MochiSingleGPUPipeline,
            T5ModelFactory,
        )
        import torch

        """Initialize the model - this runs once when the container starts"""
        print("🍡 Loading Mochi model.")

        self.num_gpus = torch.cuda.device_count()
        
        # Configure pipeline based on GPU count
        klass = MochiSingleGPUPipeline if self.num_gpus == 1 else MochiMultiGPUPipeline
        
        kwargs = dict(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(
                model_path=f"{self.model_dir}/dit.safetensors",
                lora_path=self.lora_path,
                model_dtype="bf16",
            ),
            decoder_factory=DecoderModelFactory(
                model_path=f"{self.model_dir}/decoder.safetensors",
            ),
        )

        if self.num_gpus > 1:
            assert not self.lora_path, f"Lora not supported in multi-GPU mode"
            assert not self.cpu_offload, "CPU offload not supported in multi-GPU mode"
            kwargs["world_size"] = self.num_gpus
        else:
            kwargs["cpu_offload"] = self.cpu_offload
            kwargs["decode_type"] = "tiled_spatial"
            kwargs["fast_init"] = not self.lora_path
            kwargs["strict_load"] = not self.lora_path
            kwargs["decode_args"] = dict(overlap=8)

        self.pipeline = klass(**kwargs)
        print(f"🍡 Model loaded successfully with {self.num_gpus} GPUs")

    @modal.method()
    def generate(self, 
                prompt: str,
                negative_prompt: str = "",
                width: int = 848,
                height: int = 480,
                num_frames: int = 163,
                seed: int = 1710977262,
                cfg_scale: float = 6.0,
                num_inference_steps: int = 64) -> str:
        """Generate video based on the prompt and parameters"""
        
        print("🍡 Generating video.")

        import json
        import os
        import time

        import numpy as np

        from genmo.lib.progress import progress_bar
        from genmo.lib.utils import save_video
        from genmo.mochi_preview.pipelines import linear_quadratic_schedule

        
        # Create sigma schedule
        sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025)
        cfg_schedule = [cfg_scale] * num_inference_steps

        args = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "sigma_schedule": sigma_schedule,
            "cfg_schedule": cfg_schedule,
            "num_inference_steps": num_inference_steps,
            "batch_cfg": False,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
        }

        with progress_bar(type="tqdm"):
            final_frames = self.pipeline(**args)
            final_frames = final_frames[0]

            assert isinstance(final_frames, np.ndarray)
            assert final_frames.dtype == np.float32

            # Save to mounted volume
            output_dir = "/outputs"  # Assuming this path exists in the mounted volume
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"output_{int(time.time())}.mp4")

            save_video(final_frames, output_path)
            
            # Save generation parameters
            json_path = os.path.splitext(output_path)[0] + ".json"
            json.dump(args, open(json_path, "w"), indent=4)

        print(f"🍡 Video saved to {output_path}")
        outputs_volume.commit()
        return output_path.split("/")[-1]

# Local entrypoint for using the MochiLora class
# Select the lora_path you'd want to use from the finetunes volume
# Then it with:
#   modal run main
@app.local_entrypoint()
def main(
    prompt="A pristine snowglobe featuring a winter scene sits peacefully. The glass begins to crumble into fine powder, as the entire sphere deteriorates into sparkling dust that drifts outward. The fake snow mingles with the crystalline particles, creating a glittering cloud captured in high-speed photography.",
    negative_prompt="blurry, low quality",
    width=848,
    height=480,
    num_frames=49, # (num_frames - 1) must be divisible by 6
    seed=1710977262,
    cfg_scale=6.0,
    num_inference_steps=64,
    lora_path="/finetunes/my_mochi_lora/model_2000.lora.safetensors",
    cpu_offload=True,
):
    lora = MochiLora(
        lora_path=lora_path, # your lora path
        cpu_offload=cpu_offload,
    )
    output_path = lora.generate.remote(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        seed=seed,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
    )

    local_dir = Path("/tmp/mochi")
    local_dir.mkdir(exist_ok=True, parents=True)
    local_path = local_dir / output_path
    local_path.write_bytes(b"".join(outputs_volume.read_file(output_path)))
    print(f"🍡 video saved locally at {local_path}")
