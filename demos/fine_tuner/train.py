import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from glob import glob
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import click
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from safetensors.torch import save_file
import torch
from torch import Tensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
import torch.nn.functional as F
from tqdm import tqdm

torch._dynamo.config.cache_size_limit = 32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.use_deterministic_algorithms(False)

import genmo.mochi_preview.dit.joint_model.lora as lora
from genmo.lib.progress import progress_bar
from genmo.lib.utils import Timer, save_video
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    ModelFactory,
    T5ModelFactory,
    cast_dit,
    compute_packed_indices,
    get_conditioning,
    linear_quadratic_schedule,  # used in eval'd Python code in lora.yaml
    load_to_cpu,
    move_to_device,
    sample_model,
    t5_tokenizer,
)
from genmo.mochi_preview.vae.latent_dist import LatentDistribution
from genmo.mochi_preview.vae.models import decode_latents_tiled_spatial

sys.path.append("..")

from dataset import LatentEmbedDataset


class MochiTorchRunEvalPipeline:
    def __init__(
        self,
        *,
        device_id,
        dit,
        text_encoder_factory: ModelFactory,
        decoder_factory: ModelFactory,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.tokenizer = t5_tokenizer()
        t = Timer()
        self.dit = dit
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(
                local_rank=0,
                world_size=1,
                device_id="cpu",
            )
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(local_rank=0, device_id="cpu", world_size=1)
        t.print_stats()  # type: ignore

    def __call__(self, prompt, save_path, **kwargs):
        with progress_bar(type="tqdm", enabled=True), torch.inference_mode():
            # Encode prompt with T5 XXL.
            with move_to_device(self.text_encoder, self.device, enabled=True):
                conditioning = get_conditioning(
                    self.tokenizer,
                    self.text_encoder,
                    self.device,
                    batch_inputs=False,
                    prompt=prompt,
                    negative_prompt="",
                )

            # Sample video latents from Mochi.
            with move_to_device(self.dit, self.device, enabled=True):
                latents = sample_model(self.device, self.dit, conditioning, **kwargs)

            # Decode video latents to frames.
            with move_to_device(self.decoder, self.device, enabled=True):
                frames = decode_latents_tiled_spatial(
                    self.decoder, latents, num_tiles_w=2, num_tiles_h=2, overlap=8)
            frames = frames.cpu().numpy()  # b t h w c
            assert isinstance(frames, np.ndarray)

            save_video(frames[0], save_path)


def map_to_device(x, device: torch.device):
    if isinstance(x, dict):
        return {k: map_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [map_to_device(y, device) for y in x]
    elif isinstance(x, tuple):
        return tuple(map_to_device(y, device) for y in x)
    elif isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    else:
        return x


EPOCH_IDX = 0


def infinite_dl(dl):
    global EPOCH_IDX
    while True:
        EPOCH_IDX += 1
        for batch in dl:
            yield batch


@contextmanager
def timer(description="Task", enabled=True):
    if enabled:
        start = time.perf_counter()
    try:
        yield
    finally:
        if enabled:
            elapsed = time.perf_counter() - start  # type: ignore
            print(f"{description} took {elapsed:.4f} seconds")


def get_cosine_annealing_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@click.command()
@click.option("--config-path", type=click.Path(exists=True), required=True, help="Path to YAML config file")
def main(config_path):
    mp.set_start_method("spawn", force=True)
    cfg = cast(DictConfig, OmegaConf.load(config_path))

    device_id = 0
    device_str = f"cuda:0"
    device = torch.device(device_str)

    # Verify checkpoint path exists
    checkpoint_path = Path(cfg.init_checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint file not found: {checkpoint_path}"
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get step number from checkpoint filename
    pattern = r"model_(\d+)\.(lora|checkpoint)\.(safetensors|pt)"
    match = re.search(pattern, str(checkpoint_path))
    if match:
        start_step_num = int(match.group(1))
        opt_path = str(checkpoint_path).replace("model_", "optimizer_")
    else:
        start_step_num = 0
        opt_path = ""

    print(
        f"model={checkpoint_path}, optimizer={opt_path}, start_step_num={start_step_num}"
    )

    wandb_run = None
    sample_prompts = cfg.sample.prompts

    train_vids = list(sorted(glob(f"{cfg.train_data_dir}/*.mp4")))
    train_vids = [v for v in train_vids if not v.endswith(".recon.mp4")]
    print(f"Found {len(train_vids)} training videos in {cfg.train_data_dir}")
    assert len(train_vids) > 0, f"No training data found in {cfg.train_data_dir}"
    if cfg.single_video_mode:
        train_vids = train_vids[:1]
        sample_prompts = [Path(train_vids[0]).with_suffix(".txt").read_text()]
        print(f"Training on video: {train_vids[0]}")

    train_dataset = LatentEmbedDataset(
        train_vids,
        repeat=1_000 if cfg.single_video_mode else 1,
    )
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    train_dl_iter = infinite_dl(train_dl)

    if cfg.get("wandb"):
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}-{int(time.time())}",
            config=OmegaConf.to_container(cfg),  # type: ignore
        )
        print(f"🚀 Weights & Biases run URL: {wandb_run.get_url()}")

    print("Loading model")
    patch_model_fns = []
    model_kwargs = {}
    is_lora = cfg.model.type == "lora"
    print(f"Training type: {'LoRA' if is_lora else 'Full'}")
    if is_lora:
        def mark_lora_params(m):
            lora.mark_only_lora_as_trainable(m, bias="none")
            return m

        patch_model_fns.append(mark_lora_params)
        model_kwargs = dict(**cfg.model.kwargs)
        # Replace ListConfig with list to allow serialization to JSON.
        for k, v in model_kwargs.items():
            if isinstance(v, ListConfig):
                model_kwargs[k] = list(v)

    if cfg.training.get("model_dtype"):
        assert cfg.training.model_dtype == "bf16", f"Only bf16 is supported"
        patch_model_fns.append(lambda m: cast_dit(m, torch.bfloat16))

    model = (
        DitModelFactory(
            model_path=str(checkpoint_path),
            model_dtype="bf16",
            attention_mode=cfg.attention_mode
        ).get_model(
            local_rank=0,
            device_id=device_id,
            model_kwargs=model_kwargs,
            patch_model_fns=patch_model_fns,
            world_size=1,
            strict_load=not is_lora,
            fast_init=not is_lora,  # fast_init not supported for LoRA (please someone fix this !!!)
        )
        .train()  # calling train() makes sure LoRA weights are not merged
    )

    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)
    if os.path.exists(opt_path):
        print("Loading optimizer")
        optimizer.load_state_dict(load_to_cpu(opt_path))

    scheduler = get_cosine_annealing_lr_scheduler(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=cfg.training.num_steps
    )

    print("Loading eval pipeline ...")
    eval_pipeline = MochiTorchRunEvalPipeline(
        device_id=device_id,
        dit=model,
        text_encoder_factory=T5ModelFactory(),
        decoder_factory=DecoderModelFactory(model_path=cfg.sample.decoder_path),
    )

    def get_batch() -> Tuple[Dict[str, Any], Tensor, Tensor, Tensor]:
        nonlocal train_dl_iter
        batch = next(train_dl_iter)  # type: ignore
        latent, embed = cast(Tuple[Dict[str, Any], Dict[str, Any]], batch)
        assert len(embed["y_feat"]) == 1 and len(embed["y_mask"]) == 1, f"Only batch size 1 is supported"

        ldist = LatentDistribution(latent["mean"], latent["logvar"])
        z = ldist.sample()
        assert torch.isfinite(z).all()
        assert z.shape[0] == 1, f"Only batch size 1 is supported"

        eps = torch.randn_like(z)
        sigma = torch.rand(z.shape[:1], device="cpu", dtype=torch.float32)

        if random.random() < cfg.training.caption_dropout:
            embed["y_mask"][0].zero_()
            embed["y_feat"][0].zero_()
        return embed, z, eps, sigma

    pbar = tqdm(
        range(start_step_num, cfg.training.num_steps),
        total=cfg.training.num_steps,
        initial=start_step_num,
    )
    for step in pbar:
        if cfg.sample.interval and step % cfg.sample.interval == 0 and step > 0:
            sample_dir = Path(cfg.sample.output_dir)
            sample_dir.mkdir(exist_ok=True)
            model.eval()
            for eval_idx, prompt in enumerate(sample_prompts):
                save_path = sample_dir / f"{eval_idx}_{step}.mp4"
                if save_path.exists():
                    print(f"Skipping {save_path} as it already exists")
                    continue

                sample_kwargs = {
                    k.removesuffix("_python_code"): (eval(v) if k.endswith("_python_code") else v)
                    for k, v in cfg.sample.kwargs.items()
                }
                eval_pipeline(
                    prompt=prompt,
                    save_path=str(save_path),
                    seed=cfg.sample.seed + eval_idx,
                    **sample_kwargs,
                )
                Path(sample_dir / f"{eval_idx}_{step}.txt").write_text(prompt)
            model.train()

        if cfg.training.save_interval and step > 0 and step % cfg.training.save_interval == 0:
            with timer("get_state_dict"):
                if is_lora:
                    model_sd = lora.lora_state_dict(model, bias="none")
                else:
                    # NOTE: Not saving optimizer state dict to save space.
                    model_sd, _optimizer_sd = get_state_dict(
                        model, [], options=StateDictOptions(cpu_offload=True, full_state_dict=True)
                    )

            checkpoint_filename = f"model_{step}.{'lora' if is_lora else 'checkpoint'}.pt"
            save_path = checkpoint_dir / checkpoint_filename
            if cfg.training.get("save_safetensors", True):
                save_path = save_path.with_suffix(".safetensors")
                save_file(
                    model_sd, save_path,
                    # `safetensors` only supports string-to-string metadata,
                    # so we serialize the kwargs to a JSON string.
                    metadata=dict(kwargs=json.dumps(model_kwargs)),
                )
            else:
                torch.save(model_sd, save_path)

        with torch.no_grad(), timer("load_batch", enabled=False):
            batch = get_batch()
            embed, z, eps, sigma = map_to_device(batch, device)
            embed = cast(Dict[str, Any], embed)

            num_latent_toks = np.prod(z.shape[-3:])
            indices = compute_packed_indices(device, cast(Tensor, embed["y_mask"][0]), int(num_latent_toks))

            sigma_bcthw = sigma[:, None, None, None, None]  # [B, 1, 1, 1, 1]
            z_sigma = (1 - sigma_bcthw) * z + sigma_bcthw * eps
            ut = z - eps

        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(
                x=z_sigma,
                sigma=sigma,
                packed_indices=indices,
                **embed,
                num_ff_checkpoint=cfg.training.num_ff_checkpoint,
                num_qkv_checkpoint=cfg.training.num_qkv_checkpoint,
            )
            assert preds.shape == z.shape

        loss = F.mse_loss(preds.float(), ut.float())
        loss.backward()

        log_kwargs = {
            "train/loss": loss.item(),
            "train/epoch": EPOCH_IDX,
            "train/lr": scheduler.get_last_lr()[0],
        }

        if cfg.training.get("grad_clip"):
            assert not is_lora, "Gradient clipping not supported for LoRA"
            gnorm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.training.grad_clip)
            log_kwargs["train/gnorm"] = gnorm_before_clip.item()
        pbar.set_postfix(**log_kwargs)

        if wandb_run:
            wandb_run.log(log_kwargs, step=step)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    main()