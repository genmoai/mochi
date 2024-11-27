# Mochi 1 LoRA Fine-tuner

![Mochi being made](../../assets/mochi-factory.webp)


This folder contains tools for fine-tuning the Mochi 1 model. It supports [LoRA](https://arxiv.org/abs/2106.09685) fine-tuning on a single GPU.

## Quick Start (Single GPU)
This shows you how to prepare your dataset for single GPU.

First, setup the inference code and download Mochi 1 weights following [README.md](../../README.md).
All commands below assume you are in the top-level directory of the Mochi repo.

### 1. Collect your videos and captions
Collect your videos (supported formats: MP4, MOV) into a folder, e.g. `videos/`. Then, write a detailed description of each of the videos in a txt file with the same name. For example,
```
videos/
  video_1.mp4
  video_1.txt -- One-paragraph description of video_1
  video_2.mp4
  video_2.txt -- One-paragraph description of video_2
  ...
```

### 2. Process videos and captions (About 2 minutes)
Update the paths in the command below to match your dataset. Videos are processed at 30 FPS, so make sure your videos are at least `num_frames / 30` seconds long.
```bash
bash demos/fine_tuner/preprocess.bash -v videos/ -o videos_prepared/ -w weights/ --num_frames 37
```

### 3. Fine-tune the model
Update `./demos/fine_tuner/configs/lora.yaml` to customize the fine-tuning process,
including prompts to generate at various points of the fine-tuning process and the path to your prepared videos.

Launch LoRA fine-tuning on single GPU:
```bash
bash ./demos/fine_tuner/run.bash -c ./demos/fine_tuner/configs/lora.yaml -n 1
```

Samples will be generated in `finetunes/my_mochi_lora/samples` every 200 steps.

### 4. Use your fine-tuned weights to generate videos!
Update `--lora_path` to the path of your fine-tuned weights and run:
```python
python3 ./demos/cli.py --model_dir weights/ --lora_path finetunes/my_mochi_lora/model_2000.lora.safetensors --num_frames 37 --cpu_offload --prompt "A delicate porcelain teacup sits on a marble countertop. The teacup suddenly shatters into hundreds of white ceramic shards that scatter through the air. The scene is bright and crisp with dramatic lighting."
```

You can increase the number of frames to generate a longer video. Finally, share your creations with the community by uploading your LoRA and sample videos to Hugging Face.

## System Requirements

**Single GPU:**
- 1x H100 or A100 (80 GB VRAM is recommended)
- Less VRAM is required if training with less than 1 second long videos.

**Supported video lengths:** Up to 85 frames (~2.8 seconds at 30 FPS)
- Choose a frame count in increments of 6: 25, 31, 37, ... 79, 85.
- Training on 37 frames uses 50 GB of VRAM. On 1 H100, each training step takes about 1.67 s/it,
  and you'll start seeing changes to your videos within 200-400 steps. Training for 1,000 steps takes about 30 minutes.

Settings tested on 1x H100 SXM:

| Frames | Video Length | VRAM | Time/step | num_qkv_checkpoint | num_ff_checkpoint | num_post_attn_checkpoint |
|--------|--------------|------|-----------|-------------------|-------------------|-------------------------|
| 37 frames | 1.2 second videos | 50 GB VRAM | 1.67 s/it | 48 | 48† | 48 |
| 61 frames | 2.0 second videos | 64 GB VRAM | 3.35 s/it | 48 | 48† | 48 |
| 79 frames | 2.6 second videos | 69-78 GB VRAM | 4.92 s/it | 48 | 48† | 48 |
| 85 frames | 2.8 second videos | 80 GB VRAM | 5.44 s/it | 48 | 48 | 48 |

*† As the VRAM is not fully used, you can lower `num_ff_checkpoint` to speed up training.*

## Technical Details

- LoRA fine-tuning updates the query, key, and value projection matrices, as well as the output projection matrix.
  These settings are configurable in `./demos/fine_tuner/configs/lora.yaml`.
- We welcome contributions and suggestions for improved settings.

## Known Limitations

- No support for training on multiple GPUs
- LoRA inference is restricted to 1-GPU (for now)

## Tips

- Be as descriptive as possible in your captions.
- A learning rate around 1e-4 or 2e-4 seems effective for LoRA fine-tuning.
- For larger datasets or to customize the model aggressively, increase `num_steps` in in the YAML.
- To monitor training loss, uncomment the `wandb` section in the YAML and run `wandb login` or set the `WANDB_API_KEY` environment variable.
- Videos are trimmed to the **first** `num_frames` frames. Make sure your clips contain the content you care about near the beginning.
  You can check the trimmed versions after running `preprocess.bash` to make sure they look good.
- When capturing HDR videos on an iPhone, convert your .mov files to .mp4 using the Handbrake application. Our preprocessing script won't produce the correct colorspace otherwise, and your fine-tuned videos may look overly bright.

### If you are running out of GPU memory, make sure:
- `COMPILE_DIT=1` is set in `demos/fine_tuner/run.bash`.
  This enables model compilation, which saves memory and speeds up training!
- `num_post_attn_checkpoint`, `num_ff_checkpoint`, and `num_qkv_checkpoint` are set to 48 in your YAML.
  You can checkpoint up to 48 layers, saving memory at the cost of slower training.
- If all else fails, reduce `num_frames` when processing your videos and in your YAML.
  You can fine-tune Mochi on shorter videos, and still generate longer videos at inference time.
