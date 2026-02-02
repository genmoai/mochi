#!/usr/bin/env python
"""
Quantize Mochi DiT (Genmo) weights to NF4 and save to disk.

Usage:
    python scripts/quantize_dit.py /path/to/weights/dit.safetensors

Output:
    /path/to/weights/dit_nf4.pt (~5GB instead of ~20GB)

The quantized model can be loaded with:
    model = torch.load("dit_nf4.pt", weights_only=False)
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description="Quantize Mochi DiT to NF4")
    parser.add_argument("input", help="Path to dit.safetensors")
    parser.add_argument("--output", "-o", help="Output path (default: input_nf4.pt)")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                        help="Compute dtype (default: bf16, use fp16 for faster MPS)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    input_path = args.input
    dtype_suffix = "_nf4_fp16.pt" if args.dtype == "fp16" else "_nf4.pt"
    output_path = args.output or input_path.replace(".safetensors", dtype_suffix)
    compute_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if os.path.exists(output_path) and not args.force:
        print(f"Output already exists: {output_path}")
        print("Use --force to overwrite")
        sys.exit(0)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Dtype:  {args.dtype}")
    print()

    # Import after path setup
    from mps_bitsandbytes import quantize_model, BitsAndBytesConfig
    from mps_bitsandbytes.integration import get_memory_footprint
    from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmDiTJoint

    print("Creating model skeleton...")
    model = torch.nn.utils.skip_init(AsymmDiTJoint,
        depth=48,
        patch_size=2,
        num_heads=24,
        hidden_size_x=3072,
        hidden_size_y=1536,
        mlp_ratio_x=4.0,
        mlp_ratio_y=4.0,
        in_channels=12,
        qk_norm=True,
        qkv_bias=False,
        out_bias=True,
        patch_embed_bias=True,
        timestep_mlp_bias=True,
        timestep_scale=1000.0,
        t5_feat_dim=4096,
        t5_token_length=256,
        rope_theta=10000.0,
        attention_mode="sdpa",  # Will be overridden at runtime based on device
    )

    print("Loading weights from safetensors...")
    sd = load_file(input_path)
    model.load_state_dict(sd)
    del sd  # Free memory

    print()
    print("Quantizing to NF4...")
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Skip patch embed conv and normalization layers (keep full precision)
    modules_to_skip = ["x_embedder.proj", "norm", "final_layer"]

    model = quantize_model(
        model,
        quantization_config=config,
        modules_to_not_convert=modules_to_skip,
    )

    # Report stats
    stats = get_memory_footprint(model)
    print(f"  Original (fp16): {stats['fp16_size_gb']:.2f} GB")
    print(f"  Quantized (nf4): {stats['actual_size_gb']:.2f} GB")
    print(f"  Savings: {stats['savings_pct']:.1f}%")
    print()

    print(f"Saving to {output_path}...")
    # Save entire model (not just state_dict) so we skip skeleton creation on load
    torch.save(model, output_path)

    # Check file size
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"  File size: {size_gb:.2f} GB")
    print()
    print("Done!")
    print()
    print("To use with Genmo pipeline:")
    print(f"  model = torch.load('{output_path}', weights_only=False)")
    print()
    print("Or run demos/cli.py with --quantize flag (auto-detects cached weights)")


if __name__ == "__main__":
    main()
