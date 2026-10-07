#!/usr/bin/env python
"""
Quantize Diffusers MochiTransformer3DModel to NF4 and save to disk.

Usage:
    python scripts/quantize_diffusers_transformer.py /path/to/transformer

Output:
    /path/to/transformer_nf4.pt (~5GB instead of ~20GB)

The quantized model can be loaded with:
    model = torch.load("transformer_nf4.pt", weights_only=False)
"""
import argparse
import os
import sys
import torch
import gc


def main():
    parser = argparse.ArgumentParser(description="Quantize Diffusers MochiTransformer3DModel to NF4")
    parser.add_argument("input", help="Path to transformer directory (with model files)")
    parser.add_argument("--output", "-o", help="Output path (default: input_nf4.pt)")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                        help="Compute dtype (default: bf16, use fp16 for faster MPS)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    input_path = args.input
    dtype_suffix = "_nf4_fp16.pt" if args.dtype == "fp16" else "_nf4.pt"
    output_path = args.output or os.path.join(os.path.dirname(input_path), f"transformer{dtype_suffix}")
    compute_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if not os.path.exists(input_path):
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    if os.path.exists(output_path) and not args.force:
        print(f"Output already exists: {output_path}")
        print("Use --force to overwrite")
        sys.exit(0)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Dtype:  {args.dtype}")
    print()

    from diffusers import MochiTransformer3DModel
    from mps_bitsandbytes import quantize_model, BitsAndBytesConfig

    print("Loading Diffusers MochiTransformer3DModel...")
    transformer = MochiTransformer3DModel.from_pretrained(
        input_path,
        variant="bf16",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"  Loaded: {num_params:,} params")

    # Check size before quantization
    param_bytes = sum(p.numel() * p.element_size() for p in transformer.parameters())
    print(f"  Size before: {param_bytes / 1e9:.2f} GB")

    print()
    print("Quantizing to NF4...")
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    transformer_q = quantize_model(transformer, quantization_config=config)

    # Check size after
    total_size = 0
    for param in transformer_q.parameters():
        total_size += param.numel() * param.element_size()
    for buf in transformer_q.buffers():
        total_size += buf.numel() * buf.element_size()
    print(f"  Size after: {total_size / 1e9:.2f} GB")
    print(f"  Savings: {(1 - total_size / param_bytes) * 100:.1f}%")

    print()
    print(f"Saving to {output_path}...")
    torch.save(transformer_q, output_path)

    # Verify file size
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / 1e9:.2f} GB")

    print()
    print("Done!")
    print()
    print("To use:")
    print(f"  model = torch.load('{output_path}', weights_only=False)")
    print(f"  model = model.to('mps')  # or 'cuda'")


if __name__ == "__main__":
    main()
