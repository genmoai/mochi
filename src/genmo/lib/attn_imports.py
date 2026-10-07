from contextlib import contextmanager

import torch


# ============================================================================
# MPS Flash Attention (Apple Silicon)
# ============================================================================
try:
    from mps_flash_attn import flash_attention as mps_flash_attn
    from mps_flash_attn import is_available as mps_flash_available
    HAS_MPS_FLASH = mps_flash_available()
    try:
        from mps_flash_attn import quantize_kv_nf4 as mps_quantize_kv_nf4
        from mps_flash_attn import flash_attention_nf4 as mps_flash_attn_nf4
        HAS_MPS_FLASH_NF4 = True
    except ImportError:
        mps_quantize_kv_nf4 = None
        mps_flash_attn_nf4 = None
        HAS_MPS_FLASH_NF4 = False
    if HAS_MPS_FLASH:
        nf4_status = " (with NF4 KV cache)" if HAS_MPS_FLASH_NF4 else ""
        print(f"✓ MPS Flash Attention available{nf4_status}")
except ImportError:
    mps_flash_attn = None
    mps_quantize_kv_nf4 = None
    mps_flash_attn_nf4 = None
    HAS_MPS_FLASH = False
    HAS_MPS_FLASH_NF4 = False

# ============================================================================
# CUDA Flash Attention
# ============================================================================
try:
    from flash_attn import flash_attn_varlen_func as flash_varlen_attn
except ImportError:
    flash_varlen_attn = None

try:
    from sageattention import sageattn as sage_attn
except ImportError:
    sage_attn = None

from torch.nn.attention import SDPBackend, sdpa_kernel

# Device-agnostic backend selection
training_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
eval_backends = list(training_backends)

# Only check CUDA properties if CUDA is available
if torch.cuda.is_available():
    try:
        if torch.cuda.get_device_properties(0).major >= 9.0:
            # Enable fast CuDNN attention on Hopper.
            # This gives NaN on the backward pass for some reason,
            # so only use it for evaluation.
            eval_backends.append(SDPBackend.CUDNN_ATTENTION)
    except Exception:
        pass  # No CUDA device available

@contextmanager
def sdpa_attn_ctx(training: bool = False):
    with sdpa_kernel(training_backends if training else eval_backends):
        yield
