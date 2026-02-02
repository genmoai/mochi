# Based on Llama3 Implementation.
import torch


def apply_rotary_emb_qk_real(
    xqk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor without complex numbers.

    Args:
        xqk (torch.Tensor): Query and/or Key tensors to apply rotary embeddings. Shape: (B, S, *, num_heads, D)
                            Can be either just query or just key, or both stacked along some batch or * dim.
        freqs_cos (torch.Tensor): Precomputed cosine frequency tensor.
        freqs_sin (torch.Tensor): Precomputed sine frequency tensor.

    Returns:
        torch.Tensor: The input tensor with rotary embeddings applied.
    """
    # Do RoPE math in float32 like Diffusers does (bf16 loses precision on MPS)
    orig_dtype = xqk.dtype
    xqk_even = xqk[..., 0::2].float()
    xqk_odd = xqk[..., 1::2].float()
    freqs_cos = freqs_cos.float()
    freqs_sin = freqs_sin.float()

    # Apply rotation in float32
    cos_part = xqk_even * freqs_cos - xqk_odd * freqs_sin
    sin_part = xqk_even * freqs_sin + xqk_odd * freqs_cos

    # Interleave and cast back to original dtype
    out = torch.stack([cos_part, sin_part], dim=-1).flatten(-2).to(orig_dtype)
    return out
