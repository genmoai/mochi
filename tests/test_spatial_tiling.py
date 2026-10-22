import pytest
import torch

from genmo.mochi_preview.vae.models import Decoder, apply_tiled, decode_latents_tiled_spatial


@pytest.mark.parametrize("tiles", [(2, 1), (1, 2), (2, 2), (4, 2)])
@pytest.mark.parametrize("overlap", [0, 1, 2, 3, 5, 6])
def test_identity_tiling_preserves_shape_and_values(tiles, overlap):
    x = torch.arange(3 * 12 * 16, dtype=torch.float64).reshape(1, 3, 1, 12, 16)
    result = apply_tiled(
        lambda tile: tile,
        x,
        num_tiles_w=tiles[0],
        num_tiles_h=tiles[1],
        overlap=overlap,
    )
    torch.testing.assert_close(result, x)


@pytest.mark.parametrize("tiles", [(2, 1), (1, 2), (2, 2), (4, 2)])
@pytest.mark.parametrize("overlap", [3, 5])
def test_odd_overlap_preserves_native_decoder_dimensions(tiles, overlap):
    torch.manual_seed(0)
    decoder = Decoder(
        latent_dim=2,
        base_channels=32,
        channel_multipliers=[1, 2],
        num_res_blocks=[1, 1, 1],
        temporal_expansions=[2],
        spatial_expansions=[2],
        has_attention=[False, False, False],
        output_norm=False,
    ).eval()
    z = torch.randn(1, 2, 1, 8, 8)
    frames = decode_latents_tiled_spatial(
        decoder,
        z,
        num_tiles_w=tiles[0],
        num_tiles_h=tiles[1],
        overlap=overlap,
    )
    assert frames.shape == (1, 1, 16, 16, 3)
    assert torch.isfinite(frames).all()


@pytest.mark.parametrize("overlap", [3, 5, 8])
def test_native_decoder_uses_existing_symmetric_overlap(overlap):
    torch.manual_seed(1)
    decoder = Decoder(
        latent_dim=2,
        base_channels=32,
        channel_multipliers=[1, 2],
        num_res_blocks=[1, 1, 1],
        temporal_expansions=[2],
        spatial_expansions=[2],
        has_attention=[False, False, False],
        output_norm=False,
    ).eval()
    z = torch.randn(1, 2, 1, 8, 8)
    frames = decode_latents_tiled_spatial(decoder, z, num_tiles_w=2, num_tiles_h=2, overlap=overlap)
    assert frames.shape == (1, 1, 16, 16, 3)
    even_frames = decode_latents_tiled_spatial(decoder, z, num_tiles_w=2, num_tiles_h=2, overlap=2 * (overlap // 2))
    torch.testing.assert_close(frames, even_frames, rtol=0, atol=0)


def test_eightfold_native_decoder_preserves_video_dimensions():
    decoder = Decoder(
        latent_dim=12,
        base_channels=4,
        channel_multipliers=[1, 2, 4, 6],
        num_res_blocks=[0, 0, 0, 0, 0],
        temporal_expansions=[1, 2, 3],
        spatial_expansions=[2, 2, 2],
        has_attention=[False, False, False, False, False],
        output_norm=False,
    ).eval()
    z = torch.randn(1, 12, 2, 8, 8)
    frames = decode_latents_tiled_spatial(decoder, z, num_tiles_w=2, num_tiles_h=2, overlap=3)
    assert frames.shape == (1, 7, 64, 64, 3)


def test_odd_overlap_keeps_block_aligned_tiles():
    x = torch.arange(8 * 8, dtype=torch.float64).reshape(1, 1, 1, 8, 8)

    def downsample(tile):
        assert tile.shape[-1] % 2 == tile.shape[-2] % 2 == 0
        return tile[..., ::2, ::2]

    result = apply_tiled(downsample, x, num_tiles_w=2, num_tiles_h=2, overlap=5, min_block_size=2)
    torch.testing.assert_close(result, downsample(x))
