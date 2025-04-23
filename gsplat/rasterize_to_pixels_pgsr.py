import warnings
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from .cuda._wrapper import _make_lazy_cuda_func


def get_padded_channels(channels: int, device: torch.device) -> int:
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(*backgrounds.shape[:-1], padded_channels, device=device),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0


def rasterize_to_pixels_pgsr(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    colors: Tensor,  # [C, N, channels] or [nnz, channels]
    all_maps: Tensor,  # [C, N, PGSR_MAP_DIM==5] or [nnz, PGSR_MAP_DIM]
    opacities: Tensor,  # [C, N] or [nnz]
    instrinsics: Tensor,  # [C, 4]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    masks: Optional[Tensor] = None,  # [C, tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
    render_geo: bool = True,
):
    """Rasterizes Gaussians to pixels using PGSR (https://github.com/zju3dv/PGSR/tree/main).

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        all_maps: Maps needed by PGSR. [C ,N, 2] if packed is False, [nnz, 2] if packed is True. :3 ~ normals, -1 ~ distances
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        instrinsics: Camera intrinsics. [C, 4], [fx, fy, cx, cy]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.
        render_geo: If True, the function will render PGSR geometry. Default: True.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
        - **Rendered maps**. [C, image_height, image_width, 5]
        - **Rendered plane depths**. [C, image_height, image_width, 1]
    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert all_maps.shape[0] == nnz, all_maps.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert all_maps.shape[:2] == (C, N), all_maps.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    # # Pad the channels to the nearest supported number if necessary
    # channels = colors.shape[-1]
    # padded_channels = get_padded_channels(channels, device)

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert tile_height * tile_size >= image_height, f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert tile_width * tile_size >= image_width, f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    render_colors, render_alphas, render_maps, render_plane_depths = _RasterizeToPixelsPGSR.apply(
        instrinsics.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        all_maps.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        render_geo,
    )

    return render_colors, render_alphas, render_maps, render_plane_depths


class _RasterizeToPixelsPGSR(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        instrinsics: Tensor,  # [C, 4]
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        all_maps: Tensor,  # [C, N, PGSR_MAP_DIM==5]
        opacities: Tensor,  # [C, N]
        backgrounds: Tensor,  # [C, D], Optional
        masks: Tensor,  # [C, tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
        render_geo: bool,
    ):
        (
            render_colors,
            render_alphas,
            out_all_maps,
            out_plane_depths,
            out_observe,
            last_ids,
            has_hit_any_pixels,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_pgsr_fwd")(
            instrinsics,
            means2d,
            conics,
            colors,
            all_maps,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_geo,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            # for bwd
            instrinsics,
            all_maps,
            out_all_maps,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.render_geo = render_geo

        means2d.has_hit_any_pixels = has_hit_any_pixels
        means2d.out_observe = out_observe

        render_alphas = render_alphas.float()
        return render_colors, render_alphas, out_all_maps, out_plane_depths

    @staticmethod
    def backward(ctx, v_render_colors: Tensor, v_render_alphas: Tensor, v_render_maps: Tensor, v_plane_depths: Tensor):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            # for bwd
            instrinsics,
            all_maps,
            out_all_maps,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad
        render_geo = ctx.render_geo

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_all_maps,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_pgsr_bwd")(
            instrinsics,
            means2d,
            conics,
            colors,
            all_maps,
            out_all_maps,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors,
            v_render_alphas,
            v_render_maps,
            v_plane_depths,
            absgrad,
            render_geo,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grads[6]:
            v_backgrouds = (v_render_colors * (1.0 - render_alphas).float()).sum(dim=(1, 2))
        else:
            v_backgrouds = None

        return (
            None,
            v_means2d,
            v_conics,
            v_colors,
            v_all_maps,
            v_opacities,
            v_backgrouds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
