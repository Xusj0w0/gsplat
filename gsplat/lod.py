"""Block LoD CUDA operations, built separately from the standard gsplat backend."""

from pathlib import Path
import torch

_backend = None


def backend():
    global _backend
    if _backend is None:
        from torch.utils.cpp_extension import load

        root = Path(__file__).parent / "cuda"
        _backend = load(
            name="gsplat_block_lod",
            sources=[str(root / "lod/lod.cu")],
            extra_include_paths=[
                str(root / "csrc"),
                str(root / "csrc/third_party/glm"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        )
    return _backend


class _Interpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, child, parent, weights):
        ctx.save_for_backward(child, parent, weights)
        ctx.rows = len(values)
        return backend().interpolate(values.contiguous(), child, parent, weights)

    @staticmethod
    def backward(ctx, gradient):
        child, parent, weights = ctx.saved_tensors
        return (
            backend().interpolate_backward(
                gradient.contiguous(), child, parent, weights, ctx.rows
            ),
            None,
            None,
            None,
        )


interpolate = _Interpolate.apply


@torch.compile(dynamic=True)
def _activation_gradient(raw, proxy, value, gradient):
    bounded = ((raw[:, 3:7] >= -30) & (raw[:, 3:7] <= 20)).to(raw.dtype)
    opacity = torch.where(proxy, value[:, 3] * bounded[:, 0],
                          value[:, 3] * (1 - value[:, 3]))
    q = value[:, 7:11]
    norm = raw[:, 7:11].norm(dim=-1, keepdim=True)
    gq = gradient[:, 7:11]
    gq = torch.where(norm >= 1e-12, gq - q * (gq * q).sum(-1, keepdim=True), gq)
    return torch.cat((gradient[:, :3], (gradient[:, 3] * opacity)[:, None],
                      gradient[:, 4:7] * value[:, 4:7] * bounded[:, 1:],
                      gq / norm.clamp_min(1e-12)), dim=1)


class _Activate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, proxy):
        out = backend().activate(raw.contiguous(), proxy.contiguous())
        ctx.save_for_backward(raw, proxy, out)
        return out

    @staticmethod
    def backward(ctx, gradient):
        return _activation_gradient(*ctx.saved_tensors, gradient), None


activate_geometry = _Activate.apply


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means,
        conics,
        colors,
        opacity,
        weights,
        inv_k,
        background,
        width,
        height,
        tile_size,
        offsets,
        ids,
        grad_mask,
    ):
        image, alpha, last = backend().raster_fwd(
            means,
            conics,
            colors,
            weights,
            inv_k,
            opacity,
            background,
            width,
            height,
            tile_size,
            offsets,
            ids,
        )
        ctx.save_for_backward(
            means,
            conics,
            colors,
            opacity,
            weights,
            inv_k,
            background,
            offsets,
            ids,
            alpha,
            last,
            grad_mask,
        )
        ctx.settings = width, height, tile_size
        return image, alpha

    @staticmethod
    def backward(ctx, image_grad, alpha_grad):
        (
            means,
            conics,
            colors,
            opacity,
            weights,
            inv_k,
            background,
            offsets,
            ids,
            alpha,
            last,
            grad_mask,
        ) = ctx.saved_tensors
        _, gm, gc, gcol, go = backend().raster_bwd(
            means,
            conics,
            colors,
            weights,
            inv_k,
            opacity,
            background,
            *ctx.settings,
            offsets,
            ids,
            alpha,
            last,
            image_grad.contiguous(),
            alpha_grad.contiguous(),
            False,
            grad_mask,
        )
        gb = (image_grad * (1 - alpha)).sum((1, 2)) if ctx.needs_input_grad[6] else None
        return gm, gc, gcol, go, None, None, gb, None, None, None, None, None, None


def rasterize(
    means,
    conics,
    colors,
    opacity,
    weights,
    inv_k,
    background,
    width,
    height,
    tile_size,
    offsets,
    ids,
    grad_mask=None,
):
    return _Rasterize.apply(
        means.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacity.contiguous(),
        weights.contiguous(),
        inv_k.contiguous(),
        background,
        width,
        height,
        tile_size,
        offsets,
        ids,
        grad_mask.contiguous() if grad_mask is not None else None,
    )
