// Sparse block LoD selection, interpolation and RGB alpha-corrected
// rasterization.
#include "bindings.h"
#include "lod_raster_bwd.cuh"
#include "lod_raster_fwd.cuh"
#include "utils.cuh"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>

// Inputs are activated geometry: xyz, positive falloff, positive scales, wxyz.
__device__ float projected_radius(
    const float *g, const float *v, const float *k, int width, int height
) {
    mat3<float> R(v[0], v[4], v[8], v[1], v[5], v[9], v[2], v[6], v[10]);
    vec3<float> t(v[3], v[7], v[11]), mc;
    pos_world_to_cam(R, t, glm::make_vec3(g), mc);
    // Never discard descendants because a learned proxy is behind the near
    // plane.
    if (mc.z <= .01f)
        return 1e12f;
    mat3<float> cov, cc;
    quat_scale_to_covar_preci<float>(
        glm::make_vec4(g + 7), glm::make_vec3(g + 4), &cov, nullptr
    );
    covar_world_to_cam(R, cov, cc);
    mat2<float> c2;
    vec2<float> xy;
    persp_proj<float>(mc, cc, k[0], k[4], k[2], k[5], width, height, c2, xy);
    float comp, det = add_blur(.3f, c2, comp);
    float b = .5f * (c2[0][0] + c2[1][1]);
    float radius = 3.f * sqrtf(fmaxf(0.f, b + sqrtf(fmaxf(.01f, b * b - det))));
    return isfinite(radius) ? radius : 1e12f;
}
__device__ float morph_weight(float tau, float end, float start) {
    if (end <= start + 1e-6f)
        return 1.f;
    return fminf(1.f, fmaxf(0.f, (end - tau) / (end - start)));
}
// One thread per coarse family; output is sparse in deterministic block-local
// slots.
__global__ void select_kernel(
    const float *geom,
    const int *counts,
    const float *v,
    const float *k,
    int B,
    int S,
    int M,
    int C,
    int width,
    int height,
    float tau,
    bool warm,
    int64_t *parent,
    float *weight,
    float *invk,
    int *chosen
) {
    int family = blockIdx.x * blockDim.x + threadIdx.x;
    if (family >= B * C)
        return;
    int b = family / C, c = family % C, T = S + M + C, base = b * T,
        group = M / C, span = S / M;
    int coarse = base + S + M + c;
    if (counts[coarse] == 0)
        return;
    float cr =
        warm ? 0.f : projected_radius(geom + coarse * 11, v, k, width, height);
    float maxmid = 0.f;
    // Only 4 (or configured) intermediate nodes; enforce monotone effective
    // radii.
    for (int j = c * group; j < (c + 1) * group; j++)
        if (counts[base + S + j])
            maxmid = fmaxf(
                maxmid,
                projected_radius(
                    geom + (base + S + j) * 11, v, k, width, height
                )
            );
    // A nonzero interval is essential when an optimized child exceeds its
    // parent.
    cr = fmaxf(cr, maxmid * 1.001f + 1e-5f);
    if (!warm && cr <= tau) {
        chosen[coarse] = 1;
        parent[coarse] = coarse;
        weight[coarse] = 1.f;
        invk[coarse] = 1.f;
        return;
    }
    int siblings = 0;
    for (int j = c * group; j < (c + 1) * group; j++)
        siblings += counts[base + S + j] > 0;
    for (int j = c * group; j < (c + 1) * group; j++) {
        int mid = base + S + j, kids = counts[mid];
        if (!kids)
            continue;
        float mr = projected_radius(geom + mid * 11, v, k, width, height);
        if (mr <= tau) {
            chosen[mid] = 1;
            parent[mid] = warm ? mid : coarse;
            weight[mid] =
                warm ? 1.f : morph_weight(tau, cr, fmaxf(.5f * cr, maxmid));
            invk[mid] = warm ? 1.f : 1.f / siblings;
        } else {
            float w = morph_weight(tau, mr, .5f * mr);
            for (int l = j * span; l < (j + 1) * span; l++)
                if (counts[base + l]) {
                    int id = base + l;
                    chosen[id] = 1;
                    parent[id] = mid;
                    weight[id] = w;
                    invk[id] = 1.f / kids;
                }
        }
    }
}
std::vector<torch::Tensor> lod_select(
    torch::Tensor geom,
    torch::Tensor counts,
    torch::Tensor view,
    torch::Tensor K,
    int S,
    int M,
    int C,
    int width,
    int height,
    double tau,
    bool warm
) {
    DEVICE_GUARD(geom);
    CHECK_INPUT(geom);
    CHECK_INPUT(counts);
    CHECK_INPUT(view);
    CHECK_INPUT(K);
    TORCH_CHECK(
        geom.scalar_type() == torch::kFloat32 && geom.dim() == 2 &&
            geom.size(1) == 11,
        "geometry must be float32 [N,11]"
    );
    TORCH_CHECK(
        counts.scalar_type() == torch::kInt32 && counts.numel() == geom.size(0),
        "counts must be int32 [N]"
    );
    TORCH_CHECK(
        view.scalar_type() == torch::kFloat32 && view.numel() == 16 &&
            K.scalar_type() == torch::kFloat32 && K.numel() == 9,
        "invalid camera"
    );
    TORCH_CHECK(
        S > 0 && M > 0 && C > 0 && S % M == 0 && M % C == 0 &&
            geom.size(0) % (S + M + C) == 0,
        "invalid hierarchy sizes"
    );
    TORCH_CHECK(
        width > 0 && height > 0 && tau > 0 && std::isfinite(tau),
        "invalid projection settings"
    );
    auto parent =
        torch::zeros({geom.size(0)}, geom.options().dtype(torch::kInt64));
    auto weights = torch::ones({geom.size(0)}, geom.options());
    auto invk = torch::ones_like(weights);
    auto chosen =
        torch::zeros({geom.size(0)}, geom.options().dtype(torch::kInt32));
    int B = geom.size(0) / (S + M + C);
    if (B)
        select_kernel<<<
            (B * C + 255) / 256,
            256,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            geom.data_ptr<float>(),
            counts.data_ptr<int>(),
            view.data_ptr<float>(),
            K.data_ptr<float>(),
            B,
            S,
            M,
            C,
            width,
            height,
            tau,
            warm,
            parent.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            invk.data_ptr<float>(),
            chosen.data_ptr<int>()
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {chosen, parent, weights, invk};
}
__global__ void interpolate_kernel(
    const float *x,
    const int64_t *child,
    const int64_t *par,
    const float *w,
    float *out,
    int N,
    int D
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N * D)
        return;
    int n = id / D, d = id % D;
    out[id] = w[n] * x[child[n] * D + d] + (1.f - w[n]) * x[par[n] * D + d];
}
__global__ void interpolate_backward_kernel(
    const float *grad,
    const int64_t *child,
    const int64_t *par,
    const float *w,
    float *out,
    int N,
    int D
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N * D)
        return;
    int n = id / D, d = id % D;
    atomicAdd(out + child[n] * D + d, w[n] * grad[id]);
    atomicAdd(out + par[n] * D + d, (1.f - w[n]) * grad[id]);
}
torch::Tensor lod_interpolate(
    torch::Tensor x, torch::Tensor child, torch::Tensor parent, torch::Tensor w
) {
    DEVICE_GUARD(x);
    CHECK_INPUT(x);
    CHECK_INPUT(child);
    CHECK_INPUT(parent);
    CHECK_INPUT(w);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32 && x.dim() == 2 &&
            child.scalar_type() == torch::kInt64 &&
            parent.scalar_type() == torch::kInt64 &&
            w.scalar_type() == torch::kFloat32,
        "invalid interpolation types"
    );
    TORCH_CHECK(
        child.numel() == parent.numel() && child.numel() == w.numel(),
        "invalid interpolation lengths"
    );
    auto out = torch::empty({child.numel(), x.size(1)}, x.options());
    int n = out.numel();
    if (n)
        interpolate_kernel<<<
            (n + 255) / 256,
            256,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(),
            child.data_ptr<int64_t>(),
            parent.data_ptr<int64_t>(),
            w.data_ptr<float>(),
            out.data_ptr<float>(),
            child.numel(),
            x.size(1)
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
torch::Tensor lod_interpolate_backward(
    torch::Tensor grad,
    torch::Tensor child,
    torch::Tensor parent,
    torch::Tensor w,
    int rows
) {
    DEVICE_GUARD(grad);
    CHECK_INPUT(grad);
    auto out = torch::zeros({rows, grad.size(1)}, grad.options());
    int n = grad.numel();
    if (n)
        interpolate_backward_kernel<<<
            (n + 255) / 256,
            256,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            grad.data_ptr<float>(),
            child.data_ptr<int64_t>(),
            parent.data_ptr<int64_t>(),
            w.data_ptr<float>(),
            out.data_ptr<float>(),
            child.numel(),
            grad.size(1)
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
// Gather mapped pinned host rows without materializing a scene-sized GPU table.
__global__ void
gather_kernel(const float *host, const int64_t *ids, float *out, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * d)
        out[i] = host[ids[i / d] * d + i % d];
}
torch::Tensor lod_gather(torch::Tensor host, torch::Tensor ids) {
    DEVICE_GUARD(ids);
    CHECK_INPUT(ids);
    TORCH_CHECK(
        host.device().is_cpu() && host.is_pinned() && host.is_contiguous() &&
            host.scalar_type() == torch::kFloat32 && host.dim() == 2,
        "host must be pinned float32 matrix"
    );
    TORCH_CHECK(ids.scalar_type() == torch::kInt64, "ids must be int64");
    float *mapped = nullptr;
    C10_CUDA_CHECK(
        cudaHostGetDevicePointer((void **)&mapped, host.data_ptr<float>(), 0)
    );
    auto out = torch::empty(
        {ids.numel(), host.size(1)}, ids.options().dtype(torch::kFloat32)
    );
    int n = out.numel();
    if (n)
        gather_kernel<<<
            (n + 255) / 256,
            256,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            mapped,
            ids.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            ids.numel(),
            host.size(1)
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
// Match ATen's float32 exp/division and four-lane norm reduction. Keeping
// these rounding operations explicit prevents depth-sort changes from generic
// graph fusion (approximate exp/reciprocal or contracted arithmetic).
__global__ void activate_kernel(const float* raw, const bool* proxy, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* x = raw + i * 11;
    float* y = out + i * 11;
    for (int j = 0; j < 3; ++j) y[j] = x[j];
    y[3] = proxy[i] ? expf(fminf(20.f, fmaxf(-30.f, x[3]))) :
                     __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x[3])));
    for (int j = 4; j < 7; ++j) y[j] = expf(fminf(20.f, fmaxf(-30.f, x[j])));
    float first = __fadd_rn(__fmul_rn(x[7], x[7]), __fmul_rn(x[8], x[8]));
    float second = __fadd_rn(__fmul_rn(x[9], x[9]), __fmul_rn(x[10], x[10]));
    float norm = fmaxf(sqrtf(__fadd_rn(first, second)), 1e-12f);
    for (int j = 7; j < 11; ++j) y[j] = __fdiv_rn(x[j], norm);
}

torch::Tensor lod_activate(torch::Tensor raw, torch::Tensor proxy) {
    DEVICE_GUARD(raw);
    CHECK_INPUT(raw);
    CHECK_INPUT(proxy);
    TORCH_CHECK(raw.scalar_type() == torch::kFloat32 && raw.dim() == 2 && raw.size(1) == 11,
                "geometry must be float32 [N,11]");
    TORCH_CHECK(proxy.scalar_type() == torch::kBool && proxy.numel() == raw.size(0),
                "proxy mask must have one bool per Gaussian");
    auto out = torch::empty_like(raw);
    if (raw.size(0)) activate_kernel<<<(raw.size(0)+255)/256, 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(raw.data_ptr<float>(), proxy.data_ptr<bool>(),
                                          out.data_ptr<float>(), raw.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("activate", &lod_activate);
    m.def("select", &lod_select);
    m.def("interpolate", &lod_interpolate);
    m.def("interpolate_backward", &lod_interpolate_backward);
    m.def("gather", &lod_gather);
    m.def("raster_fwd", &lod_raster_fwd_tensor);
    m.def("raster_bwd", &lod_raster_bwd_tensor);
}
