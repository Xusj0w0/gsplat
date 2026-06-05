#pragma once

#include "Common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace gsplat {

inline __device__ bool sym3_inverse_reg(
    const mat3 &C,
    mat3 &Cinv,
    float rel_jitter = 1e-6f,
    float abs_jitter = 1e-8f
) {
    mat3 Cs = 0.5f * (C + glm::transpose(C));

    float tr = Cs[0][0] + Cs[1][1] + Cs[2][2];
    float lam = fmaxf(abs_jitter, rel_jitter * fmaxf(tr, abs_jitter));

    mat3 I3 = mat3(1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f);
    mat3 Creg = Cs + lam * I3;
    Cinv = glm::inverse(Creg);
    float s = Cinv[0][0] + Cinv[1][1] + Cinv[2][2];
    return isfinite(s);
}

inline __device__ void persp_proj_radegs(
    // inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2 &cov2d,
    vec2 &mean2d,
    vec3 &ray_plane,
    vec3 &normal
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.f / z;
    float u = min(lim_x_pos, max(-lim_x_neg, x * rz));
    float v = min(lim_y_pos, max(-lim_y_neg, y * rz));
    float tx = z * u;
    float ty = z * v;

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * u * rz,
        -fy * v * rz // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2(fx * x * rz + cx, fy * y * rz + cy);

    ray_plane = vec3(0.f, 0.f, 0.f);
    normal = vec3(0.f, 0.f, 1.f);

    mat3 cov3d_inv;
    if (!sym3_inverse_reg(cov3d, cov3d_inv, 1e-6f, 1e-8f)) {
        return;
    }

    vec3 uvh = vec3(u, v, 1.f);
    vec3 Cinv_uvh = cov3d_inv * uvh;
    const float Cinv_uvh_scale =
        max(abs(Cinv_uvh.x), max(abs(Cinv_uvh.y), abs(Cinv_uvh.z)));
    if (Cinv_uvh_scale < 1e-12f || !isfinite(Cinv_uvh_scale)) {
        return;
    }
    vec3 Cinv_uvh_n = Cinv_uvh / Cinv_uvh_scale;
    const float denom = max(glm::dot(Cinv_uvh_n, uvh), 1e-7f);

    const float uu = u * u;
    const float vv = v * v;
    const float uv = u * v;
    const float l2 = uu + vv + 1.f;
    const float t = norm3df(tx, ty, z);
    const float factor = t / l2;

    mat3 nJ_T = mat3(rz, 0.f, -u * rz, 0.f, rz, -v * rz, tx / t, ty / t, z / t);
    mat3x2 nJ_inv_T = mat3x2(
        vv + 1.f, -uv, -uv, uu + 1.f, -u, -v
    ); // l^3*t^{-1} * J^{-1}_{:, :2}^T

    // Compute ray_plane & normal
    vec2 plane = nJ_inv_T * Cinv_uvh_n / denom;
    // plane: (l^3*t^{-1} * nJinvT) * |Cinv*uvh| / (uvh * |Cinv*uvh|)
    // ray_plane: l * nJinvT * |Cinv*uvh| / (uvh * |Cinv*uvh|)
    //          = t*l^{-2} * plane

    vec3 normal_r = vec3(plane.x * factor, plane.y * factor, 1.f);
    vec3 normal_c = nJ_T * (-normal_r);

    ray_plane = vec3(normal_r.x / fx, normal_r.y / fy, t);
    const float normal_c_length = glm::length(normal_c);
    if (normal_c_length > 1e-10f && isfinite(normal_c_length)) {
        const float inv_len = 1.f / normal_c_length;
        normal = normal_c * inv_len;
    }
}

inline __device__ void persp_proj_radegs_vjp(
    // fwd inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2 v_cov2d,
    const vec2 v_mean2d,
    const vec3 v_ray_plane,
    const vec3 v_normal,
    // grad inputs
    vec3 &v_mean3d,
    mat3 &v_cov3d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float u = min(lim_x_pos, max(-lim_x_neg, x * rz));
    float v = min(lim_y_pos, max(-lim_y_neg, y * rz));
    float tx = z * u;
    float ty = z * v;

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * u * rz,
        -fy * v * rz // 3rd column
    );

    vec3 v_uvh = vec3(0.f);
    float v_u = 0.f, v_v = 0.f, v_t = 0.f;
    mat3 v_nJ_T = mat3(0.f);
    float t = 0.f;
    mat3 cov3d_inv;
    const bool inv_ok = sym3_inverse_reg(cov3d, cov3d_inv, 1e-6f, 1e-8f);

    vec3 uvh = vec3(u, v, 1.f);
    vec3 Cinv_uvh = vec3(0.f);
    float Cinv_uvh_scale = 0.f;
    if (inv_ok) {
        Cinv_uvh = cov3d_inv * uvh;
        Cinv_uvh_scale =
            max(abs(Cinv_uvh.x), max(abs(Cinv_uvh.y), abs(Cinv_uvh.z)));
    }
    if (inv_ok && !(Cinv_uvh_scale < 1e-12f || !isfinite(Cinv_uvh_scale))) {
        const float uu = u * u;
        const float vv = v * v;
        const float uv = u * v;
        const float l2 = uu + vv + 1.f;
        t = norm3df(tx, ty, z);
        const float factor = t / l2;
        mat3 nJ_T =
            mat3(rz, 0.f, -u * rz, 0.f, rz, -v * rz, tx / t, ty / t, z / t);
        mat3x2 nJ_inv_T = mat3x2(vv + 1.f, -uv, -uv, uu + 1.f, -u, -v);
        const vec3 Cinv_uvh_n = Cinv_uvh / Cinv_uvh_scale;
        const float denom = max(glm::dot(Cinv_uvh_n, uvh), 1e-7f);
        const vec3 Cinv_uvh_u = Cinv_uvh / denom;
        const vec2 plane = nJ_inv_T * Cinv_uvh_u;

        const vec3 normal_r = vec3(plane.x * factor, plane.y * factor, 1.f);
        const vec3 normal_c = nJ_T * (-normal_r);
        const vec3 ray_plane = vec3(normal_r.x / fx, normal_r.y / fy, t);

        vec3 v_normal_r = vec3(0.f);
        vec3 v_normal_c = vec3(0.f);

        // +--------------+
        // |    normal    |
        // +--------------+
        const float normal_c_length = glm::length(normal_c);
        if (normal_c_length > 1e-10f && isfinite(normal_c_length)) {
            const float inv_len = 1.f / normal_c_length;
            const vec3 normal = normal_c * inv_len;

            // n = n_c / norm(n_c) = (n_c^T n_c)^{-0.5} * n_c
            // v_normal_c = v_normal * [norm(n_c)^{-1} * I - norm(n_c)^-3 * n_c n_c^T]
            const vec3 v_normal_invlen = v_normal * inv_len;
            v_normal_c =
                v_normal_invlen - normal * glm::dot(v_normal_invlen, normal);

            // n_c = -nJ_T * n_r
            // v_nJ_T = -v_normal_c * n_r^T
            // v_normal_r = -nJ_T^T * v_normal_c
            v_nJ_T = -glm::outerProduct(v_normal_c, normal_r);
            v_normal_r = -v_normal_c * nJ_T;
        }
        // +-----------------+
        // |    ray_plane    |
        // +-----------------+
        // =========== FORWARD ==============
        // n_r = [t*l^-2 * p; 1]
        // ray_plane[:2] = t*l^-2 * diag(1/fx, 1/fy) * plane
        // ray_plane[2] = t
        // =========== BACKPROP ==============
        // v_plane = t*l^-2 * (diag(fx, fy)^-1 * v_ray_plane[:2] + v_normal_r_{:2})
        // v_t = l^-2 * (diag(fx, fy)^-1 * v_ray_plane[:2] + v_normal_r_{:2})^T * plane + v_ray_plane[2]
        // v_l2 = -t*l^-4 * (diag(fx,fy)^-1 * v_ray_plane[:2] + v_normal_r[:2])^T * plane
        vec2 v_plane = factor * vec2(
                                    v_normal_r.x + v_ray_plane.x / fx,
                                    v_normal_r.y + v_ray_plane.y / fy
                                );
        float dot = glm::dot(v_plane, plane);
        v_t = dot / t + v_ray_plane.z; // v_t = dot / factor / l2
        float v_l2 = -dot / l2; // v_l2 = -dot / factor / l2 / l2 * t

        // p = J^-T * Cinv * uvh / (uvh^T * Cinv * uvh)
        // Let s = uvh^T * Cinv * uvh
        // v_J^-T = v_plane * uvh^T * Cinv / s = outerProduct<v_plane, Cinv * uvh / s>
        // v_C = Cinv * (uvh * uvh^T * Cinv - s * I) * J^-1 * v_plane * uvh^T * Cinv / s^2
        //     = outerProduct<Cinv / s * (uvh * uvh^T * Cinv / s - I) * J^-1 * v_plane, Cinv_uvh>
        //     = outerProduct<Cinv_ * (uvh * uvh^T * Cinv_ * J^-1 * v_plane - J^-1 * v_plane), Cinv_uvh>
        //     = outerProduct<Cinv_ * (uvh * dot<v_plane, J^-T * Cinv * uvh / s> - J^-1 * v_plane), Cinv_uvh>
        //     = outerProduct<Cinv_ * (uvh * dot<v_plane, plane> - J^-1 * v_plane), Cinv_uvh>
        // v_uvh = s^-1 * Cinv * J^-1 * v_plane - 2 * s^-2 * Cinv * uvh * dot<v_plane, J^-T * Cinv * uvh>
        //       = Cinv_ * (J^-1 * v_plane - 2 * uvh * dot<v_plane, J^-T * Cinv * uvh / s>)
        //       = Cinv_ * (J^-1 * v_plane - 2 * uvh * dot<v_plane, plane>)
        const float scale = max(glm::dot(Cinv_uvh, uvh), 1e-7f);
        mat3 cov3d_inv_ = cov3d_inv / scale;
        mat3x2 v_nJ_inv_T = glm::outerProduct(v_plane, Cinv_uvh_u);
        vec3 nJ_inv_v_plane = glm::transpose(nJ_inv_T) * v_plane;
        v_cov3d += glm::outerProduct(
            Cinv_uvh, cov3d_inv_ * (dot * uvh - nJ_inv_v_plane)
        );
        v_uvh += cov3d_inv_ * (nJ_inv_v_plane - 2 * dot * uvh);

        // nJ_inv_T = [v^2+1, -uv, -u; -uv, u^2+1, -v]
        // uvh = (u, v, 1)
        // l^2 = u^2 + v^2 + 1
        // v_u = -v * [v_nJ_inv_T[0,1] + v_nJ_inv_T[1,0]] - v_nJ_inv_T[0,2] + 2u * v_nJ_inv_T[1,1]
        //     + v_uvh[0] + 2u * v_l2 + z * u * l^-1 * v_t
        // v_v = -u * [v_nJ_inv_T[1,0] + v_nJ_inv_T[0,1]] - v_nJ_inv_T[1,2] + 2v * v_nJ_inv_T[0,0]
        //     + v_uvh[1] + 2v * v_l2 + z * v * l^-1 * v_t
        const float diag_sum = v_nJ_inv_T[1][0] + v_nJ_inv_T[0][1];
        v_u += -v * diag_sum + 2.f * u * (v_l2 + v_nJ_inv_T[1][1]) -
               v_nJ_inv_T[2][0] + v_uvh.x;
        v_v += -u * diag_sum + 2.f * v * (v_l2 + v_nJ_inv_T[0][0]) -
               v_nJ_inv_T[2][1] + v_uvh.y;
    }

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += vec3(
        fx * rz * v_mean2d[0],
        fy * rz * v_mean2d[1],
        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    float rz3 = rz2 * rz;
    mat3x2 v_J = v_cov2d * J * glm::transpose(cov3d) +
                 glm::transpose(v_cov2d) * J * cov3d;

    const float t3 = t * t * t;

    // J = [fx * z^-1, 0, -fx * tx * z^-2; 0, fy * z^-1, -fy * ty * z^-2]
    // nJ_T = [1/z, 0, tx/t; 0, 1/z, ty/t; -tx * z^-2, -ty * z^-2, z/t]

    // v_tx = v_J[0,2]*(-fx*z^-2)
    //      + v_nJ_T[0,2] * d(tx/t)/d(tx)       : 1/t + tx * d(1/t)/d(tx) = 1/t + tx * (-tx / t^3)
    //      + v_nJ_T[1,2] * ty * d(1/t)/d(tx)
    //      + v_nJ_T[2,0] * (-z^-2)
    //      + v_nJ_T[2,2] * z * d(1/t)/d(tx)
    //      + v_t * dt/d(tx)
    float v_tx = -v_J[2][0] * fx * rz2 +
                 v_nJ_T[2][0] * (1.f / t - tx * tx / t3) -
                 (v_nJ_T[2][1] * ty + v_nJ_T[2][2] * z) * tx / t3 -
                 v_nJ_T[0][2] * rz2 + v_t * tx / t;
    // v_ty = v_J[1,2]*(-fy*z^-2)
    //      + v_nJ_T[1,2] * d(ty/t)/d(ty)
    //      + v_nJ_T[0,2] * tx * d(1/t)/d(ty)
    //      + v_nJ_T[2,1] * (-z^-2)
    //      + v_nJ_T[2,2] * z * d(1/t)/d(ty)
    //      + v_t * dt/d(ty)
    float v_ty = -v_J[2][1] * fy * rz2 +
                 v_nJ_T[2][1] * (1.f / t - ty * ty / t3) -
                 (v_nJ_T[2][0] * tx + v_nJ_T[2][2] * z) * ty / t3 -
                 v_nJ_T[1][2] * rz2 + v_t * ty / t;

    // v_z = v_J[0,0]*fx*(-z^-2) + v_J[1,1]*fy*(-z^2) - v_J[0,2]*fx*tx*(-2z^-3) - v_J[1,2]*fy*ty*(-2z^-3)
    //     + v_nJ_T[0,0]*(-z^-2) + v_nJ_T[0,2]*tx*d(1/t)/dz + v_nJ_T[1,1]*(-z^2) + v_nJ_T[1,2]*ty*d(1/t)/dz
    //     + v_nJ_T[2,0]*(-tx)*(-2z^-3) + v_nJ_T[2,1]*(-ty)*(-2z^-3) + v_nJ_T[2,2]*(1/t - z * z / t^3)
    //     + v_t * dt/dz + v_u * du/dz + v_v * dv/dz
    float v_z = -v_J[0][0] * fx * rz2 - v_J[1][1] * fy * rz2 +
                2.f * v_J[2][0] * fx * tx * rz3 +
                2.f * v_J[2][1] * fy * ty * rz3 - v_nJ_T[0][0] * rz2 -
                v_nJ_T[1][1] * rz2 -
                (v_nJ_T[2][0] * tx + v_nJ_T[2][1] * ty) * z / t3 +
                2.f * v_nJ_T[0][2] * tx * rz3 + 2.f * v_nJ_T[1][2] * ty * rz3 +
                v_nJ_T[2][2] * (1.f / t - z * z / t3) + v_t * z / t;
    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        // without clipping, (x, y, z) -> (u, v) -> ...
        // v_x includes contributions from v_u: v_u * du/dx = v_u * d(x/z)/dx
        // v_z includes contributions from v_u: v_u * du/dz = v_u * d(x/z)/dz
        v_mean3d.x += v_tx + v_u * rz;
        v_mean3d.z -= v_u * tx * rz2;
    } else {
        // with clipping, (z,) + (u, v) -> ..., u is independent of z
        // v_z does not include contributions from v_u
        v_mean3d.z += v_tx * u;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += v_ty + v_v * rz;
        v_mean3d.z -= v_v * ty * rz2;
    } else {
        v_mean3d.z += v_ty * v;
    }
    v_mean3d.z += v_z;
}
} // namespace gsplat
