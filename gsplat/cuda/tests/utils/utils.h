#pragma once
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi_image_write.h"
#include <torch/script.h>
#include <torch/torch.h>

struct Snapshot {
  // device (optional)
  torch::Device device = torch::kCUDA;

  // tensors
  torch::Tensor bg;
  torch::Tensor means3d;
  torch::Tensor colors_precomp;
  torch::Tensor opacities;
  torch::Tensor scales;
  torch::Tensor rotations;
  torch::Tensor cov3ds_precomp;
  torch::Tensor viewmatrix;
  torch::Tensor projmatrix;
  torch::Tensor sh;
  torch::Tensor campos;
  torch::Tensor color_grad;

  // scalars
  float scale_modifier = 0.f;
  float tanfovx = 0.f;
  float tanfovy = 0.f;

  int image_height = 0;
  int image_width = 0;
  int sh_degree = 0;

  bool prefiltered = false;
  bool debug = false;
};

static torch::Tensor getTensorAttr(const torch::jit::Module &module,
                                   const std::string &name) {
  TORCH_CHECK(module.hasattr(name), "Missing attribute: ", name);
  c10::IValue iv = module.attr(name);
  TORCH_CHECK(iv.isTensor(), "Attribute '", name, "' is not a Tensor (got ",
              iv.tagKind(), ")");
  return iv.toTensor();
}

static double getScalarDoubleAttr(const torch::jit::Module &module,
                                  const std::string &name) {
  torch::Tensor t = getTensorAttr(module, name);
  TORCH_CHECK(t.numel() == 1, "Attribute '", name,
              "' must be a scalar tensor (numel==1).");
  return t.to(torch::kCPU).item<double>();
}

static int64_t getScalarIntAttr(const torch::jit::Module &module,
                                const std::string &name) {
  torch::Tensor t = getTensorAttr(module, name);
  TORCH_CHECK(t.numel() == 1, "Attribute '", name,
              "' must be a scalar tensor (numel==1).");
  return t.to(torch::kCPU).item<int64_t>();
}

static bool getScalarBoolAttr(const torch::jit::Module &module,
                              const std::string &name) {
  torch::Tensor t = getTensorAttr(module, name);
  TORCH_CHECK(t.numel() == 1, "Attribute '", name,
              "' must be a scalar tensor (numel==1).");
  return t.to(torch::kCPU).item<bool>();
}

static Snapshot loadSnapshotFromModule(const torch::jit::Module &module,
                                       torch::Device device = torch::kCUDA) {
  Snapshot s;
  s.device = device;

  // -------------------------
  // Tensors
  // -------------------------
  s.bg = getTensorAttr(module, "bg").to(device);
  s.means3d = getTensorAttr(module, "means3d").to(device);
  s.colors_precomp = getTensorAttr(module, "colors_precomp").to(device);
  s.opacities = getTensorAttr(module, "opacities").to(device);
  s.scales = getTensorAttr(module, "scales").to(device);
  s.rotations = getTensorAttr(module, "rotations").to(device);
  s.cov3ds_precomp = getTensorAttr(module, "cov3ds_precomp").to(device);
  s.viewmatrix = getTensorAttr(module, "viewmatrix").to(device);
  s.projmatrix = getTensorAttr(module, "projmatrix").to(device);
  s.sh = getTensorAttr(module, "sh").to(device);
  s.campos = getTensorAttr(module, "campos").to(device);
  s.color_grad = getTensorAttr(module, "color_grad").to(device);

  // -------------------------
  // floats (stored as double in pt, cast down)
  // -------------------------
  s.scale_modifier =
      static_cast<float>(getScalarDoubleAttr(module, "scale_modifier"));
  s.tanfovx = static_cast<float>(getScalarDoubleAttr(module, "tanfovx"));
  s.tanfovy = static_cast<float>(getScalarDoubleAttr(module, "tanfovy"));

  // -------------------------
  // ints
  // -------------------------
  s.image_height = static_cast<int>(getScalarIntAttr(module, "image_height"));
  s.image_width = static_cast<int>(getScalarIntAttr(module, "image_width"));
  s.sh_degree = static_cast<int>(getScalarIntAttr(module, "sh_degree"));

  // -------------------------
  // bools
  // -------------------------
  s.prefiltered = getScalarBoolAttr(module, "prefiltered");
  s.debug = getScalarBoolAttr(module, "debug");

  return s;
}

static bool saveTensorPNG(const torch::Tensor &t, const std::string &path) {
  torch::Tensor x = t;

  // move to CPU
  x = x.detach().to(torch::kCPU);

  // convert to HWC
  if (x.dim() == 2) {
    // [H,W] -> [H,W,1]
    x = x.unsqueeze(-1);
  } else if (x.dim() == 3) {
    // CHW -> HWC
    if (x.size(0) == 1 || x.size(0) == 3 || x.size(0) == 4) {
      // assume CHW
      x = x.permute({1, 2, 0});
    }
  } else {
    throw std::runtime_error("Unsupported tensor dim");
  }

  if (x.scalar_type() != torch::kUInt8) {
    x = x.clamp(0, 1).mul(255).to(torch::kUInt8);
  }

  x = x.contiguous();
  const int H = (int)x.size(0);
  const int W = (int)x.size(1);
  const int C = (int)x.size(2);
  const int stride_in_bytes = W * C;

  return stbi_write_png(path.c_str(), W, H, C, x.data_ptr(), stride_in_bytes) !=
         0;
}
