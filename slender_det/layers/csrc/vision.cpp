#include <torch/extension.h>
#include "corner_pool/corner_pool.h"
#include "border_align/BorderAlign.h"

namespace slender_det {

#if defined(WITH_CUDA) || defined(WITH_HIP)
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  std::ostringstream oss;

#if defined(WITH_CUDA)
  oss << "CUDA ";
#else
  oss << "HIP ";
#endif

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
#else // neither CUDA nor HIP
  return std::string("not available");
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");

  m.def("bottom_pool_forward", &bottom_pool_forward, "Bottom Pool Forward");
  m.def("bottom_pool_backward", &bottom_pool_backward, "Bottom Pool Backward");

  m.def("left_pool_forward", &left_pool_forward, "Left Pool Forward");
  m.def("left_pool_backward", &left_pool_backward, "Left Pool Backward");
  m.def("right_pool_forward", &right_pool_forward, "Right Pool Forward");
  m.def("right_pool_backward", &right_pool_backward, "Right Pool Backward");
  m.def("top_pool_forward", &top_pool_forward, "Top Pool Forward");
  m.def("top_pool_backward", &top_pool_backward, "Top Pool Backward");

  m.def("border_align_forward", &BorderAlign_Forward, "BorderAlign_Forward");
  m.def("border_align_backward", &BorderAlign_Backward, "BorderAlign_Backward");
}

} // namespace slender_det
