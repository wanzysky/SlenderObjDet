#pragma once
#include <torch/types.h>

namespace slender_det {

at::Tensor bottom_pool_forward(at::Tensor input);
at::Tensor bottom_pool_backward(at::Tensor input, at::Tensor grad_output);

at::Tensor left_pool_forward(at::Tensor input);
at::Tensor left_pool_backward(at::Tensor input, at::Tensor grad_output);

at::Tensor right_pool_forward(at::Tensor input);
at::Tensor right_pool_backward(at::Tensor input, at::Tensor grad_output);

at::Tensor top_pool_forward(at::Tensor input);
at::Tensor top_pool_backward(at::Tensor input, at::Tensor grad_output);

} // namespace slender_det