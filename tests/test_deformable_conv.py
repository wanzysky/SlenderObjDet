import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import DeformConv

import init_paths
from slender_det.modeling.grid_generator import zero_center_grid, uniform_grid


def my_dconv(feature, offset, weights):
    pad = 1
    stride = 1

    # 1, H, W, 2
    base_grid = uniform_grid(feature.shape[2:]).unsqueeze(0)
    base_offset = zero_center_grid(3)
    result = torch.zeros((1, 1, 4, 4))
    pad_result = torch.zeros((1, 1, 4, 4))
    offset = offset.view(1, 3, 3, 2, 4, 4)

    pad_feature = F.pad(feature, (1, 1, 1, 1), mode='constant')
    for i in range(9):
        # loop over kernel points
        h_i = i // 3
        w_i = i % 3

        weight = weights[:, :, h_i, w_i].view(2, 1)
        # 1, 4, 4, 2
        sample_grid = base_grid + offset[:, h_i, w_i, :, :, :].permute(0, 2, 3, 1).flip(-1)
        sample_grid = sample_grid + base_offset[h_i, w_i].view(1, 1, 1, 2)
        sample_grid = sample_grid.reshape(-1, 2)
        valid_grid_idxs = (sample_grid > -1) & (sample_grid < 4)
        valid_grid_idxs = valid_grid_idxs[:, 0] & valid_grid_idxs[:, 1]
        sample_grid = sample_grid[valid_grid_idxs].long()
        idxs = sample_grid[:, 0] + sample_grid[:, 1] * 4

        result_i = torch.zeros(4, 4).view(-1)
        result_i[valid_grid_idxs] = (feature.view(2, -1)[:, idxs] * weight).sum(0)
        result = result.view(1, 1, 4, 4) + result_i.view(1, 1, 4, 4)
        """
        pad_result = (weight.view(1, 2, 1, 1) * pad_feature[:, :, h_i:h_i+4, w_i:w_i+4]).sum(1, keepdim=True) + pad_result
        if not torch.all(pad_result == result):
            import ipdb
            ipdb.set_trace()
        """
    return result


def my_conv(feature, weights):
    pad = 1
    stride = 1

    feature = F.pad(feature, (1, 1, 1, 1), mode='constant')
    # H, W, 2
    sample_grid = uniform_grid(feature.shape[2:])
    result = torch.zeros((1, 1, 4, 4))
    for i in range(9):
        h_i = i // 3
        w_i = i % 3

        weight = weights[:, :, h_i:h_i + 1, w_i:w_i + 1]  # 1, 2, 1, 1
        result = (weight * feature[:, :, h_i:h_i + 4, w_i:w_i + 4]).sum(1, keepdim=True) + result
    return result


torch.set_default_tensor_type('torch.cuda.FloatTensor')
# 3x3 conv with stride 1, padding 1
conv = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
deform_conv = DeformConv(2, 1, 3, 1, 1, bias=False)
deform_conv.weight.data = torch.arange(9).float().reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
conv.weight.data = torch.arange(9).float().reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)

# 1, 2, 4, 4
grid = uniform_grid(4).unsqueeze(0).permute(0, 3, 1, 2)
grid = torch.stack([grid[:, 0], torch.zeros_like(grid[:, 0]) + 0.1], 1)

# 9, 2
offsets_1 = zero_center_grid(3).reshape(1, -1, 1, 1)
offsets_1 = offsets_1.repeat(1, 1, 4, 4)
offsets_2 = torch.zeros_like(offsets_1)

y_1 = deform_conv(grid, offsets_1)
y_2 = deform_conv(grid, offsets_2)
assert torch.all(torch.abs(y_2 - my_conv(grid, conv.weight.data)) < 1e-5)
assert torch.all(torch.abs(y_2 - my_dconv(grid, offsets_2, conv.weight.data)) < 1e-5)
assert torch.all(torch.abs(y_1 - my_dconv(grid, offsets_1, conv.weight.data)) < 1e-5)
