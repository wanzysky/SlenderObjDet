import torch

from detectron2.layers import DeformConv

from slender_det.modeling.grid_generator import zero_center_grid, uniform_grid

torch.set_default_tensor_type('torch.cuda.FloatTensor')
# 3x3 conv with stride 1, padding 1
deform_conv = DeformConv(2, 1, 3, 1, 1)
torch.nn.init.constant_(deform_conv.weight, 1)

# 1, 2, 4, 4
grid = uniform_grid(4).unsqueeze(0).permute(0, 3, 1, 2)
grid = torch.stack([grid[:, 0], torch.zeros_like(grid[:, 0]) + 0.1], 1)

# 9, 2
offsets_1 = zero_center_grid(3).reshape(1, -1, 1, 1)
offsets_1 = offsets_1.repeat(1, 1, 4, 4)

offsets_2 = torch.zeros_like(offsets_1)

y_1 = deform_conv(grid, offsets_1)
y_2 = deform_conv(grid, offsets_2)

import ipdb
ipdb.set_trace()
