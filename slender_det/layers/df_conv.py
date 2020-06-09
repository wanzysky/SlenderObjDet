from torch import nn

from detectron2.layers import Conv2d, ModulatedDeformConv, DeformConv


class DFConv2d(nn.Module):
    """Deformable convolution layer"""

    def __init__(
            self,
            in_channels,
            out_channels,
            with_modulated_dcn=True,
            kernel_size=3,
            stride=1,
            groups=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
            bias=False
    ):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert len(kernel_size) == 2
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            offset_base_channels = kernel_size * kernel_size

        if with_modulated_dcn:
            offset_channels = offset_base_channels * 3  # default: 27
            conv_block = ModulatedDeformConv
        else:
            offset_channels = offset_base_channels * 2  # default: 18
            conv_block = DeformConv

        self.offset = Conv2d(
            in_channels,
            deformable_groups * offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation
        )
        for l in [self.offset, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0.)

        self.conv = conv_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias
        )
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_base_channels = offset_base_channels

    def forward(self, x):
        assert x.numel() > 0, "only non-empty tensors are supported"
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset = self.offset(x)
                x = self.conv(x, offset)
            else:
                offset_mask = self.offset(x)
                split_point = self.offset_base_channels * 2
                offset = offset_mask[:, :split_point, :, :]
                mask = offset_mask[:, split_point:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            return x
