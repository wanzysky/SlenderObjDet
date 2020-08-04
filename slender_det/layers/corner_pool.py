import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

from detectron2.layers import Conv2d, get_norm
from slender_det import _C


class TopPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = _C.top_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = _C.top_pool_backward(input, grad_output)
        return output


class BottomPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = _C.bottom_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = _C.bottom_pool_backward(input, grad_output)
        return output


class LeftPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = _C.left_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = _C.left_pool_backward(input, grad_output)
        return output


class RightPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = _C.right_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        output = _C.right_pool_backward(input, grad_output)
        return output


class CornerPool(nn.Module):
    """Corner Pooling.
    Corner Pooling is a new type of pooling layer that helps a
    convolutional network better localize corners of bounding boxes.
    Please refer to https://arxiv.org/abs/1808.01244 for more details.
    Code is modified from https://github.com/princeton-vl/CornerNet-Lite.
    Args:
        mode(str): Pooling orientation for the pooling layer
            - 'bottom': Bottom Pooling
            - 'left': Left Pooling
            - 'right': Right Pooling
            - 'top': Top Pooling
    Returns:
        Feature map after pooling.
    """

    pool_functions = {
        'bottom': BottomPoolFunction,
        'left': LeftPoolFunction,
        'right': RightPoolFunction,
        'top': TopPoolFunction,
    }

    cummax_dim_flip = {
        'bottom': (2, False),
        'left': (3, True),
        'right': (3, False),
        'top': (2, True),
    }

    def __init__(self, mode):
        super(CornerPool, self).__init__()
        assert mode in self.pool_functions
        self.mode = mode
        self.corner_pool = self.pool_functions[mode]

    def forward(self, x):
        if torch.__version__ >= '1.5.0':
            dim, flip = self.cummax_dim_flip[self.mode]
            if flip:
                x = x.flip(dim)
            pool_tensor, _ = torch.cummax(x, dim=dim)
            if flip:
                pool_tensor = pool_tensor.flip(dim)
            return pool_tensor
        else:
            return self.corner_pool.apply(x)


class CornerPoolPack(nn.Module):
    def __init__(
            self, dim, pool1, pool2, first_kernel_size=3, kernel_size=3, corner_dim=128, norm="BN"
    ):
        super(CornerPoolPack, self).__init__()
        self.p1_conv1 = Conv2d(
            dim,
            corner_dim,
            first_kernel_size,
            padding=(first_kernel_size - 1) // 2,
            bias=False,
            norm=get_norm(norm, corner_dim),
            activation=F.relu_,
        )
        self.p2_conv1 = Conv2d(
            dim,
            corner_dim,
            first_kernel_size,
            padding=(first_kernel_size - 1) // 2,
            bias=False,
            norm=get_norm(norm, corner_dim),
            activation=F.relu_,
        )

        # self.p_conv1 = nn.Conv2d(corner_dim, dim, 3, padding=1, bias=False)
        # self.p_gn1 = nn.GroupNorm(num_groups=32, num_channels=dim)
        self.p_conv1 = Conv2d(corner_dim, dim, 3, padding=1, bias=False, norm=get_norm(norm, dim))
        self.conv1 = Conv2d(dim, dim, 1, bias=False, norm=get_norm(norm, dim))

        self.conv2 = Conv2d(
            dim,
            dim,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
            norm=get_norm(norm, dim),
            activation=F.relu_,
        )

        self.pool1 = pool1
        self.pool2 = pool2

    def forward(self, x):
        conv1 = F.relu_(self.conv1(x))

        # pool 1 and pool 2
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)
        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)

        out = F.relu_(p_conv1 + conv1)
        out = self.conv2(out)

        return out


class TLPool(CornerPoolPack):
    def __init__(self, dim, first_kernel_size=3, kernel_size=3, corner_dim=128):
        super(TLPool, self).__init__(
            dim, CornerPool('top'), CornerPool('left'),
            first_kernel_size, kernel_size, corner_dim
        )


class BRPool(CornerPoolPack):
    def __init__(self, dim, first_kernel_size=3, kernel_size=3, corner_dim=128):
        super(BRPool, self).__init__(
            dim, CornerPool('bottom'), CornerPool('right'),
            first_kernel_size, kernel_size, corner_dim
        )
