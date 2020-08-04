import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.layers import Conv2d, CNNBlockBase, ShapeSpec, get_norm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY, Backbone


class BasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=128, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.block = BasicBlock(out_channels, out_channels * 2, stride=2)
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = self.block(x)
        return x


def _make_layer(inp_dim, out_dim, modules):
    layers = [BasicBlock(inp_dim, out_dim)]
    layers += [BasicBlock(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)


def make_hg_layer(inp_dim, out_dim, modules):
    layers = [BasicBlock(inp_dim, out_dim, stride=2)]
    layers += [BasicBlock(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)


def _make_layer_revr(inp_dim, out_dim, modules):
    layers = [BasicBlock(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [BasicBlock(inp_dim, out_dim)]
    return nn.Sequential(*layers)


class HourglassBlock(CNNBlockBase):

    def __init__(
            self, n, dims, modules,
            make_up_layer=_make_layer,
            make_hg_layer=_make_layer,
            make_low_layer=_make_layer,
            make_hg_layer_revr=_make_layer_revr,
    ):
        super(HourglassBlock, self).__init__(dims[0], dims[0], 1)
        cur_mod = modules[0]
        next_mod = modules[1]

        cur_dim = dims[0]
        next_dim = dims[1]

        self.n = n
        self.up1 = make_up_layer(cur_dim, cur_dim, cur_mod)
        self.low1 = make_hg_layer(cur_dim, next_dim, cur_mod)
        self.low2 = HourglassBlock(
            n - 1, dims[1:], modules[1:],
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, cur_dim, cur_mod)

    def forward(self, x):
        up1 = self.up1(x)

        out = self.low1(x)
        out = self.low2(out)
        out = self.low3(out)

        out = F.interpolate(out, scale_factor=2)
        out += up1

        return out


class Hourglass(Backbone):

    def __init__(self, stem, blocks, convs, inters, convs_, inters_, out_features=None):
        super().__init__()
        self.stem = stem

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, block in enumerate(blocks):
            assert isinstance(block, CNNBlockBase), block
            name = "hourglass" + str(i + 2)
            stage = nn.ModuleDict(dict(hg=nn.Sequential(*[block, convs[i]])))

            if i < len(blocks) - 1:
                stage["inter"] = inters[i]
                stage["conv_"] = convs_[i]
                stage["inter_"] = inters_[i]

            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            self._out_feature_strides[name] = current_stride = current_stride * block.stride
            self._out_feature_channels[name] = block.out_channels

        self._size_divisibility = self.stem.stride * (2 ** 5)
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)

        if "stem" in self._out_features:
            outputs["stem"] = x

        for stage, name in self.stages_and_names:
            if hasattr(stage, "inter_"):
                inter = stage["inter_"](x)
            else:
                inter = None

            x = stage["hg"](x)
            if name in self._out_features:
                outputs[name] = x

            if inter is not None:
                x = inter + stage["conv_"](x)
                x = F.relu_(x)
                x = stage["inter"](x)

        return outputs

    @property
    def size_divisibility(self):
        return self._size_divisibility


@BACKBONE_REGISTRY.register()
def build_hourglass_backbone(cfg, input_shape):
    # need registration of new blocks/stems?
    norm = cfg.MODEL.HOURGLASS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.HOURGLASS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    out_features = cfg.MODEL.HOURGLASS.OUT_FEATURES
    stacks = cfg.MODEL.HOURGLASS.STACKS
    depth_block = cfg.MODEL.HOURGLASS.DEPTH_BLOCK
    channels_block = cfg.MODEL.HOURGLASS.CHANNELS_BLOCK
    num_conv_block = cfg.MODEL.HOURGLASS.NUM_CONV_BLOCK

    blocks = [
        HourglassBlock(
            depth_block, channels_block, num_conv_block,
            make_hg_layer=make_hg_layer
        ) for _ in range(stacks)
    ]
    in_channels = channels_block[0]
    convs = [
        Conv2d(in_channels, in_channels, kernel_size=3, bias=False, padding=1, activation=F.relu_)
        for _ in range(stacks)
    ]
    inters = [BasicBlock(in_channels, in_channels) for _ in range(stacks - 1)]
    convs_ = [Conv2d(in_channels, in_channels, kernel_size=1) for _ in range(stacks - 1)]
    inters_ = [Conv2d(in_channels, in_channels, kernel_size=1) for _ in range(stacks - 1)]

    return Hourglass(stem, blocks, convs, inters, convs_, inters_, out_features=out_features)
