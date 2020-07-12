from functools import partial

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import fvcore.nn.weight_init as weight_init
from detectron2.layers import (
    cat,
    Conv2d as VanillaConv2d,
    get_norm
)
from slender_det.layers import Conv1d as VanillaConv1d


def Conv2d(in_channels, out_channels, *args, **kwargs):
    norm_desc = kwargs.pop("norm", "")
    norm_desc = "SyncBN"
    norm = get_norm(norm_desc, out_channels)

    return VanillaConv2d(
        in_channels, out_channels, *args, norm=norm, **kwargs)


def Conv1d(in_channels, out_channels, *args, **kwargs):
    norm_desc = kwargs.pop("norm", "")
    norm_desc = "BN1d"
    norm = get_norm(norm_desc, out_channels)

    return VanillaConv1d(
        in_channels, out_channels, *args, norm=norm, **kwargs)

class VisualTransformer(nn.Module):
    """
    Implementation of Visual Transformers: (https://arxiv.org/abs/2006.03677).
    """

    def __init__(
        self,
        in_channels,
        inner_channels,
        length,
        dynamic,
        size,
        downasamples
    ):
        pos_encoding = DownsampleEmbedding(
            in_channels=length, size=size, downasamples=downasamples)
        self.tokenizer = Tokenizer(
            in_channels, inner_channels, length,
            dynamic=dynamic, pos_encoding=pos_encoding)
        self.transformer = Transformer(inner_channels)
        self.projecter = Projector(inner_channels, in_channels)

    def forward(self, x, prev_tokens=None):
        tokens = self.tokenizer(x, prev_tokens)
        tokens = self.transformer(tokens)
        x = self.projecter(x, tokens)
        return x, tokens


class Tokenizer(nn.Module):
    """
    Static and dynamic tokenization in visual transformers.
    """

    def __init__(
            self, in_channels, inner_channels, length,
            pos_encoding,
            groups=16, heads=16,
            dynamic=False, bias=False):
        """
        Args:
            in_channels (int): C in the paper, indicating the number of
                channels of input feature.
            inner_channels (int): C_T in the paper, indicating the number
                of channels of W_V.
            length (int): L in the paper, the size of tokens.
            groups (int): groups of convolution layers. It is not specifyed
                in the paper but found in the pseudo-code, where group-convs
                are applied in part of the convolutions.
            heads (int): number of heads.
            dynamic (bool): dynamic(True) or static(False) tokenization.
            bias (bool): whether to use bias in convolution layers. Acoording
                to the equations in the paper, convolutions should be rid of
                bias. However, it's not clear in the pseudo-code.
        """
        super(Tokenizer, self).__init__()
        self.dynamic = dynamic
        self.pos_encoding = pos_encoding
        self.heads = heads
        self.inner_channels = inner_channels

        if not dynamic:
            # use static weights to compute token coefficients.
            self.conv_token_coef = Conv2d(
                in_channels, length, bias=bias,
                kernel_size=1, stride=1, padding=0, norm="GN")
        else:
            # use previous tokens to compute a query weight,
            # which is then used to compute token coefficients.
            self.conv_query = Conv1d(
                inner_channels, in_channels, bias=bias,
                kernel_size=1, stride=1, padding=0, norm="GN")
            self.conv_key = Conv2d(
                in_channels, in_channels, groups=groups, bias=bias,
                kernel_size=1, padding=0, stride=1, norm="GN")

        # CAUTION: this implementation is according to the pseudo-code
        # in the appendix of the paper. It is slightly different with
        # the paper: V should have been in C_T channels.
        self.conv_value = Conv2d(
            in_channels, in_channels, groups=groups,
            kernel_size=1, stride=1, padding=0, bias=bias, norm="GN")
        self.conv_token = Conv1d(
            in_channels + self.pos_encoding.pos_dim, inner_channels,
            kernel_size=1, stride=1, padding=0, norm="GN")

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, tokens=None):
        """
        Args:
            x (Tensor): N, C, H, W.
            token (Tensor): N, C_T, L
        """
        if self.dynamic:
            assert tokens
            L = tokens.size(2)
            T_a, T_b = tokens[:, :, :L//2], tokens[:, :, L//2:]
            query = self.conv_query(T_a)
            N, C, L_a = query.shape
            query = query.view(N, self.heads, C//self.heads, L_a)
            N, C, H, W = x.shape
            assert C % self.heads == 0
            # H, h, C//h, HW
            key = self.conv_key(x).view(N, self.heads, C//self.heads, -1)
            # (N, h, HW, C//h) x (N, h, C//h, L_a) --> (N, h, HW, L_a)
            token_coef = torch.matmul(key.permute(0, 1, 3, 2), query)
            token_coef = token_coef / np.sqrt(C / self.heads)
        else:
            token_coef = self.conv_token_coef(x)
            N, L, H, W = token_coef.shape
            # N, 1, HW, L
            token_coef = token_coef.view(
                N, 1, L, -1).permute(0, 1, 3, 2)
            token_coef = token_coef / np.sqrt(x.size(1))

        token_coef = F.softmax(token_coef, dim=2)
        N, C, H, W = x.shape

        # N, h, C//h, HW
        value = self.conv_value(x).view(
            N, self.heads, C//self.heads, -1)
        # N, h, C//h, L -> N, C, L
        tokens = torch.matmul(value, token_coef).view(N, C, -1)

        pos_encoding = self.pos_encoding(token_coef, (H, W))
        tokens = cat((tokens, pos_encoding), dim=1)

        if self.dynamic:
            tokens = torch.cat((T_b, self.conv_token(tokens)), dim=2)
        else:
            tokens = self.conv_token(tokens)
        return tokens


class DownsampleEmbedding(nn.Module):
    """
    Position embedding using downasample and weight sum.
    """
    def __init__(
            self, in_channels, size, downasamples):
        super(DownsampleEmbedding, self).__init__()
        self.size = size
        self.downasample_conv = nn.Sequential(
            *[Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=1, stride=2, norm="GN"
            ) for _ in range(downasamples)])
        self.pos_dim = size ** 2 // (4 ** downasamples)
        self.pos_conv = Conv1d(
            self.pos_dim, self.pos_dim,
            kernel_size=1, stride=1, padding=0, norm="GN")

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, token_coef, input_size):
        H, W = input_size
        # h == 1 for static
        N, h, HW, L = token_coef.shape
        # N, L, h, HW -> NL, h, H, W
        token_coef = token_coef.permute(0, 3, 1, 2).reshape(N*L, h, H, W)
        token_coef = F.interpolate(
            token_coef, size=(self.size, self.size))
        token_coef = self.downasample_conv(token_coef)

        token_coef = token_coef.view(N, L, -1).permute(0, 2, 1)
        return self.pos_conv(token_coef)


class Transformer(nn.Module):
    def __init__(self, CT, heads=16, kqv_groups=8):
        super(Transformer, self).__init__()
        self.k_conv = Conv1d(
            CT, CT//2, groups=kqv_groups,
            kernel_size=1, stride=1, padding=0, norm="GN")
        self.q_conv = Conv1d(
            CT, CT//2, groups=kqv_groups,
            kernel_size=1, stride=1, padding=0, norm="GN")
        self.v_conv = Conv1d(
            CT, CT, groups=kqv_groups,
            kernel_size=1, stride=1, padding=0, norm="GN")
        self.ff_conv = Conv1d(
            CT, CT,
            kernel_size=1, stride=1, padding=0, norm="GN")
        
        self.heads = 16
        self.CT = CT

    def forward(self, tokens):
        N = tokens.shape[0]
        # N, h, CT//2//h, L
        k = self.k_conv(tokens).view(
            N, self.heads, self.CT//2//self.heads, -1)
        # N, h, CT//2//h, L
        q = self.q_conv(tokens).view(
            N, self.heads, self.CT//2//self.heads, -1)
        # N, h, CT//2//h, L
        v = self.v_conv(tokens).view(
            N, self.heads, self.CT//self.heads, -1)

        # N, h, L, L
        kq = torch.matmul(k.permute(0, 1, 3, 2), q)
        kq = F.softmax(kq / np.sqrt(kq.size(2)), dim=2)

        # N, CT, L
        kqv = torch.matmul(v, kq).view(N, self.CT, -1)
        tokens = tokens + kqv
        tokens = tokens + self.ff_conv(tokens)
        return tokens


class Projector(nn.Module):
    def __init__(self,
                 token_channels,
                 feature_channels,
                 out_channels,
                 heads=16, groups=16):
        super(Projector, self).__init__()
        self.proj_value_conv = Conv1d(
            token_channels, out_channels,
            kernel_size=1, padding=0, stride=1, norm="GN")
        self.proj_key_conv = Conv1d(
            token_channels, feature_channels,
            kernel_size=1, stride=1, padding=0, norm="GN")
        self.proj_query_conv = Conv2d(
            feature_channels, feature_channels, groups=groups,
            kernel_size=1, padding=0, stride=1, norm="GN")
        self.lateral_conv = None
        if not out_channels == feature_channels:
            self.lateral_conv = Conv2d(
                feature_channels, out_channels, groups=groups,
                kernel_size=1, padding=0, stride=1, norm="GN")
        self.heads = heads

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, feature, tokens):
        N, _, L = tokens.shape

        # N, h, C//h, L
        proj_v = self.proj_value_conv(
            tokens).view(N, self.heads, -1, L)
        # N, h, C_T//h, L
        proj_k  = self.proj_key_conv(
            tokens).view(N, self.heads, -1, L)
        # N, CT, H, W
        proj_q = self.proj_query_conv(feature)

        N, C, H, W = proj_q.shape
        # N, h, C_T//h, HW --> N, h, HW, C_T//h
        proj_q = proj_q.view(
            N, self.heads, C//self.heads, H*W).permute(0, 1, 3, 2)
        # N, h, HW, L
        proj_coef = F.softmax(
            torch.matmul(proj_q, proj_k) / np.sqrt(C / self.heads),
            dim=3)
        # N, h, C//h, HW
        proj = torch.matmul(proj_v, proj_coef.permute(0, 1, 3, 2))
        _, _, H, W = feature.shape
        proj = proj.view(N, -1, H, W)
        if self.lateral_conv is not None:
            feature = self.lateral_conv(feature)
        return feature + proj
