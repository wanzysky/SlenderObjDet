import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.layers import get_norm, cat

from .transformer import Transformer


class TransformerNonLocal(nn.Module):
    def __init__(self, cfg, in_channels):
        super(TransformerNonLocal, self).__init__()
        hidden_dim = cfg.MODEL.DPM.HIDDEN_DIM
        dropout = cfg.MODEL.DPM.DROPOUT
        dim_feedforward = cfg.MODEL.DPM.DIM_FEEDFORWARD
        nheads = cfg.MODEL.DPM.NHEADS
        enc_layers = cfg.MODEL.DPM.ENC_LAYERS
        dec_layers = cfg.MODEL.DPM.DEC_LAYERS
        num_queries = cfg.MODEL.DPM.NUM_OBJECT_QUERIES

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, padding=0))
        self.pos_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, padding=0)
        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=False,
            return_intermediate_dec=True,
        )
        self.non_local = NonLocalBlock2D(hidden_dim, hidden_dim)

    def forward(self, features, masks, poses):
        relations = []
        for src, mask, pos in zip(features, masks, poses):
            assert mask is not None
            n, c, h, w = src.shape
            query = self.input_proj(src)
            memory = self.transformer(
                query, mask[:, 1::2, 1::2], self.query_embed.weight,
                self.pos_proj(pos), encoder_only=True)
            memory = F.interpolate(memory, size=(h, w))
            relation = self.non_local(memory, True)
            relations.append(relation)

        return relations


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Conv2dNonLocal(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Conv2dNonLocal, self).__init__()
        hidden_dim = cfg.MODEL.DPM.HIDDEN_DIM

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_dim,
                kernel_size=2, padding=0, stride=2))
        self.non_local = NonLocalBlock2D(hidden_dim+in_channels, hidden_dim)

    def forward(self, features, masks, poses):
        relations = []
        for src, mask, pos in zip(features, masks, poses):
            assert mask is not None
            n, c, h, w = src.shape
            memory = self.conv(src)
            memory = cat([memory, pos], dim=1)
            memory = F.interpolate(memory, size=(h, w))
            relation = self.non_local(memory, True)
            relations.append(relation)

        return relations


class NonLocalBlock2D(nn.Module):
    """
    Nonlocal using embeded gaussian.
    """
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=False):
        super(NonLocalBlock2D, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = "GN"

        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels,
                          out_channels=self.in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0), get_norm(bn, self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels,
                               out_channels=self.in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, attention_only=False):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        if attention_only:
            return f_div_C

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
