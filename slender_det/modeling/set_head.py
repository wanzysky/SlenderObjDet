import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.layers import get_norm, cat

from .transformer import Transformer, MLP


class TransformerSetHead(nn.Module):
    def __init__(self, cfg, in_channels, num_classes=80):
        super(TransformerSetHead, self).__init__()
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
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0))
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
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, features, masks, poses):
        hs = self.transformer(
            self.input_proj(features), masks,
            self.query_embed.weight, poses)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


