import torch
import torch.nn as nn
import torch.nn.functional as F


class CornerNet(nn.Module):

    def __init__(self):
        super(CornerNet, self).__init__()
