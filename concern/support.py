import os
from collections import Iterable
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
import cv2
import matplotlib.pyplot as plt


def any_of(array, function, *args, **kwargs):
    if not isinstance(array, Iterable):
        return function(array, *args, **kwargs)
    for item in array:
        if function(*args, item, **kwargs):
            return True
    return False


def between(a, a_range: tuple) -> bool:
    if isinstance(a, np.ndarray):
        return np.logical_and(a >= a_range[0], a <= a_range[1])
    return a >= a_range[0] and a <= a_range[1]


def all_the_same(a_list: list) -> bool:
    for item in a_list:
        if not item == a_list[0]:
            return False
    return True


def make_dual(item_or_tuple) -> tuple:
    if isinstance(item_or_tuple, tuple):
        return item_or_tuple
    return (item_or_tuple, item_or_tuple)


def ratio_of_bbox(bbox):
    """
    Args:
        bbox: box in form (x0, y0, x1, y1).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    return min(w, h) / max(w, h)


def ratio_of_polygon(polygon):
    """
    Args:
        polygon (n, 2): a set of points.
    """

    polygon = np.concatenate(polygon, 0).reshape(-1, 2)
    hull = cv2.convexHull(polygon.astype(np.float32)).reshape(-1, 2)
    if hull.shape[0] < 3:
        return ratio_of_bbox([
            polygon[:, 0].min(),
            polygon[:, 1].min(),
            polygon[:, 0].max(),
            polygon[:, 1].max()
        ])
    rect = cv2.minAreaRect(hull.astype(np.float32))
    w, h = rect[1]
    return min(w, h) / max(w, h)


def fig2image(fig: plt.Figure):
    fig.canvas.draw()
    buff, (width, height) = fig.canvas.print_to_buffer()
    image = np.fromstring(buff, dtype=np.uint8).reshape(height, width, 4)
    return image


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()



@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

