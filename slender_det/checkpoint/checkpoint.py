import os
import torch
from fvcore.common.checkpoint import Checkpointer
from torch.utils import model_zoo

from detectron2.checkpoint import DetectionCheckpointer as _DetectionCheckpointer
from detectron2.utils import comm
from slender_det.utils.file_io import PathManager


class DetectionCheckpointer(_DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager


def load_from_http(filename, map_location=None, model_dir=None):
    """load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.
    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (string, optional): directory in which to save the object,
            Default: None
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    rank, world_size = comm.get_rank(), comm.get_world_size()
    rank = int(os.environ.get("LOCAL_RANK", rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(
            filename, model_dir=model_dir, map_location=map_location
        )
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(
                filename, model_dir=model_dir, map_location=map_location
            )
    return checkpoint


def load_checkpoint_from_http(
    model,
    filename,
    map_location=None,
):
    checkpointer = Checkpointer(model)
    checkpoint = load_from_http(filename, map_location=map_location)
    
    checkpointer.logger.info("[Checkpointer] Loading from {} ...".format(filename))
    incompatible = checkpointer._load_model(checkpoint={"model": checkpoint})
    
    # handle some existing subclasses that returns None
    if incompatible is not None:
        checkpointer._log_incompatible_keys(incompatible)
