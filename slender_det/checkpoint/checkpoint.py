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
