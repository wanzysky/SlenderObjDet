from detectron2.data import transforms as T
from . import transforms as T_local 

__all__ = [
    "build_augmentation",
]
def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    augmentation = []
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        augmentation = [T_local.ResizeLongestEdge(min_size, max_size, sample_style)]
    if is_train:
        augmentation.append(T.RandomFlip())
    return augmentation