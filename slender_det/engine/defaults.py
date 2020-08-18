import logging

from slender_det.modeling import build_model
from slender_det.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer


class BaseTrainer(DefaultTrainer):
    """
    A simple warpper class for using our models/datasets/evaluators
    """

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`slender_det.modeling.build_model`.
        Overwrite it for using our own model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`slender_det.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`slender_det.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)
