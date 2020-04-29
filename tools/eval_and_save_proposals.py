import os

from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator



class ProposalVisibleEvaluator(DefaultTrainer):
    """
    Overide trainer for saving or showing detailed results, eg, proposals.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator for scrutinizing proposals.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        return evaluator
