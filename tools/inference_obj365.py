#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import time
import datetime
import os
import json
from collections import OrderedDict
import itertools
import torch
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, default_setup, hooks, launch

from slender_det.engine import BaseTrainer
from slender_det.config import get_cfg
from slender_det.evaluation.obj365 import inference_on_dataset


class Trainer(BaseTrainer):
    """
    We use the "BaseTrainer" which contains pre-defined default logic for standard training workflow
    """

    @classmethod
    def test(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        output_dir = os.path.join(cfg.OUTPUT_DIR, 'inference')

        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            coco_dict = inference_on_dataset(
                model, data_loader, distributed=True, output_dir=output_dir)

            if not comm.is_main_process():
                return {}

            file_path = os.path.join(output_dir, "{}_with_masks.json".format(dataset_name))
            logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_dict))
                f.flush()

        return {}


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = Trainer.build_model(cfg)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS)
    Trainer.test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
