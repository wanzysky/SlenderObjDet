from detectron2.evaluation.testing import flatten_results_dict

import logging
import torch
import numpy as np

results = dict(
    list=[1, 2, 3, 4],
    tensor=torch.zeros((2, 3)),
    numpy=np.zeros((2, 3)),
    valid=int(8)
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('detectron2')
flattened_results = flatten_results_dict(results)
valid = dict()
for k, v in flattened_results.items():
    try:
        valid[k] = float(v)
    except (ValueError, TypeError):
        logger.info("Skip put {}: {} to tensorboard".format(k, type(v)))

print(valid)
