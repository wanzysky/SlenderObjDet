import ipdb

import torch
from fvcore.common.file_io import PathManager


def load_coco(file_path):
    with PathManager.open(file_path, "rb") as f:
        result = torch.load(f)

    outs = []
    metrics = result['bbox']
    names = ['AP', 'AP50', 'AP75']
    outs.extend([metrics[name] for name in names])

    metrics = result['ar']
    names = ['mAR-all areas@100', 'mAR- 0  - 1/5@100', 'mAR-1/5 - 1/3@100', 'mAR-1/3 - 3/1@100']
    outs.extend([metrics[name].item() for name in names])

    info = ''
    for out in outs:
        info += '{:.3f} '.format(out)

    print(info)


def load_obj365(file_path):
    with PathManager.open(file_path, "rb") as f:
        result = torch.load(f)

    outs = []
    metrics = result['ar']
    names = [
        'mAR-all areas@100',
        'AR- 0  - 1/5@100', 'mAR- 0  - 1/5@100',
        'AR-1/5 - 1/3@100', 'mAR-1/5 - 1/3@100',
        'AR-1/3 - 3/1@100','mAR-1/3 - 3/1@100',
    ]
    outs.extend([metrics[name].item() for name in names])

    info = ''
    for out in outs:
        info += '{:.3f} '.format(out)

    print(info)


if __name__ == '__main__':
    import fire

    fire.Fire()
