from detectron2.data import build_detection_train_loader, build_detection_test_loader

from . import mappers


def get_dataset_mapper(dataset_name):
    if "coco" in dataset_name:
        return getattr(mappers, "DatasetMapper", None)
    elif "objects365" in dataset_name:
        return getattr(mappers, "OssMapper", None)
    else:
        return getattr(mappers, "DatasetMapper", None)


def build_train_loader(cfg, mapper=None):
    if mapper is None:
        mapper = get_dataset_mapper(cfg.DATASETS.TRAIN[0])(cfg, True)

    return build_detection_train_loader(cfg, mapper=mapper)


def build_test_loader(cfg, dataset_name, mapper=None):
    if mapper is None:
        mapper = get_dataset_mapper(dataset_name)(cfg, False)

    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
