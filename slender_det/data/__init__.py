from detectron2.data import (
    DatasetCatalog,
    Metadata,
    MetadataCatalog,
    get_detection_dataset_dicts,
)

# ensure the builtin datasets are registered
from . import datasets, mappers, transforms
from .build import build_test_loader as build_detection_test_loader
from .build import build_train_loader as build_detection_train_loader
