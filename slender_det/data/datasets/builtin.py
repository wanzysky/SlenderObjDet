import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta

from .obj365 import load_obj365_json
from .builtin_meta import _get_obj365_metadata

def register_obj365_instances(name, metadata, json_file, image_root):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_obj365_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_obj365_coco_type(name, metadata, json_file, image_root):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_all_obj365(root):
    SPILTS = {
        "objects365_train": (
            "objects365_raw_data/objects365/train",
            "obj365/annotations/objects365_train_20190423.json"
        ),
        "objects365_val": (
            "objects365_raw_data/objects365/val",
            "obj365/annotations/objects365_val_20190423.json"),
        "coco_objects365_val_with_masks": (
            "objects365_raw_data/objects365/val-with-coco",
            "obj365/annotations/coco_obj365_slender_val_20200820.json"
        )
    }
    for key, (image_root, json_file) in SPILTS.items():
        # Assume pre-defined datasets live in `./datasets`.
        if 'coco' in key:
            register_obj365_coco_type(
                key,
                _get_coco_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                image_root,
            )
        else:
            register_obj365_instances(
                key,
                _get_obj365_metadata(key),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                image_root,
            )


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_obj365(_root)
