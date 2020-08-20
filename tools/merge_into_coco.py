import json
import datetime
import copy

from fire import Fire

from slender_det.evaluation.coco import COCO
from concern.smart_path import smart_path


def main(input_path, coco_json_path, output_path):
    origin_dict = json.load(smart_path(input_path).open("rt"))
    coco_dict = json.load(smart_path(coco_json_path).open("rt"))
    origin_coco = COCO(input_path)
    coco = COCO(coco_json_path)

    anns = []
    images = list()
    image_ids = set()
    for ann in origin_dict["annotations"]:
        ratio = COCO.compute_ratio(ann)["ratio"]
        if ratio > 1 / 3:
            continue
        ann["ratio"] = ratio
        ann["id"] = ann["id"] + 1 << 31

        image_id = ann["image_id"] + 1 << 31
        image = origin_coco.imgs[ann["image_id"]]

        image["id"] = image_id
        ann["image_id"] = image_id

        anns.append(ann)

        if not image_id in image_ids:
            images.append(image)
            image_ids.add(image_id)

    info = dict(date_created=str(datetime.datetime.now()),
                description="Merged from {} and {}.".format(
                    input_path, coco_json_path))
    new_coco_dict = dict(info=copy.deepcopy(info),
                         categories=copy.deepcopy(coco_dict["categories"]),
                         licenses=None,
                         annotations=copy.deepcopy(coco_dict["annotations"]) +
                         copy.deepcopy(anns),
                         images=copy.deepcopy(coco_dict["images"]) +
                         copy.deepcopy(images))

    with open(output_path, "wt") as writer:
        json.dump(new_coco_dict, writer)


Fire(main)
