from functools import lru_cache
from collections import defaultdict

import numpy as np
import cv2
import geojson
import boto3
from tqdm import tqdm
import matplotlib.pyplot as plt

from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import DatasetCatalog, MetadataCatalog

from concern.smart_path import smart_path
from concern.support import fig2image, between
from concern import webcv2
from slender_det.evaluation.coco import COCO
from ._setup import setup

IMAGE_PATH = smart_path('s3://wanzhaoyi-oss/xview/raw/train_images')

@lru_cache(maxsize=2)
def get_image(image_id):
    image_path = IMAGE_PATH.joinpath(image_id)
    with image_path.open('rb') as reader:
        data = reader.read()
        image = cv2.imdecode(np.fromstring(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    print('loading', image_id)
    return image


def main():
    parser = default_argument_parser()
    args = parser.parse_args()
    cfg = setup(args)
    dataset = cfg.DATASETS.TEST[0]
    dicts = list(DatasetCatalog.get(dataset))

    metadata = MetadataCatalog.get(dataset)
    labels = metadata.thing_classes
    ratios = {"0-1/5": [0, 1/5], "0-1/3": [1/5, 1/3], "1/3-1": [1/3, 1]}
    # ratios = {"1/3-1": [1/3, 1], "1/5-1/3": [1/5, 1/3], "0-1/5": [0, 1/5]}
    bars = dict()
    for key in ratios.keys():
        bars[key] = [0 for _ in range(len(labels))]


    all_ratios = []
    for dic in tqdm(dicts):
        for obj in dic["annotations"]:
            ratio = COCO.compute_ratio(obj, oriented=True)["ratio"]
            for key, ratio_range in ratios.items():
                if between(ratio, ratio_range):
                    bars[key][obj["category_id"]] += 1
            all_ratios.append(ratio)

    fig, ax = plt.subplots()
    ax.set_yscale("symlog")
    prev = np.zeros((len(labels), ))
    for key, bar in bars.items():
        bar = np.array(bar)
        ax.bar(labels, bar, bottom=prev, label=key)
        prev = prev + bar
    ax.legend()
    fig.set_size_inches(18.5, 10.5)
    ax.set_xticklabels(labels, rotation="vertical")
    group = fig2image(fig)
    webcv2.imshow("group", group)

    fig, ax = plt.subplots()
    all_ratios = sorted(all_ratios)
    numbers = [0]
    tick = 0.01
    seperator = tick
    count = 0

    for r in all_ratios:
        count += 1
        while seperator < r:
            seperator += tick
            numbers.append(count)

    ax.plot(np.arange(0, 1, tick), np.array(numbers) / count)
    number = fig2image(fig)
    webcv2.imshow("number", number)
    webcv2.waitKey()

if __name__ == '__main__':
    main()
