from functools import lru_cache

import numpy as np
import cv2
import geojson
import boto3
from tqdm import tqdm
import matplotlib.pyplot as plt

from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import DatasetCatalog, MetadataCatalog

from concern.smart_path import smart_path
from concern.support import fig2image
from concern import webcv2
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

    fig, ax = plt.subplots(3, 1)

    xs = []
    ys = []
    for dic in tqdm(dicts):
        for obj in dic["annotations"]:
            xs.append(obj["bbox"][3] * obj["bbox"][2])
            ratio = obj["bbox"][2] / obj["bbox"][3]
            if ratio <= 1:
                ratio = ratio
            else:
                ratio = 1 / ratio + 1
            ys.append(ratio)

    ax.scatter(xs, ys, s=0.01)
    image = fig2image(fig)
    webcv2.imshow("fig.png", image)
    webcv2.waitKey()

if __name__ == '__main__':
    main()
