from functools import lru_cache

import numpy as np
import cv2
import geojson
import boto3
from fire import Fire
from tqdm import tqdm
import matplotlib.pyplot as plt

from smart_path import smart_path
from neupeak.utils import webcv2

IMAGE_PATH = smart_path('s3://wanzhaoyi-oss/xview/raw/train_images')

@lru_cache(maxsize=2)
def get_image(image_id):
    image_path = IMAGE_PATH.joinpath(image_id)
    with image_path.open('rb') as reader:
        data = reader.read()
        image = cv2.imdecode(np.fromstring(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    print('loading', image_id)
    return image


def main(annotation_path, show=False):
    with smart_path(annotation_path).open('rb') as reader:
        ann = geojson.load(reader)
        last_image_id = None
        ratios = []
        for feature in tqdm(ann['features']):
            coords = np.array([int(num) for num in feature['properties']['bounds_imcoords'].split(",")])
            h, w = coords[3] - coords[1], coords[2] - coords[0]
            ratios.append(min(h/w, w/h))
            if not show:
                continue
            image_id = feature['properties']['image_id']
            if not image_id == last_image_id:
                webcv2.imshow('image', image)
                image = get_image(image_id)
                if last_image_id is not None:
                    webcv2.waitKey()
            image = cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), color=(255, 0, 0), thickness=2)
            last_image_id = image_id
    ratios = sorted(ratios)
    total = len(ratios)
    xs, ys = [], []
    previous_r = None
    previous_i = 0
    for i, r in enumerate(ratios):
        if (previous_r is None or
            r - previous_r > 0.005 or 
            i - previous_i > 1e4):
            xs.append(r)
            ys.append(i/total)
            previous_r = r
            print(r, ',', i/total)
            previous_i = i

    print(len(xs))
    plt.plot(xs, ys)
    plt.savefig('ax.png')
    plt.show()

if __name__ == '__main__':
    Fire(main)
