import json
import time

from pycocotools.coco import COCO as Base
from collections import defaultdict

from slender_det.structures.masks import PolygonMasks


class COCO(Base):
    oriented = False

    def __init__(self, annotation_file=None, oriented=False):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :param oriented (bool): Use oriented box for computing ratio or not.
        :return:
        """
        self.oriented = oriented
        super(COCO, self).__init__(annotation_file)

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                if ann["bbox"][2] * ann["bbox"][3] < 96**2:
                    continue
                imgToAnns[ann['image_id']].append(ann)
                self.compute_ratio(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    @classmethod
    def compute_ratio(self, ann):
        if ann["iscrowd"]:
            ann["ratio"] = ann["bbox"][2] / ann["bbox"][3]
            return ann

        segmentations = PolygonMasks([ann["segmentation"]])
        ann["ratio"] = segmentations.get_ratios(oriented=self.oriented)[0]
        return ann
