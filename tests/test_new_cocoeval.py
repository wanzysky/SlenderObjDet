import fire

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import init_paths
from slender_det.evaluation.cocoeval import COCOeval


def main(
        path_to_gt='datasets/coco/annotations/instances_val2017.json',
        path_to_res=None
):
    coco_gt = COCO(path_to_gt)
    if path_to_res is not None:
        coco_result = coco_gt.loadRes(path_to_res)
        coco_eval = COCOeval(coco_gt, coco_result, iouType='bbox')

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    fire.Fire(main)
