import argparse
import pickle
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.structures import Keypoints
import sys
sys.path.append("..")
from utils.visualizer import Visualizer
from concern import webcv2


def create_instances(predictions, image_size):
    ret = Instances(image_size)
    
    
    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    box_ratio = (bbox[:, 0] - bbox[:, 2]) / (bbox[:, 1] - bbox[:, 3])
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels
    ret.box_ratio = box_ratio

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret

def create_instances_with_anchor(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["anchor"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])
    
    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
#    parser.add_argument("--proposal", required=True, help="Pickle file storing proposals")
    parser.add_argument("--result", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--show", action="store_true", help="show imags via webcv")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--iou-threshold", default=0.05, type=float, help="confidence threshold")
    parser.add_argument("--multisave", action="store_true", help="save visualization results in multiple forms")
    args = parser.parse_args()

    interest = set(['baseball bat', 'knife', 'bench'])

    logger = setup_logger()
    
    #load instance
    with PathManager.open(args.result, "r") as f:
        predictions = json.load(f)
    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)
    
    
    
    #load dataset
    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)
    
    #split visualization dir by ratio
    ratios_ranges = [(0, 0.2), (0.2, 0.33), (0.33,3), (3,5), (5,1000)]
    if (args.multisave):
        os.makedirs(os.path.join(args.output, 'ratio_0'), exist_ok=True)#(0, 0.2), (5,1000)
        os.makedirs(os.path.join(args.output, 'ratio_1'), exist_ok=True)#(0.2, 0.33), (3,5)
        os.makedirs(os.path.join(args.output, 'ratio_2'), exist_ok=True)#(0.33,3)
        os.makedirs(os.path.join(args.output, 'concat'), exist_ok=True)#rows=3,columns=2,concat_result
    ratio_counts=[0]*3
    bar=tqdm.tqdm(dicts)
    for dic in bar:
        bar.set_description("img num of ratios "+str(ratio_counts))
#        flag = False
#        for ann in dic['annotations']:
#            category = metadata.get('thing_classes')[ann['category_id']]
#            if category in interest:
#                flag = True
#        if not flag:
#            continue
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])
        
        vis_pred=[]
        vis = Visualizer(img, metadata)
        selected_preds=vis.select_predictions_by_ratio(pred_by_image[dic["image_id"]],ratios_ranges)
        instances_results=[]
        instances_anchors=[]
        has_ratio=''
        ratio_concate=[[] for _ in range(0,len(selected_preds))]
        for ratio_i,pred in enumerate(selected_preds):
            #prediction
            predictions = create_instances(pred, img.shape[:2])
            if len(predictions)>=1:
                has_ratio+='+'+str(ratio_i)
            instances_results.append(predictions)
            vis = Visualizer(img, metadata)
            pred_result = vis.draw_instance_predictions(predictions).get_image()
            vis_pred.append(pred_result)
            ratio_concate[ratio_i].append(pred_result)
            
            
            #anchor
            predictions = create_instances_with_anchor(pred, img.shape[:2])
            instances_anchors.append(predictions)
            vis = Visualizer(img, metadata)
            pred_result = vis.draw_instance_predictions(predictions).get_image()
            vis_pred.append(pred_result)
            ratio_concate[ratio_i].append(pred_result)
            ratio_concate_pred = vis.smart_concatenate(ratio_concate[ratio_i], min_side=1960, num_rows=1, num_colums=2)
            if (args.multisave and has_ratio.find('+'+str(ratio_i))!=-1):
                cv2.imwrite(os.path.join(args.output,'ratio_'+str(ratio_i), basename), ratio_concate_pred[:, :, ::-1])
        concat_pred = vis.smart_concatenate(vis_pred, min_side=1960, num_rows=3, num_colums=2)
        cv2.imwrite(os.path.join(args.output, 'concat', basename+has_ratio+'.jpg'), concat_pred[:, :, ::-1])

        for ratio_i,ratio_count in enumerate(ratio_counts):
            if has_ratio.find('+'+str(ratio_i))!=-1:
                ratio_counts[ratio_i]+=1
    print(ratio_counts)
        
