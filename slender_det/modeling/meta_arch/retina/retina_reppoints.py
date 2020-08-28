import cv2
import math
import numpy as np
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, DeformConv
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet, permute_to_N_HWA_K

from slender_det.modeling.meta_arch.reppoints import flat_and_concate_levels
from slender_det.modeling.grid_generator import zero_center_grid, uniform_grid

__all__ = ["ReppointsRetinaNet", "ReppointsRetinaNetHead"]


@META_ARCH_REGISTRY.register()
class ReppointsRetinaNet(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features = cfg.MODEL.RETINANET.IN_FEATURES
        self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.num_points = cfg.MODEL.PROPOSAL_GENERATOR.NUM_POINTS

        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = ReppointsRetinaNetHead(cfg, feature_shapes)
        grid = uniform_grid(2048)
        self.register_buffer("grid", grid)
        self.point_strides = [8, 16, 32, 64, 128]
        self.loss_normalizer = 20  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9
        input_shape = self.backbone.output_shape()
        self.strides = [input_shape[f].stride for f in self.in_features]

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        self.vis_period = 1024

        # Assigning init box labels.
        if cfg.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE == 'points':
            from slender_det.modeling.matchers.rep_matcher import rep_points_match
            self.matcher = rep_points_match
        elif cfg.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE == 'nearest_points':
            from slender_det.modeling.matchers.rep_matcher import nearest_point_match
            self.matcher = nearest_point_match
        else:
            assert cfg.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE == 'inside'
            from slender_det.modeling.matchers.rep_matcher import inside_match
            self.matcher = inside_match

        # Used for matching refine box labels.
        self.bbox_matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        # no anchors!
        # anchors = self.anchor_generator(features)

        # pred_deltas changes from [dx,dy,dw,dh] of retina to [x1,y1,x2,y2]
        # List[[N,C,H,W]]
        logits, offsets_init, offsets_refine = self.head(features)

        # List[[H*W,2]] List[[H*W]]
        point_centers, strides = self.get_center_grid(features)

        # List[[N,4,H,W]]
        init_boxes = self.points2bbox(point_centers, offsets_init, [1, 2, 4, 8, 16])
        refine_boxes = self.points2bbox(point_centers, offsets_refine, [1, 2, 4, 8, 16])
        # flatten point_centers, strides
        point_centers = torch.cat(point_centers, 0)
        strides = torch.cat(strides, 0)
        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            # init_bbox and refine_bbox are assign by different matcher: nearest vs IoU
            gt_init_objectness, gt_init_offsets, gt_cls, gt_refine_offsets = \
                self.get_ground_truth(point_centers, strides,
                                      flat_and_concate_levels(init_boxes), gt_instances)

            storage = get_event_storage()
            # This condition is keeped as the code is from RetinaNet in D2.
            if storage.iter % self.vis_period == 0:
                results = self.inference(logits, init_boxes, refine_boxes, images.image_sizes)
                self.visualize_training(batched_inputs, results)

            # self.may_visualize_gt(
            #     batched_inputs, gt_init_objectness.bool(),
            #     gt_init_offsets, gt_cls, gt_refine_offsets,
            #     point_centers,
            #     flat_and_concate_levels(init_boxes),
            #     flat_and_concate_levels(refine_boxes),
            #     flat_and_concate_levels(logits))

            losses = self.losses(
                flat_and_concate_levels(logits),
                flat_and_concate_levels(init_boxes),
                flat_and_concate_levels(refine_boxes),
                gt_init_objectness,
                gt_init_offsets,
                gt_cls,
                gt_refine_offsets,
                strides)

            return losses

        else:
            results = self.inference(logits, init_boxes, refine_boxes, images.image_sizes)
            #            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(
            self,
            pred_logits,
            pred_init_boxes,
            pred_refine_boxes,
            gt_init_objectness,
            gt_init_bboxes,
            gt_cls: torch.Tensor,
            gt_refine_bboxes,
            strides):
        """
        Loss computation.
        Args:
            pred_logits: (N, X, C). Classification prediction, where X is the number
                of positions from all feature levels, C is the number of object classes.
            pred_init_boxes: (N, X, 4). Init box prediction.
            pred_refine_boxes: (N, X, 4). Refined box prediction.
            gt_init_objectness: (N, X). Foreground/background classification for initial
                prediction.
            gt_init_bboxes: (N, X, 4). Initial box prediction.
            gt_cls: (N, X), Long. GT for box classification, -1 indicates ignoring.
            gt_refine_bboxes: (N, X, 4). Refined box prediction.
            strides: (X). Scale factor at each position.
        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls", "loss_localization_init", and "loss_localization_refine".
        """

        valid_idxs = gt_cls >= 0
        foreground_idxs = valid_idxs.logical_and(gt_cls != self.num_classes)
        num_foreground = foreground_idxs.sum().item() / gt_init_bboxes.shape[0]
        get_event_storage().put_scalar("num_foreground", num_foreground)

        gt_cls_target = torch.zeros_like(pred_logits)
        gt_cls_target[foreground_idxs, gt_cls[foreground_idxs]] = 1

        self.loss_normalizer = (
                self.loss_normalizer_momentum * self.loss_normalizer
                + (1 - self.loss_normalizer_momentum) * num_foreground
        )

        loss_cls = sigmoid_focal_loss_jit(
            pred_logits[valid_idxs],
            gt_cls_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum"
        ) / max(1, self.loss_normalizer)

        strides = strides[None].repeat(pred_logits.shape[0], 1)
        init_foreground_idxs = gt_init_objectness > 0
        coords_norm_init = strides[init_foreground_idxs].unsqueeze(-1) * 4
        loss_localization_init = smooth_l1_loss(
            pred_init_boxes[init_foreground_idxs] / coords_norm_init,
            gt_init_bboxes[init_foreground_idxs] / coords_norm_init,
            0.11, reduction='sum') / max(1, gt_init_objectness.sum()) * 0.5

        coords_norm = strides[foreground_idxs].unsqueeze(-1) * 4
        loss_localization_refine = smooth_l1_loss(
            pred_refine_boxes[foreground_idxs] / coords_norm,
            gt_refine_bboxes[foreground_idxs] / coords_norm,
            0.11, reduction="sum") / max(1, self.loss_normalizer)

        return {"loss_cls": loss_cls,
                "loss_localization_init": loss_localization_init,
                "loss_localization_refine": loss_localization_refine}

    def inference(self, logits, init_boxes, refine_boxes, image_sizes):
        results = []

        for img_idx, image_size in enumerate(image_sizes):
            logits_per_image = [logits_per_level[img_idx] for logits_per_level in logits]
            init_boxes_per_image = [init_boxes_per_level[img_idx]
                                    for init_boxes_per_level in init_boxes]
            refine_boxes_per_image = [refine_boxes_per_level[img_idx]
                                      for refine_boxes_per_level in refine_boxes]

            results_per_image = self.inference_single_image(
                logits_per_image, init_boxes_per_image, refine_boxes_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, logits, init_boxes, refine_boxes, image_size):
        boxes_all = []
        init_boxes_all = []
        class_idxs_all = []
        scores_all = []
        for logit, init_box, refine_box in zip(logits, init_boxes, refine_boxes):
            scores, cls = logit.sigmoid().max(0)
            cls = cls.view(-1)
            scores = scores.view(-1)
            init_box = init_box.view(4, -1).permute(1, 0)
            refine_box = refine_box.view(4, -1).permute(1, 0)

            predicted_prob, topk_idxs = scores.sort(descending=True)
            num_topk = min(self.topk_candidates, cls.size(0))
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            init_box_topk = init_box[topk_idxs]
            refine_box_topk = refine_box[topk_idxs]
            cls_topk = cls[topk_idxs]
            score_topk = scores[topk_idxs]

            boxes_all.append(refine_box_topk)
            init_boxes_all.append(init_box_topk)
            class_idxs_all.append(cls_topk)
            scores_all.append(score_topk)

        boxes_all, scores_all, class_idxs_all, init_boxes_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all, init_boxes_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        result.init_boxes = init_boxes_all[keep]
        return result

    def get_center_grid(self, features):
        '''
            Returns:
                points_centers: List[[H*W,2]]
                strides: List[[H*W]]
        '''
        point_centers = []
        strides = []
        for f_i, feature in enumerate(features):
            height, width = feature.shape[2:]
            stride = self.strides[f_i]
            # HxW, 2
            grid = self.grid[:height, :width].reshape(-1, 2)
            strides.append(torch.full((grid.shape[0],), stride, device=grid.device))
            point_centers.append(grid * stride)
            # point_centers.append(grid * stride)
        return point_centers, strides

    def points2bbox(self, base_grids: List[torch.Tensor], deltas: List[torch.Tensor], point_strides=(1, 1, 1, 1, 1)):
        """
        Args:
            base_grids: List[[H*W,2]] coordinate of each feature map
            deltas: List[[N,C,H,W]] offsets
            point_strides (tuple[Int]) :
        Returns:
            bboxes: List[[N,4,H,W]]

        """

        bboxes = []
        use_2points = False
        # For each level
        for i in range(len(deltas)):
            """
            delta: (N, C, H_i, W_i),
            C=4 or 18 
            """
            delta = deltas[i]
            if use_2points:
                delta = delta[:, :4, :, :]
            N, C, H_i, W_i = delta.shape
            # (1, 2, H_i, W_i), grid for this feature level.
            base_grid = base_grids[i].view(1, H_i, W_i, 2).permute(0, 3, 1, 2)

            # (N*C/2, 2, H_i, W_i)
            delta = delta.view(-1, C // 2, 2, H_i, W_i).reshape(-1, 2, H_i, W_i)
            # (N, C/2, 2, H_i, W_i)
            points = (delta * point_strides[i] + base_grid).view(-1, C // 2, 2, H_i, W_i)
            pts_x = points[:, :, 0, :, :]
            pts_y = points[:, :, 1, :, :]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]

            bbox = torch.cat(
                [bbox_left, bbox_up, bbox_right, bbox_bottom],
                dim=1)
            bboxes.append(bbox)
        return bboxes

    @torch.no_grad()
    def get_ground_truth(self, centers: torch.Tensor, strides, init_boxes, gt_instances):
        """
        Get gt according to the init box prediction.
        The labels for init boxes are generated from point-based distance matching,
        and the labels refine boxes are generated from the init boxes using the same way
        with RetinaNet, where the init boxes are regarded as anchors.
        Args:
            centers: (X, 2), center coordinates for points in all feature levels.
            strides: (X), strides for each point in all feature levels.
            init_boxes: (N, X, 4), init box predection.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.
        Returns:
            Tensor (N, X):
                Foreground/background label for init boxes. It is used to select positions
                where the init box regression loss is computed.
            Tensor (N, X, 4):
                Label for init boxes, will be masked by binary label above.
            Tensor (N, X):
                Classification label at all positions,
                including values -1 for ignoring, [0, self.num_classes -1] fore foreground positions,
                and self.num_classes for background positions.
            Tensor (N, X, 4):
                Label for refine boxes, only foreground positions are considered.
        """
        # the init_bbox uses point-based nearest assign, the refine_bbox uses IoU based assign
        init_objectness_labels = []
        init_bbox_labels = []
        cls_labels = []
        refine_bbox_labels = []
        for i, targets_per_image in enumerate(gt_instances):
            image_size = targets_per_image.image_size
            centers_invalid = (centers[:, 0] >= image_size[1]).logical_or(
                centers[:, 1] >= image_size[0])

            init_objectness_label, init_bbox_label = self.matcher(
                centers, strides, targets_per_image.gt_boxes)
            init_objectness_label[centers_invalid] = 0

            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes,
                Boxes(init_boxes[i]))
            gt_matched_idxs, bbox_mached = self.bbox_matcher(match_quality_matrix)
            cls_label = targets_per_image.gt_classes[gt_matched_idxs]
            cls_label[bbox_mached == 0] = self.num_classes
            cls_label[centers_invalid] = -1
            refine_bbox_label = targets_per_image.gt_boxes[gt_matched_idxs]

            init_objectness_labels.append(init_objectness_label)
            init_bbox_labels.append(init_bbox_label)
            cls_labels.append(cls_label)
            refine_bbox_labels.append(refine_bbox_label.tensor)

        return torch.stack(init_objectness_labels), \
               torch.stack(init_bbox_labels), \
               torch.stack(cls_labels), \
               torch.stack(refine_bbox_labels)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        if self.training:
            storage = get_event_storage()
        max_boxes = 100

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)

        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()
        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        if self.training:
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            vis_img = np.vstack((anno_img, prop_img))
            vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
            vis_img = vis_img.transpose(2, 0, 1)
            storage.put_image(vis_name, vis_img)
        '''
        else:
            webcv2.imshow('result', prop_img)
            webcv2.waitKey()
        '''

    def may_visualize_gt(
            self,
            batched_inputs,
            init_objectness,
            init_bbox,
            refine_objectness,
            refine_boxes,
            centers,
            pred_init_boxes,
            pred_refine_boxes,
            logits):
        """
        Visualize initial and refine boxes using mathced labels for filtering.
        The prediction at positive positions are shown.
        """
        if self.training:
            storage = get_event_storage()
            if not storage.iter % self.vis_period == 0:
                return

        from detectron2.utils.visualizer import Visualizer
        image_index = 0
        img = batched_inputs[image_index]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)

        v_init = Visualizer(img, None)
        v_init = v_init.overlay_instances(
            boxes=Boxes(init_bbox[image_index][init_objectness[image_index]].cpu()))
        init_image = v_init.get_image()

        v_refine = Visualizer(img, None)
        v_refine = v_refine.overlay_instances(
            boxes=Boxes(refine_boxes[image_index][refine_objectness[image_index] > 0].cpu()))
        refine_image = v_refine.get_image()

        if self.training:
            vis_img = np.vstack((init_image, refine_image))
            vis_img = vis_img.transpose(2, 0, 1)
            storage.put_image("TOP: init gt boxes; Bottom: refine gt boxes", vis_img)

        vp_init = Visualizer(img, None)
        selected_centers = centers[init_objectness[image_index]].cpu().numpy()
        vp_init = vp_init.overlay_instances(
            boxes=Boxes(pred_init_boxes[image_index][init_objectness[image_index]].detach().cpu()),
            labels=logits[image_index][init_objectness[image_index]].sigmoid().max(1)[0].detach().cpu())
        init_image = vp_init.get_image()

        for point in selected_centers:
            init_image = cv2.circle(init_image, tuple(point), 3, (255, 255, 255))

        vp_refine = Visualizer(img, None)
        foreground_idxs = (refine_objectness[image_index] >= 0).logical_and(
            refine_objectness[image_index] < self.num_classes)
        selected_centers = centers[foreground_idxs].cpu().numpy()
        vp_refine = vp_refine.overlay_instances(
            boxes=pred_refine_boxes[image_index][foreground_idxs].detach().cpu(),
            labels=logits[image_index][foreground_idxs].sigmoid().max(1)[0].detach().cpu())
        refine_image = vp_refine.get_image()
        for point in selected_centers:
            refine_image = cv2.circle(refine_image, tuple(point), 3, (255, 255, 255))

        vis_img = np.vstack((init_image, refine_image))
        # vis_img = np.vstack((init_image.get(), refine_image.get()))
        if self.training:
            vis_img = vis_img.transpose(2, 0, 1)
            storage.put_image("TOP: init pred boxes; Bottom: refine pred boxes", vis_img)
        # NOTE: This is commented temporarily. Uncomment it if
        # eagerly visualization is desired.
        '''
        else:
            webcv2.imshow('pred', vis_img)
            webcv2.waitKey()
        '''


class ReppointsRetinaNetHead(nn.Module):

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # the same as RetinaNetHead, we replace the cls_score net to logits net, which utilizes the deform_conv
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        # please add it in cfg!
        self.num_points = cfg.MODEL.PROPOSAL_GENERATOR.NUM_POINTS
        self.point_feat_channels = 256
        self.cls_out_channels = num_classes - 1  # maybe not right
        #        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        #        # fmt: on
        #        assert (
        #            len(set(num_anchors)) == 1
        #        ), "Using different number of anchors between levels is not currently supported!"
        #        num_anchors = num_anchors[0]

        # dcn_base_offset
        self.dcn_kernel = int(np.sqrt(9))
        # 1 for kernel 3.
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("dcn_base_offset", dcn_base_offset)

        self.gradient_mul = 0.1

        self.cls_conv = nn.Sequential(
            *self.stacked_convs())
        self.reg_conv = nn.Sequential(
            *self.stacked_convs())

        self.deform_cls_conv = DeformConv(
            self.point_feat_channels,
            self.point_feat_channels,
            self.dcn_kernel, 1, self.dcn_pad)
        self.deform_reg_conv = DeformConv(
            self.point_feat_channels,
            self.point_feat_channels,
            self.dcn_kernel, 1, self.dcn_pad)

        points_out_dim = 2 * self.num_points
        self.offsets_init = nn.Sequential(
            nn.Conv2d(self.point_feat_channels,
                      self.point_feat_channels,
                      3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.point_feat_channels,
                      points_out_dim,
                      1, 1, 0))

        self.offsets_refine = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.point_feat_channels,
                      points_out_dim,
                      1, 1, 0))
        self.logits = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.point_feat_channels,
                      self.cls_out_channels,
                      1, 1, 0))

        bias_init = float(-np.log((1 - 0.01) / 0.01))
        for modules in [
            self.cls_conv,
            self.reg_conv,
            self.offsets_init,
            self.offsets_refine,
            self.deform_cls_conv,
            self.deform_reg_conv]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        for module in self.logits.modules():
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias, bias_init)

    def stacked_convs(self, layers=3):
        convs = []
        for _ in range(layers):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.point_feat_channels,
                        self.point_feat_channels,
                        kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(32, self.point_feat_channels),
                    nn.ReLU(inplace=True)
                ))
        return convs

    def forward(self, features):

        logits = []
        offsets_refine = []
        cls_features = [self.cls_conv(f) for f in features]
        reg_features = [self.reg_conv(f) for f in features]

        offsets_init = [self.offsets_init(f) for f in reg_features]

        logits = []
        offsets_refine = []
        offsets_init_9points = []  # only used in dcn_offset generation
        if self.num_points == 2:
            for i in range(len(cls_features)):
                offsets_init_9points_i = self.gen_grid_from_reg(offsets_init[i])
                offsets_init_9points.append(offsets_init_9points_i)
        else:
            offsets_init_9points = offsets_init
        for i in range(len(cls_features)):
            pts_out_init_grad_mul = (1 - self.gradient_mul) * offsets_init_9points[i].detach() \
                                    + self.gradient_mul * offsets_init_9points[i]
            # N, 18, H, W --> N, 9, 2(x, y), H, W --> N, 9, 2(y, x), H, W
            # BUGGY: assuming self.num_points == 9
            pts_out_init_grad_mul = pts_out_init_grad_mul.reshape(
                pts_out_init_grad_mul.size(0),
                9, 2,
                *pts_out_init_grad_mul.shape[-2:]
            ).flip(2)
            pts_out_init_grad_mul = pts_out_init_grad_mul.reshape(
                -1, 18, *pts_out_init_grad_mul.shape[-2:])
            dcn_offset = pts_out_init_grad_mul - self.dcn_base_offset

            logits.append(
                self.logits(self.deform_cls_conv(cls_features[i], dcn_offset)))
            offsets_refine.append(
                self.offsets_refine(
                    self.deform_reg_conv(reg_features[i], dcn_offset)) +
                # self.deform_reg_conv(reg_features[i], pts_out_init_grad_mul)) +
                offsets_init[i].detach())
        return logits, offsets_init, offsets_refine

    def gen_grid_from_reg(self, reg):
        b, c, h, w = reg.shape
        grid_left = reg[:, [0], ...]
        grid_top = reg[:, [1], ...]
        grid_width = reg[:, [2], ...] - reg[:, [0], ...]
        grid_height = reg[:, [3], ...] - reg[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_xy = torch.stack([grid_x, grid_y], dim=2)
        grid_xy = grid_xy.view(b, -1, h, w)
        return grid_xy
