"""
Complete implementation of RepPoints detector (https://arxiv.org/pdf/1904.11490).

"""
from typing import List
import time

import cv2
import torch
from torch import nn
import numpy as np
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, RetinaNet
from detectron2.modeling.backbone import build_backbone
from detectron2.layers import DeformConv, cat, batched_nms
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.matcher import Matcher
from detectron2.utils.events import get_event_storage
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.modeling.postprocessing import detector_postprocess

import concern.webcv2 as webcv2
from slender_det.modeling.grid_generator import zero_center_grid, uniform_grid


def flat_and_concate_levels(tensor_list: List[torch.Tensor]):
    """
    Flat tensors in different spatial sizes and concat them.
    Args:
        tensor_list: A list of tensors with the same shape
            in the first two dimensions(N, C, H_i, W_i).
    Returns:
        Concatenated tensor (N, X, C).
    """
    if len(tensor_list) < 1:
        return tensor_list

    N, C = tensor_list[0].shape[:2]
    tensor_list = [t.view(N, C, -1).permute(0, 2, 1) for t in tensor_list]

    return torch.cat(tensor_list, dim=1)


@META_ARCH_REGISTRY.register()
class RepPointsDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Configurations in common with RetinaNet are inherited.
        # fmt: off
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features = cfg.MODEL.RETINANET.IN_FEATURES
        self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        # The RepPoints configurations. They are currently fixed as
        # we haven't met the need to change super parameters.
        self.point_feat_channels = 256
        self.num_stacked_convs = 3
        self.num_points = 9
        self.gradient_mul = 0.1
        self.point_base_scale = 4
        self.point_strides = [8, 16, 32, 64, 128]
        self.use_sigmoid_cls = True
        self.sampling = False
        self.loss_cls = "focal_loss"
        self.loss_bbox_init = dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        self.loss_bbox_refine = dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        self.use_grid_points = False
        self.center_init = True
        self.vis_period = 1024

        self.backbone = build_backbone(cfg)
        self.transform_method = "minmax"

        if self.transform_method == "moment":
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = 0.01

        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes

        # 3 for 9 points representation.
        self.dcn_kernel = int(np.sqrt(self.num_points))
        # 1 for kernel 3.
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)

        # we use deformable conv to extract points features
        assert self.dcn_kernel * self.dcn_kernel == self.num_points, \
            "The points number should be a square number."
        assert self.dcn_kernel % 2 == 1, \
            "The points number should be an odd square number."

        # The base for deformable conv offsets.
        # There are two implementations for dcn_base, where the code from
        # official implementation is currently used.
        # The alternative one is `zero_center_grid(self.dcn_kernel).view(1, -1, 1, 1)`
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("dcn_base_offset", dcn_base_offset)

        self.init_layers()

        input_shape = self.backbone.output_shape()
        self.strides = [input_shape[f].stride for f in self.in_features]
        grid = uniform_grid(2048)
        self.register_buffer("grid", grid)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        self.loss_normalizer = 20  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

        # Assigning init box labels.
        if cfg.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE == 'points':
            from slender_det.modeling.matcher import rep_points_match
            self.matcher = rep_points_match
        elif cfg.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE == 'nearest_points':
            from slender_det.modeling.matcher import nearest_point_match
            self.matcher = nearest_point_match
        else:
            assert cfg.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE == 'inside'
            from slender_det.modeling.matcher import inside_match
            self.matcher = inside_match

        # Used for matching refine box labels.
        self.bbox_matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

    def init_layers(self):
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

        points_out_dim = 4 if self.use_grid_points else 2 * self.num_points
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

    @property
    def device(self):
        return self.pixel_mean.device

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
            strides.append(torch.full((grid.shape[0], ), stride, device=grid.device))
            point_centers.append(grid * stride)
            #point_centers.append(grid * stride)
        return point_centers, strides
        
    def points2bbox(self, base_grids: List[torch.Tensor], deltas: List[torch.Tensor], point_strides=[1,1,1,1,1]):
        '''
            Args:
                base_grids: List[[H*W,2]] coordinate of each feature map
                deltas: List[[N,C,H,W]] offsets
            Returns:
                bboxes: List[[N,4,H,W]]
        '''
        bboxes = []
        # For each level
        for i in range(len(deltas)):
            """
            delta: (N, C, H_i, W_i),
            C=4 or 18 
            """
            delta = deltas[i]
            N, C, H_i, W_i = delta.shape
            # (1, 2, H_i, W_i), grid for this feature level.
            base_grid = base_grids[i].view(1, H_i, W_i, 2).permute(0, 3, 1, 2)

            # (N*C/2, 2, H_i, W_i)
            delta = delta.view(-1, C//2, 2, H_i, W_i).view(-1, 2, H_i, W_i)
            # (N, C/2, 2, H_i, W_i)
            points = (delta * point_strides[i] + base_grid).view(-1, C//2, 2, H_i, W_i)
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
        
#    def get_center_grid(self, features):
#        point_centers = []
#        strides = []
#        for f_i, feature in enumerate(features):
#            height, width = feature.shape[2:]
#            stride = self.strides[f_i]
#            # HxW, 2
#            grid = self.grid[:height, :width].reshape(-1, 2)
#            strides.append(torch.full((grid.shape[0], ), stride, device=grid.device))
#            point_centers.append(grid * stride)
#        return torch.cat(point_centers, dim=0), torch.cat(strides, dim=0)
#
#    def points2bbox(self, base_grids: torch.Tensor, deltas: List[torch.Tensor]):
#        bboxes = []
#        H_i, W_i = float("inf"), float("inf")
#        start = 0
#        # For each level
#        for delta in deltas:
#            """
#            delta: (N, 18, H_i, W_i), 
#            """
#            # Assuming that strides go from small to large
#            assert delta.size(-2) < H_i
#            assert delta.size(-1) < W_i
#            H_i, W_i = delta.shape[-2:]
#            # (1, 2, H_i, W_i), grid for this feature level.
#            base_grid = base_grids[start:start + H_i * W_i].view(1, H_i, W_i, 2).permute(0, 3, 1, 2)
#            start += H_i * W_i
#
#            # (N*9, 2, H_i, W_i)
#            delta = delta.view(-1, 9, 2, H_i, W_i).view(-1, 2, H_i, W_i)
#            # (N, 9, 2, H_i, W_i)
#            points = (delta + base_grid).view(-1, 9, 2, H_i, W_i)
#            pts_x = points[:, :, 0, :, :]
#            pts_y = points[:, :, 1, :, :]
#            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
#            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
#            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
#            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
#
#            bbox = torch.cat(
#                [bbox_left, bbox_up, bbox_right, bbox_bottom],
#                dim=1)
#            bboxes.append(bbox)
#            continue
#
#            # (N, 1, 2, H_i, W_i)
#            points_mean = points.mean(dim=1, keepdim=True)
#            # (N, 1, 2, H_i, W_i)
#            points_std = torch.std(points - points_mean, dim=1, keepdim=True)
#            # (2)
#            moment_transfer = (self.moment_transfer * self.moment_mul) + (
#                self.moment_transfer.detach() * (1 - self.moment_mul))
#            half_shape = points_std * torch.exp(moment_transfer)[None, None, :, None, None]
#            # (N, 4, H_i, W_i)
#            bbox = torch.cat(
#                [points_mean[:, :, 0] - half_shape[:, :, 0],
#                 points_mean[:, :, 1] - half_shape[:, :, 1],
#                 points_mean[:, :, 0] + half_shape[:, :, 0],
#                 points_mean[:, :, 1] + half_shape[:, :, 1]],
#                dim=1)
#            bboxes.append(bbox)
#        return bboxes

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

        return torch.stack(init_objectness_labels),\
            torch.stack(init_bbox_labels),\
            torch.stack(cls_labels),\
            torch.stack(refine_bbox_labels)

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

        init_foreground_idxs = gt_init_objectness > 0
        strides = strides[None].repeat(pred_logits.shape[0], 1)
        coords_norm_init = strides[init_foreground_idxs].unsqueeze(-1) * 4
        loss_localization_init = smooth_l1_loss(
            pred_init_boxes[init_foreground_idxs] / coords_norm_init,
            gt_init_bboxes[init_foreground_idxs] / coords_norm_init,
            0.11, reduction='sum') / max(1, gt_init_objectness.sum()) * 0.5

        coords_norm_refine = strides[foreground_idxs].unsqueeze(-1) * 4
        loss_localization_refine = smooth_l1_loss(
            pred_refine_boxes[foreground_idxs] / coords_norm_refine,
            gt_refine_bboxes[foreground_idxs] / coords_norm_refine,
            0.11, reduction="sum") / max(1, self.loss_normalizer)

        return {"loss_cls": loss_cls,
                "loss_localization_init": loss_localization_init,
                "loss_localization_refine": loss_localization_refine}

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

    def visualize_refine_boxes(
            self,
            batched_inputs,
            centers,
            refine_logits,
            refine_boxes):
        """
        Visualize refine boxes with scores, only used for evaluation.
        """

        from detectron2.utils.visualizer import Visualizer
        image_index = 0
        img = batched_inputs[image_index]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)

        vp_refine = Visualizer(img, None)
        scores, refine_objectness = refine_logits.sigmoid().max(2)
        _, foreground_idxs = scores[image_index].sort(descending=True)
        foreground_idxs = foreground_idxs[:100]
        # foreground_idxs = (refine_objectness[image_index] >= 0).logical_and(refine_objectness[image_index] < self.num_classes).logical_and(scores[image_index] > 0.1)

        selected_centers = centers[foreground_idxs].cpu().numpy()
        vp_refine = vp_refine.overlay_instances(
            boxes=refine_boxes[image_index][foreground_idxs].detach().cpu(),
            labels=scores[image_index][foreground_idxs].detach().cpu())
        refine_image = vp_refine.get_image()
        for point in selected_centers:
            refine_image = cv2.circle(refine_image, tuple(point), 3, (255, 255, 255))
        # NOTE: This is commented temporarily. Uncomment it if
        # eagerly visualization is desired.
        '''
        if not self.training:
            webcv2.imshow("pred", vis_img)
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
        
    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate.
            Args:
                center_list: List[[H*W,2]] center coordinate of each fpn level
                pred_list: List[[N,C,H,W]] C=4 or 18, pred offset
            Returns:
                pts_list: List[[N,C,H,W]] C=4 or 18
                
        """
        #important!
        #in rpd, the coordinates of prediction points is xy form , original is yx form!
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_shift = pred_list[i_lvl]
            N,C,H,W = pts_shift.shape
            #[H*W,2]->[N,H*W,C]
            pts_center = center_list[i_lvl][:, :2].repeat(
                N, 1, self.num_points)
            #[N,C,H,W]->[N,H,W,C]->[N,H*W,C]
            xy_pts_shift = pts_shift.permute(0, 2, 3, 1).view(
                N, -1, 2 * self.num_points)
            pts_lvl = xy_pts_shift * self.point_strides[i_lvl] + pts_center
            pts_list.append(pts_lvl.permute(0, 2, 1).view(N,C,H,W))
        return pts_list

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
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        cls_features = [self.cls_conv(f) for f in features]
        reg_features = [self.reg_conv(f) for f in features]

        offsets_init = [self.offsets_init(f) for f in reg_features]

        logits = []
        offsets_refine = []
        for i in range(len(cls_features)):
            pts_out_init_grad_mul = (1 - self.gradient_mul) * offsets_init[i].detach()\
                + self.gradient_mul * offsets_init[i]
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
                    #self.deform_reg_conv(reg_features[i], pts_out_init_grad_mul)) +
                offsets_init[i].detach())
        

        point_centers, strides = self.get_center_grid(features)
#        #before offset_to_pts: feature level offset,
#        #after offset_to_pts: image level offset
#        init_points = self.offset_to_pts(point_centers, offsets_init)
#        refine_points = self.offset_to_pts(point_centers, offsets_refine)
        init_boxes = self.points2bbox(point_centers, offsets_init, [1,2,4,8,16])
        refine_boxes = self.points2bbox(point_centers, offsets_refine, [1,2,4,8,16])
        #flatten point_centers, strides
        point_centers = torch.cat(point_centers,0)
        strides = torch.cat(strides,0)
        if self.training:
            gt_init_objectness, gt_init_offsets, gt_cls, gt_refine_offsets =\
                self.get_ground_truth(point_centers, strides,
                                      flat_and_concate_levels(init_boxes), gt_instances)

            storage = get_event_storage()
            # This condition is keeped as the code is from RetinaNet in D2.
            if storage.iter % self.vis_period == 0:
                results = self.inference(logits, init_boxes, refine_boxes, images.image_sizes)
                self.visualize_training(batched_inputs, results)

            self.may_visualize_gt(
                batched_inputs, gt_init_objectness.bool(),
                gt_init_offsets, gt_cls, gt_refine_offsets,
                point_centers,
                flat_and_concate_levels(init_boxes),
                flat_and_concate_levels(refine_boxes),
                flat_and_concate_levels(logits))

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
            self.visualize_refine_boxes(
                batched_inputs,
                point_centers,
                flat_and_concate_levels(logits),
                flat_and_concate_levels(refine_boxes))
            results = self.inference(logits, init_boxes, refine_boxes, images.image_sizes)
            self.visualize_training(batched_inputs, results)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

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
            # The following code is the decoding procedure of RetinaNet in D2.
            # However, it fails to handle the predictions though I thought it could.
            """
            cls = logit.flatten().sigmoid()

            # pre nms
            num_topk = min(self.topk_candidates, cls.size(0))

            predicted_prob, topk_idxs = cls.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            points_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            init_box = init_box.reshape(4, -1).clone()
            refine_box = refine_box.reshape(4, -1).clone()
            init_box = init_box[:, points_idxs].permute(1, 0)
            refine_box_topk = refine_box[:, points_idxs].permute(1, 0)

            boxes_all.append(refine_box_topk)
            init_boxes_all.append(init_box)
            class_idxs_all.append(classes_idxs)
            scores_all.append(predicted_prob)
            """

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

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
