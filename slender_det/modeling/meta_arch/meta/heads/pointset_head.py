from typing import List, Tuple, Optional, Dict
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.layers import ShapeSpec, get_norm, cat, batched_nms
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances, ImageList, pairwise_iou

from .meta_head import HeadBase, MEAT_HEADS_REGISTRY
from .utils import grad_mul, ShiftGenerator


@MEAT_HEADS_REGISTRY.register()
class PointSetHead(HeadBase):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)
        head_params = cfg.MODEL.META_ARCH
        self.num_points = head_params.NUM_POINTS

        self.point_base_scale = head_params.POINT_BASE_SCALE
        self.transform_method = head_params.TRANSFORM_METHOD
        self.moment_mul = head_params.MOMENT_MUL
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)
        else:
            self.moment_transfer = None

        # init bbox pred
        self.loc_init_conv = nn.Conv2d(self.feat_channels, self.loc_feat_channels, 3, 1, 1)
        self.loc_init_out = nn.Conv2d(self.loc_feat_channels, self.num_points * 2, 1, 1, 0)

        # make feature adaptive layer
        self.cls_conv, self.loc_refine_conv = self.make_feature_adaptive_layers()
        self._prepare_offset()

        self.cls_out = nn.Conv2d(self.feat_channels, self.num_classes, 1, 1, 0)
        self.loc_refine_out = nn.Conv2d(self.loc_feat_channels, self.num_points * 2, 1, 1, 0)

        self.shift_generator = ShiftGenerator(input_shape)

        self._init_weights()

    def _prepare_offset(self):
        if self.feat_adaption == "Unsupervised Offset":
            self.offset_conv = nn.Conv2d(self.feat_channels, 18, 1, 1, 0)
        elif self.feat_adaption == "Split Unsup Offset":
            self.offset_conv_cls = nn.Conv2d(self.feat_channels, 18, 1, 1, 0)
            self.offset_conv_loc = nn.Conv2d(self.feat_channels, 18, 1, 1, 0)
        elif self.feat_adaption == "Supervised Offset":
            self.dcn_kernel = int(np.sqrt(self.num_points))
            self.dcn_pad = int((self.dcn_kernel - 1) / 2)
            dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
            dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
            dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
            dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
            self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

    def _init_weights(self):
        # Initialization
        for modules in [
            self.loc_init_conv, self.loc_init_out, self.cls_conv,
            self.loc_refine_conv, self.cls_out, self.loc_refine_out
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)

        if self.feat_adaption == "Unsupervised Offset":
            nn.init.normal_(self.offset_conv.weight, mean=0, std=0.01)
            nn.init.constant_(self.offset_conv.bias, 0)
        elif self.feat_adaption == "Split Unsup Offset":
            for l in [self.offset_conv_cls, self.offset_conv_loc]:
                nn.init.normal_(l.weight, mean=0, std=0.01)
                nn.init.constant_(l.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - self.prior_prob) / self.prior_prob))
        nn.init.constant_(self.cls_out.bias, bias_value)

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            gt_instances: Optional[List[Instances]] = None
    ):
        cls_outs, pts_outs_init, pts_outs_refine = self._forward(features)
        center_pts = self.shift_generator(features)

        if self.training:
            return self.losses(center_pts, cls_outs, pts_outs_init, pts_outs_refine, gt_instances)
        else:
            results = self.inference(center_pts, cls_outs, pts_outs_init, pts_outs_refine, images)
            return results

    def _forward(self, features):
        cls_outs = []
        loc_outs_init = []
        loc_outs_refine = []

        if self.feat_adaption == "Supervised Offset":
            dcn_base_offsets = self.dcn_base_offset.type_as(features[0])
        else:
            dcn_base_offsets = None

        for l, feature in enumerate(features):
            cls_feat = feature
            loc_feat = feature

            for cls_conv in self.cls_subnet:
                cls_feat = cls_conv(cls_feat)
            for loc_conv in self.loc_subnet:
                loc_feat = loc_conv(loc_feat)

            loc_out_init = self.loc_init_out(F.relu_(self.loc_init_conv(loc_feat)))
            loc_outs_init.append(loc_out_init)

            if self.feat_adaption == "Empty":
                cls_feat_fa = self.cls_conv(cls_feat)
                loc_feat_fa = self.loc_refine_conv(loc_feat)
            elif self.feat_adaption == "Unsupervised Offset":
                # TODO: choose a better input info for generating offsets
                dcn_offsets = self.offset_conv(loc_feat)
                cls_feat_fa = self.cls_conv(cls_feat, dcn_offsets)
                loc_feat_fa = self.loc_refine_conv(loc_feat, dcn_offsets)
            elif self.feat_adaption == "Split Unsup Offset":
                # TODO: split
                dcn_offsets_cls = self.offset_conv_cls(loc_feat)
                cls_feat_fa = self.cls_conv(cls_feat, dcn_offsets_cls)

                dcn_offsets_loc = self.offset_conv_loc(loc_feat)
                loc_feat_fa = self.loc_refine_conv(loc_feat, dcn_offsets_loc)
            elif self.feat_adaption == "Supervised Offset":
                # build offsets for deformable conv
                loc_out_init_grad_mul = grad_mul(loc_out_init, self.gradient_mul)
                # TODOs: commpute offset for different methods
                dcn_offsets = loc_out_init_grad_mul - dcn_base_offsets
                # get adaptive feature map
                cls_feat_fa = self.cls_conv(cls_feat, dcn_offsets)
                loc_feat_fa = self.loc_refine_conv(loc_feat, dcn_offsets)
            else:
                raise RuntimeError("Got {}".format(self.feat_adaption))

            cls_outs.append(self.cls_out(F.relu_(cls_feat_fa)))
            if self.res_refine:
                loc_out_refine = self.loc_refine_out(F.relu_(loc_feat_fa)) + loc_out_init.detach()
            else:
                loc_out_refine = self.loc_refine_out(F.relu_(loc_feat_fa))

            loc_outs_refine.append(loc_out_refine)

        return cls_outs, loc_outs_init, loc_outs_refine

    def losses(self, center_pts, cls_outs, pts_outs_init, pts_outs_refine, targets):
        """
        Args:
            center_pts: (list[list[Tensor]]): a list of N=#image elements. Each
                is a list of #feature level tensors. The tensors contains
                shifts of this image on the specific feature level.
            cls_outs: List[Tensor], each item in list with
                shape:[N, num_classes, H, W]
            pts_outs_init: List[Tensor], each item in list with
                shape:[N, num_points*2, H, W]
            pts_outs_refine: List[Tensor], each item in list with
            shape:[N, num_points*2, H, W]
            targets: (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.
                Specify `targets` during training only.
        Returns:
            dict[str:Tensor]:
                mapping from a named loss to scalar tensor
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_outs]
        assert len(featmap_sizes) == len(center_pts[0])

        pts_dim = 2 * self.num_points

        cls_outs = [
            cls_out.permute(0, 2, 3, 1).reshape(cls_out.size(0), -1, self.num_classes)
            for cls_out in cls_outs
        ]
        pts_outs_init = [
            pts_out_init.permute(0, 2, 3, 1).reshape(pts_out_init.size(0), -1, pts_dim)
            for pts_out_init in pts_outs_init
        ]
        pts_outs_refine = [
            pts_out_refine.permute(0, 2, 3, 1).reshape(pts_out_refine.size(0), -1, pts_dim)
            for pts_out_refine in pts_outs_refine
        ]

        cls_outs = torch.cat(cls_outs, dim=1)
        pts_outs_init = torch.cat(pts_outs_init, dim=1)
        pts_outs_refine = torch.cat(pts_outs_refine, dim=1)

        pts_strides = []
        for i, s in enumerate(center_pts[0]):
            pts_strides.append(
                cls_outs.new_full((s.size(0),), self.fpn_strides[i]))
        pts_strides = torch.cat(pts_strides, dim=0)

        center_pts = [
            torch.cat(c_pts, dim=0).to(cls_outs.device) for c_pts in center_pts
        ]

        pred_cls = []
        pred_init = []
        pred_refine = []

        target_cls = []
        target_init = []
        target_refine = []

        num_pos_init = 0
        num_pos_refine = 0

        for img, (per_center_pts, cls_prob, pts_init, pts_refine, per_targets) in enumerate(
                zip(center_pts, cls_outs, pts_outs_init, pts_outs_refine, targets)):
            assert per_center_pts.shape[:-1] == cls_prob.shape[:-1]

            gt_bboxes = per_targets.gt_boxes.to(cls_prob.device)
            gt_labels = per_targets.gt_classes.to(cls_prob.device)

            pts_init_bbox_targets, pts_init_labels_targets = \
                self.point_targets(per_center_pts, pts_strides, gt_bboxes.tensor, gt_labels)

            # per_center_pts, shape:[N, 18]
            per_center_pts_repeat = per_center_pts.repeat(1, self.num_points)

            normalize_term = self.point_base_scale * pts_strides
            normalize_term = normalize_term.reshape(-1, 1)

            # bbox_center = torch.cat([per_center_pts, per_center_pts], dim=1)
            per_pts_strides = pts_strides.reshape(-1, 1)
            pts_init_coordinate = pts_init * per_pts_strides + \
                                  per_center_pts_repeat
            init_bbox_pred = self.pts_to_bbox(pts_init_coordinate)

            foreground_idxs = (pts_init_labels_targets >= 0) & \
                              (pts_init_labels_targets != self.num_classes)

            pred_init.append(init_bbox_pred[foreground_idxs] / normalize_term[foreground_idxs])
            target_init.append(pts_init_bbox_targets[foreground_idxs] / normalize_term[foreground_idxs])
            num_pos_init += foreground_idxs.sum()

            # A another way to convert predicted offset to bbox
            # bbox_pred_init = self.pts_to_bbox(pts_init.detach()) * \
            #     per_pts_strides
            # init_bbox_pred = bbox_center + bbox_pred_init

            pts_refine_bbox_targets, pts_refine_labels_targets = \
                self.bbox_targets(init_bbox_pred, gt_bboxes, gt_labels)

            pts_refine_coordinate = pts_refine * per_pts_strides + per_center_pts_repeat
            refine_bbox_pred = self.pts_to_bbox(pts_refine_coordinate)

            # bbox_pred_refine = self.pts_to_bbox(pts_refine) * per_pts_strides
            # refine_bbox_pred = bbox_center + bbox_pred_refine

            foreground_idxs = (pts_refine_labels_targets >= 0) & \
                              (pts_refine_labels_targets != self.num_classes)

            pred_refine.append(refine_bbox_pred[foreground_idxs] / normalize_term[foreground_idxs])
            target_refine.append(pts_refine_bbox_targets[foreground_idxs] / normalize_term[foreground_idxs])
            num_pos_refine += foreground_idxs.sum()

            gt_classes_target = torch.zeros_like(cls_prob)
            gt_classes_target[foreground_idxs, pts_refine_labels_targets[foreground_idxs]] = 1
            pred_cls.append(cls_prob)
            target_cls.append(gt_classes_target)

        pred_cls = torch.cat(pred_cls, dim=0)
        pred_init = torch.cat(pred_init, dim=0)
        pred_refine = torch.cat(pred_refine, dim=0)

        target_cls = torch.cat(target_cls, dim=0)
        target_init = torch.cat(target_init, dim=0)
        target_refine = torch.cat(target_refine, dim=0)

        loss_cls = sigmoid_focal_loss_jit(
            pred_cls,
            target_cls,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum") / max(1, num_pos_refine.item()) * self.loss_cls_weight

        loss_pts_init = smooth_l1_loss(
            pred_init, target_init, beta=0.11, reduction='sum') / max(
            1, num_pos_init.item()) * self.loss_loc_init_weight

        loss_pts_refine = smooth_l1_loss(
            pred_refine, target_refine, beta=0.11, reduction='sum') / max(
            1, num_pos_refine.item()) * self.loss_loc_refine_weight

        return {
            "loss_cls": loss_cls,
            "loss_pts_init": loss_pts_init,
            "loss_pts_refine": loss_pts_refine
        }

    def pts_to_bbox(self, points):
        """
        Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_x = points[:, 0::2]
        pts_y = points[:, 1::2]

        if self.transform_method == "minmax":
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_top = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_top, bbox_right, bbox_bottom], dim=1)
        elif self.transform_method == "partial_minmax":
            bbox_left = pts_x[:, :4].min(dim=1, keepdim=True)[0]
            bbox_right = pts_x[:, :4].max(dim=1, keepdim=True)[0]
            bbox_top = pts_y[:, :4].min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y[:, :4].max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_top, bbox_right, bbox_bottom], dim=1)
        elif self.transform_method == "moment":
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_std = pts_x.std(dim=1, keepdim=True)
            pts_y_std = pts_y.std(dim=1, keepdim=True)
            moment_transfer = self.moment_transfer * self.moment_mul + \
                              self.moment_transfer.detach() * (1 - self.moment_mul)
            moment_transfer_width = moment_transfer[0]
            moment_transfer_height = moment_transfer[1]
            half_width = pts_x_std * moment_transfer_width.exp()
            half_height = pts_y_std * moment_transfer_height.exp()
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ], dim=1)
        else:
            raise ValueError

        return bbox

    @torch.no_grad()
    def point_targets(self, points, pts_strides, gt_bboxes, gt_labels):
        """
        Target assign: point assign. Compute corresponding GT box and classification targets
        for proposals.
        Args:
            points: pred boxes
            pts_strides: boxes stride of current point(box)
            gt_bboxes: gt boxes
            gt_labels: gt labels
        Returns:
            assigned_bboxes, assigned_labels
        """
        if points.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        points_lvl = torch.log2(pts_strides).int()
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()
        num_gts, num_points = gt_bboxes.shape[0], points.shape[0]

        # assign gt box
        gt_bboxes_ctr_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)

        scale = self.point_base_scale

        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        assigned_gt_inds = points.new_zeros((num_points,), dtype=torch.long)
        assigned_gt_dist = points.new_full((num_points,), float('inf'))
        points_range = torch.arange(points.shape[0])

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            lvl_points = points[lvl_idx, :]
            gt_point = gt_bboxes_ctr_xy[[idx], :]
            gt_wh = gt_bboxes_wh[[idx], :]

            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            min_dist, min_dist_index = torch.topk(points_gt_dist, 1, largest=False)
            min_dist_points_index = points_index[min_dist_index]
            less_than_recorded_index = min_dist < assigned_gt_dist[min_dist_points_index]
            min_dist_points_index = min_dist_points_index[less_than_recorded_index]

            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[less_than_recorded_index]

        assigned_bboxes = gt_bboxes.new_zeros((num_points, 4))
        assigned_labels = gt_labels.new_full((num_points,), self.num_classes)

        pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze().long()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
            assigned_bboxes[pos_inds] = gt_bboxes[assigned_gt_inds[pos_inds] - 1]

        return assigned_bboxes, assigned_labels

    @torch.no_grad()
    def bbox_targets(self,
                     candidate_bboxes,
                     gt_bboxes,
                     gt_labels,
                     pos_iou_thr=0.5,
                     neg_iou_thr=0.4,
                     gt_max_matching=True):
        """
        Target assign: MaxIoU assign
        Args:
            candidate_bboxes:
            gt_bboxes:
            gt_labels:
            pos_iou_thr:
            neg_iou_thr:
            gt_max_matching:
        Returns:
        """
        if candidate_bboxes.size(0) == 0 or gt_bboxes.tensor.size(0) == 0:
            raise ValueError('No gt or anchors')

        candidate_bboxes[:, 0].clamp_(min=0)
        candidate_bboxes[:, 1].clamp_(min=0)
        candidate_bboxes[:, 2].clamp_(min=0)
        candidate_bboxes[:, 3].clamp_(min=0)

        num_candidates = candidate_bboxes.size(0)

        overlaps = pairwise_iou(Boxes(candidate_bboxes), gt_bboxes)
        assigned_labels = overlaps.new_full((overlaps.size(0),), self.num_classes, dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=0)

        bg_inds = max_overlaps < neg_iou_thr
        assigned_labels[bg_inds] = self.num_classes

        fg_inds = max_overlaps >= pos_iou_thr
        assigned_labels[fg_inds] = gt_labels[argmax_overlaps[fg_inds]]

        if gt_max_matching:
            fg_inds = torch.nonzero(overlaps == gt_max_overlaps)[:, 0]
            assigned_labels[fg_inds] = gt_labels[argmax_overlaps[fg_inds]]

        assigned_bboxes = overlaps.new_zeros((num_candidates, 4))

        fg_inds = (assigned_labels >= 0) & (assigned_labels != self.num_classes)
        assigned_bboxes[fg_inds] = gt_bboxes.tensor[argmax_overlaps[fg_inds]]

        return assigned_bboxes, assigned_labels

    def inference(self, center_pts, cls_outs, pts_outs_init, pts_outs_refine,
                  images):
        """
        Argumments:
            cls_outs, pts_outs_init, pts_outs_refine:
                Same as the output of :`RepPointsHead.forward`
            center_pts: (list[list[Tensor]]): a list of N=#image elements. Each
                is a list of #feature level tensors. The tensors contains
                shifts of this image on the specific feature level.
        Returns:
            results (List[Instances]): a list of #images elements
        """
        assert len(center_pts) == len(images)
        results = []
        cls_outs = [
            x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
            for x in cls_outs
        ]
        # pts_outs_init = [
        #     x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_points * 2)
        #     for x in pts_outs_init
        # ]
        pts_outs_refine = [
            x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_points * 2)
            for x in pts_outs_refine
        ]

        pts_strides = []
        for i, s in enumerate(center_pts[0]):
            pts_strides.append(cls_outs[0].new_full((s.size(0),), self.fpn_strides[i]))
        # pts_strides = torch.cat(pts_strides, dim=0)

        for img_idx, center_pts_per_image in enumerate(center_pts):
            image_size = images.image_sizes[img_idx]
            cls_outs_per_img = [
                cls_outs_per_level[img_idx] for cls_outs_per_level in cls_outs
            ]
            pts_outs_refine_per_img = [
                pts_outs_refine_per_level[img_idx]
                for pts_outs_refine_per_level in pts_outs_refine
            ]
            results_per_img = self.inference_single_image(
                cls_outs_per_img, pts_outs_refine_per_img, pts_strides,
                center_pts_per_image, tuple(image_size))
            results.append(results_per_img)
        return results

    def inference_single_image(self, cls_logits, pts_refine, pts_strides, points, image_size):
        """
        Single-image inference. Return bounding-box detection results by
        thresholding on scores and applying non-maximum suppression (NMS).
        Arguemnts:
            cls_logits (list[Tensor]): list of #feature levels. Each entry
                contains tensor of size (H x W, K)
            pts_refine (list[Tensor]): Same shape as 'cls_logits' except that K
                becomes 2 * num_points.
            pts_strides (list(Tensor)): list of #feature levels. Each entry
                contains tensor of size (H x W, )
            points (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the points for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.
        Returns:
            Same as `inference`, but only for one image
        """
        assert len(cls_logits) == len(pts_refine) == len(pts_strides)
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for cls_logits_i, pts_refine_i, points_i, pts_strides_i in zip(
                cls_logits, pts_refine, points, pts_strides):
            bbox_pos_center = torch.cat([points_i, points_i], dim=1)
            bbox_pred = self.pts_to_bbox(pts_refine_i)
            bbox_pred = bbox_pred * pts_strides_i.reshape(-1, 1) + bbox_pos_center
            bbox_pred[:, 0].clamp_(min=0, max=image_size[1])
            bbox_pred[:, 1].clamp_(min=0, max=image_size[0])
            bbox_pred[:, 2].clamp_(min=0, max=image_size[1])
            bbox_pred[:, 3].clamp_(min=0, max=image_size[0])

            # (HxWxK, )
            point_cls_i = cls_logits_i.flatten().sigmoid_()

            # keep top k scoring indices only
            num_topk = min(self.topk_candidates, point_cls_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = point_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            point_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            predicted_boxes = bbox_pred[point_idxs]

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]

        return result
