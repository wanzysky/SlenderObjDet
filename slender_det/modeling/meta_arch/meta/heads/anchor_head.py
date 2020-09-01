from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss, giou_loss

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.matcher import Matcher
from detectron2.utils.events import get_event_storage
from detectron2.layers import ShapeSpec, cat, batched_nms
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances

from .meta_head import HeadBase, MEAT_HEADS_REGISTRY
from .utils import permute_to_N_HWA_K


@MEAT_HEADS_REGISTRY.register()
class AnchorHead(HeadBase):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)
        head_params = cfg.MODEL.META_ARCH

        self.box_reg_loss_type = head_params.BBOX_REG_LOSS_TYPE

        # init bbox pred
        self.loc_init_conv = nn.Conv2d(self.feat_channels, self.loc_feat_channels, 3, 1, 1)
        self.loc_init_out = nn.Conv2d(self.loc_feat_channels, self.num_anchor * 4, 1, 1, 0)

        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)
        self.num_anchor = self.anchor_generator.num_cell_anchors
        assert self.num_anchor == 1

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=head_params.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            head_params.IOU_THRESHOLDS,
            head_params.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        # make feature adaptive layer
        self.cls_conv, self.loc_refine_conv = self.make_featre_adaptive_layers()
        self._make_offset()

        self.cls_out = nn.Conv2d(self.feat_channels, self.num_classes, 1, 1, 0)
        self.loc_refine_out = nn.Conv2d(self.loc_feat_channels, self.num_anchor * 4, 1, 1, 0)

        self._init_weights()

        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    def _prepare_offset(self):
        if self.feat_adaptive == "Unsupervised Offset":
            self.offset_conv = nn.Conv2d(self.feat_channels, 18, 1, 1, 0)
        elif self.feat_adaptive == "Supervised Offset":
            self.dcn_kernel = 3
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

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - self.prior_prob) / self.prior_prob))
        nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features, gt_instances, images):
        cls_outs, loc_outs_init, loc_outs_refine = self._forward(features)
        # compute ground truth location (x, y)
        shapes = [feature.shape[-2:] for feature in features]
        anchors = self.anchor_generator(features)

        if self.training:
            return self.losses(anchors, cls_outs, loc_outs_init, loc_outs_refine, gt_instances)
        else:
            results = self.inference(anchors, cls_outs, loc_outs_init, loc_outs_refine, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def _forward(self, features):
        cls_outs = []
        loc_outs_init = []
        loc_outs_refine = []

        dcn_base_offsets = self.dcn_base_offset.type_as(features[0])
        for l, feature in enumerate(features):
            cls_feat = feature
            loc_feat = feature

            for cls_conv in self.cls_subnet:
                cls_feat = cls_conv(cls_feat)
            for loc_conv in self.loc_subnet:
                loc_feat = loc_conv(loc_feat)

            loc_out_init = self.loc_init_out(F.relu_(self.loc_init_conv(loc_feat)))
            loc_outs_init.append(loc_out_init)

            if self.feat_adaption == "None":
                cls_feat_fa = self.cls_conv(cls_feat)
                loc_feat_fa = self.loc_refine_conv(loc_feat)
            elif self.feat_adaption == "Unsupervised Offset":
                # TODO: choose a better input info for generating offsets
                dcn_offsets = self.offset_conv(loc_feat)
                cls_feat_fa = self.cls_conv(cls_feat, dcn_offsets)
                loc_feat_fa = self.loc_refine_conv(loc_feat, dcn_offsets)
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

            loc_out_refine = self.reg_refine_out(F.relu_(loc_feat_fa)) + loc_out_init.detach()
            loc_outs_refine.append(loc_out_refine)

        return cls_outs, reg_outs_init, reg_outs_refine

    def losses(self, anchors, pred_logits, pred_anchor_deltas_init, pred_anchor_deltas, gt_instances):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas_init = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas_init]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[:, :-1]
        # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) * self.loss_cls_weight

        if self.box_reg_loss_type == "smooth_l1":
            loss_loc_init = smooth_l1_loss(
                cat(pred_anchor_deltas_init, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
                beta=self.smooth_l1_loss_beta,
                reduction="sum",
            )
            loss_loc_refine = smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
                beta=self.smooth_l1_loss_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            pred_boxes_init = [
                self.box2box_transform.apply_deltas(k, anchors)
                for k in cat(pred_anchor_deltas_init, dim=1)
            ]
            pred_boxes = [
                self.box2box_transform.apply_deltas(k, anchors)
                for k in cat(pred_anchor_deltas, dim=1)
            ]
            loss_loc_init = giou_loss(
                torch.stack(pred_boxes_init)[pos_mask], torch.stack(gt_boxes)[pos_mask], reduction="sum"
            )
            loss_loc_refine = giou_loss(
                torch.stack(pred_boxes)[pos_mask], torch.stack(gt_boxes)[pos_mask], reduction="sum"
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        return {
            "loss_cls": loss_cls / self.loss_normalizer,
            "loss_loc_init": loss_loc_init / self.loss_normalizer * self.loss_loc_init_weight,
            "loss_loc_refine": loss_loc_refine / self.loss_normalizer * self.loss_loc_refine_weight,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            list[Tensor]:
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """
        anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def inference(self, anchors, class_logits, anchor_deltas_init, anchor_deltas, image_sizes):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
             class_logits, anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        for img_idx, image_size in enumerate(image_sizes):
            class_logits_per_image = [x[img_idx] for x in class_logits]
            deltas_init_per_image = [x[img_idx] for x in anchor_deltas_init]
            deltas_per_image = [x[img_idx] for x in anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, class_logits_per_image,
                deltas_init_per_image, deltas_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors, box_cls, box_delta_init, box_delta, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta_init (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result
