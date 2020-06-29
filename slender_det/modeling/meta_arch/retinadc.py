import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

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

__all__ = ["DeformableConvRetinaNet","DeformableConvRetinaNetHead"]

@META_ARCH_REGISTRY.register()
class DeformableConvRetinaNet(RetinaNet):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = DeformableConvRetinaNetHead(cfg, feature_shapes)
        
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

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
            
            
class DeformableConvRetinaNetHead(nn.Module):

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        #the same as RetinaNetHead, we replace the cls_score net to logits net, which utilizes the deform_conv
        # fmt: off
        in_channels      = input_shape[0].channels
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs        = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
#        self.cls_score = nn.Conv2d(
#            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
#        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        
        
        #Deform_conv block, added as a second stage refinement. The implementation follows reppoints.
        self.dcn_kernel = 3
        self.dcn_pad = 1
        self.point_base_scale = 4
        self.gradient_mul = 0.1
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("dcn_base_offset", dcn_base_offset)
        
        self.deform_cls_conv = DeformConv(
            self.in_channels,
            self.in_channels,
            self.dcn_kernel, 1, self.dcn_pad)
        self.deform_reg_conv = DeformConv(
            self.in_channels,
            self.in_channels,
            self.dcn_kernel, 1, self.dcn_pad)
        self.offsets_refine = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.num_anchors*self.in_channels,
                      num_anchors * 4,
                      1, 1, 0))
        self.logits = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.num_anchors*self.in_channels,
                      num_anchors * num_classes,
                      1, 1, 0))
                      
        bias_init = float(-np.log((1 - 0.01) / 0.01))
        for modules in [
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

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        
        logits = []
        offsets_final = []
        

        cls_features = [self.cls_subnet(f) for f in features]
        reg_features = [self.bbox_subnet(f) for f in features]
        
        offsets_init = [self.bbox_pred(f) for f in reg_features]#[dx,dy,dw,dh], [B,C=num_anchors * 4,H,W]
        #transform [dx,dy,dw,dh] to [dx1,dy1,dx2,dy2]
        offsets_init_xyxy = []
        scale = self.point_base_scale / 2
        points_init = self.dcn_base_offset / self.dcn_base_offset.max() * scale
        # original bbox_init is not suitable to retina, because of the different anchor size
        #and ratio, so bbox_init should be initalized by 9 shapes.
        
        #this code is from detectron2.modeling.anchor_generator.DefaultAnchorGenerator.generate_cell_anchors
        # all fpn feature share the same size!! because the bbox_init is only used for dcn offset computation
        sizes = [2, 2 * 2**(1.0/3), 2 * 2**(2.0/3)]
        aspect_ratios = [0.5, 1.0, 2.0]
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        anchors = features[0].new_tensor(anchors)
        offset_stride=[1,2,4,8,16]
        for i in range(len(features)):
            box_reg_i = offsets_init[i]
            b, c, h, w = box_reg_i.shape #c=4*num_anchors
            if c != 4:
                reg=box_reg_i.view(-1,4,h,w)
                bbox_init = anchors.repeat(b,1).view(-1,4,1,1)
            else:
                reg=box_reg_i
                bbox_init = anchors.view(-1,4,1,1)
            box_reg_i_xyxy, bbox_out_init = self.gen_grid_from_reg(reg, bbox_init.detach()) #box_reg_i_xyxy: [b,18,h,w]
            offsets_init_xyxy.append(box_reg_i_xyxy)
            offsets_init_grad_mul = (1 - self.gradient_mul) * offsets_init_xyxy[i].detach()\
                + self.gradient_mul * offsets_init_xyxy[i]
            # b*9, C=18, H, W --> N, C/2, 2(x, y), H, W --> N, C/2, 2(y, x), H, W
            N, C, H, W = offsets_init_grad_mul.shape #N is batch_size*num_anchors
            batch_size=N//self.num_anchors
            offsets_init_grad_mul = offsets_init_grad_mul.reshape(
                offsets_init_grad_mul.size(0),
                -1, 2,
                *offsets_init_grad_mul.shape[-2:]
            ).flip(2)
            offsets_init_grad_mul = offsets_init_grad_mul.reshape(
                -1, 18, *offsets_init_grad_mul.shape[-2:])

            dcn_offset = (offsets_init_grad_mul - self.dcn_base_offset)
            deform_cls_conv = []
            deform_reg_conv = []
            for conv_i in range(self.num_anchors):
                deform_cls_conv.append(self.deform_cls_conv(cls_features[i], dcn_offset[conv_i*batch_size:(conv_i+1)*batch_size,:,:,:]))
                deform_reg_conv.append(self.deform_reg_conv(reg_features[i], dcn_offset[conv_i*batch_size:(conv_i+1)*batch_size,:,:,:]))
            cat_cls = torch.cat(deform_cls_conv, 1) # b,num_anchors*inchannel,h,w
            cat_reg = torch.cat(deform_reg_conv, 1) # b,num_anchors*inchannel,h,w
            logits.append(self.logits(cat_cls)) # b,num_anchors*classes,h,w
            offsets_refine_delta = self.offsets_refine(cat_reg) # b,num_anchors*4,h,w
            #merge the two stage offset.
            offset1 = box_reg_i.view(-1,4,h,w)
            offset2 = offsets_refine_delta.view(-1,4,h,w)
            offsets_final_delta_x = offset1[:,[0],:,:] + torch.exp(offset1[:,[2],:,:]) * offset2[:,[0],:,:]
            offsets_final_delta_y = offset1[:,[1],:,:] + torch.exp(offset1[:,[3],:,:]) * offset2[:,[1],:,:]
            offsets_final_delta_w = offset1[:,[2],:,:] + offset2[:,[2],:,:]
            offsets_final_delta_h = offset1[:,[3],:,:] + offset2[:,[3],:,:]
            offsets_final_delta = torch.cat([offsets_final_delta_x,offsets_final_delta_y,offsets_final_delta_w,offsets_final_delta_h], dim=1)
            offsets_final_delta = offsets_final_delta.view(-1,c,h,w) * offset_stride[i]
            offsets_final.append(offsets_final_delta)

         #retinahead version      
#        for feature in features:
#            logits.append(self.cls_score(self.cls_subnet(feature)))
#            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, offsets_final


    def gen_grid_from_reg(self, reg, previous_boxes):
        """
        Base on the previous bboxes and regression values, we compute the
            regressed bboxes and generate the grids on the bboxes.
        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, c, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox
