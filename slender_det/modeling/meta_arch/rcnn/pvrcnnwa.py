import torch
from typing import Dict, List, Optional, Tuple, Union

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.meta_arch import GeneralizedRCNN, META_ARCH_REGISTRY

from slender_det.modeling.detector_postprocessing_with_anchor import detector_postprocess_with_anchor
from .pvrcnn import ProposalVisibleRCNN


@META_ARCH_REGISTRY.register()
class ProposalVisibleRCNNWithAnchor(ProposalVisibleRCNN):
    def _postprocess(self, instances, proposals, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, proposal_per_image, input_per_image, image_size in zip(
                instances, proposals, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess_with_anchor(results_per_image, height, width)
            processed_results.append(
                {"instances": r, "proposals": detector_postprocess_with_anchor(proposal_per_image, height, width)})
        return processed_results
