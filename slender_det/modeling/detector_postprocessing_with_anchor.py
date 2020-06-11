import torch
import torch.nn.functional as F

from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom


def detector_postprocess_with_anchor(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        raise KeyError("key{pred_boxes/proposal_boxes} not found!"
                       "Please check your output boxes.")

    # add
    if results.has("anchors"):
        valid_mask = torch.isfinite(results.anchors.tensor).all(dim=1)
        if not valid_mask.all():
            print(results.anchors.tensor)
        anchor_boxes = results.anchors
        anchor_boxes.scale(scale_x, scale_y)
        anchor_boxes.clip(results.image_size)
    if results.has("proposals"):
        valid_mask = torch.isfinite(results.proposals.tensor).all(dim=1)
        if not valid_mask.all():
            print(results.proposals.tensor)
        proposal_boxes = results.proposals
        proposal_boxes.scale(scale_x, scale_y)
        proposal_boxes.clip(results.image_size)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results
