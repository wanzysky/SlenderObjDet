import torch


def iou_loss(pred, target, weight=None, loss_type="iou"):
    pred_left, pred_top = pred[:, 0], pred[:, 1]
    pred_right, pred_bottom = pred[:, 2], pred[:, 3]

    target_left, target_top = target[:, 0], target[:, 1]
    target_right, target_bottom = target[:, 2], target[:, 3]

    target_area = (target_left + target_right) * (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)

    ac_uion = g_w_intersect * g_h_intersect + 1e-7
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion
    if loss_type == 'iou':
        losses = -torch.log(ious)
    elif loss_type == 'linear_iou':
        losses = 1 - ious
    elif loss_type == 'giou':
        losses = 1 - gious
    else:
        raise NotImplementedError

    if weight is not None:
        return (losses * weight).sum()
    else:
        assert losses.numel() != 0
        return losses.sum()


def box_iou_loss(pred, target, weight=None, loss_type="iou"):
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_right - target_left) * (target_bottom - target_top)
    pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)

    w_intersect = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
    g_w_intersect = torch.max(pred_right, target_right) - torch.min(pred_left, target_left)
    h_intersect = torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)
    g_h_intersect = torch.max(pred_bottom, target_bottom) - torch.min(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect + 1e-7
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    if loss_type == 'iou':
        losses = -torch.log(ious)
    elif loss_type == 'linear_iou':
        losses = 1 - ious
    elif loss_type == 'giou':
        losses = 1 - gious
    else:
        raise NotImplementedError

    if weight is not None:
        return (losses * weight).sum()
    else:
        assert losses.numel() != 0
        return losses.sum()

def anchor_iou_loss(pred, target, weight=None, num_anchors=9, loss_type="iou"):
    pred_left, pred_top = pred[:, 0], pred[:, 1]
    pred_right, pred_bottom = pred[:, 2], pred[:, 3]

    target_left, target_top = target[:, 0], target[:, 1]
    target_right, target_bottom = target[:, 2], target[:, 3]

    target_area = (target_left + target_right) * (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)

    ac_uion = g_w_intersect * g_h_intersect + 1e-7
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    if loss_type == 'iou':
        losses = -torch.log(ious)
    elif loss_type == 'linear_iou':
        losses = 1 - ious
    elif loss_type == 'giou':
        losses = 1 - gious
    else:
        raise NotImplementedError
    norm_losses = losses.view(-1,num_anchors)
    norm_losses = softmax(1 / norm_losses).view(-1)
    if weight is not None:
        return norm_losses, (losses * weight).sum()
    else:
        assert losses.numel() != 0
        return norm_losses, losses.sum()

def softmax(x):
    x_exp = torch.exp(x)
    x_sum = torch.sum(x_exp, 1)
    return x_exp/x_sum[:,None]