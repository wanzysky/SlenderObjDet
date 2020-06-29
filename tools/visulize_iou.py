import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from concern.support import fig2image
from concern import webcv2


def l2(p1, p2):
    return p1 * p2

def l1(p1, p2):
    return p1 + p2

def iou(box_1, box_2, distance=l2):
    area_1 = distance(box_1[2] - box_1[0], box_1[3] - box_1[1])
    area_2 = distance(box_2[2] - box_2[0], box_2[3] - box_2[1])

    wh = - np.maximum(box_1[:2], box_2[:2]) + np.minimum(box_1[2:], box_2[2:])
    wh = (wh > 0) * wh

    intersection = distance(wh[0], wh[1])
    union = area_1 + area_2 - intersection

    return intersection / union

def vis_boxes(box_1, box_2, steps=100, distance=l2):
    diff = (box_2 - box_1) / 100

    xs = []
    ious = []
    losses = []
    fig, iou_scores = plt.subplots()
    max_val = max(box_1.max(), box_2.max()) * 4
    iou_scores.set_xlim(-0.5, 1.5)
    loss_scores = iou_scores.twinx()
    iou_scores.set_ylabel("IoU with %s" % str(distance))
    loss_scores.set_ylabel("L1 Loss")

    for i in range(steps):
        iou_i = iou(box_1, box_2, distance)
        xs.append(i/steps)
        ious.append(iou_i)
        losses.append(np.abs(box_1 - box_2).sum())

        if i % (steps // 3) == 0:
            patch = plt.Rectangle(
                (i/steps, 0),
                width=(box_1[2] - box_1[0])/max_val,
                height=(box_1[3] - box_1[1])/max_val,
                fill=False)
            iou_scores.add_patch(patch)
            patch = plt.Rectangle(
                (i/steps, 0),
                width=(box_2[2] - box_2[0])/max_val,
                height=(box_2[3] - box_2[1])/max_val,
                fill=False)
            iou_scores.add_patch(patch)

        box_1 += diff
        # examples.add_patch(patch)

    losses = np.array(losses)
    iou_scores.plot(xs, ious, "r", label="IoU")
    loss_scores.plot(xs, losses, "b", label="L1 loss")
    iou_scores.legend(loc="lower right")
    loss_scores.legend()

    image = fig2image(fig)
    webcv2.imshow("image", image)
    webcv2.waitKey()


if __name__ == "__main__":
    b_2 = np.array([0, 0, 10, 80], dtype=np.float)
    b_1 = np.array([0, 0, 20, 20], dtype=np.float)
    vis_boxes(b_1, b_2)

    b_2 = np.array([0, 0, 10, 10], dtype=np.float)
    b_1 = np.array([0, 0, 10, 80], dtype=np.float)
    vis_boxes(b_1, b_2, distance=l1)
