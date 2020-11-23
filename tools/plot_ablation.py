from fire import Fire
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np

from concern.support import fig2image
from concern import webcv2

Path = mpath.Path


def wise(v):
    if v == 1:
        return "CCW"
    else:
        return "CW"


def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))


inside_vertices = make_circle(0.5)
outside_vertices = make_circle(1.0)
codes = np.ones(len(inside_vertices),
                dtype=mpath.Path.code_type) * mpath.Path.LINETO
codes[0] = mpath.Path.MOVETO
vertices = np.concatenate((outside_vertices[::1], inside_vertices[::1]))
# The codes will be all "LINETO" commands, except for "MOVETO"s at the
# Create the Path object
all_codes = np.concatenate((codes, codes))
# Create the Path object
path = mpath.Path(vertices, all_codes)
# path = mpatches.PathPatch(path, facecolor='#885500', edgecolor='black')


def separator(ax, x):
    ax.axvline(x=x, linestyle="-", linewidth=0.5, color="0.0")
    x += 1
    return x


def main(file_path="train_log/Ablation.csv"):
    csv = pd.read_csv(file_path)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    tax = ax.twinx()
    ax.set_ylim(37, 40)
    tax.set_ylim(0, 60)
    plt.style.use(['ieee', "no-latex"])
    ax.set_ylabel("COCO mAP")

    c = (0, 200 / 255, 0, 0.7)

    tax.set_ylabel("COCO+ mAR", c="g")

    x = 1
    xs = []
    baselines = []
    baseline_names = ["original RepPoints", "original FCOS", "original RetinaNet"]
    # baseline_names = ["original RepPoints", "original RetinaNet"]

    # RepPoints
    rows = [9, 8, 7, 5]
    base = 8
    for r in rows:
        if r == base:
            baselines.append(ax.scatter(x,
                       csv["mAP"][r],
                       marker="o",
                       s=64))
            ax.scatter(x,
                       csv["mAP"][r],
                       marker=".",
                       s=4,
                       c="k")
        else:
            if r == rows[-1]:
                ours = ax.scatter(x,
                           csv["mAP"][r],
                           marker="*",
                           color="red",
                           s=64)
            else:
                ax.scatter(x,
                           csv["mAP"][r],
                           marker="o",
                           s=64,
                           facecolor="#cccccc",
                           edgecolor="#cccccc")
            ax.scatter(x,
                       csv["mAP"][r],
                       marker=".",
                       s=4,
                       c="k")
        tax.scatter(x, csv["AR0-0.2@100 1"][r], marker="_", c=c)
        tax.scatter(x, csv["AR0.2-0.3*@100 1"][r], marker="_", c=c)
        tax.scatter(x, csv["AR0.3*-1@100 1"][r], marker="_", c=c)
        tax.plot([x, x, x], [
            csv["AR0-0.2@100 1"][r], csv["AR0.2-0.3*@100 1"][r],
            csv["AR0.3*-1@100 1"][r]
        ], c=c, linestyle="--", linewidth=0.5)
        xs.append(x)
        x += 1

    x = separator(ax, x)

    # FCOS
    rows = [16, 14, 13, 11]
    base = 16
    for r in rows:
        if r == base:
            baselines.append(ax.scatter(x,
                       csv["mAP"][r],
                       marker="o",
                       s=64))
            ax.scatter(x,
                       csv["mAP"][r],
                       marker=".",
                       s=4,
                       c="k")
        else:
            if r == rows[-2]:
                ours = ax.scatter(x,
                           csv["mAP"][r],
                           marker="*",
                           color="red",
                           s=64)
            else:
                ax.scatter(x,
                           csv["mAP"][r],
                           marker="o",
                           s=64,
                           facecolor="#cccccc",
                           edgecolor="#cccccc")
            ax.scatter(x,
                       csv["mAP"][r],
                       marker=".",
                       s=4,
                       c="k")
        tax.scatter(x, csv["AR0-0.2@100 1"][r], marker="_", c=c)
        tax.scatter(x, csv["AR0.2-0.3*@100 1"][r], marker="_", c=c)
        tax.scatter(x, csv["AR0.3*-1@100 1"][r], marker="_", c=c)
        tax.plot([x, x, x], [
            csv["AR0-0.2@100 1"][r], csv["AR0.2-0.3*@100 1"][r],
            csv["AR0.3*-1@100 1"][r]
        ], c=c, linestyle="--", linewidth=0.5)
        xs.append(x)
        x += 1

    x = separator(ax, x)

    # RetinaNet
    rows = [15, 3, 2, 0]
    base = 15
    for r in rows:
        if r == base:
            baselines.append(ax.scatter(x,
                       csv["mAP"][r],
                       marker="o",
                       s=64))
            ax.scatter(x,
                       csv["mAP"][r],
                       marker=".",
                       s=4,
                       c="k")
        else:
            if r == rows[-1]:
                ours = ax.scatter(x,
                           csv["mAP"][r],
                           marker="*",
                           color="red",
                           s=64)
            else:
                ax.scatter(x,
                           csv["mAP"][r],
                           marker="o",
                           s=64,
                           facecolor="#cccccc",
                           edgecolor="#cccccc")
            ax.scatter(x,
                       csv["mAP"][r],
                       marker=".",
                       s=4,
                       c="k")
        tax.scatter(x, csv["AR0-0.2@100 1"][r], marker="_", c=c)
        tax.scatter(x, csv["AR0.2-0.3*@100 1"][r], marker="_", c=c)
        tax.scatter(x, csv["AR0.3*-1@100 1"][r], marker="_", c=c)
        tax.plot([x, x, x], [
            csv["AR0-0.2@100 1"][r], csv["AR0.2-0.3*@100 1"][r],
            csv["AR0.3*-1@100 1"][r]
        ], c=c, linestyle="--", linewidth=0.5)
        xs.append(x)
        x += 1

    x = separator(ax, x)

    # SOTA
    sota_names = ["FSAF", "FasterRCNN", "FreeAnchor", "ATSS", "AugFPN"]
    sota_aps  = [37.2, 38.3, 38.7, 39.3, 39.4]

    for name, ap in zip(sota_names, sota_aps):
        ax.scatter(x,
                   ap,
                   marker="o",
                   s=64,
                   facecolor="#cccccc",
                   edgecolor="#cccccc")
        ax.scatter(x,
                   ap,
                   marker=".",
                   s=4,
                   c="k")
        ax.annotate(name, (x + 0.2, ap - 0.02))
        x += 0.5

    ax.legend(
        handles=baselines + [ours],
        labels=baseline_names + ["our recommendations"],
        loc=(0.02, 1.02),
        ncol=len(baselines) + 1)

    tax.tick_params(axis='y', labelcolor="g")
    ax.set_xlim(0, x + 1)
    ax.set_xticks(xs)
    ax.set_xlabel("Feature Adaption")
    # ax.set_xticklabels(
    #     ["I+L", "I+J+L", "I+J", "I+K"]
    #     + ["", "I", "J", "K"]
    #     + ["", "I", "J", "K"]
    #     )
    ax.set_xticklabels(
        ["M1", "M2", "M3", "M4"]
        + ["M5", "M6", "M7", "M8"]
        + ["M9", "M10", "M11", "M12"]
    )
    # ax.text("RepPoints", 0.2, )
    ax.grid(which="both", linestyle="--", axis="y")
    ax.text(2-0.2, 39.9, "RepPoints")
    ax.text(7+0.1, 39.9, "FCOS")
    ax.text(12, 39.9, "RetinaNet")
    ax.text(17+0.1, 39.9, "SOTA")
    fig.savefig("train_log/Ablation.pdf")
    image = fig2image(fig)
    webcv2.imshow("", image)
    webcv2.waitKey()
    import ipdb
    ipdb.set_trace()


Fire(main)
