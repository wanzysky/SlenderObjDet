import pickle

from fire import Fire
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from concern.smart_path import smart_path
from concern.support import fig2image
from concern import webcv2


def scatter_with_markers(handler, xs, ys, markers, *args, **kwargs):
    assert len(xs) == len(markers)
    assert len(ys) == len(markers)

    handler.plot([xs[0], xs[-1]], [ys[0], ys[0]], "-", linewidth=0.5, color="gray")
    for x, y, m in zip(xs, ys, markers):
        handler.scatter(x, y, *args, marker=m, **kwargs)


def title(ax, content, x=0.5, y=0.9):
    ax.set_title(content, x=x, y=y)


def main(
        result_path="train_log/retinanet/R_50_FPN/baseline/inference/instances_predictions_coco.pkl",
        result_path_2="../train_log/retinanet/R_50_FPN/baseline/inference/coco_objects365_val_with_masks/instances_predictions.pkl"):
    result_path = smart_path(result_path)
    results = pickle.loads(result_path.read_bytes())

    result_path_2 = smart_path(result_path_2)
    results_2 = pickle.loads(result_path_2.read_bytes())
    # shape (T, K, R, A), where T, K, R, A are number of thresholds,
    # classes, ratios, areas, respectively.
    ar = results["ar"]
    stats = results["ar"]["ar-stats"]["recalls"] * 100
    ar_2 = results_2["ar"]
    stats_2 = results_2["ar"]["ar-stats"]["recalls"] * 100
    plt.style.use(['ieee', "no-latex"])
    fig, axs = plt.subplots(3, 3, sharey=True, figsize=(9, 9))
    markers = ['*', '_', '+', 'x']

    # plot overall AR
    xs = np.arange(4) + 1
    ys = np.array([ar["AR@100"],
         ar["AR- 0  - 1/5@100"],
         ar["AR-1/5 - 1/3@100"],
         ar["AR-1/3 - 3/1@100"]])
    scatter_with_markers(axs[0, 0], xs, ys, markers, c="r")
    axs[0, 0].plot(xs, ys, linestyle="dotted")

    axs[0, 0].set_xticks(xs)
    title(axs[0, 0], "all objects")
    axs[0, 0].set_ylabel("mAR-COCO")
    axs[0, 0].set_xlabel("slenderness")
    axs[0, 0].set_xticklabels(["all", "XS", "S", "R"])

    # merge
    [ax.remove() for ax in axs[0, 1:]]
    ax = fig.add_subplot(axs[0, 2].get_gridspec()[0, 1:])
    
    # plot thresholds
    T = stats.shape[0]
    xs = np.array([1])
    x_labels = []
    thresh  = 0.5
    stride = 0.05
    xticks = []
    for i in range(T):
        ys = stats[i, :-1, 0:4, 0].mean(0)
        xs = np.arange(4) + xs.max() + 1
        scatter_with_markers(ax, xs, ys, markers, c='r')
        ax.plot(xs, ys, c='black', linestyle="dotted")
        x_labels += ["", str(int(thresh * 100) / 100), "", ""]
        thresh += stride
        xticks.append(xs)

    ax.set_xticks(np.concatenate(xticks))
    ax.set_xticklabels(x_labels)
    title(ax, "all objects")
    ax.set_xlabel("threshold")
    ax.legend(
        [Line2D([0], [0], color="b", linewidth=1, linestyle="none", marker=m, c="r")
            for m in markers],
        ["all", "XS", "S", "R"], loc="upper right")

    # plot overall AR
    xs = np.arange(4) + 1
    ys = np.array([ar_2["AR@100"],
         ar_2["AR- 0  - 1/5@100"],
         ar_2["AR-1/5 - 1/3@100"],
         ar_2["AR-1/3 - 3/1@100"]])
    ax = axs[1, 0]
    scatter_with_markers(ax, xs, ys, markers, c="g")
    ax.plot(xs, ys, linestyle="dotted")

    ax.set_xticks(xs)
    title(ax, "all objects")
    ax.set_ylabel("mAR-COCO+")
    ax.set_xlabel("slenderness")
    ax.set_xticklabels(["all", "XS", "S", "R"])

    # merge
    [ax.remove() for ax in axs[1, 1:]]
    ax = fig.add_subplot(axs[1, 2].get_gridspec()[1, 1:])
    
    # plot thresholds
    T = stats.shape[0]
    xs = np.array([1])
    x_labels = []
    thresh  = 0.5
    stride = 0.05
    xticks = []
    for i in range(T):
        ys = stats_2[i, :-1, 0:4, 0].mean(0)
        xs = np.arange(4) + xs.max() + 1
        scatter_with_markers(ax, xs, ys, markers, c='g')
        ax.plot(xs, ys, c='black', linestyle="dotted")
        x_labels += ["", str(int(thresh * 100) / 100), "", ""]
        thresh += stride
        xticks.append(xs)

    ax.set_xticks(np.concatenate(xticks))
    ax.set_xticklabels(x_labels)
    title(ax, "all objects")
    ax.set_xlabel("threshold")
    ax.legend(
        [Line2D([0], [0], color="b", linewidth=1, linestyle="none", marker=m, c="g")
            for m in markers],
        ["all", "XS", "S", "R"], loc="upper right")

    # plot small objects
    ax = axs[2, 0]
    xs = np.arange(4)
    ys = np.array([stats[:, :-1, 0, 1].mean(),
        stats[:, :-1, 1, 1].mean(),
        stats[:, :-1, 2, 1].mean(),
        stats[:, :-1, 3, 1].mean()])
    ys_2 = np.array([stats_2[:, :-1, 0, 1].mean(),
        stats_2[:, :-1, 1, 1].mean(),
        stats_2[:, :-1, 2, 1].mean(),
        stats_2[:, :-1, 3, 1].mean()])
    # scatter_with_markers(ax, xs, ys, markers, c='g')
    scatter_with_markers(ax, xs, ys_2, markers, c='g')
    # ax.plot(xs, ys, c="g", linestyle="dotted")
    ax.plot(xs, ys_2, c="black", linestyle="dotted")
    ax.set_xticks(xs)
    title(ax, "small objects")
    ax.set_ylabel("mAR-COCO+")
    ax.set_xlabel("slenderness")
    ax.set_xticklabels(["all", "XS", "S", "R"])
    
    # plot medium objects
    ax = axs[2, 1]
    xs = np.arange(4)
    ys = np.array([stats[:, :-1, 0, 2].mean(),
        stats[:, :-1, 1, 2].mean(),
        stats[:, :-1, 2, 2].mean(),
        stats[:, :-1, 3, 2].mean()])
    ys_2 = np.array([stats_2[:, :-1, 0, 2].mean(),
        stats_2[:, :-1, 1, 2].mean(),
        stats_2[:, :-1, 2, 2].mean(),
        stats_2[:, :-1, 3, 2].mean()])
    # scatter_with_markers(ax, xs, ys, markers, c='g')
    scatter_with_markers(ax, xs, ys_2, markers, c='g')
    # ax.plot(xs, ys, c="g", linestyle="dotted")
    ax.plot(xs, ys_2, c="black", linestyle="dotted")
    ax.set_xticks(xs)
    title(ax, "medium objects")
    ax.set_xlabel("slenderness")
    ax.set_xticklabels(["all", "XS", "S", "R"])

    # plot large objects
    ax = axs[2, 2]
    xs = np.arange(4)
    ys = np.array([stats[:, :-1, 0, 3].mean(),
        stats[:, :-1, 1, 3].mean(),
        stats[:, :-1, 2, 3].mean(),
        stats[:, :-1, 3, 3].mean()])
    ys_2 = np.array([stats_2[:, :-1, 0, 3].mean(),
        stats_2[:, :-1, 1, 3].mean(),
        stats_2[:, :-1, 2, 3].mean(),
        stats_2[:, :-1, 3, 3].mean()])
    # scatter_with_markers(ax, xs, ys, markers, c='g')
    scatter_with_markers(ax, xs, ys_2, markers, c='g')
    # ax.plot(xs, ys, c="g", linestyle="dotted")
    ax.plot(xs, ys_2, c="black", linestyle="dotted")
    ax.set_xticks(xs)
    title(ax, "large objects")
    ax.set_xlabel("slenderness")
    ax.set_xticklabels(["all", "XS", "S", "R"])

    group = fig2image(fig)
    webcv2.imshow("group", group)
    webcv2.waitKey()



Fire(main)
