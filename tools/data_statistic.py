from pylab import subplots_adjust
from functools import lru_cache
from collections import defaultdict

import numpy as np
import cv2
import geojson
import boto3
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import DatasetCatalog, MetadataCatalog

from concern.smart_path import smart_path
from concern.support import fig2image, between
from concern import webcv2
from slender_det.evaluation.coco import COCO
from slender_det.engine import BaseTrainer

from ._setup import setup
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
matplotlib.use('Agg')
#IMAGE_PATH = smart_path('s3://wanzhaoyi-oss/xview/raw/train_images')


@lru_cache(maxsize=2)
def get_image(image_id):
    image_path = IMAGE_PATH.joinpath(image_id)
    with image_path.open('rb') as reader:
        data = reader.read()
        image = cv2.imdecode(np.fromstring(data, dtype=np.uint8),
                             cv2.IMREAD_UNCHANGED)
    print('loading', image_id)
    return image


def PlotPie(dataset, ratios):
    dicts = list(DatasetCatalog.get(dataset))

    metadata = MetadataCatalog.get(dataset)
    labels = metadata.thing_classes
    ratios_num = dict()
    for key in ratios.keys():
        ratios_num[key] = 0

    for dic in tqdm(dicts):
        for obj in dic["annotations"]:
            ratio = COCO.compute_ratio(obj, oriented=True)["ratio"]
            for key, ratio_range in ratios.items():
                if between(ratio, ratio_range):
                    ratios_num[key] += 1

    plt.pie(ratios_num.values(), labels=ratios_num.keys(), autopct='%1.2f%%')
    plt.savefig("PieChart.jpg")
    return ratios_num


def PlotPiecewiseBars(dataset, num_subfigs, sorted=False):
    ratios = {
        "0-1/5": [0, 1 / 5],
        "1/5-1/3": [1 / 5, 1 / 3],
        "1/3-1": [1 / 3, 1]
    }
    dicts = list(DatasetCatalog.get(dataset))

    metadata = MetadataCatalog.get(dataset)
    labels = metadata.thing_classes
    num_thing_per_fig = len(labels) // num_subfigs
    bars = dict()
    for key in ratios.keys():
        bars[key] = [0 for _ in range(len(labels))]

    all_ratios = []
    for dic in tqdm(dicts):
        for obj in dic["annotations"]:
            ratio = COCO.compute_ratio(obj, oriented=True)["ratio"]
            for key, ratio_range in ratios.items():
                if between(ratio, ratio_range):
                    bars[key][obj["category_id"]] += 1
            all_ratios.append(ratio)
    slender_ratios = [0 for _ in range(len(labels))]
    for i in range(len(labels)):
        slender_ratios[i] = bars["0-1/5"][i] / \
            (bars["0-1/5"][i]+bars["1/5-1/3"][i]+bars["1/3-1"][i])
    slender_ratios = np.array(slender_ratios)
    if sorted == True:
        sorted_indexes = np.argsort(slender_ratios)
    fig, axes = plt.subplots(num_subfigs, 1)
    for fig_i in range(num_subfigs):
        if sorted == True:
            index_range = sorted_indexes[num_thing_per_fig *
                                         fig_i:num_thing_per_fig * (fig_i + 1)]
        else:
            index_range = np.arrange(num_thing_per_fig * fig_i,
                                     num_thing_per_fig * (fig_i + 1))
        label = []
        for ind_i in index_range:
            label.append(labels[ind_i])

        axes[fig_i].set_yscale("symlog")
        prev = np.zeros((len(label), ))
        for key, bar in bars.items():
            bar = np.array(bar)
            axes[fig_i].bar(label, bar[index_range], bottom=prev, label=key)
            prev = prev + bar[index_range]
        axes[fig_i].legend()
        fig.set_size_inches(15, 15)
        axes[fig_i].set_xticklabels(label, rotation="vertical")
    plt.savefig("barchart.jpg")


def PlotGradientColorBars(dataset,
                          num_subfigs,
                          cmap,
                          sorted=False,
                          include_all=False):
    ratios = {
        "0-1/5": [0, 1 / 5],
        "1/5-1/3": [1 / 5, 1 / 3],
        "1/3-1": [1 / 3, 1]
    }
    dicts = list(DatasetCatalog.get(dataset))

    metadata = MetadataCatalog.get(dataset)
    labels = metadata.thing_classes
    bars = dict()
    for key in ratios.keys():
        bars[key] = [0 for _ in range(len(labels))]

    for dic in tqdm(dicts):
        for obj in dic["annotations"]:
            ratio = COCO.compute_ratio(obj, oriented=True)["ratio"]
            for key, ratio_range in ratios.items():
                if between(ratio, ratio_range):
                    bars[key][obj["category_id"]] += 1
    slender_ratios = np.zeros(len(labels))
    num_all_ratios = np.zeros(len(labels))

    num_slender_ratios = np.array(bars["0-1/5"])

    for i in range(len(labels)):
        slender_ratios[i] = bars["0-1/5"][i] / \
            (bars["0-1/5"][i]+bars["1/5-1/3"][i]+bars["1/3-1"][i])
    for key, bar in bars.items():
        num_all_ratios = num_all_ratios + np.array(bar)

    slender_ratio_all = num_slender_ratios.sum() / num_all_ratios.sum()
    if sorted == True:
        sorted_indexes = np.argsort(slender_ratios)
        min_ratio = slender_ratios[sorted_indexes[0]]
        max_ratio = slender_ratios[sorted_indexes[-1]]
    else:
        sorted_indexes = np.arange(len(labels))
        min_ratio = slender_ratios.min()
        max_ratio = slender_ratios.max()
    if include_all == True:
        slender_ratios = np.append(slender_ratios, slender_ratio_all)
        num_all_ratios = np.append(num_all_ratios, num_all_ratios.sum())
        sorted_indexes = np.insert(sorted_indexes, 0, len(labels))
        labels.append('all')

    fig, axes = plt.subplots(num_subfigs, 1)

    norm = mpl.colors.Normalize(vmin=min_ratio, vmax=max_ratio)
    subplots_adjust(left=0.15,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0,
                    hspace=0.6)
    num_thing_per_fig = len(labels) // num_subfigs
    for fig_i in range(num_subfigs):
        start = num_thing_per_fig * fig_i
        end = num_thing_per_fig * (fig_i + 1)

        if fig_i == num_subfigs - 1:
            end = len(labels)
        index_range = sorted_indexes[start:end]

        label = []
        for ind_i in index_range:
            label.append(labels[ind_i])

        axes[fig_i].set_yscale("symlog")
        prev = np.zeros((len(label), ))
        bar_colors = [cmap(norm(x)) for x in slender_ratios[index_range]]
        axes[fig_i].bar(label,
                        num_all_ratios[index_range],
                        bottom=prev,
                        color=bar_colors,
                        edgecolor=(0, 0, 0))
        axes[fig_i].set_xticklabels(label, rotation="vertical")
    fig.set_size_inches(12, 12)
    position = fig.add_axes([0., 0.1, 0.03, 0.8])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=position,
                 orientation='vertical')
    plt.savefig("barchart.png")


def PlotCOCOAndCOCOPlus(datasets,
                          num_subfigs,
                          cmap,
                          sorted=False,
                          include_all=False,
                          show_part=True):
    ratios = {
        "0-1/5": [0, 1 / 5],
        "1/5-1/3": [1 / 5, 1 / 3],
        "1/3-1": [1 / 3, 1]
    }
    total_slender_ratios = []
    total_num_all_ratios = []
    total_sorted_indexes = []
    total_labels = []
    min_ratios = []
    max_ratios = []
    symbols = ["","+"]
    for dataset_i,dataset in enumerate(datasets):
        
        dicts = list(DatasetCatalog.get(dataset))
    
        metadata = MetadataCatalog.get(dataset)
        labels = metadata.thing_classes
        for i in range(len(labels)):
            labels[i]=labels[i]+symbols[dataset_i]
        bars = dict()
        for key in ratios.keys():
            bars[key] = [0 for _ in range(len(labels))]
    
        for dic in tqdm(dicts):
            for obj in dic["annotations"]:
                ratio = COCO.compute_ratio(obj, oriented=True)["ratio"]
                for key, ratio_range in ratios.items():
                    if between(ratio, ratio_range):
                        bars[key][obj["category_id"]] += 1
        slender_ratios = np.zeros(len(labels))
        num_all_ratios = np.zeros(len(labels))
    
        num_slender_ratios = np.array(bars["0-1/5"])
    
        for i in range(len(labels)):
            slender_ratios[i] = bars["0-1/5"][i] / \
                (bars["0-1/5"][i]+bars["1/5-1/3"][i]+bars["1/3-1"][i])
        for key, bar in bars.items():
            num_all_ratios = num_all_ratios + np.array(bar)
    
        slender_ratio_all = num_slender_ratios.sum() / num_all_ratios.sum()
        if sorted == True:
            sorted_indexes = np.argsort(slender_ratios)
            min_ratio = slender_ratios[sorted_indexes[0]]
            max_ratio = slender_ratios[sorted_indexes[-1]]
        else:
            sorted_indexes = np.arange(len(labels))
            min_ratio = slender_ratios.min()
            max_ratio = slender_ratios.max()
        if include_all == True:
            slender_ratios = np.append(slender_ratios, slender_ratio_all)
            num_all_ratios = np.append(num_all_ratios, num_all_ratios.sum())
            sorted_indexes = np.insert(sorted_indexes, 0, len(labels))
            labels.append('all'+symbols[dataset_i])
        total_slender_ratios.append(slender_ratios)
        total_num_all_ratios.append(num_all_ratios)
        total_sorted_indexes.append(sorted_indexes)
        total_labels.append(labels)
        min_ratios.append(min_ratio)
        max_ratios.append(max_ratio)
        
    if show_part:
        show_num = 10
        show_indexes = np.append(np.arange(0,show_num+1),np.arange(-show_num,0))
    else:
        show_indexes = np.arrange(0,len(total_labels[0]))
    min_ratio = min(min_ratios)
    max_ratio = max(max_ratios)
    
    fig, axes = plt.subplots(num_subfigs, 1)

    norm = mpl.colors.Normalize(vmin=min_ratio, vmax=max_ratio)
    subplots_adjust(left=0.15,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0,
                    hspace=0.6)
    num_thing_per_fig = len(show_indexes) // num_subfigs
    first_row_plus_num = len(show_indexes) % num_subfigs
    show_label = []
    show_slender_ratio = [] 
    show_num_all_ratio = []
    start = 0
    end = 0
    show_x = []
    for fig_i in range(num_subfigs):
        start = end
        end += num_thing_per_fig
        if fig_i == 0:
            end += first_row_plus_num
            
        index_range = total_sorted_indexes[0][show_indexes[start:end]]
        label = []
        slender_ratio = [] 
        num_all_ratio = []
        coor_x = []
        count=0
        for ind_i in index_range:
            for dataset_i in range(len(datasets)):
                label.append(total_labels[dataset_i][ind_i])
                slender_ratio.append(total_slender_ratios[dataset_i][ind_i])
                num_all_ratio.append(total_num_all_ratios[dataset_i][ind_i])
                coor_x.append(count)
                c = 1.5 if dataset_i == len(datasets)-1 else 1
                count += c
        show_label.append(label)
        show_slender_ratio.append(slender_ratio)
        show_num_all_ratio.append(num_all_ratio)
        show_x.append(coor_x)

    plt.tick_params(labelsize=16)
    bar_ec = [(0.73,0.73,0.73,1.0),(0.5,0.5,0.5,0.8)]
    for fig_i in range(num_subfigs): 
        axes[fig_i].set_yscale("symlog")
        prev = np.zeros((len(show_label[fig_i]), ))
        
        bar_colors = [cmap(norm(x)) for x in show_slender_ratio[fig_i]]
        bars = axes[fig_i].bar(show_x[fig_i],
                        show_num_all_ratio[fig_i],
                        bottom=prev,
                        color=bar_colors,
                        edgecolor=bar_ec[fig_i],
                        width=1)
        for i,bar in enumerate(bars):
            if i%2!=0:
                bar.set_hatch('\\')
        xticks = show_x[fig_i][::2]+np.ones(len(show_x[fig_i][::2]))*0.5
        axes[fig_i].set_xticks(xticks)
        if fig_i==1:
            axes[fig_i].xaxis.tick_top()
            axes[fig_i].set_xticklabels(show_label[fig_i][::2], fontdict = {'fontsize':14}, rotation=45)
        else:
            axes[fig_i].set_xticklabels(show_label[fig_i][::2], fontdict = {'fontsize':14}, rotation=315)
        legend_labels = ['COCO', 'COCO+']
        hatches = ['x','\\']
        patches = []
        patches.append(mpatches.Patch(fc='white',label="COCO",ec="grey"))
        patches.append(mpatches.Patch(fc='white',label="COCO+",ec="grey",hatch='\\'))
        if fig_i ==0:
            axes[fig_i].legend(loc="upper center",handles=patches,prop={'size': 24},ncol=2)
    fig.set_size_inches(12, 12)
    position = fig.add_axes([0., 0.1, 0.03, 0.8])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=position,
                 orientation='vertical')
    plt.savefig("coco_coco+.png")
    
    
def PlotAll(dataset):

    dicts = list(DatasetCatalog.get(dataset))

    metadata = MetadataCatalog.get(dataset)
    labels = metadata.thing_classes
    ratios = {
        "0-1/5": [0, 1 / 5],
        "1/5-1/3": [1 / 5, 1 / 3],
        "1/3-1": [1 / 3, 1]
    }

    bars = dict()
    for key in ratios.keys():
        bars[key] = [0 for _ in range(len(labels))]

    all_ratios = []
    ratio_counts = {k: 0 for k in ratios.keys()}
    for dic in tqdm(dicts):
        for obj in dic["annotations"]:
            ratio = COCO.compute_ratio(obj, oriented=True)["ratio"]
            for key, ratio_range in ratios.items():
                if between(ratio, ratio_range):
                    bars[key][obj["category_id"]] += 1
                    ratio_counts[key] += 1
            all_ratios.append(ratio)

    print("images", len(dicts))
    print("counts", ratio_counts)

    fig, ax = plt.subplots()
    ax.set_yscale("symlog")
    prev = np.zeros((len(labels), ))
    for key, bar in bars.items():
        bar = np.array(bar)
        ax.bar(labels, bar, bottom=prev, label=key)
        prev = prev + bar
    ax.legend()
    fig.set_size_inches(18.5, 10.5)
    ax.set_xticklabels(labels, rotation="vertical")
    group = fig2image(fig)
    # cv2.imwrite('./group.png', group)
    webcv2.imshow("group", group)

    fig, ax = plt.subplots()
    all_ratios = sorted(all_ratios)
    numbers = [0]
    tick = 0.01
    seperator = tick
    count = 0

    for r in all_ratios:
        count += 1
        while seperator < r:
            seperator += tick
            numbers.append(count)

    ax.plot(np.arange(0, 1, tick), np.array(numbers) / count)
    ax.set_xlabel("slenderness")
    ax.set_title("cumulative distribution function")
    number = fig2image(fig)
    # cv2.imwrite('./number.png', number)
    webcv2.imshow("number", number)
    webcv2.waitKey()


def main():
    parser = default_argument_parser()
    args = parser.parse_args()
    cfg = setup(args)
    #plot coco and coco+
    datasets = cfg.DATASETS.TEST
    cmap = cm.get_cmap('YlGn')
    color_list = cmap(np.linspace(0,1,256)[::-1])
    new_color1 = color_list[:50:10]
    new_color2 = color_list[50:160:3]
    new_color3 = color_list[160::]
    new_color = np.concatenate((new_color1,new_color2,new_color3),axis=0)
    
    cmap = ListedColormap(new_color)
    PlotCOCOAndCOCOPlus(datasets, 2, cmap,sorted=True,include_all=True,show_part=True)
#    dataset = cfg.DATASETS.TEST[0]
#
#    PlotAll(dataset)


if __name__ == '__main__':
    main()
