import os
import numpy as np

import matplotlib.figure as mplfigure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _setup():
    data = dict()
    data["tags"] = ["mAP", "mAP0", "mAP1", "mAP2", "mAR", "mAR0", "mAR1", "mAR2", "AR", "AR0", "AR1", "AR2"]
    data["reppoints-50"] = [38.0905, 28.8, 38.3, 37.6, 52.3019, 26.034, 43.425, 53.047, 57.0442, 40.189, 53.105, 59.234]
    data["faster-50"] = [37.8081, 27.7, 37.3, 37.2, 52.2889, 24.546, 41.744, 52.868, 56.1357, 38.225, 51.787, 58.495]
    data["fcos-50"] = [37.6416, 27.4, 37.2, 37.5, 55.5287, 24.412, 43.134, 56.962, 57.8819, 34.712, 51.434, 61.096]
    data["retina-50"] = [36.2174, 26.7, 35.5, 36.3, 53.2837, 22.769, 40.566, 54.607, 57.2151, 35.141, 51.116, 60.268]

    data["reppoints-101"] = []
    data["faster-101"] = [40.0379, 31.8, 39.6, 39.3, 54.0286, 27.234, 43.380, 54.542, 57.5305, 40.538, 53.242, 59.800]
    data["fcos-101"] = [39.7901, 30.0, 38.7, 39.7, 57.2863, 25.471, 43.982, 59.415, 59.1991, 36.790, 52.669, 62.365]
    data["retina-101"] = [38.8686, 28.6, 38.9, 38.9, 55.5379, 24.635, 44.211, 56.915, 58.6286, 37.947, 52.948, 61.484]

    data["reppoints-152"] = []
    data["faster-152"] = [46.5451, 36.1, 46.7, 45.6, 58.9371, 30.471, 49.697, 59.483, 60.7258, 44.947, 56.308, 62.919]
    data["fcos-152"] = [46.7312, 36.7, 46.5, 46.4, 62.0342, 31.221, 51.320, 63.517, 63.1898, 42.747, 57.618, 65.991]
    data["retina-152"] = [45.4311, 34.0, 46.8, 45.0, 60.6029, 28.588, 50.576, 61.809, 62.4203, 43.122, 57.081, 65.092]
    return data


def scatter_with_markers(ax, xs, ys, markers, *args, **kwargs):
    assert len(xs) == len(markers), "{} {}".format(len(xs), len(markers))
    assert len(ys) == len(markers), "{} {}".format(len(ys), len(markers))

    ret = []
    for x, y, m in zip(xs, ys, markers):
        ret.append(ax.scatter(x, y, *args, marker=m, **kwargs))
    return ret


def plot(style=('science', 'no-latex'), save_dir=None):
    data = _setup()

    fname = "plot.png"
    if save_dir is not None:
        fname = os.path.join(save_dir, fname)

    plot_data = dict()

    for name, value in data.items():
        # if 'reppoints' in name:
        #     if 'reppints' not in plot_data:
        #         plot_data['reppoints'] = [value[0]]
        #     else:
        #         plot_data['reppoints'].append(value[0])
        if 'faster' in name:
            if 'faster' not in plot_data:
                plot_data['faster'] = [value]
            else:
                plot_data['faster'].append(value)
        elif 'fcos' in name:
            if 'fcos' not in plot_data:
                plot_data['fcos'] = [value]
            else:
                plot_data['fcos'].append(value)
        elif 'retina' in name:
            if 'retina' not in plot_data:
                plot_data['retina'] = [value]
            else:
                plot_data['retina'].append(value)
        else:
            tags = value

    print(plot_data)

    with plt.style.context(style):
        # fig = mplfigure.Figure()
        # ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], xlabel='backbone', ylabel='mAP')
        # idx = 0
        # ax.plot(
        #     [_[idx] for _ in plot_data['fcos']],
        #     label='focs-all'
        # )
        # ax.plot([_[idx] for _ in plot_data['retina']], label='retina-all')
        # ax.plot([_[idx] for _ in plot_data['faster']], label='faster-all')
        #
        # idx = 1
        # ax.plot([_[idx] for _ in plot_data['fcos']], label='focs-<1/5')
        # ax.plot([_[idx] for _ in plot_data['retina']], label='retina-<1/5')
        # ax.plot([_[idx] for _ in plot_data['faster']], label='faster-<1/5')
        #
        # ax.set_xticks([0, 1, 2, 3])
        # ax.set_xticklabels(["50", "101", "152"])
        # ax.legend()
        # fig.savefig(fname)

        fig = mplfigure.Figure()
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], xlabel='depth', ylabel='mAR')
        markers = ['*', 'd', '.']

        xs = [1, 1, 1]
        ys = [plot_data['faster'][0][i] for i in [5, 6, 7]]
        ret = scatter_with_markers(ax, xs, ys, markers, c="green")
        line1 = ax.plot(xs, ys, linestyle='dotted', color='green')

        xs = [2, 2, 2]
        ys = [plot_data['faster'][1][i] for i in [5, 6, 7]]
        scatter_with_markers(ax, xs, ys, markers, c="green")
        ax.plot(xs, ys, linestyle='dotted', color='green')

        xs = [3, 3, 3]
        ys = [plot_data['faster'][2][i] for i in [5, 6, 7]]
        scatter_with_markers(ax, xs, ys, markers, c="green")
        ax.plot(xs, ys, linestyle='dotted', color='green')

        ##########
        name = 'retina'
        linestyle = 'dotted'
        color = 'red'
        xs = [1.25, 1.25, 1.25]
        ys = [plot_data[name][0][i] for i in [5, 6, 7]]
        scatter_with_markers(ax, xs, ys, markers, c=color)
        line2 = ax.plot(xs, ys, linestyle=linestyle, color=color)

        xs = [2 + 0.25, 2 + 0.25, 2 + 0.25]
        ys = [plot_data[name][1][i] for i in [5, 6, 7]]
        scatter_with_markers(ax, xs, ys, markers, c=color)
        ax.plot(xs, ys, linestyle=linestyle, color=color)

        xs = [3 + 0.25, 3 + 0.25, 3 + 0.25]
        ys = [plot_data[name][2][i] for i in [5, 6, 7]]
        scatter_with_markers(ax, xs, ys, markers, c=color)
        ax.plot(xs, ys, linestyle=linestyle, color=color)

        ##########
        name = 'fcos'
        linestyle = 'dotted'
        color = 'blue'
        xs = [0.75, 0.75, 0.75]
        ys = [plot_data[name][0][i] for i in [5, 6, 7]]
        scatter_with_markers(ax, xs, ys, markers, c=color)
        line2 = ax.plot(xs, ys, linestyle=linestyle, color=color)

        xs = [2 - 0.25, 2 - 0.25, 2 - 0.25]
        ys = [plot_data[name][1][i] for i in [5, 6, 7]]
        scatter_with_markers(ax, xs, ys, markers, c=color)
        ax.plot(xs, ys, linestyle=linestyle, color=color)

        xs = [3 - 0.25, 3 - 0.25, 3 - 0.25]
        ys = [plot_data[name][2][i] for i in [5, 6, 7]]
        scatter_with_markers(ax, xs, ys, markers, c=color)
        ax.plot(xs, ys, linestyle=linestyle, color=color)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0", "50", "101", "152"])
        leg1 = ax.legend(ret, ["0-1/5", "1/5-1/3", "1/3-1"], loc='upper left')
        ax.add_artist(leg1)

        ax.legend(
            [Line2D([0], [0], color=color, linewidth=1, linestyle='dotted')
             for color in ['green', 'red', 'blue']],
            ['faster', 'retina', 'fcos'],
            loc='lower left'
        )
        fig.savefig(fname)


if __name__ == '__main__':
    import fire

    fire.Fire()
