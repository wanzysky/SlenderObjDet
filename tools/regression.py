# This file is used to analyze the relation between "aspect ratio" and "map".
# We just compare two methods currently, "FPN" and "Retina".
import fire

import csv
import json

import numpy as np
import scipy.stats as st


def regression(x, y) -> str:
    slope, intercept, r_value, p_value, std_err = st.linregress(x, y)

    info = ''
    info += 'slope    : {}'.format(slope)
    info += '\nintercept: {}'.format(intercept)
    info += '\nr_value  : {}'.format(r_value)
    return info


def test_regression():
    import numpy as np

    x = np.arange(0, 10)
    y = 3 * x + 2.5

    print('x: {}'.format(x))
    print('y: {}'.format(y))

    return regression(x, y)


def plot(x, y) -> bool:
    return True


def data_prepare(file_path1, file_path2) -> dict:
    # prepare data for analysis
    # 1. load "FPN" data
    with open(file_path1, mode='r') as csv_fp:
        reader = csv.DictReader(csv_fp)
        data = [row for row in reader]

    categories = [_['category'] for _ in data]
    xs = [float(_['std_ratio']) for _ in data]
    ys = [float(_['map']) for _ in data]
    fpn_data = dict(categories=categories, xs=xs, ys=ys)

    # 2. load "Retina" data
    with open(file=file_path2, mode='r') as fp:
        line = fp.readline()
        data_line = json.loads(line)
        data = data_line

    EXT = 'bbox/AP-'
    ys = [float(data[EXT + category]) for category in categories]
    retina_data = dict(categories=categories, xs=xs, ys=ys)

    return dict(
        fpn_data=fpn_data,
        retina_data=retina_data,
    )


def main(file_path1, file_path2):
    data = data_prepare(file_path1, file_path2)

    for name, value in data.items():
        print(name)
        info = regression(np.array(value['xs']), np.array(value['ys']))
        print(info)


if __name__ == '__main__':
    fire.Fire()
