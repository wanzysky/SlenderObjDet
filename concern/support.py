import  torchvision.models.resnet
import numpy as np


def between(a, a_range: tuple) -> bool:
    if isinstance(a, np.ndarray):
        return np.logical_and(a >= a_range[0], a <= a_range[1])
    return a >= a_range[0] and a <= a_range[1]

def all_the_same(a_list: list) -> bool:
    for item in a_list:
        if not item == a_list[0]:
            return False
    return True

def make_dual(item_or_tuple) -> tuple:
    if isinstance(item_or_tuple, tuple):
        return item_or_tuple
    return (item_or_tuple, item_or_tuple)
