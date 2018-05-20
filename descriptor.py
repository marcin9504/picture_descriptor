import math
from collections import defaultdict

import cv2
import numpy as np
from scipy.stats.stats import pearsonr

from brief import Brief
from util import get_sample


class Descriptor(dict):
    CIRCLE_HIST = 'circle_hist_distance'
    HU_MOMENTS = 'hu_moments'
    AVERAGE = 'average'
    MAX_ON_CIRCLE = 'max_on_circle'
    BRIEF = 'brief'


brief = None


def extract(img, points):
    global brief
    if brief is None:
        brief = Brief(512, 32)

    descryptors = []
    for point in points:
        des = Descriptor()
        # des[Descriptor.CIRCLE_HIST] = circle_hist_extract(img, point)
        des[Descriptor.HU_MOMENTS] = hu_extract(img, point)
        des[Descriptor.AVERAGE] = average_extract(img, point)
        # des[Descriptor.MAX_ON_CIRCLE] = max_on_circle_extract(img, point)
        des[Descriptor.BRIEF] = brief.extract(img, point)
        descryptors.append(des)
    return descryptors


def distance(des1, des2):
    scores = []
    # scores.append(circle_hist_distance(des1[Descriptor.CIRCLE_HIST], des2[Descriptor.CIRCLE_HIST]))
    scores.append(hu_distance(des1[Descriptor.HU_MOMENTS], des2[Descriptor.HU_MOMENTS]))
    scores.append(average_distance(des1[Descriptor.AVERAGE], des2[Descriptor.AVERAGE]))
    # scores.append(max_on_circle_distance(des1[Descriptor.MAX_ON_CIRCLE], des2[Descriptor.MAX_ON_CIRCLE]))
    scores.append(brief.compare(des1[Descriptor.BRIEF], des2[Descriptor.BRIEF]))
    # for idx, d in enumerate(scores):
    #     if d not in range(0, 1):
    #         scores[idx] = 0.5

    return sum(scores) / float(len(scores))


def max_on_circle_extract(img, point):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = point
    sample = get_sample(img, x, y)
    w, h = sample.shape
    dict_brightness = {}
    center_x, center_y = w // 2, h // 2

    for i in range(w):
        for j in range(h):
            key = math.floor(math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2))
            if key == 0:
                continue
            val = dict_brightness.get(key, [0, 0])
            if sample[i, j] > val[1]:
                val[1] = sample[i, j]
                sin = (j - center_y) / key
                alpha = math.asin(sin)
                val[0] = alpha

            dict_brightness[key] = val
    des = [0 for _ in range(len(dict_brightness))]
    for key, value in dict_brightness.items():
        des[key - 1] = value[0]
    return des[:32]


def max_on_circle_distance(des1, des2):
    corr = []
    for scale in (0.7, 0.8, 0.9, 1):
        rescaled_length = math.floor(len(des1) * scale)
        ten_degrees = math.pi / 36
        for factor in range(0, 18):
            factor = ten_degrees * factor
            tmp_v = []
            for i in des1:
                value = i + factor
                if value > math.pi / 2:
                    value -= math.pi
                tmp_v.append(value)
            des1 = tmp_v
            corr.append(abs(pearsonr(des1[:rescaled_length], rescale(des2, rescaled_length))[0]))
            corr.append(abs(pearsonr(des2[:rescaled_length], rescale(des1, rescaled_length))[0]))
    if np.isnan(np.max(corr)):
        return 0.5
    else:
        return 1-np.max(corr)


def average_extract(img, point):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = point
    sample = get_sample(img, x, y)
    des = np.average(sample)
    return des


def average_distance(des1, des2):
    threshold = 64
    return min(1, abs(des1 - des2) / threshold)


def circle_hist_extract(img, point):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = point
    sample = get_sample(img, x, y)
    w, h = sample.shape
    dict_brightness = defaultdict(int)
    normalise_count = defaultdict(int)
    center_x, center_y = w // 2, h // 2

    for i in range(w):
        for j in range(h):
            key = math.floor(math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2))
            dict_brightness[key] += sample[i, j]
            normalise_count[key] += 1

    des = np.zeros(len(dict_brightness))
    for key, value in normalise_count.items():
        des[key] = dict_brightness[key] / value
    return des[:32]


def rescale(vector, length):
    r = length
    split_arr = np.linspace(0, len(vector), num=r + 1, dtype=int)
    dwnsmpl_subarr = np.split(vector, split_arr[1:])
    dwnsmpl_arr = (list(np.mean(item) for item in dwnsmpl_subarr[:-1]))
    return dwnsmpl_arr


def circle_hist_distance(des1, des2):
    corr = []
    for scale in (0.7, 0.8, 0.9, 1, 0.5, 0.6, 0.25):
        rescaled_length = math.floor(len(des1) * scale)
        corr.append(abs(pearsonr(des1[:rescaled_length], rescale(des2, rescaled_length))[0]))
        corr.append(abs(pearsonr(des2[:rescaled_length], rescale(des1, rescaled_length))[0]))
    return 1-max(corr)


max_diff = 0


def hu_distance(des1, des2):
    des1 = -np.sign(des1) * np.log10(np.abs(des1))
    des2 = -np.sign(des2) * np.log10(np.abs(des2))
    diff = np.abs(np.sum(des1 - des2)) / 160.0
    global max_diff
    return diff


def hu_extract(img, point):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/256
    y, x = point
    sample = get_sample(img, x, y)
    des = cv2.HuMoments(cv2.moments(sample))
    return des


if __name__ == '__main__':
    def build_checkerboard(w, h):
        re = np.r_[w * [0, 1]]  # even-numbered rows
        ro = np.r_[w * [1, 0]]  # odd-numbered rows
        return np.row_stack(h * (re, ro))


    img1 = build_checkerboard(64, 64).astype(np.uint16) * 256
    img2 = (np.random.rand(64, 64) * 256).astype(np.uint16)
    des1 = hu_extract(img1, (32, 32))
    des2 = hu_extract(img2, (32, 32))
    print(hu_distance(des1, des2))
