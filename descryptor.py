import math

import cv2
import numpy as np
from scipy.stats.stats import pearsonr

def cut_circle(img):
    w, h = img.shape
    a, b = w / 2, h / 2
    n = min(w, h)
    y, x = np.ogrid[-a:n - a, -b:n - b]
    mask = x * x + y * y <= n * n / 4
    img[~mask] = 0
    return img


class Descriptor(dict):
    CIRCLE_HIST = 'circle_hist_distance'
    HU_MOMENTS = 'hu_moments'
    AVERAGE = 'average'


def extract(img, points):
    # return circle_hist_extract(img, points) #uncomment it to use only one method
    descryptors = []
    for point in points:
        des = Descriptor()
        # des[Descriptor.CIRCLE_HIST] = circle_hist_extract(img, point)
        # des[Descriptor.HU_MOMENTS] = hu_extract(img, point)
        des[Descriptor.AVERAGE] = average_extract(img, point)
        descryptors.append(des)
    return descryptors


def distance(des1, des2):
    # return circle_hist_distance(des1,des2) #uncomment it to use only one method
    scores = []
    # scores.append(circle_hist_distance(des1[Descriptor.CIRCLE_HIST], des2[Descriptor.CIRCLE_HIST]))
    # scores.append(hu_distance(des1[Descriptor.HU_MOMENTS], des2[Descriptor.HU_MOMENTS]))
    scores.append(average_distance(des1[Descriptor.AVERAGE], des2[Descriptor.AVERAGE]))

    # for idx, d in enumerate(scores):
    #     if d not in range(0, 1):
    #         scores[idx] = 0.5

    return sum(scores) / float(len(scores))


def average_extract(img, point):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = point
    sample = get_sample(img, x, y)
    des = np.average(sample[sample > 0])
    return des


def average_distance(des1, des2):
    threshold = 64
    return min(1, abs(des1 - des2) / threshold)


def circle_hist_extract(img, point):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = point
    sample = get_sample(img, x, y)
    w, h = sample.shape
    dict_brightness = {}
    normalise_count = {}
    center_x, center_y = w // 2, h // 2

    for i in range(w):
        for j in range(h):
            key = math.floor(math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2))
            val = dict_brightness.get(key, 0)
            count_val = normalise_count.get(key, 0)
            dict_brightness[key] = val + sample[i, j]
            normalise_count[key] = count_val + 1

    des = [0 for _ in range(len(dict_brightness))]
    for key, value in normalise_count.items():
        des[key] = dict_brightness[key] / value
    return des


def rescale(vector, length):
    r = length
    split_arr = np.linspace(0, len(vector), num=r + 1, dtype=int)
    dwnsmpl_subarr = np.split(vector, split_arr[1:])
    dwnsmpl_arr = (list(np.mean(item) for item in dwnsmpl_subarr[:-1]))
    return dwnsmpl_arr


def circle_hist_distance(des1, des2):
    # return abs(pearsonr(des1, des2)[0])
    corr = []
    for scale in (0.7, 0.8, 0.9, 1):
        rescaled_length = math.floor(len(des1) * scale)
        corr.append(abs(pearsonr(des1[:rescaled_length], rescale(des2, rescaled_length))[0]))
        corr.append(abs(pearsonr(des2[:rescaled_length], rescale(des1, rescaled_length))[0]))
    if np.isnan(max(corr)):
        return 0.5
    else:
        return max(corr)


def hu_distance(des1, des2):
    threshold = 50
    s = 0
    for i in range(len(des1)):
        diff = abs(des1[i] - des2[i])
        if diff == 0:
            continue
        num = -sign(diff) * math.log10(diff)
        s += num
    if s < threshold:
        return 0
    else:
        return 1


def hu_extract(img, point):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = point
    sample = get_sample(img, x, y)
    des = cv2.HuMoments(cv2.moments(sample))
    return des


def normalise_image(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def get_sample(img, x, y, r=32, normalize=True):
    sample = img[y - r:y + r, x - r:x + r]
    sample = cut_circle(sample)
    if normalize:
        sample = normalise_image(sample)
    return sample


def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 0.
    else:
        return x
