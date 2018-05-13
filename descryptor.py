import math

import cv2
import numpy as np
import math
from scipy.stats.stats import pearsonr


def cut_circle(img):
    w, h = img.shape
    a, b = w / 2, h / 2
    n = min(w, h)
    y, x = np.ogrid[-a:n - a, -b:n - b]
    mask = x * x + y * y <= n * n / 4
    img[~mask] = 0
    return img


def extract(img, points):
    return circle_hist_extract(img, points)
    # return hu_extract(img, points)
    # return sift_extract(img, points)


def distance(des1, des2):
    return circle_hist_distance(des1, des2)
    # return hu_distance(des1, des2)
    # return sift_distance(des1, des2)


def circle_hist_extract(img, points):
    descriptors = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for point in points:
        y, x = point
        sample = get_sample(img, x, y)
        w, h = sample.shape
        dict = {}
        normalise_count = {}
        center_x, center_y = w // 2, h // 2

        for i in range(w):
            for j in range(h):
                key = math.floor(math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2))
                val = dict.get(key, 0)
                count_val = normalise_count.get(key, 0)
                dict[key] = val + img[i, j]
                normalise_count[key] = count_val + 1

        des = [0 for i in range(len(dict))]
        for key, value in normalise_count.items():
            des[key] = dict[key] / value

        descriptors.append(des)
        # print(des)
    return descriptors


def circle_hist_distance(des1, des2):
    return abs(pearsonr(des1, des2)[0])


def sift_extract(img, points):
    descriptors = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(400)
    for point in points:
        y, x = point
        sample = get_sample(img, x, y)
        kp, des = sift.detectAndCompute(sample, None)
        descriptors.append(des)
    return descriptors


def sift_distance(des1, des2):
    threshold = 1 / 30
    if des1 is None or des2 is None:
        print("None")
        return 1
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    # for match in matches:
    #     print(match.distance)
    # matches = sorted(matches, key=lambda x: x.distance)
    return float(threshold) / len(matches)


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


def hu_extract(img, points):
    descriptors = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for point in points:
        y, x = point
        sample = get_sample(img, x, y)
        des = cv2.HuMoments(cv2.moments(sample))
        descriptors.append(des)
    return descriptors


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
