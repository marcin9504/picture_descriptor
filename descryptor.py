import math

import cv2
import numpy as np


def extract(img, points):
    descriptors = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    for point in points:
        y, x = point
        sample = img[y - 32:y + 32, x - 32:x + 32]
        des = cv2.HuMoments(cv2.moments(img))
        # des = np.average(sample)
        # kp, des = sift.detectAndCompute(sample, None)
        descriptors.append(des)
    return descriptors


def distance(des1, des2):
    # print(des1)
    # print(des2)
    # return np.abs(des1 - des2)

    s = 0
    for i in range(len(des1)):
        diff = abs(des1[i] - des2[i])
        if diff == 0:
            continue
        num = -sign(diff) * math.log10(diff)
        s += num

    return s


def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 0.
    else:
        return x
