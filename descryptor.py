import math

import cv2
import numpy as np


def extract(img, points):
    # return hu_extract(img, points)
    return sift_extract(img, points)


def distance(des1, des2):
    # return hu_distance(des1, des2)
    return sift_distance(des1, des2)


def sift_extract(img, points):
    descriptors = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(400)
    for point in points:
        y, x = point
        sample = img[y - 32:y + 32, x - 32:x + 32]
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
    # matches = sorted(matches, key=lambda x: x.distance)
    # print(matches)
    # print(len(matches))
    print(len(matches))
    return threshold / len(matches)


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
        sample = img[y - 32:y + 32, x - 32:x + 32]
        des = cv2.HuMoments(cv2.moments(sample))
        descriptors.append(des)
    return descriptors


def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 0.
    else:
        return x
