import cv2
import numpy as np

def extract(img, points):
    descriptors = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    for point in points:
        y, x = point
        sample = img[y-32:y+32,x-32:x+32]
        des = np.average(sample)
        # kp, des = sift.detectAndCompute(sample, None)
        descriptors.append(des)
    return descriptors


def distance(des1, des2):
    print(des1)
    print(des2)
    # if np.array(des1).shape != np.array(des2).shape:
    #     return 0
    return np.abs(des1 - des2)
