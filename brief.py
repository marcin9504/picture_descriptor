from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial.distance import hamming

from util import rotate, get_sample, cut_circle


class Brief(object):
    """Descriptor is array of 0 and 1 from comparing pixel intensity in random points"""

    def __init__(self, points_num, radius):
        self.points_num = points_num * 2
        self.points = self._get_random_points(radius - 1)
        print(np.max(self.points))

    def _get_random_points(self, radius):
        points = np.sort(np.random.rand(self.points_num, 2), axis=1)
        points = np.apply_along_axis(lambda p: (p[1] * radius *
                                                np.cos(2 * np.pi * p[0] / p[1]), p[1] *
                                                radius * np.sin(2 * np.pi * p[0] / p[1])), 1, points)
        points = np.round(points).astype(int)
        return points

    def extract(self, img, point):
        y, x = point
        # hist = self.circle_hist_extract(img, point)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = get_sample(img, y, x, normalize=False)
        angle = self.extract_dominant_gradient(img)
        img = rotate(img, angle)
        y = x = 32
        des = np.zeros(self.points_num, dtype=np.uint8)
        for idx, (p1, p2) in enumerate(zip(self.points[0::2], self.points[1::2])):
            p1y, p1x = p1
            p2y, p2x = p2
            pos1 = (y + p1y, x + p1x)
            pos2 = (y + p2y, x + p2x)
            if img[pos1] > img[pos2]:
                des[idx] = 1
        return des

    def compare(self, des1, des2):
        return hamming(des1, des2) / self.points_num

    def reject_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    def extract_dominant_gradient(self, img):
        img = img.astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        mag = cut_circle(mag)
        std = np.std(mag) * 3
        mask = mag > std
        mag = mag[mask]
        angle = cut_circle(angle)
        angle = angle[mask]
        return np.average(angle, weights=mag)

    def circle_hist_extract(self, img, point):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y, x = point
        sample = get_sample(img, x, y, r=32)
        w, h = sample.shape
        dict_brightness = defaultdict(int)
        normalise_count = defaultdict(int)
        center_x, center_y = w // 2, h // 2

        for i in range(w):
            for j in range(h):
                key = np.floor(np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)).astype(int)
                dict_brightness[key] += sample[i, j]
                normalise_count[key] += 1

        des = np.zeros(len(dict_brightness))
        for key, value in normalise_count.items():
            des[key] = dict_brightness[key] / value
        return des[15:32]


if __name__ == '__main__':
    b = Brief(1, 32)
    img = cv2.imread('cropped.png')
    des1 = b.extract(img, (32, 32))
    img = cv2.imread('rotated.png')
    des2 = b.extract(img, (32, 32))
    print(b.compare(des1, des2))
   