import cv2
import numpy as np
from scipy.spatial.distance import hamming


class Brief(object):
    """Descriptor is array of 0 and 1 from comparing pixel intensity in random points"""

    def __init__(self, points_num, radius):
        self.points_num = points_num * 2
        self.points = self._get_random_points(radius - 1)

    def _get_random_points(self, radius):
        points = np.sort(np.random.rand(self.points_num, 2), axis=1)
        points = np.apply_along_axis(lambda p: (p[1] * radius *
                                                np.cos(2 * np.pi * p[0] / p[1]), p[1] *
                                                radius * np.sin(2 * np.pi * p[0] / p[1])), 1, points)
        return points

    def extract(self, img, point):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        des = np.zeros(self.points_num,dtype=np.uint8)
        y, x = point
        for idx, (p1, p2) in enumerate(zip(self.points[0::2], self.points[1::2])):
            p1y, p1x = p1
            p2y, p2x = p2
            pos1 = tuple(np.round([y + p1y, x + p1x]).astype(int))
            pos2 = tuple(np.round([y + p2y, x + p2x]).astype(int))
            if img[pos1] > img[pos2]:
                des[idx] = 1
        return des

    @staticmethod
    def compare(des1, des2):
        return hamming(des1, des2)


if __name__ == '__main__':
    b = Brief(512, 32)
    import matplotlib.pyplot as plt
    plt.scatter(b.points[:, 0], b.points[:, 1])
    plt.show()
    img = cv2.imread('tmp.jpg')
    b.extract(img, (32, 32))
