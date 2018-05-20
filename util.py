from math import pi
import cv2
import numpy as np

""" Utility Functions """


def load_image(img_path, shape=None):
    img = cv2.imread(img_path)
    if shape is not None:
        img = cv2.resize(img, shape)

    return img


def save_image(img_path, img):
    cv2.imwrite(img_path, img)


def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))


def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))


def deg_to_rad(deg):
    return deg * pi / 180.0


def rad_to_deg(rad):
    return rad * 180.0 / pi


def rotate(img, deg):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def cut_circle(img):
    if len(img.shape) == 3:
        w, h, _ = img.shape
    else:
        w, h  = img.shape
    a, b = w / 2, h / 2
    r = min(w, h)
    y, x = np.ogrid[-a:r - a, -b:r - b]
    mask = x * x + y * y <= r * r / 4
    img[~mask] = 0
    return img


def get_sample(img, x, y, r=32, normalize=True):
    sample = img[y - r:y + r, x - r:x + r]
    sample = cut_circle(sample)
    if normalize:
        sample = normalise_image(sample)
    return sample


def normalise_image(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
