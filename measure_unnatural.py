import os
import random

import cv2
import numpy as np

import descriptor
import roc_plot
from image_transformaer import ImageTransformer

SAMPLE_SIZE = 128
SAMPLES_NUM = 200


def random_sample(img, size):
    rows, cols = img.shape
    y, x = random.randint(size / 2, rows - size / 2), \
           random.randint(size / 2, cols - size / 2)

    y_from, y_to, x_from, x_to = round(y - size / 2), \
                                 round(y + size / 2), \
                                 round(x - size / 2), \
                                 round(x + size / 2)
    return img[int(y_from):int(y_to), int(x_from):int(x_to)]


def blur(img, kernel_size):
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img


def rotate(img, angle):
    w, h = img.shape
    rot_mat = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, (h, w), flags=cv2.INTER_LINEAR)
    return img


def jpg_compress(img, quality):
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imread('tmp.jpg')


def zoom(img, scale):
    w, h = img.shape
    return cv2.resize(img, (int(scale * h), int(scale * w)), interpolation=cv2.INTER_CUBIC)


def gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def view_point(img, angle):
    trans = ImageTransformer(img)
    return trans.rotate_along_axis(phi=angle)


def random_transformation(img):
    trans = [(blur, range(5, 21, 4)),
             (rotate, range(0, 91, 15)),
             (view_point, range(0, 55, 7)),
             (zoom, [0.5, 1.2, 1.5, 2, 4]),
             (zoom, [0.5]),
             (gamma, [0.5, 0.6, 0.75, 1.2, 1.5, 1.7]),
             (jpg_compress, [5, 10, 30, 55, 80])
             ]
    f, val = random.choice(trans)
    img = f(img, random.choice(val))
    return img


def save_samples(samples):
    for i, sample in enumerate(samples):
        w, h = sample.shape
        sample = sample[int((w - SAMPLE_SIZE) / 2):int(w - (w - SAMPLE_SIZE) / 2),
                 int((h - SAMPLE_SIZE) / 2):int(h - (h - SAMPLE_SIZE) / 2)]
        cv2.imwrite(os.path.join('samples', f'{i}.png'), sample)


def main():
    # np.random.seed(56)
    # random.seed(56)
    orginal_set = 'bikes'
    img = cv2.imread(os.path.join(orginal_set, 'img1.ppm'))
    orginal_samples = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orginal = random_sample(img, SAMPLE_SIZE)
    for rot in range(0, 91, 10):
        orginal_samples.append(rotate(orginal, rot))
    for rot in range(0, 55, 7):
        orginal_samples.append(view_point(orginal, rot))
    for kernel_size in range(5, 21, 4):
        orginal_samples.append(blur(orginal, kernel_size))
    for scale in [0.5, 0.7, 0.8, 1.2, 1.5, 2, 3, 4, 5]:
        orginal_samples.append(zoom(orginal, scale))
    for g in [0.5, 0.6, 0.75, 1.2, 1.5, 1.7]:
        orginal_samples.append(gamma(orginal, g))
    for quality in [5, 10, 30, 55, 80]:
        orginal_samples.append(jpg_compress(orginal, quality))
    sets = [
        ('bikes', 'ppm'),
        ('bark', 'ppm'),
        ('boat', 'pgm'),
        ('leuven', 'ppm'),
        ('trees', 'ppm'),
        ('ubc', 'ppm'),
        ('wall', 'ppm')
    ]
    used_set = sets[0]  # samples used from one set
    # sets = filter(lambda x: x[0] != original_set, sets)
    images = [f'img{i+1}' for i in range(6)]
    other_samples = []
    for _ in range(len(orginal_samples)):
        # rand_set = random.choice(sets)
        rand_set = used_set
        rand_img = random.choice(images)
        rand_img = cv2.imread(os.path.join(rand_set[0], f'{rand_img}.{rand_set[1]}'))
        rand_img = cv2.cvtColor(rand_img, cv2.COLOR_BGR2GRAY)
        rand_img = random_sample(rand_img, SAMPLE_SIZE)
        rand_img = random_transformation(rand_img)
        other_samples.append(rand_img)
    for idx, sample in enumerate(orginal_samples):
        if len(sample.shape) == 3:
            orginal_samples[idx] = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    for idx, sample in enumerate(other_samples):
        if len(sample.shape) == 3:
            other_samples[idx] = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    save_samples(orginal_samples + other_samples)

    y_true = []
    y_score = []
    des = []
    for sample in orginal_samples:
        sample_center = [[int(sample.shape[0] / 2)] * 2]
        des.append(descriptor.extract(sample, sample_center)[0])
    for sample in other_samples:
        sample_center = [[int(sample.shape[0] / 2)] * 2]
        des.append(descriptor.extract(sample, sample_center)[0])
    for i, d1 in enumerate(des):
        for j, d2 in enumerate(des):
            if i < len(des) / 2 and j < len(des) / 2:
                y_true.append(0)
            else:
                y_true.append(1)
            y_score.append(descriptor.distance(d1, d2))
            # print(f'\r {i*len(des)+j} out of {len(des)**2}', end='')
    print(y_score)
    # for y_t, y_s in zip(y_true, y_score):
    #     print(y_t, y_s)
    roc_plot.draw(y_true, y_score)


if __name__ == '__main__':
    main()
