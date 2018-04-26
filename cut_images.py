import json
import uuid

import cv2
import random
import numpy as np
from numpy.linalg import inv
import glob2
import os
import sys


def get_files(set_name, ext='ppm'):
    return glob2.glob('{}/img[0-9].{}'.format(set_name, ext))


def coord_to_dict(pos_id, set_name, file, x, y, origin, coord_class):
    file = os.path.basename(file)
    return {
        'id': pos_id,
        'set': set_name,
        'file': file,
        'pos': {'x': x, 'y': y},
        'origin': origin,
        'class': coord_class
    }


def make_samples(set_name, points):
    img_names = get_files(set_name, 'png')
    sample_size = 100
    rows, cols, channels = cv2.imread(img_names[0]).shape
    samples_num = 10
    for sample_num in range(samples_num):
        while True:
            y, x = random.randint(sample_size / 2, rows - sample_size / 2), \
                   random.randint(sample_size / 2, cols - sample_size / 2)
            samples = []
            samples_points = []
            samples_class = str(uuid.uuid4())
            first_id = str(uuid.uuid4())
            for sample_idx, img_name in enumerate(img_names, 1):
                if sample_idx >= 2:
                    M = read_matrix(set_name, sample_idx)
                    pos = np.array([x, y, 1])
                    pos = np.dot(M, pos)
                    pos /= pos[2]
                    pos = pos[:2]
                else:
                    pos = np.array([x, y])
                new_x, new_y = pos
                img = cv2.imread(img_name)
                y_from, y_to, x_from, x_to = round(new_y - sample_size / 2), \
                                             round(new_y + sample_size / 2), \
                                             round(new_x - sample_size / 2), \
                                             round(new_x + sample_size / 2)
                if x_from < 0 or x_to > img.shape[1]:
                    break
                if y_from < 0 or y_to > img.shape[0]:
                    break
                sample = img[y_from:y_to, x_from:x_to]
                samples.append(sample)
                samples_points.append(coord_to_dict(str(uuid.uuid4()) if sample_idx == 1 else first_id,
                                                    set_name,
                                                    img_name,
                                                    x, y,
                                                    first_id,
                                                    samples_class))
                print('sample', sample_num, 'from image', sample_idx, 'from set', set_name)
            if len(samples) != len(img_names):
                continue
            for sample_idx, sample in enumerate(samples):
                cv2.imwrite('samples/{}_{}_from_{}.jpg'.format(set_name, sample_num, sample_idx), sample)
            break
        points += samples_points
    return points


def read_matrix(set_name, idx):
    return np.genfromtxt('{}/H1to{}p'.format(set_name, idx), delimiter='  ')


def unwrap(set_name, ext):
    img_names = get_files(set_name, ext)
    rows, cols, ch = cv2.imread(img_names[0]).shape
    for idx, img_name in enumerate(img_names, 1):
        img = cv2.imread(img_name)
        if idx >= 2:
            M = read_matrix(set_name, idx)
            M = inv(M)
            img_unwrapped = cv2.warpPerspective(img, M, (cols, rows))
        else:
            img_unwrapped = img
        cv2.imwrite('{}/unwrapped_img{}.png'.format(set_name, idx), img_unwrapped)
        cv2.imwrite('{}/img{}.png'.format(set_name, idx), img)


def process_sets():
    sets = [
        ('bikes', 'ppm'),
        ('bark', 'ppm'),
        ('boat', 'pgm'),
        ('graf', 'ppm'),
        ('leuven', 'ppm'),
        ('trees', 'ppm'),
        ('ubc', 'ppm'),
        ('wall', 'ppm')
    ]
    points = []
    for dir, ext in sets:
        unwrap(dir, ext)
        points = make_samples(dir, points)
    if not os.path.isdir('samples'):
        print('samples dir not found!', file=sys.stderr)
    with open('points.json', 'w') as output_file:
        json.dump(points, output_file, indent=2)


if __name__ == '__main__':
    process_sets()
