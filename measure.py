import os
import random

import cv2
import pandas as pd

import descryptor
import roc_plot


def get_file_name(file):
    return os.path.splitext(os.path.basename(file))[0]


def main():
    all_points = pd.read_json('points.json')
    img_class = random.choice(all_points['class'].unique())
    points = all_points[all_points['class'] == img_class]
    points = points.append(all_points[all_points['class'] != img_class].sample(6))
    images = {}
    for idx, point in points.iterrows():
        if (point['set'], point['file']) not in images:
            images[(point['set'], point['file'])] = (cv2.imread(os.path.join(point['set'], point['file'])))
    descriptors = []
    for idx, point in points.iterrows():
        x = point['pos']['x']
        y = point['pos']['y']
        descriptors.extend(descryptor.extract(images[(point['set'], point['file'])], [[y, x]]))
    y_scores = []
    y_true = []
    for i, des1 in enumerate(descriptors):
        for j, des2 in enumerate(descriptors):
            y_scores.append(descryptor.distance(des1, des2))
            y_true.append(0 if points.iloc[i]['class'] == points.iloc[j]['class'] else 1)

    for y_t, y_s in zip(y_true, y_scores):
        print(y_t, y_s)
    roc_plot.draw(y_true, y_scores)


if __name__ == '__main__':
    main()
