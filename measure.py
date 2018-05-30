import os
import random

import cv2
import pandas as pd

import descriptor
import roc_plot


def get_file_name(file):
    return os.path.splitext(os.path.basename(file))[0]


def main():
    random.seed(3)
    all_points = pd.read_json('points.json')
    img_class = random.choice(all_points['class'].unique())
    points = all_points[all_points['class'] == img_class]
    points = points.append(all_points[all_points['class'] != img_class].sample(30))
    images = {}
    for idx, point in points.iterrows():
        if (point['set'], point['file']) not in images:
            img = cv2.imread(os.path.join(point['set'], point['file']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images[(point['set'], point['file'])] = img
    descriptors = []
    for idx, point in points.iterrows():
        x = point['pos']['x']
        y = point['pos']['y']
        img = images[(point['set'], point['file'])]
        descriptors.extend(descriptor.extract(img, [[y, x]]))

    y_scores = []
    y_true = []
    for i, des1 in enumerate(descriptors):
        for j, des2 in enumerate(descriptors):
            y_scores.append(descriptor.distance(des1, des2))
            y_true.append(0 if points.iloc[i]['class'] == points.iloc[j]['class'] else 1)

    for y_t, y_s in zip(y_true, y_scores):
        print(y_t, y_s)
    roc_plot.draw(y_true, y_scores)

if __name__ == '__main__':
    main()
