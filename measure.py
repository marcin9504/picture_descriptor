import json
import random

import cv2
import glob2
from sklearn.metrics import roc_auc_score

import descryptor
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

def get_file_name(file):
    return os.path.splitext(os.path.basename(file))[0]

def main():
    dir = 'set0'
    all_points = pd.read_json('points.json')
    img_class = random.choice(all_points['class'].unique())
    points = all_points[all_points['class'] == img_class].sample(2)
    points = points.append(all_points[all_points['class'] != img_class].sample(4))
    images = {}
    for idx, point in points.iterrows():
        if (point['set'], point['file']) not in images:
            images[(point['set'], point['file'])] = (cv2.imread(os.path.join(point['set'], point['file'])))
    descriptors = []
    for idx, point in points.iterrows():
        x = point['pos']['x']
        y = point['pos']['y']
        descriptors += descryptor.extract(images[(point['set'], point['file'])],[[y,x]])
    y_scores = []
    y_true = []
    print(descriptors)
    for des in descriptors:
        y_scores.append(descryptor.distance(descriptors[0], des))

    print(y_scores)
    score = roc_auc_score((points['class'] == img_class).astype(int), y_scores)
    print(score)
    plt.plot(score)
    plt.show()

if __name__ == '__main__':
    main()
