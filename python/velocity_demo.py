import os
from evaluate.velocity import VeloEval
import copy
import numpy as np

dataset_path = '/mnt/truenas/scratch/chenyangli/benchmark/v2/clips/'
folder_path = os.listdir(dataset_path)
annotations = [os.path.join(dataset_path, x, 'annotation.json') for x in folder_path]

gt = VeloEval.load_annotation(annotations)

pred = copy.deepcopy(gt)
for idx in range(len(pred)):
    for j in range(len(pred[idx])):
        pred[idx][j]["velocity"][0] += np.random.normal(0, 0.5)
        pred[idx][j]["velocity"][1] += np.random.normal(0, 0.5)
        pred[idx][j]["position"][0] += np.random.normal(0, 0.5)
        pred[idx][j]["position"][1] += np.random.normal(0, 0.5)

VeloEval.accuracy(pred, gt)

