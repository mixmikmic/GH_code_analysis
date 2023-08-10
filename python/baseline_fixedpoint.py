import pandas as pd
import numpy as np

base_path = 'C:\\Users\\Roman Bolzern\\Desktop\\D4\\'
train = pd.read_csv(base_path + 'training.csv', sep=";", parse_dates=["start","end","peak"], index_col="id")
test = pd.read_csv(base_path + 'test.csv', sep=";", parse_dates=["start","end","peak"], index_col="id")

# optimal fixed point prediction, solved by optimization
# Always predict the same small flare ("B9")
predict_val = 9.17647058823529E-07

# evaluate
print('Mean absolute errors:')
print(f'train: {np.mean(np.abs(train.peak_flux-predict_val))}')
print(f'test:  {np.mean(np.abs(test.peak_flux-predict_val))}')

from utils.statistics import *

y_pred = np.repeat(predict_val, len(train.peak_flux))
print(f'TSS train: {true_skill_statistic(train.peak_flux, y_pred)}')
print(f'HSS train: {heidke_skill_score(train.peak_flux, y_pred)}')

y_pred = np.repeat(predict_val, len(test.peak_flux))
print(f'TSS train: {true_skill_statistic(test.peak_flux, y_pred)}')
print(f'HSS train: {heidke_skill_score(test.peak_flux, y_pred)}')



