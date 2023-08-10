import pandas as pd
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import normalize, minmax_scale

df = pd.read_csv('datasets/dataset2.csv')

df['average_montly_hours'][:10]

hours = df['average_montly_hours'].values

result = normalize(df['average_montly_hours'].astype(float).values.reshape(1,-1), norm='l2', axis=1).reshape(-1,1)

result

stats.describe(result)



