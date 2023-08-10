import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fall_times = np.loadtxt('falltimes.csv', delimiter=',', skiprows=1)
print('The size of the data array is {}' .format(np.shape(fall_times)))

print(fall_times[3, 0])

print(fall_times[:, 0])

import pandas as pd
fall_times = pd.read_csv('falltimes.csv')

print(fall_times.head())

positions = fall_times.columns.values
print(positions)

print(fall_times[positions[0]])
print(type(fall_times[positions[0]]))

print(fall_times[positions[0]][0])



