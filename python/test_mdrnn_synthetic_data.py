from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# little path hack to get robojam from one directory up in the filesystem.
from context import * # imports robojam
# import robojam # alternatively do this.
import tensorflow as tf

# data:
x_y_t_log = robojam.generate_synthetic_3D_data()
loader = robojam.SequenceDataLoader(num_steps=65, batch_size=64, corpus=x_y_t_log)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns

def plot_3D_data(perf):
    """Plot in 3D."""
    perf = perf.T
    perf_df = pd.DataFrame({'x':perf[0], 'y':perf[1], 't':perf[2]})
    perf_df['time'] = perf_df.t.cumsum()
    # Plot in 2D
    plt.figure(figsize=(8, 8))
    p = plt.plot(perf_df.x, perf_df.y, '.r-')
    plt.show()
    # Plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(perf_df.time, perf_df.x, perf_df.y, '.r-')
    plt.show()
    
# Plot the sample data
plot_3D_data(x_y_t_log[:200])

# Hyperparameters
H_UNITS = 128
N_MIX = 8
BATCH = 64
SEQ_LEN = 64
# Setup network
net = robojam.MixtureRNN(mode=robojam.NET_MODE_TRAIN, n_hidden_units=H_UNITS, n_mixtures=N_MIX, batch_size=BATCH, sequence_length=SEQ_LEN)

# Train
# 1 epoch is too few, about 30 works well.
losses = net.train(loader, 1, saving=True)

first_touch = np.array([(np.random.rand()-0.5), (np.random.rand()-0.5), (0.01 + (np.random.rand()-0.5)*0.005)])
print("Test Input:",first_touch)
# Load running network.
net = robojam.MixtureRNN(mode=robojam.NET_MODE_RUN, n_hidden_units=128, n_mixtures=8, batch_size=1, sequence_length=1)
with tf.Session() as sess:
    perf = net.generate_performance(first_touch,1000,sess)
print("Test Output:")
perf_df = pd.DataFrame({'a':perf.T[0], 'b':perf.T[1], 't':perf.T[2]})
perf_df['time'] = perf_df.t.cumsum()
print(perf_df.describe())

# plots
plot_3D_data(perf[50:200])

