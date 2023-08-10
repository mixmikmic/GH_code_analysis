# First, let's load all the dependencies we will need.
# The utils module contains all the functions for loading
# and processing data; this tutorial will focus on building
# the deep learning models.

from __future__ import division
from __future__ import print_function

import os
import sys

from imp import reload
import utils; reload(utils)

import keras
import numpy as np
import matplotlib.pyplot as plt

# Checks that you've got Keras 2.0.0 installed (for compatibility).
assert keras.__version__ == '2.0.2', 'Invalid Keras version.'

# Plots the first 5 samples.
fig = plt.figure(figsize=(15, 6))
for i, (c, s) in zip(range(1, 4), utils.iterate_files()):
    plt.subplot(3, 1, i)
    utils.plot_sample(c['0'], s['0'])

plt.tight_layout()
plt.show()

del i, c, s

# Gets five examples of each type.
yes_cal, no_cal, yes_spikes, no_spikes = [], [], [], []
for calcium, spikes, did_spike in utils.partition_data(10, spike_n=1):
    if did_spike:
        yes_cal.append(calcium)
        yes_spikes.append(spikes)
    else:
        no_cal.append(calcium)
        no_spikes.append(spikes)
    
    if len(yes_spikes) > 5 and len(no_spikes) > 5:
        break

# Plot the data where no spike was observed on the left,
# and the data where a spike was observed on the right.
fig = plt.figure(figsize=(7, 9))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    utils.plot_sample(no_cal[i], no_spikes[i], t_start=-10, t_end=9, sampling_rate=1)
    plt.title('Sample %d, no associated spike' % i)
    plt.subplot(5, 2, 2 * i + 2);
    utils.plot_sample(yes_cal[i], yes_spikes[i], t_start=-10, t_end=9, sampling_rate=1)
    plt.title('Sample %d, associated spike' % i)

plt.tight_layout()
plt.show()

del yes_cal, no_cal, yes_spikes, no_spikes

calcium, did_spike = utils.load_dataset()
print('Size of the dataset:')
print('    calcium: %d samples of length %d' % (calcium.shape[0], calcium.shape[1]))
print('    did_spike: %d samples' % did_spike.shape[0])

del calcium, did_spike

def build_model(input_len):
    input_calcium = keras.layers.Input(shape=(input_len,), name='input_calcium')
    
    # This adds some more features that the model can use.
    calcium_rep = keras.layers.Reshape((input_len, 1))(input_calcium)
    calcium_delta = utils.DeltaFeature()(calcium_rep)
    calcium_quad = utils.QuadFeature()(calcium_rep)
    calcium_delta_quad = utils.QuadFeature()(calcium_delta)
    x = keras.layers.Concatenate()([calcium_rep, calcium_delta, calcium_quad])
    
    # This is the single LSTM layer that performs the classification.
    x = keras.layers.LSTM(64, return_sequences=False)(x)
    
    output_pred = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=[input_calcium], outputs=[output_pred])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_evenly_split_dataset(num_samples):
    """Gets an evenly-split sample of the data."""
    
    calcium, did_spike = utils.load_dataset()

    spike_idxs = np.arange(calcium.shape[0])[did_spike == 1]
    nospike_idxs = np.arange(calcium.shape[0])[did_spike == 0]
    spike_idxs = np.random.choice(spike_idxs, num_samples // 2)
    nospike_idxs = np.random.choice(nospike_idxs, num_samples // 2)
    idxs = np.concatenate([spike_idxs, nospike_idxs])

    return calcium[idxs], did_spike[idxs]

NUM_LSTM_TRAIN = 10000
calcium, did_spike = get_evenly_split_dataset(NUM_LSTM_TRAIN)
model = build_model(calcium.shape[1])

# Trains the model for a single pass.
model.fit([calcium], [did_spike], epochs=5, verbose=2)
print('Done')

preds = model.predict(calcium)
p_s, p_n = preds[did_spike == 1], preds[did_spike == 0]
n_total = calcium.shape[0]

# Computes confusion matrix values.
ss, ns = np.sum(p_s > 0.5) / n_total, np.sum(p_s <= 0.5) / n_total
sn, nn = np.sum(p_n > 0.5) / n_total, np.sum(p_n <= 0.5) / n_total

print('                     spike    no spike')
print('predicted spike    | %.3f  | %.3f' % (ss, ns))
print('predicted no spike | %.3f  | %.3f' % (sn, nn))

plt.figure(figsize=(10, 10))

# Gets the false positives and false negatives.
c_s, c_n = calcium[did_spike == 1, :], calcium[did_spike == 0, :]
p_sf, p_nf = np.squeeze(p_s), np.squeeze(p_n)
ns_calc, sn_calc = c_s[p_sf <= 0.5], c_n[p_nf > 0.5]

d = calcium.shape[1] / 2
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    utils.plot_sample(calcium=ns_calc[i],
                      t_start=-d,
                      t_end=d - 1,
                      sampling_rate=1)
    plt.title('Sample %d, false positive' % i)
    
    plt.subplot(5, 2, 2 * i + 2)
    utils.plot_sample(calcium=sn_calc[i],
                      t_start=-d,
                      t_end=d - 1,
                      sampling_rate=1)
    plt.title('Sample %d, false negative' % i)

plt.tight_layout()
plt.show()

