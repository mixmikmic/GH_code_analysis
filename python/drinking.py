get_ipython().magic('matplotlib inline')
from IPython.display import Image
Image('./drinking/water_usage_exp1.png')

# for y in [w1, w4, w7]:
#     plt.plot(t, y, 'g')
# for y in [w2, w5, w6, w8]:
#     plt.plot(t, y, 'r')
# plt.ylabel('Water remaining (g)')
# plt.xlabel('Day')
# plt.xticks([i[1] for i in e], ['End Night %s' % i for i in range(len(e))], rotation=90, size=8)
# for i in e:
#     plt.axvspan(i[0], i[1], color='gray', alpha=.3)
# plt.show()



Image('./drinking/water_usage_example.png')



from IPython.display import Image
Image('./drinking/m1_k5_sl200.png')

get_ipython().magic('matplotlib inline')
from bcp.feature_extraction import trace_to_signals_matrix
from bcp.stats import centered_moving_average
from bcp.preprocess import (weight_sensor_positive_spikes, smooth_positive_spikes)
import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt

w1 = np.load('../data/exp1/Water_1.npy')
t = np.load('../data/exp1/time.npy')

# We are going to use only a small bit of the water trace.
start_ind = 110000
end_ind = 130000

# The signal_length is the length of the vectors we will use for the inputs to k-means.
# K is the number of vector centroids we have. 
signal_length = 100
k = 5

# Calculate the vectors for input to kmeans.
m_start_ind = start_ind / signal_length
m_end_ind = end_ind / signal_length

# Kmeans with no prior data smoothing. 
w1_sm = trace_to_signals_matrix(w1, signal_length, regularization_value=.001)
cb, _ = kmeans(w1_sm, k)
obs = w1_sm[m_start_ind: m_end_ind]
m, d = vq(obs, cb)

# Kmeans with center moving average applied to data
# The cma_radius indicates the window of the moving average around the given point. 
cma_radius = 5
w1_cma = centered_moving_average(w1, cma_radius)
# set the edges of the signals to what they would otherwise be.
# without this, there are significant effects on the set of kmeans.
w1_cma[:cma_radius] = w1[:cma_radius]
w1_cma[-cma_radius:] = w1[-cma_radius:]
w1_cma_sm = trace_to_signals_matrix(w1_cma, signal_length, regularization_value=.001)
cb_cma, _ = kmeans(w1_cma_sm, k)
obs_cma = w1_cma_sm[m_start_ind: m_end_ind]
m_cma, d_cma = vq(obs_cma, cb_cma)

# Kmeans with removal of spiking values as illustrated in trace above at roughly t=112000.
threshold = .3
spikes = weight_sensor_positive_spikes(w1, t, threshold)
backward_window = 10
forward_window = 5
w1_psr = smooth_positive_spikes(w1, spikes, backward_window, forward_window)
w1_psr_sm = trace_to_signals_matrix(w1_psr, signal_length, regularization_value=.001)
cb_psr, _ = kmeans(w1_psr_sm, k)
obs_psr = w1_psr_sm[m_start_ind: m_end_ind]
m_psr, d_psr = vq(obs_psr, cb_psr)

# Kmeans with removal of spiking values as illustrated in trace above at roughly t=112000
# and cma
threshold = .3
spikes = weight_sensor_positive_spikes(w1, t, threshold)
backward_window = 10
forward_window = 5
w1_psr = smooth_positive_spikes(w1, spikes, backward_window, forward_window)
cma_radius = 5
w1_psr_cma = centered_moving_average(w1_psr, cma_radius)
w1_psr_cma[:cma_radius] = w1[:cma_radius]
w1_psr_cma[-cma_radius:] = w1[-cma_radius:]
w1_psr_cma_sm = trace_to_signals_matrix(w1_psr_cma, signal_length, regularization_value=.001)
cb_psr_cma, _ = kmeans(w1_psr_cma_sm, k)
obs_psr_cma = w1_psr_cma_sm[m_start_ind: m_end_ind]
m_psr_cma, d_psr_cma = vq(obs_psr_cma, cb_psr_cma)

# plot results
ms = [m, m_cma, m_psr, m_psr_cma]
cbs = [cb, cb_cma, cb_psr, cb_psr_cma]
ws = [w1, w1_cma, w1_psr, w1_psr_cma]

colors = [plt.cm.Paired(i/float(k)) for i in range(k)]

f, axarr = plt.subplots(nrows=4, ncols=2, figsize=(25,20))

for i in range(4):
    axarr[i,0].plot(t[start_ind:end_ind], ws[i][start_ind:end_ind])
    for n in range(len(ms[i])):
        xmin_ind = start_ind + n * signal_length
        xmax_ind = xmin_ind + signal_length
        axarr[i, 0].axvspan(t[xmin_ind], t[xmax_ind], color=colors[ms[i][n]], alpha=.2)
    axarr[i, 0].set_xlim(start_ind, end_ind)
    axarr[i, 0].set_ylim(319.8, 320.6)

    for j in range(k):
        axarr[i, 1].plot(cbs[i][j], lw=2, color=colors[j])

plt.show()
# Number in each class in the entire data from mouse 1, and in just the subset we classified.
#print np.bincount(vq(_w1, cb)[0], minlength=k)
#print np.bincount(m, minlength=k)





get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

w1 = np.load('../data/exp1/Water_1.npy')
t = np.load('../data/exp1/time.npy')

start = 120000
stop = 150000
data = w1[start:stop]

fs = np.fft.fft(data)
f, axarr = plt.subplots(nrows=3, figsize=(20,10))
axarr[0].plot(t[start:stop], data)
axarr[0].set_ylabel('Mouse 1\nwater trace')
axarr[0].set_xlabel('Time since start (s)')

axarr[1].plot(np.log10(np.abs(fs)))
axarr[1].set_ylabel('log10(amplitdues)\nFourier spectrum')
axarr[1].set_xlabel('Component')

axarr[2].plot(np.log10(np.abs(fs[:100])))
axarr[2].set_ylabel('fs[:100]\nlog10(amplitudes)')
axarr[2].set_xlabel('Component')

plt.subplots_adjust(hspace=.4)
plt.show()
print 'Top 10 Components (Magnitudes, Frequencies)'
for i in range(10):
    print '\t'.join(map(str, ['Component %s' % i, np.abs(fs[i])*(2./(stop-start)**.5), i/float(stop-start)]))

from IPython.display import Image
Image('./drinking/w1_fourier_spectrum.png')

# start = 0
# stop = 2342138
# data = w1[start:stop]

# fs = np.fft.fft(data)
# f, axarr = plt.subplots(nrows=3, figsize=(20,10))
# axarr[0].plot(t[start:stop], data)
# axarr[0].set_ylabel('Mouse 1\nwater trace')
# axarr[0].set_xlabel('Time since start (s)')

# axarr[1].plot(np.log10(np.abs(fs)))
# axarr[1].set_ylabel('log10(amplitdues)\nFourier spectrum')
# axarr[1].set_xlabel('Component')

# axarr[2].plot(np.log10(np.abs(fs[:100])))
# axarr[2].set_ylabel('fs[:100]\nlog10(amplitudes)')
# axarr[2].set_xlabel('Component')

# plt.subplots_adjust(hspace=.4)
# plt.show()
# print 'Top 10 Components (Magnitudes, Frequencies)'
# for i in range(10):
#     print '\t'.join(map(str, ['Component %s' % i, np.abs(fs[i])*(2./(stop-start)**.5), i/float(stop-start)]))

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from bcp.stats import centered_moving_average

w1 = np.load('/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/Water_1.npy')
t = np.load('/Users/wdwvt/Desktop/Sonnenburg/cumnock/bcp/data/exp1/time.npy')

data = w1[120000:150000]
cma_data = centered_moving_average(data, 5)

f, axarr = plt.subplots(nrows=2)
axarr[0].plot(np.log10(np.abs(np.fft.fft(data))))
axarr[1].plot(np.log10(np.abs(np.fft.fft(cma_data))))
plt.show()



