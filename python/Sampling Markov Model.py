import numpy as np
import hmm_sampling as hmms
from matplotlib import pyplot as plt
from IPython.display import YouTubeVideo
get_ipython().magic('matplotlib inline')

YouTubeVideo('34Noy-7bPAo')

runing_seconds = 40

samples_per_second = 1
states_1 = hmms.get_state_probabilities(samples_per_second, runing_seconds)
x_1 = np.linspace(0,runing_seconds, len(states_1[0]))
curves = plt.plot(x_1, np.array(states_1).T)
plt.legend(curves, ('State 1','State 2','State 3','State 4'), fontsize = 'x-small', loc = 'best')
plt.xlabel('seconds [1 sample/second]')
plt.ylabel('Probability')
plt.show()

samples_per_second = 0.25
states_025 = hmms.get_state_probabilities(samples_per_second, runing_seconds)
x_025 = np.linspace(0,runing_seconds, len(states_025[0]))
curves = plt.plot(x_025, np.array(states_025).T)
plt.legend(curves, ('State 1','State 2','State 3','State 4'), fontsize = 'x-small', loc = 'best')
plt.xlabel('seconds [0.25 samples/second]')
plt.ylabel('Probability')
plt.show()

samples_per_second = 0.5
states_05 = hmms.get_state_probabilities(samples_per_second, runing_seconds)
x_05 = np.linspace(0,runing_seconds, len(states_05[0]))
curves = plt.plot(x_05, np.array(states_05).T)
plt.legend(curves, ('State 1','State 2','State 3','State 4'), fontsize = 'x-small', loc = 'best')
plt.xlabel('seconds [0.5 samples/second]')
plt.ylabel('Probability')
plt.show()

samples_per_second = 1000
states_1000 = hmms.get_state_probabilities(samples_per_second, runing_seconds)
x_1000 = np.linspace(0,runing_seconds, len(states_1000[0]))
curves = plt.plot(x_1000, np.array(states_1000).T)
plt.legend(curves, ('State 1','State 2','State 3','State 4'), fontsize = 'x-small', loc = 'best')
plt.xlabel('seconds [1000 samples/second]')
plt.ylabel('Probability')
plt.show()

f, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize = (20,10))
ax = ax.flatten()
titles = ['State 1 probability', 'State 2 probability', 'State 3 probability', 'State 4 probability']
for i in range(4):
    ax[i].set_title(titles[i])
    ax[i].plot(x_025, states_025[i],color='g', label = '0.25 samples/sec')
    ax[i].plot(x_05, states_05[i],color='y', label = '0.5 samples/sec')
    ax[i].plot(x_1, states_1[i],color='r', label = '1 samples/sec')
    ax[i].plot(x_1000, states_1000[i],color='b', label = '1000 samples/sec')
    ax[i].legend(fontsize = 'x-small')
plt.show()

runing_seconds = 100
samples_per_second = 1000
states_1000 = hmms.get_state_probabilities(samples_per_second, runing_seconds)
x_1000 = np.linspace(0,runing_seconds, len(states_1000[0]))
curves = plt.plot(x_1000, np.array(states_1000).T)
plt.legend(curves, ('State 1','State 2','State 3','State 4'), fontsize = 'x-small', loc = 'best')
plt.xlabel('seconds [1000 samples/second]')
plt.ylabel('Probability')
plt.show()

