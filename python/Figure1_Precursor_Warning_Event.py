import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'font.size': 20})

events = np.load('Datasets\DataExamples.npy') # features, # samples
labels = np.load('Datasets\LabelExamples.npy') # samples, 1
times = np.load('Datasets\TimeExamples.npy') # features, # samples
times = (times - times[0,:]) * 3600 * 24 # set time to 0 and in seconds

sample1 = 8 # index
sample2 = 1 # index

precursor = int(256 * times.shape[0] / (2 * (256 + 60)))
warning = precursor + int(60 * times.shape[0] / (2 * (256 + 60)))

fig = plt.figure(figsize=(15, 10))
fig.add_subplot(2, 1, 1)
plt.plot(times[:precursor,sample1], events[:precursor, sample1], '-')
plt.plot(times[precursor:warning, sample1], events[precursor:warning, sample1], '-')
plt.plot(times[warning:,sample1], events[warning:, sample1], '-')
plt.title('Label: ' + str(labels[sample1]))
plt.legend(['Precursor period', 'Warning period', 'Event period'])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude ')
plt.axis([0, 632, -10000, 10000])

fig.add_subplot(2, 1, 2)
plt.plot(times[:precursor,sample2], events[:precursor, sample2], '-')
plt.plot(times[precursor:warning, sample2], events[precursor:warning, sample2], '-')
plt.plot(times[warning:,sample2], events[warning:, sample2], '-')
plt.title('Label: ' + str(labels[sample2]))
plt.legend(['Precursor period', 'Warning period', 'Event period'])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude ')
plt.axis([0, 632, -10000, 10000])
plt.tight_layout()
plt.savefig('Images\data_example_fig1.png', bbox_inches='tight', dpi=400)



