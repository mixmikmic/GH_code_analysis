import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def unpack(ls):
    names, tups = zip(*ls)
    dens = [list(zip(*x)[0]) for x in tups]
    return names, dens
# Get the data, values calculated on cortex
import cPickle as pkl
with open('distant/raw_no_denoise.pkl', 'rb') as f:
    res_raw = pkl.load(f)
    raw_values = list(zip(*res_raw)[1])
with open('distant/wavelet_denoise.pkl', 'rb') as f:
    res_den = pkl.load(f)
with open('distant/preprocessed.pkl', 'rb') as f:
    res_pre = pkl.load(f)
with open('distant/pca.pkl', 'rb') as f:
    res_pca = pkl.load(f)
with open('distant/rpca.pkl', 'rb') as f:
    res_rpca = pkl.load(f)
names, dens = unpack(res_den)
_, pre = unpack(res_pre)
pca_names, pca = unpack(res_pca)
pca_names = ['p' + str(x) for x in pca_names]
rpca_names, rpca = unpack(res_rpca)
rpca_names = ['rp' + str(x) for x in rpca_names]
dens.insert(0, raw_values)
dens.extend(pca)
dens.extend(rpca)
dens.extend(pre)
dens = np.array(dens).T
names = ['raw'] + list(names) + list(pca_names) + list(rpca_names) + ['CMI pre']

# Plotting
plt.plot(range(dens.shape[1]), dens[0, :], '-', color = 'red', label='frob correl')
plt.plot(range(dens.shape[1]), dens[1, :], '-', color = 'green', label = 'diff number 3 cycles')
plt.plot(range(dens.shape[1]), dens[2, :], '-', color = 'blue', label = 'diff number 4 cycles')
axes = plt.gca()
axes.set_ylim([np.min(dens) - .01, np.max(dens) + .01])
axes.margins(0.1, 0.1)
plt.ylabel('Discriminibility')
plt.xlabel('Denoising Method')
plt.title('Denoising Benchmark')
legend = plt.legend(title = 'Metrics', loc='upper center', bbox_to_anchor=(1.2,1), frameon=True, borderpad=1)
legend.get_frame().set_facecolor('white')
plt.xticks(range(dens.shape[1]), names)
plt.show()

names, dens = unpack(res_den)
dens = np.array(dens).T
for i in range(dens.shape[0]):
    dens[i, :] = dens[i, :] - np.max(dens[i, :])
plt.plot(range(dens.shape[1]), dens[0, :], '-', color = 'red', label='frob correl')
plt.plot(range(dens.shape[1]), dens[1, :], '-', color = 'green', label = 'diff number 3 cycles')
plt.plot(range(dens.shape[1]), dens[2, :], '-', color = 'blue', label = 'diff number 4 cycles')
axes = plt.gca()
axes.set_ylim([np.min(dens) - .0001, np.max(dens) + .0001])
axes.margins(0.1, 0.1)
plt.ylabel('Difference from Max Discriminibility')
plt.xlabel('Denoising Method')
plt.title('Denoising Benchmark: Close up on wavelet based')
legend = plt.legend(title = 'Metrics', loc='upper center', bbox_to_anchor=(1.2,1), frameon=True, borderpad=1)
legend.get_frame().set_facecolor('white')
plt.xticks(range(dens.shape[1]), names)
plt.show()

