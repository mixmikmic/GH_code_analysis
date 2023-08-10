get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
sys.path.append("..")
from script.dynamic_SAX import Dynamic_SAX
from bisect import insort_left as sinsert
from bisect import bisect_left as sindex

data1 = np.asarray([2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34, 1.42, 3.68, 5.65, 4.23, 6.81])
data2 = np.asarray([0.50, 1.29, 2.58, 3.83, 3.25, 4.25, 3.83, 5.63, 6.44, 6.25, 8.75, 8.83, 3.25, 0.75, 0.72, 1.02, 4.56, 3.21, 5.51, 4.95])
time_series = np.asarray([data1,data2])
plt.plot(range(time_series.shape[1]),time_series[0,:])
plt.plot(range(time_series.shape[1]),time_series[1,:])

alphabet = range(10)
alphabet_size = len(alphabet)
nb_interval = 10
length_interval = 2
window_size = nb_interval * length_interval
window = time_series[:,:window_size].T                                                
sorted_distribution = np.sort(window, axis = 0, kind = 'mergesort')
index_oldest = 0                                                                                     
global_mean = window.mean(axis = 0)                                          
global_variance = window.var(axis = 0)
percentils_index = map(lambda x : (int(x),x%1), [1.*i * (window_size - 1) / alphabet_size for i in xrange(1, alphabet_size)])

window = (window - global_mean) / np.sqrt(global_variance)
sorted_distribution  = (sorted_distribution - global_mean) / np.sqrt(global_variance)
percentils = [[(sorted_distribution[i + 1][k] * j + sorted_distribution[i][k] * (1 - j)) for i, j in percentils_index] for k in xrange(window.shape[1])]
subwin_means = np.asarray(map(lambda xs: xs.mean(axis = 0), np.vsplit(window, nb_interval)))
SAX = np.zeros(subwin_means.shape)
sorted_distribution = sorted_distribution.T.tolist()

for i in xrange(SAX.shape[0]):
    for j in xrange(SAX.shape[1]):
        SAX[i][j] = alphabet[sindex(percentils[j],subwin_means[i][j])]

print SAX

n_point = np.asarray([7.2, 3.8])

print "former mean : ", global_mean
print "former variance : ", global_variance

new_point = (n_point - global_mean) * 1./ np.sqrt(global_variance)
removed_point = window[index_oldest]
temp_mean = global_mean
global_mean = temp_mean + (new_point - removed_point) * 1. * np.sqrt(global_variance) / window_size
global_variance = global_variance + (new_point**2 -removed_point**2 - 2*temp_mean*(new_point - removed_point) - (new_point - removed_point)**2 * 1. / window_size) *1. /window_size 

print "new mean : ", global_mean
print "new variance : ", global_variance
print "reality : ", (np.asarray([7.2, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34, 1.42, 3.68, 5.65, 4.23, 6.81]).std())**2


a = np.asarray([2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34, 1.42, 3.68, 5.65, 4.23, 6.81])
a_moy = a.mean()
a_var = a.var()
a_std = a.std()
a_norm = (a - a.mean())*1./a.std()
a_nmoy = a_norm.mean()
n = 7.2
n_norm = (n - a.mean())*1./a.std()
n_norm
temp = a[0]
temp_n = a_norm[0]
a[0] = n
a_norm[0] = n_norm
print a.mean()
print a_moy + (n - temp)*1./20
print a_moy + (n_norm - temp_n)*1.*a_std/20
print a.var()
print a_var + ((n_norm**2 - temp_n**2 + 2*a_moy*(-temp_n + n_norm))*a_var - 2*a_moy*(n_norm - temp_n)*a_std - ((n_norm - temp_n) * a_var)**2 * 1./20)*1./20

