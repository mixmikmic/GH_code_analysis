from __future__ import print_function, division
import numpy as np
import scipy as sp
np.set_printoptions(suppress=True, precision=3)
np.random.seed(200)

# x[3] is a bogus datapoint where a scraper gave a dumb datapoint
x = np.array([10, 11, 10, 100001, 9, 10, 11])
mean_x = x.mean()
print('Mean x = %.1f' % mean_x)
std_x = x.std(ddof=1)
print('StDev x = %.1f' % std_x)
z_score = (x - mean_x)/std_x
print(z_score)

median_x = np.median(x)
print('Median x = %.1f' % median_x)
zm_score = (x - median_x)/std_x
print(zm_score)

q75, q25 = np.percentile(x, [75, 25])
iqr = q75 - q25
fake_std = iqr/1.349
print('IQR = %.1f' % iqr)
zmi_score = (x - median_x)/fake_std
print(zmi_score)

x2 = [9, 10, 11, 100001]
q75, q25 = np.percentile(x2, [75 ,25])
iqr = q75 - q25
fake_std = iqr/1.349
print('IQR = %.2f :(' % iqr)
zmi2 = (x2 - np.median(x2))/fake_std
print(zmi2)

x3 = [1000, 9, 9, 10, 11, 100001]
q75, q25 = np.percentile(x3, [75, 25])
iqr = q75 - q25
fake_std = iqr/1.349
print('IQR = %.2f :(' % iqr)
zmi3 = (x3 - np.median(x3))/fake_std
print(zmi3)

x4 = np.array([1000, 9, 9, 9, 10, 11, 100001])
q75, q25 = np.percentile(x4, [75, 25])
iqr = q75 - q25
fake_std = iqr/1.349
print('IQR = %.2f :(' % iqr)
zmi4 = (x4 - np.median(x4))/fake_std
print(zmi4)

print('Abs median difference')
print(np.abs(x4 - np.median(x4)))
MAD = np.median(np.abs(x4 - np.median(x4)))
print('MAD = %.3f' % MAD)

# Don't implement these in your code, there's a link to a more computationally efficent method below
# plus this is not vectorized
def find_Sn(x):
    n = len(x)
    outer_med_array = np.empty(n)
    for ind in xrange(n):
        outer_med_array[ind] = np.median(np.abs(x-x[ind]))
    return 1.1926*np.median(outer_med_array)
print('S_%d = %.3f' % (len(x4), find_Sn(x4)))

# Don't do this naive implementation either
from math import floor
from sklearn.cross_validation import LeavePOut
def find_Qn(x):
    n = len(x)
    h = floor(n/2) + 1
    k = int(h*(h-1)/2)
    nchoose2 = int(n*(n-1)/2)
    diff_array = np.empty(nchoose2)
    # There's probably a better/faster way to get the n-choose-2 combinations of indicies, but I'm lazy
    # and I know this works
    lpo = LeavePOut(n, p=2)
    for ind, (train, test) in enumerate(lpo):
        diff_array[ind] = np.abs(x[test[1]] - x[test[0]])
    diff_array.sort()
    print('k = %d' % k)
    print('Sorted array:')
    print(diff_array)
    return 2.2219*diff_array[k-1]  # Since Python zero-indexes, use k-1 for kth order stat
print('Q_%d = %.3f' % (len(x4), find_Qn(x4)))

def find_Pn(x):
    n = len(x)
    h = floor(n/2) + 1
    k = int(h*(h-1)/2)
    nchoose2 = int(n*(n-1)/2)
    sum_array = np.empty(nchoose2)
    # There's probably a better/faster way to get the n-choose-2 combinations of indicies, but I'm lazy
    # and I know this works
    lpo = LeavePOut(n, p=2)
    for ind, (train, test) in enumerate(lpo):
        sum_array[ind] = float(x[test[1]] + x[test[0]])/2
    q75, q25 = np.percentile(sum_array, [75, 25])
    return 1.048*(q75 - q25)
print('P_%d = %.3f' % (len(x4), find_Pn(x4)))

x5 = np.array([10, 9, 9, 9, 10, 11, 100001, 11, 10, 12, 8, 10, 9, 11])
print('P_%d = %.3f' % (len(x5), find_Pn(x5)))

np.random.seed(200)
poisson_test = sp.stats.poisson.rvs(1, size=50)
print('S_%d = %.3f' % (len(poisson_test), find_Sn(poisson_test)))
print('Q_%d = %.3f' % (len(poisson_test), find_Qn(poisson_test)))
print('P_%d = %.3f' % (len(poisson_test), find_Pn(poisson_test)))

# Naive approach, not the clever algorithm.
def find_HL(x):
    n = len(x)
    h = floor(n/2) + 1
    k = int(h*(h-1)/2)
    nchoose2 = int(n*(n-1)/2)
    sum_array = np.empty(nchoose2)
    # There's probably a better/faster way to get the n-choose-2 combinations of indicies, but I'm lazy
    # and I know this works
    lpo = LeavePOut(n, p=2)
    for ind, (train, test) in enumerate(lpo):
        sum_array[ind] = float(x[test[1]] + x[test[0]])/2
    return np.median(sum_array)
print("HL of x_4 = %.3f. We're right at the breakpoint here." % (find_HL(x4)))
print('HL of x_5 = %.3f' % (find_HL(x5)))

q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
       0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
       0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
       0.277, 0.273, 0.269, 0.266, 0.263, 0.26
      ]

q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
       0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
       0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
       0.308, 0.305, 0.301, 0.29
      ]

q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
       0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
       0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
       0.384, 0.38, 0.376, 0.372
       ]

Q90 = {n:q for n,q in zip(range(3,len(q90)+1), q90)}
Q95 = {n:q for n,q in zip(range(3,len(q95)+1), q95)}
Q99 = {n:q for n,q in zip(range(3,len(q99)+1), q99)}

x2.sort()
print(x2)

Q_min = (x2[1]-x2[0])/(x2[-1]-x2[0])
print('Q_min = %.3f' % Q_min)
Q_max = (x2[-1]-x2[-2])/(x2[-1]-x2[0])
print('Q_max = %.3f' % Q_max)
print(Q_max > Q95[len(x2)])

from scipy import stats
def Grubbs_outlier_test(y_i, alpha=0.95):
    """
    Perform Grubbs' outlier test.
    
    ARGUMENTS
    y_i (list or numpy array) - dataset
    alpha (float) - significance cutoff for test

    RETURNS
    G_i (list) - Grubbs G statistic for each member of the dataset
    Gtest (float) - rejection cutoff; hypothesis that no outliers exist if G_i.max() > Gtest
    no_outliers (bool) - boolean indicating whether there are no outliers at specified
    significance level
    index (int) - integer index of outlier with maximum G_i
    
    Code from https://github.com/choderalab/cadd-grc-2013/blob/master/notebooks/Outliers.ipynb
    """
    s = y_i.std()
    G_i = np.abs(y_i - y_i.mean()) / s
    N = y_i.size
    t = stats.t.isf(1 - alpha/(2*N), N-2)
    # Upper critical value of the t-distribution with N − 2 degrees of freedom and a
    # significance level of α/(2N)
    Gtest = (N-1)/np.sqrt(N) * np.sqrt(t**2 / (N-2+t**2))    
    G = G_i.max()
    index = G_i.argmax()
    no_outliers = (G > Gtest)
    return [G_i, Gtest, no_outliers, index]

x_grubbs1 = np.array([9, 10, 11, 9, 10, 100000, 11])
G1, Gtest, noout, indexval = Grubbs_outlier_test(x_grubbs1)
print(G1)
print("%.3f" % Gtest)
print(noout)
print(indexval)



