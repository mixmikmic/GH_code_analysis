# import libraries for matrix manipulation and data visualization

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data

data = pd.read_csv('activity_recognition/1.csv')

data.head()

# renaming columns
data.columns = ['sample', 'x', 'y', 'z', 'target']
data.head()

# calculate and add norms for each data point
data['m'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)

# check for missing data
nan_flag = False
for c in data.columns:
    if any(data[c] == np.nan):
        print c, 'contains NaNs'
        nan_flag = True
if not nan_flag:
    print 'No missing values.'

# examine class distribution
data['target'].value_counts()/float(len(data))

plt.style.use('fivethirtyeight')

for c in range(1, 8):
    plt.plot(data[data['target'] == c]['m'], linewidth=.5)
plt.xlim(0, len(data))
plt.ylim(3400, 4100)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    # helper function to return coefficients for scipy.lfilter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # applies a lowpass filter
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    # helper function to return coefficients for scipy.lfilter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    # applies a highpass filter
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 6
fs = 52.0       # sample rate, Hz
cutoff = 2.     # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.plot(0.5*fs*w/np.pi, np.abs(h))
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 5)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')

# test the filter on a sample of the dataset
filter_test_data = data[data['target'] == 2]['z']
filter_test_x = xrange(len(filter_test_data))
filter_test_y = butter_lowpass_filter(filter_test_data, cutoff, fs, order)
plt.plot(filter_test_x, filter_test_data, linewidth=1, label='data')
plt.plot(filter_test_x, filter_test_y, linewidth=2, label='filtered data')
plt.xlabel('t')
plt.legend(loc='upper left')
plt.xlim(0, len(filter_test_data))
plt.ylim(1900, 2300)

# apply the filters
for c in ['x', 'y', 'z', 'm']:
    data[c+'l'] = butter_lowpass_filter(data[c], cutoff, fs, order)
    data[c+'h'] = butter_highpass_filter(data[c], cutoff, fs, order)

from scipy.signal import argrelmin, argrelmax

def rms(series):
    # returns root mean square value of a series
    return np.sqrt((series**2).mean())


def min_max_mean(series):
    # returns the average of the differences between local mins/maxs
    mins = argrelmin(series)[0]    # indices of the local minima
    maxs = argrelmax(series)[0]    # local maxima
    min_max_sum = 0
    # build the sums, then take the average
    if len(mins) <= len(maxs):
        for j, arg in enumerate(mins):
            min_max_sum += series[maxs[j]] - series[arg]
    else:
        for j, arg in enumerate(maxs):
            min_max_sum += series[arg] - series[mins[j]]
    return min_max_sum/float(min(len(mins), len(maxs)))


def extract_features(data, y, num_windows):
    window_len = len(data)/(num_windows/2)
    i = 0    # initialize index
    features = []
    targets = []
    
    for n in range(num_windows):
        # isolate window
        win = data[i:i+window_len]
        
        # extract target
        target = int(y[i:i+window_len].mode())
        targets.append(target)
        
        for c in data.columns:
            # extract features for each series
            s = np.array(win[c])
            rms_val = rms(s)
            min_max = min_max_mean(s)
            mean = s.mean()
            std = s.std()
            new_features = [rms_val, min_max, mean, std]
            features.append(new_features)
        # update index
        i += window_len/2
    features = np.array(features)
    features.shape = num_windows, 48
    targets = np.array(targets)
    return features, targets

features = data.drop(['sample', 'target'], axis=1)
targets = data['target']

X, y = extract_features(features, targets, 208)

from sklearn.cross_validation import train_test_split, cross_val_score

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=1)

# majority class prediction
from collections import Counter

label_counts = Counter(y_test)
null = label_counts.most_common()[0][1]/float(len(y_test))
print null

import sklearn.metrics as met

# helper function to quickly build different models
def model_build(model):
    model.fit(X_train, y_train)
    return model

# helper function to handle evaluation
def model_eval(model):
    pred = model.predict(X_test)
    print 'Accuracy:\n-----------------------------'
    print met.accuracy_score(y_test, pred)
    print '\nConfusion Matrix:\n-----------------------------'
    print met.confusion_matrix(y_test, pred)
    print '\nClassification Report:\n-----------------------------'
    print met.classification_report(y_test, pred)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = model_build(KNeighborsClassifier())
model_eval(knn)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = model_build(RandomForestClassifier(n_estimators=20))
model_eval(rfc)

# Blending the models by voting
from sklearn.ensemble import VotingClassifier

vote = model_build(VotingClassifier(estimators=[('rf', rfc), ('knn', knn)]))
model_eval(vote)

# Random Forest with weighted classes

rfc_bal = model_build(RandomForestClassifier(n_estimators=20, class_weight='balanced'))
model_eval(rfc_bal)

