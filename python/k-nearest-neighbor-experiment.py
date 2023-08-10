get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import preprocessing

labels = []
examples = []
print "GETTING DATASET"
print
# Replace filename with the path to the CSV where you have the year predictions data saved.
filename = "/mnt/c/Users/Aumit/Desktop/YearPredictionMSD.txt/yp.csv"
with open(filename, 'r') as f:
    for line in f:
        content = line.split(",")
        
        labels.append(float(content[0]))

        content.pop(0)

        # If we wanted pure lists
        content = [float(elem) for elem in content]
        #content = map(float, content)

        # If we want a list of numpy arrays, not necessary
        #npa = np.asarray(content, dtype=np.float64)

        examples.append(content)

print "SPLITTING TRAINING AND TEST SETS"
print 
# Turning lists into numpy arrays
total_array = np.array(examples)

# Scale the features so they have 0 mean
#total_scaled = preprocessing.scale(total_array)

# Numpy array of the labels 
total_labels = np.array(labels)

# Split training and test:
training_examples = total_array[:200000]
#training_examples = random.sample(total_array, 10)
training_labels = total_labels[:200000]

# Use the following 1000 examples as text examples
test_examples = total_array[200000:205000]
test_labels = total_labels[200000:205000]

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

indices

distances

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=50)

neigh.fit(training_examples, training_labels)

print training_examples[0]

predictions = neigh.predict(test_examples)

neigh.score(test_examples, test_labels)

predictions.shape

test_labels.shape

squared_dist = []
test_examples = test_examples.astype(float)
test_labels = test_labels.astype(float)
for x, y in zip(test_labels, predictions):
    print str(x) + " --- " + str(y) + " --- " + str(abs(x - y))
    #print y
    #print np.square(x-y)
    #squared_dist.append(np.square(x-y))
    #squared_dist.append(np.linalg.norm(test_labels - predictions))
    squared_dist.append(abs(x - y))
    #print type(y)
#squared_dist

np.mean(squared_dist)

tot = 0
for j in squared_dist:
    tot += j
mean_val = tot/(len(squared_dist))
print mean_val
    

acc_count = 0
for w, p in zip(test_labels, predictions):
    if(abs(w - p) <= 10):
        acc_count += 1

acc_count/float(len(test_labels)) * 100

# Let's create some bins
hist_helper_preds = []
hist_labels = []
for r in predictions:
    if 1920 <= r <= 1930:
        hist_helper_preds.append("1920-1930")
    elif 1931 <= r <= 1940:
        hist_helper_preds.append("1931-1940")
    elif 1941 <= r <= 1950:
        hist_helper_preds.append("1941-1940")
    elif 1951 <= r <= 1960:
        hist_helper_preds.append("1951-1960")
    elif 1961 <= r <= 1970:
        hist_helper_preds.append("1961-1970")
    elif 1971 <= r <= 1980:
        hist_helper_preds.append("1971-1980")
    elif 1981 <= r <= 1990:
        hist_helper_preds.append("1981-1990")
    elif 1991 <= r <= 2000:
        hist_helper_preds.append("1991-2000")
    elif 2001 <= r <= 2011:
        hist_helper_preds.append("2001-2011")
    
for q in test_labels:
    if 1920 <= q <= 1930:
        hist_labels.append("1920-1930")
    elif 1931 <= q <= 1940:
        hist_labels.append("1931-1940")
    elif 1941 <= q <= 1950:
        hist_labels.append("1941-1940")
    elif 1951 <= q <= 1960:
        hist_labels.append("1951-1960")
    elif 1961 <= q <= 1970:
        hist_labels.append("1961-1970")
    elif 1971 <= q <= 1980:
        hist_labels.append("1971-1980")
    elif 1981 <= q <= 1990:
        hist_labels.append("1981-1990")
    elif 1991 <= q <= 2000:
        hist_labels.append("1991-2000")
    elif 2001 <= q <= 2011:
        hist_labels.append("2001-2011")

import pandas 
from collections import Counter
#range_counts = Counter(hist_helper)
#df = pandas.DataFrame.from_dict(range_counts, orient='index')
conv_range_preds = dict((h, hist_helper_preds.count(h)) for h in hist_helper_preds)
conv_range_labels = dict((i, hist_labels.count(i)) for i in hist_labels)
print "Prediction freqs"
print conv_range_preds

print "Label freqs"
print conv_range_labels
#df.plot(kind='bar')
#plt.hist(helper, bins = 10)

#bins = np.arange(-100, 100, 5) # fixed bin size
centers = range(len(conv_range_preds))
centers_labels = range(len(conv_range_labels))

plt.figure(figsize=(10, 3)) 
#plt.bar(centers_labels, conv_range_labels.values(), tick_label=conv_range_labels.keys(),  align='center', width=0.3)
plt.bar(centers, conv_range_preds.values(), tick_label=conv_range_preds.keys() ,align='center', width=0.3)

plt.ylim([0, 200])
#plt.hist(np.array(hist_helper))

plt.figure(figsize=(10, 3)) 
plt.bar(centers_labels, conv_range_labels.values(), tick_label=conv_range_labels.keys(),  align='center', width=0.3)
#plt.bar(centers, conv_range_preds.values(), align='center', width=0.3)

plt.ylim([0, 200])

total_array = np.array(examples)

# Scale the features so they have 0 mean#total_scaled = preprocessing.scale(total_array)
total_scaled = preprocessing.scale(total_array)
# Numpy array of the labels 
total_labels = np.array(labels)
# 
# Split training and test:
# Increase or decrease these sizes
# Currently using first 10000 examples as training data
# Last 1000 as test data
training_examples = total_scaled[:200000]
#training_examples = random.sample(total_array, 10)
training_labels = total_labels[:200000]

# Use the following 1000 examples as text examples
test_examples = total_scaled[200000:205000]
test_labels = total_labels[200000:205000]

neigh_scaled = KNeighborsClassifier(n_neighbors=100)
neigh_scaled.fit(training_examples, training_labels)
predictions_scaled = neigh.predict(test_examples)
neigh_scaled.score(test_examples, test_labels)

squared_dist_scaled = []
test_examples = test_examples.astype(float)
test_labels = test_labels.astype(float)
for f, e in zip(test_labels, predictions):
    print str(f) + " --- " + str(e) + "---" + str(abs(f - e))
    #print y
    #print np.square(x-y)
    #squared_dist.append(np.square(x-y))
    #squared_dist.append(np.linalg.norm(test_labels - predictions))
    squared_dist_scaled.append(abs(f - e))
    #print type(y)
#squared_dist

np.mean(squared_dist_scaled)

acc_count_scaled = 0
for w, p in zip(test_labels, predictions_scaled):
    if(abs(w - p) <= 10):
        acc_count_scaled += 1

acc_count_scaled/float(len(test_labels)) * 100

