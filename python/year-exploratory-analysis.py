get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

labels = []
examples = []

# Replace filename with the path to the CSV where you have the year predictions data saved.
filename = "/mnt/c/Users/Aumit/Desktop/YearPredictionMSD.txt/yp.csv"
with open(filename, 'r') as f:
    for line in f:
        content = line.split(",")
        
        labels.append(int(content[0]))

        content.pop(0)

        content = [float(elem) for elem in content]

        # If we want a list of numpy arrays, not necessary
        #npa = np.asarray(content, dtype=np.float64)

        examples.append(content)

total_array = np.array(examples)

total_array

total_labels = np.array(labels)

total_labels

training_examples = total_array[:10000]

training_labels = total_labels[:10000]

plt.hist(training_labels, bins='auto')

plt.show()

training_labels.shape

plt.hist(training_labels, bins=100)

plt.hist(total_labels, bins=100)

test_examples = total_array[-1000:]
test_labels = total_labels[-1000:]

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(100, 100, 100, 100), random_state=1)

clf.fit(training_examples, training_labels)   

y_pred = clf.predict(test_examples)

plt.hist(y_pred, bins=100)

plt.hist(test_labels, bins=100)

plt.hist(training_labels, bins=95)

clf = MLPClassifier(solver='sgd', alpha=10,
                     hidden_layer_sizes=(100, 100, 100, 100), random_state=1)

clf.fit(training_examples, training_labels)                         

y_pred = clf.predict(test_examples)

plt.hist(y_pred, bins=95)

clf = MLPClassifier(solver='sgd', alpha=0.0001,
                     hidden_layer_sizes=(100, 100, 100, 100), random_state=1)

clf.fit(training_examples, training_labels)     

y_pred = clf.predict(test_examples)

plt.hist(y_pred, bins = 95)

training_examples = total_array[:100000]
#training_examples = random.sample(total_array, 10)
training_labels = total_labels[:100000]

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(100, 100, 100, 100), random_state=1)

clf.fit(training_examples, training_labels)  

y_pred = clf.predict(test_examples)

plt.hist(y_pred, bins=95)

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(100, 100, 100, 100), random_state=1, learning_rate="adaptive")

clf.fit(training_examples, training_labels) 

y_pred = clf.predict(test_examples)

plt.hist(y_pred, bins = 95)

label_counts = {}

for item in labels:
    if item not in label_counts:
        label_counts[item] = 1
    else:
        label_counts[item] += 1

label_counts

import math 

familiarity = []
hotttness = []
count = 0

# Replace filename with the path to the CSV where you have the year predictions data saved.
filename = "/mnt/c/Users/Aumit/Documents/GitHub/million-song-analysis/fam_vs_hot.csv"
with open(filename, 'r') as f:
    for line in f:
        if count == 0:
            count += 1
            continue
        else:
        
            content = line.split(",")

            # temp1 = float(content) 
            familiarity.append(float(content[0]))

            #content.pop(0)

            #content = [float(elem) for elem in content]

            # If we want a list of numpy arrays, not necessary
            #npa = np.asarray(content, dtype=np.float64)

            hotttness.append(float(content[1]))
        count += 1
print len(familiarity)
print len(hotttness)
for elem, elem1 in zip(familiarity, hotttness):
    if math.isnan(elem) or math.isnan(elem1):
        ind = familiarity.index(elem)
        ind1 = hotttness.index(elem1)
        del familiarity[ind]
        del hotttness [ind1]

print len(familiarity)
print len(hotttness)

#[value for value in familiarity if not math.isnan(value)]
#[value1 for value1 in hotttness if not math.isnan(value1)]
        
        
total_hotttness = np.array(hotttness)
total_familiarity = np.array(familiarity)


# fit with np.polyfit
m, b = np.polyfit(total_familiarity, total_hotttness, 1)

plt.plot(total_familiarity, total_hotttness, '.')
plt.xlabel("Familiarity")
plt.ylabel("Hotttness")
plt.title("Artist Familiarity vs. Artist Hotttness")
plt.plot(total_familiarity, m*total_familiarity + b, '-')
plt.yticks(np.arange(min(total_hotttness), max(total_hotttness), 0.4))

