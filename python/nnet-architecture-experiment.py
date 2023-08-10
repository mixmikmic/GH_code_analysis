get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

import warnings
warnings.filterwarnings("ignore")

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
total_labels = np.array(labels)
# Split training and test:
training_examples = total_array[:100000]
#training_examples = random.sample(total_array, 10)
training_labels = total_labels[:100000]

test_examples = total_array[-1000:]
test_labels = total_labels[-1000:]

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(100), random_state=1)

clf.fit(training_examples, training_labels)   

y_pred = clf.predict(test_examples)

plt.hist(y_pred, bins = 100)

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(10, 10), random_state=1)

clf.fit(training_examples, training_labels) 

y_pred = clf.predict(test_examples)

plt.hist(y_pred, bins=100)

