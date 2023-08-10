import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Load the dataset
newsgroups = np.load('./resources/newsgroup.npz')

# Define the model
model = MLPClassifier(
    activation='relu',  # Rectifier Linear Unit activation
    hidden_layer_sizes=(5000,),  # 1 hidden layer of size 5000
    max_iter=5,  # Each epochs takes a lot of time so we keep it to 5
    batch_size=100,  # The batch size is set to 100 elements
    solver='adam')  # We use the adam solver

model.fit(newsgroups['train_data'],
          newsgroups['train_target'])

accuracy = accuracy_score(
    newsgroups['test_target'],
    model.predict(newsgroups['test_data']))

print("Accuracy: %.2f\n" % accuracy)

print("Classification Report\n=====================")
print(classification_report(
    newsgroups['test_target'],
    model.predict(newsgroups['test_data'])))

