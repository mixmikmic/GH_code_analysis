import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

# Setting some Pandas options.
pd.set_option('display.precision', 2)
pd.set_option('display.max_columns', None)

# Path to the directory containing datasets.
path = '../tests/fixtures/'

# Load both trained model and scaler.
model = joblib.load('../gabbar/trained/model.pkl')
scaler = joblib.load('../gabbar/trained/scaler.pkl')

# Load dataset used for regression testing.
regression = pd.read_csv(path + 'regression.csv')
regression.head(10)

# Prepare training vectors and target values for the model.
non_training_attributes = ['changeset_id', 'changeset_harmful']

X = regression.drop(non_training_attributes, axis=1)
y = regression['changeset_harmful']

# Scale training vectors.
X_scaled = scaler.transform(X)

# Get predictions from the model.
regression['prediction'] = model.predict(X_scaled)

regression.head(10)

# What does the confusion matrix look like?
matrix = confusion_matrix(y, regression['prediction'])

matrix = pd.DataFrame(matrix, index=['Labelled good', 'Labelled harmful'], columns=['Predicted good', 'Predicted harmful'])
matrix

