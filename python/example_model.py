import numpy as np
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import cPickle
import os
import json
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
random.seed(0)
# Force matplotlib to not use any Xwindows backend.

# Read the data that Ajay labeled. Convert 1,000 to 1000
from IPython.core.debugger import Tracer
training_data_path = os.path.join(os.getcwd(), 'numerai_training_data.csv')
prediction_data_path = os.path.join(os.getcwd(), 'numerai_tournament_data.csv')
print("Loading data...")
# Load the data from the CSV files
training_data = pd.read_csv(training_data_path, header=0)
prediction_data = pd.read_csv(prediction_data_path, header=0)

# Some clean up. Replace #DIV/0! with 0
# I think 0 is a reasonable, non-biasing number because if, e.g. #Months is 0, a spend per month of 0 is reasonable
training_data.replace(to_replace='#DIV/0!',value='0',inplace=True)
training_data.fillna(0, inplace=True)
prediction_data.replace(to_replace='#DIV/0!',value='0',inplace=True)
prediction_data.fillna(0, inplace=True)
# Transform the loaded CSV data into numpy arrays
features = [f for f in list(training_data) if "feature" in f]
X = training_data[features]
Y = training_data["target"]
X_test = prediction_data[features]
Y_test = prediction_data["target"]
ids = prediction_data["id"]
X.sort_index(axis=1, inplace=True)
X_test.sort_index(axis=1, inplace=True)

X.info()

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

training_data.reset_index( drop = True, inplace = True )
Y = le.fit_transform(Y)

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in X.columns]

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(training_data[col].values, training_data['target'].values)[0,1])
    
ind = np.arange(len(labels))
width = 0.3
fig, ax = plt.subplots(figsize=(10,15))
rects = ax.barh(ind, np.array(values), color='y')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient")
#autolabel(rects)
plt.show()

cols_to_use = ['feature2', 'feature10', 'feature12', 'feature21']

temp_df = training_data[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

X.fillna(0, inplace=True)

#*** Split into training and testing data
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
X_train.sort_index(axis=1, inplace=True)
X_test.sort_index(axis=1,inplace=True)
X_val.sort_index(axis=1,inplace=True)
print (len(X_train))
print (len(X_val))
print (len(X_test))

unique, counts = np.unique(y_train, return_counts=True)
print unique
print counts
dict(zip(unique, counts))

from xgboost import XGBClassifier
model = XGBClassifier() 
model.fit(X_train, y_train)

#scores = cross_val_score(model, X, y, cv=5)
#print (scores.mean())

#*** Test
y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

#*** Get Accuracy
from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_val, y_val_pred)
train_acc = accuracy_score(y_train, y_train_pred)
print ('Train accuracy: ', train_acc)
print ('Test accuracy: ', test_acc)

print("Predicting...")
# Your trained model is now used to make predictions on the numerai_tournament_data
# The model returns two columns: [probability of 0, probability of 1]
# We are just interested in the probability that the target is 1.
y_test_pred = model.predict_proba(X_test)
results = y_test_pred[:, 1]
results_df = pd.DataFrame(data={'probability':results})
joined = pd.DataFrame(ids).join(results_df)

# save the classifier
stats = {"train accuracy": train_acc,"test accuracy":test_acc, 'label':'initial model',}
predictions_path = os.path.join(os.getcwd(), 'predictions.csv')
joined.to_csv(predictions_path, index=False)
model_filename = os.path.join(os.getcwd(),'model.dat')
pickle.dump(model, open(model_filename, 'wb'))
stats_filename = os.path.join(os.getcwd(),'stats.json')
with open(stats_filename, 'wb') as f:
    f.write(json.dumps(stats))
print stats

#scores

unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))

unique, counts = np.unique(y_val, return_counts=True)
dict(zip(unique, counts))

import xgboost as xgb
xgb.plot_importance(model)



