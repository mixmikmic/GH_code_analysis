# Imports
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd

# Training Data
training_raw = pd.read_table("../data/training_data.dat")
df_training = pd.DataFrame(training_raw)

# test Data
test_raw = pd.read_table("../data/test_data.dat")
df_test = pd.DataFrame(test_raw)
df_test.head()

# target names
target_categories = ['Unclassified','Art','Aviation','Boating','Camping /Walking /Climbing','Collecting']
target_values = ['1','528','529','530','531','532']

# features
feature_names = ['Barcode','Description','UnitRRP']

# Extract features from panda and convert into a dictionary
x_df_training = df_training[feature_names].T.to_dict().values()
x_df_training[:2]

# Create vectorizer class
# We use sparse = False so we get an array rather than scipy.sparse matrix
# This is so we can see the values
vectorizer = DictVectorizer( sparse = False )

# Create a feature to indices mapping
vectorizer.fit( x_df_training )

# vectorizer all the dictionaries
vec_x_df_training = vectorizer.transform( x_df_training )
vec_x_df_training[0]

# Extract target results from panda
target = df_training["CategoryID"].values

# Create classifier class
model = DecisionTreeClassifier(random_state=511)

# train model
model.fit(vec_x_df_training, target)

# Extract test data
x_df_test = df_test[feature_names].T.to_dict().values()
x_df_test[0]

# vectorizer test data
vec_x_df_test = vectorizer.transform( x_df_test )
vec_x_df_test[0]

# Test the model
expected = df_test["CategoryID"].values
predicted = model.predict(vec_x_df_test)

print(metrics.classification_report(expected, predicted,    target_names=target_categories))

print(metrics.confusion_matrix(expected, predicted))

metrics.accuracy_score(expected, predicted, normalize=True, sample_weight=None)

df_training[df_training["CategoryID"]==529][:15]

df_test[df_test["CategoryID"]==529][:15]





