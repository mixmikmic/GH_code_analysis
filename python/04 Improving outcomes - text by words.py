# Imports
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

# Training Data
training_raw = pd.read_table("../data/training_data.dat")
df_training = pd.DataFrame(training_raw)

# test Data
test_raw = pd.read_table("../data/test_data.dat")
df_test = pd.DataFrame(test_raw)

# target names
target_categories = ['Unclassified','Art','Aviation','Boating','Camping /Walking /Climbing','Collecting']
target_values = ['1','528','529','530','531','532']

# features
feature_names_integers = ['Barcode','UnitRRP']
training_data_integers = df_training[feature_names_integers].values
training_data_integers[:3]

df_training['Description'][:3]

# Rather than Vectorizing the string as a whole do each word
count_vect = CountVectorizer()
count_vect.fit(df_training['Description'])
training_data_description_vect_matrix = count_vect.transform(df_training['Description'])
training_data_description_vect_matrix.shape

training_data_description_vect_matrix.__class__

training_data_description_vect_matrix

print('Description: "{}" - "todays" word number: {} "pilot" word number: {}').format(
    df_training['Description'][0],count_vect.vocabulary_.get(u'todays'),count_vect.vocabulary_.get(u'pilot'))

# So we work with the vectorized text along side the barcode and price, convert it to an array
training_data_description_vect = training_data_description_vect_matrix.toarray()
training_data_description_vect[0][201:220]

# Using numpy's hstack append the vectorized text to the barcode and price
training_data_combined = np.hstack((training_data_integers,training_data_description_vect))
training_data_combined[0][201:220]

# Train the model
model = DecisionTreeClassifier(random_state=511)
target = df_training["CategoryID"].values
model.fit(training_data_combined, target)

# Do all this again for the test data
test_data_integers = df_test[feature_names_integers].values
test_data_description_vect_matrix = count_vect.transform(df_test['Description'])
test_data_description_vect = test_data_description_vect_matrix.toarray()
test_data_combined = np.hstack((test_data_integers,test_data_description_vect))
test_data_combined[0][:20]

predicted = model.predict(test_data_combined)

expected = df_test["CategoryID"].values

print(metrics.classification_report(expected, predicted,    target_names=target_categories))

print(metrics.confusion_matrix(expected, predicted))

metrics.accuracy_score(expected, predicted, normalize=True, sample_weight=None)

count_vect_stop = CountVectorizer(stop_words='english')
count_vect_stop.fit(df_training['Description'])
training_data_stop_description_vect_matrix = count_vect_stop.transform(df_training['Description'])
training_data_stop_description_vect = training_data_stop_description_vect_matrix.toarray()
training_data_stop_combined = np.hstack((training_data_integers,training_data_stop_description_vect))
model = DecisionTreeClassifier(random_state=511)
model.fit(training_data_stop_combined, target)
test_data_stop_integers = df_test[feature_names_integers].values
test_data_stop_description_vect_matrix = count_vect_stop.transform(df_test['Description'])
test_data_stop_description_vect = test_data_stop_description_vect_matrix.toarray()
test_data_stop_combined = np.hstack((test_data_stop_integers,test_data_stop_description_vect))
predicted_stop = model.predict(test_data_stop_combined)
metrics.accuracy_score(expected, predicted_stop, normalize=True, sample_weight=None)

count_vect_stop.get_stop_words()

count_vect_stop = CountVectorizer(stop_words=['the'])
count_vect_stop.fit(df_training['Description'])
training_data_stop_description_vect_matrix = count_vect_stop.transform(df_training['Description'])
training_data_stop_description_vect = training_data_stop_description_vect_matrix.toarray()
training_data_stop_combined = np.hstack((training_data_integers,training_data_stop_description_vect))
model = DecisionTreeClassifier(random_state=511)
model.fit(training_data_stop_combined, target)
test_data_stop_integers = df_test[feature_names_integers].values
test_data_stop_description_vect_matrix = count_vect_stop.transform(df_test['Description'])
test_data_stop_description_vect = test_data_stop_description_vect_matrix.toarray()
test_data_stop_combined = np.hstack((test_data_stop_integers,test_data_stop_description_vect))
predicted_stop = model.predict(test_data_stop_combined)
metrics.accuracy_score(expected, predicted_stop, normalize=True, sample_weight=None)



