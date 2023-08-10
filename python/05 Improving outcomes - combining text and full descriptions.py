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

df_training['NewDescription'] = df_training['Description'].replace({' ': ''}, regex=True)
df_training['FullDescription'] = df_training[['Description','NewDescription']].apply(lambda x: ' '.join(x), axis=1)
df_training.head()

df_test['NewDescription'] = df_test['Description'].replace({' ': ''}, regex=True)
df_test['FullDescription'] = df_test[['Description','NewDescription']].apply(lambda x: ' '.join(x), axis=1)
df_test.head()

# Rather than Vectorizing the string as a whole do each word in the new full description
count_vect = CountVectorizer()
count_vect.fit(df_training['FullDescription'])
training_data_description_vect_matrix = count_vect.transform(df_training['FullDescription'])
training_data_description_vect_matrix.shape

# So we work with the vectorized text along side the barcode and price, convert it to an array
training_data_description_vect = training_data_description_vect_matrix.toarray()
training_data_description_vect[0][420:]

print('Description: "{}" - "todays" word number: {} "pilot" word number: {}  "todayspilot" word number: {}').format(
    df_training['Description'][0],count_vect.vocabulary_.get(u'todays'),count_vect.vocabulary_.get(u'pilot'),
    count_vect.vocabulary_.get(u'todayspilot'))

# Using numpy's hstack append the vectorized text to the barcode and price
training_data_combined = np.hstack((training_data_integers,training_data_description_vect))
training_data_combined[0][:20]

# Train the model
model = DecisionTreeClassifier(random_state=511)
target = df_training["CategoryID"].values
model.fit(training_data_combined, target)

# Do all this again for the test data
test_data_integers = df_test[feature_names_integers].values
test_data_description_vect_matrix = count_vect.transform(df_test['FullDescription'])
test_data_description_vect = test_data_description_vect_matrix.toarray()
test_data_combined = np.hstack((test_data_integers,test_data_description_vect))
test_data_combined[0][:20]

predicted = model.predict(test_data_combined)
expected = df_test["CategoryID"].values

print(metrics.classification_report(expected, predicted,    target_names=target_categories))

print(metrics.confusion_matrix(expected, predicted))

metrics.accuracy_score(expected, predicted, normalize=True, sample_weight=None)

df_training.head()

df_training['Description'].str.split()

df_training['Description'].str.split().apply(lambda x: ' '.join(x))

# Lets try sorting the description
for index, row in df_training.iterrows():
    str_list = row['Description'].lower().split()
    str_list.sort()
    result = ''.join(str_list)
    df_training.set_value(index,'SortedDescription', result)
df_training.head()

df_training['FullDescription2'] = df_training[['Description','SortedDescription']].apply(lambda x: ' '.join(x), axis=1)
df_training.head()

# Rather than Vectorizing the string as a whole do each word in the new full description
count_vect = CountVectorizer()
count_vect.fit(df_training['FullDescription2'])
training_data_description_vect_matrix = count_vect.transform(df_training['FullDescription2'])
training_data_description_vect_matrix.shape

training_data_description_vect = training_data_description_vect_matrix.toarray()
training_data_combined = np.hstack((training_data_integers,training_data_description_vect))
# Train the model
model = DecisionTreeClassifier(random_state=511)
model.fit(training_data_combined, target)

for index, row in df_test.iterrows():
    str_list = row['Description'].lower().split()
    str_list.sort()
    result = ''.join(str_list)
    df_test.set_value(index,'SortedDescription', result)
df_test['FullDescription2'] = df_test[['Description','SortedDescription']].apply(lambda x: ' '.join(x), axis=1)
test_data_description_vect_matrix = count_vect.transform(df_test['FullDescription2'])
test_data_description_vect = test_data_description_vect_matrix.toarray()
test_data_combined = np.hstack((test_data_integers,test_data_description_vect))

predicted = model.predict(test_data_combined)
print(metrics.classification_report(expected, predicted,    target_names=target_categories))
print(metrics.confusion_matrix(expected, predicted))
metrics.accuracy_score(expected, predicted, normalize=True, sample_weight=None)



