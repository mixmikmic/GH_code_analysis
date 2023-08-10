#first, import the necessary modules
import pandas
import numpy as np
#scikit-learn is a huge libaray. We import what we need.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC #Linear Suppot Vector Classifier
from sklearn.naive_bayes import MultinomialNB #Naive Bayes classifier
from sklearn.neighbors import KNeighborsClassifier #nearest neighbors classifier
from sklearn.metrics import accuracy_score #to asses the accuracy of the algorithm
from sklearn.model_selection import cross_val_score #to compute cross validation for assessment purposes
from sklearn.cross_validation import cross_val_score #to compute cross validation for assessment purposes

#read our texts and turn them into lists
import os
review_path = '../data/poems/reviewed/'
random_path = '../data/poems/random/'
review_files = os.listdir(review_path)
random_files = os.listdir(random_path)

review_texts = [open(review_path+file_name).read() for file_name in review_files]
random_texts = [open(random_path+file_name).read() for file_name in random_files]

review_texts[0] #notice the strange output here. These poems are saved in a bag of words format

#transform and concat these lists into a Pandas dataframe
df1 = pandas.DataFrame(review_texts, columns = ['body'])
df1['label'] = "review"
df2 = pandas.DataFrame(random_texts, columns = ['body'])
df2['label'] = "random"
df = pandas.concat([df1,df2])
df

##EX: Output some summary statistics for this dataframe. How many poems with the review label, and how many with the random label?
##What is the total number of words in each category? What is the average number of words per poem in each category?

print(df['label'].value_counts())

df['tokens'] = df['body'].str.split()
df['tokens'] = df['tokens'].str.len()
grouped = df.groupby('label')
print(grouped['tokens'].sum())
print(grouped['tokens'].mean())

#randomize our rows
df = df.sample(720, random_state=0)
df

#create two new dataframes
df_train = df[:500]
df_test = df[500:]
print(df_test['label'].value_counts())
df_train['label'].value_counts()

#transform the 'body' column into a document term matrix
tfidfvec = TfidfVectorizer(stop_words = 'english', min_df = 1, binary=True)
countvec = CountVectorizer(stop_words = 'english', min_df = 1, binary=True)

training_dtm_tf = countvec.fit_transform(df_train.body)
test_dtm_tf = countvec.transform(df_test.body)

#create an array for labels
training_labels = df_train.label
test_labels = df_test.label
test_labels.value_counts()

#define a container for our chosen algorithm, in this case multinomial naive bayes
nb = MultinomialNB()

#fit a model on our training set
nb.fit(training_dtm_tf, training_labels)

#predict the labels on the test set using the trained model
predictions_nb = nb.predict(test_dtm_tf) 
predictions_nb

accuracy_score(predictions_nb, test_labels)

#import from sklearn.metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

#precision
precision_score(test_labels, predictions_nb, labels=['random', 'review'], average=None)

recall_score(test_labels, predictions_nb, labels=['random', 'review'], average=None)

f1_score(test_labels, predictions_nb, labels=['random', 'review'], average=None)

confusion_matrix(test_labels, predictions_nb, labels=['random', 'review'])

