# Tables, Queries, and Stats
import pandas as pd
import numpy as np

# Plotting
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sbn

# Data Partitioning
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

# Decision Tree Model and Plotting
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# Ensemble Methods
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# Error Analysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('../data/conversion_data.csv')
df_orig = df.copy()

df.head()

# Create dummy variables for Category values
df_is_country = pd.get_dummies(df['country'])
df_source = pd.get_dummies(df['source'])

# Add columns to dataframe
df[df_is_country.columns] = df_is_country
df[df_source.columns] = df_source

# Drop original columns
df = df.drop('country', axis=1)
df = df.drop('source', axis=1)

df.head()

# Create a data subset to work with
df_stats = df_orig.copy()
df_stats = df_stats.drop(['new_user', 'country', 'source', 'converted'], axis=1)

fig, ax = plt.subplots(figsize=(8,6))
sbn.boxplot(data=df_stats)

df_stats_of_converted = df_orig.copy()
df_stats_of_converted = df_stats_of_converted.drop(['country', 'new_user', 'source'], axis=1)

fig, ax = plt.subplots(figsize=(8,6))
sbn.boxplot(x='converted', y='total_pages_visited', data=df_stats_of_converted)

def get_features(df, label):
    df_features = df.copy()
    df_features = df_features.drop(label, axis=1)
    return df_features
    
def get_labels(df, label):
    return pd.DataFrame(df[label])

def partition_data(features, labels):
    return train_test_split(features, labels, 
                           test_size=0.3,
                           random_state=549)

all_features = get_features(df, 'converted')
all_labels = get_labels(df, 'converted')
train_features, test_features, train_labels, test_labels = partition_data(all_features, all_labels)
train_labels = np.ravel(train_labels)
test_labels = np.ravel(test_labels)

def train_and_test_model(model):
    print()
    print(model)
    print()
    print('Training model...')
    fitted = model.fit(train_features, train_labels)
    print('Test the model.')
    predicted = fitted.predict(test_features)
    predicted_binary = predicted.round()
    accuracy = accuracy_score(predicted_binary, test_labels)
    print('This model\'s accuracy is {}'.format(accuracy))
    return predicted, accuracy

def get_predictions_and_scores(all_models):
    '''
    Store the predictions and scores in a list of tuples, that can be sorted based on a key. The key is the
    accuracy of each model.
    
    Returns the highest score and corresponding prediction of labels.
    '''
    predictions_and_accuracies = [train_and_test_model(model) for model in all_models]
    return max(predictions_and_accuracies, key=lambda x: x[1])

all_models = [LinearRegression(), LogisticRegression(), BernoulliNB()]
max_score_for_prediction = get_predictions_and_scores(all_models)

print('The best model has a score of {:.2f} percent.'.format(max_score_for_prediction[1]*100))

# Refer to Seaborn Heatmap docs here to select color map:
# http://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap
def plot_confusion_matrix(cm, labels,
                          title='Confusion matrix',
                         cmap=None):
    
    plt.figure(figsize=(8,8))

    sbn.heatmap(cm,
               annot=True,
               fmt='d',
               linewidths=0.5,
               cmap=cmap)
    
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(test_labels, max_score_for_prediction[0])
plot_confusion_matrix(cm, 
                     labels=['No Conversion','Conversion'],
                     cmap="YlGnBu")

baseline_accuracy = accuracy_score(np.zeros(len(test_labels)), test_labels)
print('The baseline for predicting no conversions for all instances is {:.2f}.'.format(baseline_accuracy*100))

def run_model(model):
    print()
    print('Training the model (Fitting to the training data) ')
    fitted = model.fit(train_features, train_labels)
    print('Fitted model: {}'.format(fitted))
    predicted_labels = fitted.predict(test_features)
    accuracy = accuracy_score(predicted_labels, test_labels)
    print('The accuracy for the decision tree classifier is {}'.format(accuracy*100))
    return fitted

def decision_tree():
    return run_model(DecisionTreeClassifier(max_leaf_nodes=9))

dt_classifier = decision_tree()

feature_names = all_features.columns
label_names = all_labels.columns
dot_data = export_graphviz(dt_classifier, out_file=None,
               feature_names=feature_names,
                           # check to see if I need to dummy variablize the labels before
                           # running Decision Tree
               class_names=['no conversion', 'conversion'],
               filled=True, rounded=True,
               special_characters=True)
#graph = graphviz.Source(dot_data)
graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())

run_model(BaggingClassifier(DecisionTreeClassifier(max_leaf_nodes=20),
         max_features=.9, max_samples=0.9));

from math import sqrt
sqrt(len(df)) / len(df)

run_model(RandomForestClassifier(
    n_estimators=100, max_leaf_nodes=20)
    );



