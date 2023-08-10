import pandas as pd

# Columns in Data With Categorical Values- Must LabelEncode them
categorical_cols = ['hitpoint', 'outside.sideline', 
                    'outside.baseline', 'same.side', 
                    'previous.hitpoint', 
                    'server.is.impact.player', 'outcome', 
                    'gender']

# Columns in the Data That Should Be Scaled
scaled_data = ['serve', 'rally', 'speed', 'net.clearance', 
               'distance.from.sideline', 'depth', 
               'player.distance.travelled', 
               'player.impact.depth', 
               'player.impact.distance.from.center', 
               'player.depth', 
               'player.distance.from.center', 
               'previous.speed', 'previous.net.clearance', 
               'previous.distance.from.sideline', 
               'previous.depth', 'opponent.depth', 
               'opponent.distance.from.center', 
               'previous.time.to.net']


train_data = pd.read_csv('tennis_data/mens_train_file.csv')
train_data.head()

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# Encode Categorical Data
def encode_data(data):
    d = defaultdict(LabelEncoder)
    data[categorical_cols] = data[categorical_cols].apply(lambda x: d[x.name].fit_transform(x))
    return data

encode_data(train_data).head()

from sklearn.model_selection import train_test_split
import numpy as np

train_data.drop('id', 1, inplace=True)   
train_data.drop('gender', 1, inplace=True)
train_data.drop('train', 1, inplace=True)

# Split into training and validation sets
train_mens, val_mens = train_test_split(train_data, 
                                        shuffle = True,
                                        test_size=0.2,
                                        random_state=42
                                        )


# Split data into input and outputs
X_train = train_mens.loc[:, train_mens.columns != 'outcome']
y_train = train_mens['outcome']
X_val = val_mens.loc[:, val_mens.columns != 'outcome']
y_val = val_mens['outcome']

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(),
    SVC(probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_val)
    acc = accuracy_score(y_val, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_val)
    ll = log_loss(y_val, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)

