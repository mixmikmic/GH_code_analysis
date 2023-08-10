import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)

## Loading data and creating features. NOTE: df_test won't get used in this notebook

df_train = pd.read_csv('../data/titanic_train.csv')
df_test = pd.read_csv('../data/titanic_test.csv')

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = df_all.replace([np.inf, -np.inf], np.nan)

df_all['Age'].fillna(df_all['Age'].mean(), inplace=True)

df_all['Fare'] = df_all['Fare'].fillna(df_all['Fare'].mean())

df_all['has_cabin'] = df_all['Cabin'].apply(lambda val: 0 if pd.isnull(val) else 1)

df_all.shape, df_test.shape

df_all.columns

df_all = df_all[['Age', 'SibSp', 'Parch', 'Survived', 'Embarked', 'Pclass', 'Sex',
                 'Fare', 'has_cabin', 'PassengerId']]

df_all.set_index('PassengerId', inplace=True)

df_all['Sex'] = df_all['Sex'].map({'male':0, 'female':1})

df_all = pd.concat([df_all, pd.get_dummies(df_all['Embarked'])], axis=1)
df_all.drop('Embarked', axis=1, inplace=True)

df_train = df_all[:df_train.shape[0]]
df_test = df_all[df_train.shape[0]:]

df_train.shape, df_test.shape

df_train.head(5)

X_train, X_test = train_test_split(df_train, test_size=0.3)

Y_train = X_train.pop('Survived')
Y_test = X_test.pop('Survived')

## build a classifier model

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

lw = 1 # line width for plt

model = RandomForestClassifier()
model = model.fit(X_train, Y_train)

# predict_proba will give probability for each class 0, 1.
# We only need valeus for calss 1
y_pred_test = model.predict_proba(X_test)[:,1]

# we will compute ROC and Auc_ROC values

# we are setting thresholds as 0 to 1 with an increment of 0.05
# 1.05 has been taken as an outside of the boundary threshold
thresholds = list(reversed([i/100 for i in range(0, 106, 5)]))

total_positive = len(Y_test[Y_test == 1])
total_negative = len(Y_test[Y_test == 0])

total_positive, total_negative # total number of 

# The idea is to find out at each threshold value what percentage of true positive and false positive 
# cases are detected
# Ideally we want a threshold for which all all true positive cases are detected and no true negative 
# cases are detected
tpr_computed = []
fpr_computed = []
for t_val in thresholds:
    values = y_pred_test >= t_val
    positive_count = sum(Y_test[values] == 1)
    negative_count = len(Y_test[values]) - sum(Y_test[values] == 1)
    tpr_computed.append(positive_count/total_positive)
    fpr_computed.append(negative_count/total_negative)

prev_fpr_value = 0
prev_tpr_value = 0
roc_auc_computed = 0
for fpr, tpr in zip(fpr_computed, tpr_computed):
    # for each change in fpr there is a change in tpr
    # this will form a rectangle and we will compute the area of the rectangle.
    # plus the top part of the rectangle is sort of like a traiangle we will adjust for that.
    fpr_change = (fpr - prev_fpr_value)
    tpr_change = tpr - prev_tpr_value
    roc_auc_computed += prev_tpr_value * fpr_change + 1/2 * fpr_change * tpr_change
    prev_fpr_value = fpr
    prev_tpr_value = tpr

roc_auc_computed # the auc that we have computed

plt.plot(fpr_computed, tpr_computed, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_computed)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Let's compute the metrics using sklearn library
from sklearn.metrics import roc_curve, auc, roc_auc_score

roc_auc_score(Y_test, y_pred_test)

fpr, tpr, _ = roc_curve(Y_test, y_pred_test)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkgreen',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

plt.plot(fpr_computed, tpr_computed, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_computed)
plt.plot(fpr, tpr, color='darkgreen',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Almost overlaps

print("Threshold | TPR  | FPR")
for a, b, c in zip(tpr_computed, fpr_computed, thresholds):
    print("     {0:.2f} |".format(c), "{0:.2f} |".format(a), "{0:.2f}".format(b))





