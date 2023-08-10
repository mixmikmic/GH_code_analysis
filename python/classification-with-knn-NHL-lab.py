import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict, KFold
import sklearn.preprocessing as preproc
from sklearn import metrics


get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

# web location:
web_csv = 'https://raw.githubusercontent.com/josephofiowa/GA-DSI/master/NHL_Data_GA.csv'
local_csv = './datasets/NHL_Data_GA.csv'

# A:
nhl_df = pd.read_csv(local_csv)
nhl_df.head()

nhl_df.Rank.unique()

# A:
# Dataset looks neat as fuck
nhl_df.info()

nhl_df.describe()

# get top correlation predictors with Rank target
nhl_df_corr = nhl_df.corr()**2
top_ten_pred = nhl_df_corr['Rank'].sort_values(ascending=False)[1:].head(10)
print(top_ten_pred)

def plot_heatmap(df):
    
    corr = df.corr()**2

    fig, ax = plt.subplots(figsize=(20,10))

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    ax = sns.heatmap(corr, mask=mask, ax=ax, annot=True)

    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14)

    plt.show()

plot_heatmap(nhl_df[top_ten_pred.index.tolist()])

# PTS, GF%, CA60, GA60, GA, SF60
plot_heatmap(nhl_df[['PTS','CA60','GA','SF60']])

# A:
y = nhl_df['Rank']
y.unique()

# 3 classes

# A:
y.value_counts()/y.value_counts().sum()

# A:
# X = nhl_df[['PTS','CA60','GA','SF60']]
X = nhl_df[['CF%', 'GF', 'Sh%', 'PDO']]

print(X.shape)
print(y.shape)

# A:
knn1 = KNeighborsClassifier(n_neighbors=1)

knn1_model = knn1.fit(X, y)

# A:
knn1_predictions = knn1_model.predict(X)
acc = metrics.accuracy_score(y.values, knn1_predictions)
print(acc)

# overfitting

# A:
# STEP 1: split X and y into training and testing sets (using random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, random_state=99, test_size=0.5)

# STEP 2: train the model on the training set (using K=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# STEP 3: test the model on the testing set, and check the accuracy
y_pred_class = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

# A:
knn = KNeighborsClassifier(n_neighbors=X_train.shape[0])
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

# A:
# get scores for each K
scores = []
for n in range(1,X_train.shape[0]+1):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred_class = knn.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred_class)
    scores.append(score)

print(len(range(1,len(scores)+1)), len(scores))

# plot scores against k
plt.plot(range(1, X_train.shape[0]+1), scores)

# A:
# since proportion of each class is around the same, no need for stratified sampling
def knn_cross_val_no_stratify(X, y, folds=5, shuf=False):
    max_k = np.floor(X.shape[0] - X.shape[0]/float(folds))
    kf_shuffle = KFold(n_splits=folds, shuffle=shuf)
    scores = []
    for n in range(1,int(max_k)+1):
        knn = KNeighborsClassifier(n_neighbors=n)
        scores.append(np.mean(cross_val_score(knn, X, y, cv=kf_shuffle)))
    
    return scores

# plot score against neighbors
scores = knn_cross_val_no_stratify(X, y, shuf=True)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(1,len(scores)+1), scores, lw=3)

# A:
scaler = preproc.StandardScaler()
Xs = scaler.fit_transform(X)

scores_Xs = knn_cross_val_no_stratify(Xs, y, shuf=True)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(1,len(scores)+1), scores, lw=3)
ax.plot(range(1,len(scores_Xs)+1), scores_Xs, lw=3, color='darkred')



