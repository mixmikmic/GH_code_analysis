import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
df_wine = pd.read_csv('../data/wine.data')
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
                   'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanolds', 
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 
                   'Hue', 'OD280/OD315 of diluted wines', 'Proline']
# Only use class 2 and 3
df_wine = df_wine[df_wine['Class label']!=1]
y = le.fit_transform(df_wine['Class label'].values)
X = df_wine[['Alcohol', 'Hue']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.tree     import DecisionTreeClassifier
from sklearn.metrics  import accuracy_score

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)

# Fit the decision tree
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred  = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test  = accuracy_score(y_test,  y_test_pred)
print 'Decision tree train/test accuracies %.3f/%.3f'%(tree_train, tree_test)

from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(base_estimator=tree, 
                        n_estimators=500, 
                        max_samples=1., 
                        max_features=1., 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=1, 
                        random_state=1)

# Fit the bagging method
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred  = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test  = accuracy_score(y_test,  y_test_pred)
print 'Bagging tree train/test accuracies %.3f/%.3f'%(bag_train, bag_test)

# Plot the decision regions
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(10,5))

for idx, clf, label in zip([0, 1], [tree, bag], ['Decision', 'Bagging']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter( X_train[y_train==0,0], X_train[y_train==0,1], c='blue', marker='^', s=50)
    axarr[idx].scatter( X_train[y_train==1,0], X_train[y_train==1,1], c='red',  marker='o', s=50) 
    axarr[idx].set_title(label)
    
plt.text(10,  -0.8, s='Hue',     ha='center', va='center', fontsize=12)
plt.text(4.5, 1,    s='Alcohol', ha='center', va='center', fontsize=12, rotation=90)
plt.show()



