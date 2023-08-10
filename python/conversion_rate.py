import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('~/Downloads/conversion_data.csv')
df['country'] = df['country'].astype('category')
df['source'] = df['source'].astype('category')

df.head()

sorted(df['age'],reverse=True)[:10]

print(df.loc[df['age']==123])
print(df.loc[df['age']==111])

df.drop([90928,295581],inplace=True)

sns.barplot(x='country',y='converted',data=df)

sns.barplot(x='source',y='converted',data=df)

data_pages = df.groupby('total_pages_visited')[['converted']].mean()

data_ages = df.groupby('age')[['converted']].mean()

plt.plot(data_pages.index,data_pages['converted'])

plt.plot(data_ages.index,data_ages['converted'])

df_country = pd.get_dummies(df['country'])
df = pd.concat([df, df_country], axis=1)
df_source = pd.get_dummies(df['source'])
df = pd.concat([df, df_source], axis=1)
df.head()

X_train, X_test, y_train, y_test = train_test_split(df[['total_pages_visited','China','Germany','UK','US','Ads','Direct','Seo','age','new_user']], df['converted'], test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, np.array(y_train))

forest.score(X_test,y_test)

1-sum(df['converted'])/len(df['converted'])

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[['China','Germany','UK','US','Ads','Direct','Seo','age','new_user']], df['converted'], test_size=0.33, random_state=42)
forest = RandomForestClassifier(random_state=42,class_weight={0: 0.7, 1: 0.3})
forest.fit(X_train, np.array(y_train))

forest.score(X_test,y_test)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3,class_weight={0: 0.7, 1: 0.3})
clf.fit(X_train, y_train)

tree.export_graphviz(clf,out_file='tree.dot')

from io import StringIO
import pydotplus
from IPython.display import Image
out = StringIO()
tree.export_graphviz(clf, out_file = out,feature_names=['China','Germany','UK','US','Ads','Direct','Seo','age','new_user'],  
                         class_names=['converted'])
 
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())



