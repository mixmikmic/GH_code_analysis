import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
df.head()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# compute X & y
X = df.loc[:, 2:].values
y = df.loc[:, 1].values

# transform y from ['M','B'] a.k.a malign & Benign to [0,1]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

p = Pipeline([('scl', StandardScaler()),
              ('pca', PCA(n_components=2)), 
              ('lr', LogisticRegression(random_state=1))])

p.fit(X_train, y_train)
print "Train accuracy: %.3f" %(p.score(X_train, y_train))
print "Test accuracy: %.3f" %(p.score(X_test, y_test))

from IPython.display import Image
Image("/Users/surthi/gitrepos/ml-notes/images/ModelEvaluation.jpg")

