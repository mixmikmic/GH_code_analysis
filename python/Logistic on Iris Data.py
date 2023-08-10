import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris_data = load_iris()

features = iris_data.feature_names
nbFeatures = np.shape(features)[0]

data = iris_data.data
nbPoints = np.shape(data)[0]

targets = iris_data.target

import plotly.plotly as py
from plotly.tools import FigureFactory as FF
plotly.tools.set_credentials_file(username='juanjo.neri', api_key='$*n^o%ne$of$your$(:bus#in!ess')
## Great idea plotly! now everyone on github knows my credentials!

featuresMatrix = np.append(features, data).reshape((nbPoints + 1, nbFeatures)) # Add title lables
targetsColumn = np.append(['Type'], targets).reshape((nbPoints + 1, 1))

allTogether = np.append(featuresMatrix, targetsColumn, 1)

table = FF.create_table(allTogether[0:nbPoints:10])
py.iplot(table)

# I think this data is in memory like 3 times with different names already
X = iris_data.data
y = iris_data.target

# This is just following the usual workflow of sk
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X, y)
predictions = logreg.predict(X)

# So sad we have to use pretty table again :(
from prettytable import PrettyTable as PT

## Reduce data for output
predictions_min = predictions[0:nbPoints:10]
targets_min = targets[0:nbPoints:10]

lables = ['expected', 'actual']

t = PT(lables)
# Magic numbers everywhere!
for i in range(int(nbPoints/10)):
    t.add_row( [ predictions_min[i], targets_min[i] ] )

print(t)

# How much accuracy does that account for?
# See if we can do better

from sklearn import metrics
print(str(int(100*metrics.accuracy_score(targets, predictions))) + "% accuracy on train data") #nice cast nesting btw



