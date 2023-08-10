sc

from sklearn import grid_search, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

digits = datasets.load_digits()

X, y = digits.data, digits.target

# 864 combinations
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [10, 20, 40, 80]}

from datetime import datetime
init = datetime.today()

gs = grid_search.GridSearchCV(RandomForestClassifier(), param_grid=param_grid)

gs.fit(X, y)

end = datetime.today()

(end - init).seconds

from sklearn import grid_search, datasets
from sklearn.ensemble import RandomForestClassifier

from spark_sklearn import GridSearchCV

digits = datasets.load_digits()

X, y = digits.data, digits.target

# 864 combinations
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [10, 20, 40, 80]}

from datetime import datetime
init = datetime.today()

gs = GridSearchCV(sc, RandomForestClassifier(), param_grid=param_grid)

gs.fit(X, y)

end = datetime.today()

(end - init).seconds

sc.getConf().getAll()



