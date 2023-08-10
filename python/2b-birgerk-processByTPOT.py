import pandas as pd
from tpot import TPOTRegressor
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.preprocessing import Imputer

feature_matrix = pd.read_csv('./example_data.csv', index_col=0)

X = feature_matrix.drop('MEAN(invoices.MEAN(item_purchases.UnitPrice))', axis=1)
y = feature_matrix['MEAN(invoices.MEAN(item_purchases.UnitPrice))']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

y_hat = tpot.predict(X_test)
print("R2 score:", sklearn.metrics.r2_score(y_test, y_hat))

tpot.export('./tpot_pipeline.py')

# %load ./tpot_pipeline.py
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target =     train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    RobustScaler(),
    LinearSVR(C=15.0, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.0001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR

training_features, testing_features, training_target, testing_target =     train_test_split(X_train, y_train, random_state=42)

exported_pipeline = make_pipeline(
    RobustScaler(),
    LinearSVR(C=15.0, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.0001)
)

exported_pipeline.fit(training_features, training_target)

important_coefs = pd.Series(data=exported_pipeline.steps[1][1].coef_, index=X.columns)
sorted_coef = important_coefs.sort_values(ascending=False)

sorted_coef

