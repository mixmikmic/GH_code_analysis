import pandas as pd
import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.preprocessing import Imputer

feature_matrix = pd.read_csv('./example_data.csv', index_col=0)

X = feature_matrix.drop('MEAN(invoices.MEAN(item_purchases.UnitPrice))', axis=1)
y = feature_matrix['MEAN(invoices.MEAN(item_purchases.UnitPrice))']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X.values, y.values, random_state=1)

automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120, per_run_time_limit=30,
        tmp_folder='/tmp/autoslearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out')
automl.fit(X_train, y_train, dataset_name='retail')

y_hat = automl.predict(X_test)

list(automl._automl._automl.models_.values())[0]

print("R2 score:", sklearn.metrics.r2_score(y_test, y_hat))

