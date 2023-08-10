import os
import pandas as pd

DATA_PATH = "datasets/human_resource_analytics"

def load_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, "HR_comma_sep.csv")
    return pd.read_csv(csv_path)

hra = load_data()
hra

hra.info()

hra["salary"].value_counts()

hra["sales"].value_counts()

hra.describe()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
hra.hist(bins=50, figsize=(20, 15))
plt.show()

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(hra, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")

import numpy as np

hra["satisfaction_cat"] = np.ceil(hra["satisfaction_level"] * 10) / 10

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(hra, hra["satisfaction_cat"]):
    strat_train_set = hra.loc[train_index]
    strat_test_set = hra.loc[test_index]

print(len(strat_train_set), "train +", len(strat_test_set), "test")

strat_train_set["satisfaction_cat"].value_counts() / len(strat_train_set)

strat_test_set["satisfaction_cat"].value_counts() / len(strat_test_set)

for set in (strat_train_set, strat_test_set):
    set.drop(["satisfaction_cat"], axis=1, inplace=True)

hra = strat_train_set

corr_matrix = hra.corr()
corr_matrix["left"].sort_values(ascending=False)

from pandas.tools.plotting import scatter_matrix

attributes = ["left", "time_spend_company", "average_montly_hours", "number_project", "last_evaluation", "promotion_last_5years", "Work_accident", "satisfaction_level"]
scatter_matrix(hra[attributes], figsize=(12, 8))

hra = strat_train_set.drop("left", axis=1)
hra_labels = strat_train_set["left"].copy()

import pandas
from collections import Counter

department_counts = Counter(hra["sales"])
df = pandas.DataFrame.from_dict(department_counts, orient='index')

df.plot(kind='bar')

# attribution: http://stackoverflow.com/a/22097018/137996
deparments, department_enums = np.unique(hra["sales"], return_inverse=True)
print("deparments", deparments)
#hra["department_enums"] = department_enums
hra.plot(kind="scatter", x="department_enums", y="satisfaction_level", alpha=0.1)

# attribution: http://stackoverflow.com/a/22097018/137996
salary, salary_enums = np.unique(hra["salary"], return_inverse=True)
print("salary", salary)
#hra["salary_enums"] = salary_enums
hra.plot(kind="scatter", x="salary_enums", y="satisfaction_level", alpha=0.01)

hra = hra.drop("salary_enums", axis=1).drop("department_enums", axis=1)

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
salary_cat = hra["salary"]
salary_cat_encoded = encoder.fit_transform(salary_cat)
salary_cat_encoded

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
salary_cat_1hot = encoder.fit_transform(salary_cat)
salary_cat_1hot

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer, Imputer

hra_numbers = hra.drop("salary", axis=1).drop("sales", axis=1)
num_attribs = list(hra_numbers)

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

salary_pipeline = Pipeline([
    ('selector', DataFrameSelector(["salary"])),
    ('label_binarizer', LabelBinarizer())
])

department_pipeline = Pipeline([
    ('selector', DataFrameSelector(["sales"])),
    ('label_binarizer', LabelBinarizer())
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("salary_pipeline", salary_pipeline),
    ("department_pipeline", department_pipeline),
])

hra_prepared = full_pipeline.fit_transform(hra)
hra_prepared.shape

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(hra_prepared, hra_labels)

def test_some_data(model):
    some_data = hra.iloc[:5]
    some_labels = hra_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("Predictions:\t", model.predict(some_data_prepared))
    print("Labels:\t\t", list(some_labels))

test_some_data(lin_reg)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def rmse(predictions):
    mse = mean_squared_error(hra_labels, predictions)
    return np.sqrt(mse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(hra_prepared, hra_labels)

tree_hra_predictions = tree_reg.predict(hra_prepared)
rmse(tree_hra_predictions)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, hra_prepared, hra_labels,
                        scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)

lin_scores = cross_val_score(lin_reg, hra_prepared, hra_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(hra_prepared, hra_labels)
forest_predictions = forest_reg.predict(hra_prepared)
rmse(forest_predictions)

forest_scores = cross_val_score(forest_reg, hra_prepared, hra_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

test_some_data(forest_reg)

from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_estimators": [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]}
]

grid_search_one = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search_one.fit(hra_prepared, hra_labels)
grid_search_one.best_params_

param_grid = [
    {"n_estimators": [30, 60, 120, 240], 'max_features': [6, 7, 8]}
]

grid_search_two = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search_two.fit(hra_prepared, hra_labels)
grid_search_two.best_params_

param_grid = [
    {"n_estimators": range(122, 130), 'max_features': [8]}
]

grid_search_three = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search_three.fit(hra_prepared, hra_labels)
grid_search_three.best_params_

def cross_val_scores(regressor):
    scores = cross_val_score(regressor, hra_prepared, hra_labels, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    return rmse_scores

cross_val_scores(grid_search_three.best_estimator_)

from sklearn.preprocessing import LabelBinarizer
salary_encoder = LabelBinarizer()
salary_1hot = salary_encoder.fit_transform(hra["salary"])
salary_1hot

from sklearn.preprocessing import LabelBinarizer
department_encoder = LabelBinarizer()
department_1hot = department_encoder.fit_transform(hra["sales"])
department_1hot

feature_importances = grid_search_three.best_estimator_.feature_importances_
print(feature_importances)

attributes = num_attribs + list(salary_encoder.classes_) + list(department_encoder.classes_)
sorted(zip(feature_importances, attributes), reverse=True)

from sklearn.preprocessing import FunctionTransformer

def get_class_index(encoder, class_name):
    return encoder.classes_.tolist().index(class_name)

technical_index = get_class_index(department_encoder, "technical")
sales_index = get_class_index(department_encoder, "sales")
support_index = get_class_index(department_encoder, "support")

department_pipeline_two = Pipeline([
    ('selector', DataFrameSelector(["sales"])),
    ('label_binarizer', LabelBinarizer()),
    ('output_selector', FunctionTransformer(lambda X: X[:,[technical_index, sales_index, support_index]])),
])

full_pipeline_two = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("salary_pipeline", salary_pipeline),
    ("department_pipeline", department_pipeline_two),
])

hra_prepared_two = full_pipeline_two.fit_transform(hra)

param_grid = [
    {"n_estimators": range(50, 111, 5), 'max_features': range(1, 9)}
]

grid_search_four = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search_four.fit(hra_prepared_two, hra_labels)
grid_search_four.best_params_

def cross_val_scores(regressor, data=hra_prepared):
    scores = cross_val_score(regressor, data, hra_labels, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    return rmse_scores

cross_val_scores(forest_reg, hra_prepared_two)

best_search = grid_search_three
final_model = best_search.best_estimator_

X_test = strat_test_set.drop("left", axis=1)
y_test = strat_test_set["left"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse



