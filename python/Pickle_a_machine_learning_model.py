from IPython.core.display import HTML
css_file = 'style.css'
HTML(open(css_file, 'r').read())

from pandas import read_excel, get_dummies, concat
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

import pickle

data = read_excel("Pickle_model.xlsx", sheetname = 1)

data.head()

# The outcome vector
y = data.pop("Outcome")

# Variable data types
# Note that the Outcome variable has been removed by the .pop() method above
data.dtypes

# Defining a value_counts type function for categorical values
def describe_categorical(X):
    """
    Returns the .describe method values when called on categorical variables in a dataset.
    """
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))

describe_categorical(data)

# Creating dummy variables
categorical_variables = ["Cat1", "Cat2", "Cat3"]

for variable in categorical_variables:
    data[variable].fillna("Missing", inplace = True)
    dummies = get_dummies(data[variable], prefix = variable)
    data = concat([data, dummies], axis = 1)
    data.drop([variable], axis = 1, inplace = True)

data.head()

model = RandomForestRegressor(100, oob_score = True, n_jobs = -1, random_state = 42)
model.fit(data, y)
print("Area under the curve: ", roc_auc_score(y, model.oob_prediction_))

# Open the file to save as pkl file
# The wb stands for write and binary
decision_tree_model_pkl = open("Random_forest_regressor_model.pkl", "wb")

# Write to the file (dump the model)
# Open the file to save as pkl file
pickle.dump(model, decision_tree_model_pkl)

# Close the pickle file
decision_tree_model_pkl.close()



