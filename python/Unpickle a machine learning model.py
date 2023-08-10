from IPython.core.display import HTML
css_file = 'style.css'
HTML(open(css_file, 'r').read())

from numpy import array

import pickle

# Opening the pickle file
# The rb stands for read binary
model_pkl = open("Random_forest_regressor_model.pkl", "rb")

# Reading the model
model = pickle.load(model_pkl)

# Calling the model
model

# Confirming the number of features
model.n_features_

# The importance of each feature
model.feature_importances_

# Testing the probability of a positive outcome of a new example
new_patient = array([[3, 16, 9, 22, 1, 0, 0, 0, 1, 0, 0, 1, 0]])
model.predict(new_patient)



