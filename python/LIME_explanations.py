import keras
import pickle
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular

model = keras.models.load_model('../models/model_4.h5') 

X_test = pickle.load(open('X_test.p', 'rb'))
cat_cols = pickle.load(open('cat_cols.p', 'rb'))

names = X_test.columns
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_test),
                                                   feature_names=names,
                                                   categorical_features=cat_cols,
                                                   class_names=['r'],
                                                   verbose=True,
                                                   mode='regression')

y_preds = model.predict(np.array(X_test))

lowest = np.argsort(y_preds,axis=0)[:50]
highest =  np.argsort(y_preds,axis=0)[::-1][:50]

obs = [x[0] for x in list(lowest)] + [x[0] for x in list(highest)]

len(obs)

def predict_modified(X):
    """
    This wrapper function takes a numpy array X, predicts values for the 
    array using the keras model specified above, and then converts these 
    predictions from an array of 1-D arrays returned by Keras into a single
    array.
    """
    predicted_vals = model.predict(X) # X is already a numpy array
    return np.array([x[0] for x in predicted_vals[:,]])

get_ipython().run_cell_magic('time', '', 'explanation_dict = {}\nfor i in obs:\n    print("Getting explanation for observation ", str(i))\n    exp = explainer.explain_instance(np.array(X_test)[i,:], predict_modified, num_features=5)\n    explanation_dict[i] = exp.as_list()')

pickle.dump(explanation_dict, open('new_lime_explanations_dict.p', 'wb'))

explanation_dict



