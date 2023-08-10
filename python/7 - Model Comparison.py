import os
import time
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./data/sf/data_clean_engineered.csv')
features = [feature for feature in df.columns if feature != 'price']
X = df[features]
y = df['price']

def load_models():
    model_filenames = glob.glob('./models/sf/*.pkl')
    models = []
    for filename in model_filenames:
        # skip the simple linear model
        if os.path.basename(filename) == 'simple_linear.pkl':
            continue
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            models.append(model)
    return models
models = load_models()

# try brand new data
actual_price = '$899,000'
sqft = 1430
bed = 3
bath = 1
property_type = 'house'
postal_code = '94110'
new_data = {'sqft': sqft,
            'bed': bed,
            'bath': bath,
            'property_type_{}'.format(property_type): 1,
            'postal_code_{}'.format(postal_code): 1
           }
new_df = pd.get_dummies(pd.DataFrame(data=[new_data], columns=X.columns).fillna(0))

for model in models:
    predicted_price = model.predict(new_df)
    print("method: {}".format(model.__class__))
    print("predicted price: ${}M".format(predicted_price[0]/1e6))
print("actual price: {}".format(actual_price))

from IPython.display import YouTubeVideo
YouTubeVideo(id='Un9zObFjBH0')

def avg_prediction(models: list) -> float:
    """Get average prediction from a list of models"""
    predictions = []
    for model in models:
        predictions.append(model.predict(new_df))
    predictions = np.array(predictions) # convert to numpy array
    return np.average(predictions)

avg_pred = avg_prediction(models)
print(f"average predicted price: ${avg_pred/1e6}M")



