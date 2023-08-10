from sml import execute

query = 'READ "../data/auto-mpg.csv" (separator = "\s+", header = None)'

execute(query, verbose=True)

import pandas as pd

#Names of all of the columns
names = [
       'mpg'
    ,  'cylinders'
    ,  'displacement'
    ,  'horsepower'
    ,  'weight'
    ,  'acceleration'
    ,  'model_year'
    ,  'origin'
    ,  'car_name'
]

#Import dataset
data = pd.read_csv('../data/auto-mpg.csv', sep = '\s+', header = None, names = names)

data.head()





