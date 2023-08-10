import pandas as pd
cars = pd.read_csv('auto.csv')

print(cars.head())

#print out unique values in 'origin' column. These will be the classes. 
unique_origin = cars['origin'].unique()
print(unique_origin)

#create dummy variables for cylinder column
dummy_cylinders = pd.get_dummies(cars['cylinders'],prefix='cylin') 
cars = pd.concat([cars,dummy_cylinders],axis=1)

#create dummy variables for yea column
dummy_years = pd.get_dummies(cars['year'],prefix='yr')
cars = pd.concat([cars,dummy_years],axis=1)

cars=cars.drop('cylinders',axis=1)
cars=cars.drop('year',axis=1)

print(cars.head())

#randomize the data set
import numpy as np
shuffled_rows = np.random.permutation(cars.index)
shuffled_cars = cars.iloc[shuffled_rows]

#split shuffled_cars into 2 DFs: 
#1. a train DF with 70% of the observations
#2. a test DF with the other 30% of observations
highest_train_row = int(cars.shape[0] * .70)
train = shuffled_cars.iloc[0:highest_train_row]
test = shuffled_cars.iloc[highest_train_row:]

from sklearn.linear_model import LogisticRegression

unique_origins = cars["origin"].unique()
unique_origins.sort()


models = {}
features = [c for c in train.columns if c.startswith("cyl") or c.startswith("yr")]

for origin in unique_origins:
    model = LogisticRegression()
    
    X_train = train[features]
    y_train = train["origin"] == origin #this will effectively create binary classes, either the origin (1) or not (0)

    model.fit(X_train, y_train)
    models[origin] = model #put model for particular origin class in the 'models' dictionary

testing_probs = pd.DataFrame(columns=unique_origins)

for origin in unique_origins:
    # Select testing features.
    X_test = test[features]   
    # Compute probability of observation being in the origin.
    testing_probs[origin] = models[origin].predict_proba(X_test)[:,1]

testing_probs.head()

predicted_origins = testing_probs.idxmax(axis=1)
print(predicted_origins[0:10])

