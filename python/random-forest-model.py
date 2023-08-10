import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geohash

featuredDataset = pd.read_csv('featured-dataset.csv')
featuredDataset = featuredDataset.drop(featuredDataset.columns[0], axis=1)
featuredDataset.head(5)

featuredDataset.shape


# Get the longitude and latitude from the geohash
def decodegeo(geo, which):
    if len(geo) >= 6:
        geodecoded = geohash.decode(geo)
        return geodecoded[which]
    else:
        return 0
    
def further_data_prep(df):  

    df['start_lat'] = df['location_start'].apply(lambda geo: decodegeo(geo, 0))
    df['start_lon'] = df['location_start'].apply(lambda geo: decodegeo(geo, 1))
    df['end_lat'] = df['location_end'].apply(lambda geo: decodegeo(geo, 0))
    df['end_lon'] = df['location_end'].apply(lambda geo: decodegeo(geo, 1))
    
    return df

featuredDataset = further_data_prep(featuredDataset)
featuredDataset.head(5)
featuredDataset.columns

columns_all_features = featuredDataset.columns
columns_X = ['day_num', 'x_start', 'y_start', 'z_start']
columns_y = ['end_lat', 'end_lon']
X = featuredDataset[columns_X]
y = featuredDataset[columns_y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print ('X: ({}, {})'.format(*X.shape))
print ('y: ({}, {})'.format(*y.shape))
print ('X_train: ({}, {})'.format(*X_train.shape))
print ('y_train: ({}, {})'.format(*y_train.shape))
print ('X_test: ({}, {})'.format(*X_test.shape))
print ('y_test: ({}, {})'.format(*y_test.shape))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
# Set the parameters by cross-validation
tuned_parameters = {'n_estimators': [2, 5, 10, 20, 40], 'max_depth': [None, 1, 2, 3, 4], 'min_samples_split': [2, 3, 4, 5, 6]}


# clf = ensemble.RandomForestRegressor(n_estimators=500, n_jobs=1, verbose=1)
gridCV = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, n_jobs=-1, verbose=1)
gridCV.fit(X_train, y_train)
print (gridCV.best_estimator_)

reg = gridCV.best_estimator_
training_accuracy = reg.score(X_train, y_train)
valid_accuracy = reg.score(X_test, y_test)
rmsetrain = np.sqrt(mean_squared_error(reg.predict(X_train),y_train))
rmsevalid = np.sqrt(mean_squared_error(reg.predict(X_test),y_test))
print (" R^2 (train) = %0.6f, R^2 (valid) = %0.6f, RMSE (train) = %0.6f, RMSE (valid) = %0.6f" % (training_accuracy, valid_accuracy, rmsetrain, rmsevalid))

importances = reg.feature_importances_
std = np.std([tree.feature_importances_ for tree in reg.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
feature_names = X_train.columns
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names)
plt.xlim([-1, X_train.shape[1]])
plt.show()

sampleds = pd.DataFrame(featuredDataset, columns=(columns_X + columns_y))
sampleds = sampleds.sample(10)
sampleds

y_pred = reg.predict(sampleds.iloc[:,:-2])
y_pred

from sklearn.externals import joblib
joblib.dump(reg, 'random_forest_model.pkl') 

