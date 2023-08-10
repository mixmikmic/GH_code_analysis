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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Funtion for cross-validation over a grid of parameters

def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)
    print ("BEST", gs.best_params_, gs.best_score_, gs.cv_results_)
    best = gs.best_estimator_
    return best

# Create a k-Nearest Neighbors Regression estimator
knn_estimator = KNeighborsRegressor()
#knn_parameters = {"n_neighbors": [1,2,5,10,20,50,100]}
knn_parameters = {"n_neighbors": [1,2,5]}
knn_best = cv_optimize(knn_estimator, knn_parameters, X_train, y_train, score_func='neg_mean_squared_error')

# Fit the best Random Forest and calculate R^2 values for training and test sets
knn_reg=knn_best.fit(X_train, y_train)
knn_training_accuracy = knn_reg.score(X_train, y_train)
knn_test_accuracy = knn_reg.score(X_test, y_test)
print ("############# based on standard predict ################")
print ("R^2 on training data: %0.8f" % (knn_training_accuracy))
print ("R^2 on test data:     %0.8f" % (knn_test_accuracy))

# Calculate the Root Mean Squared Error
np.sqrt(mean_squared_error(knn_reg.predict(X_test),y_test))

sampleds = pd.DataFrame(featuredDataset, columns=(columns_X + columns_y))
sampleds = sampleds.sample(10)
sampleds

y_pred = knn_reg.predict(sampleds.iloc[:,:-2])
y_pred

from sklearn.externals import joblib
joblib.dump(knn_reg, 'k_nearest_model.pkl') 

