import pandas as pd
import numpy as np

import brewer2mpl
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.grid_search import GridSearchCV

train = pd.read_csv('train.csv')

train.info()

train.ix[:,:15].hist(figsize=(16,18),bins=50)
plt.show()

def plotc(c1,c2):
    colors = brewer2mpl.get_map('Set3', 'qualitative', 7).mpl_colors
    
    fig = plt.figure(figsize=(16,8))
    #colors = np.array(list(train.Cover_Type.values))
    
    plt.scatter(c1, c2,s=150, c=colors)
    plt.xlabel(c1.name)
    plt.ylabel(c2.name)
    
plotc(train.Elevation, train.Vertical_Distance_To_Hydrology)

def select_features(data):
    return [feat for feat in data.columns if feat not in ['Id', 'Cover_Type']]

def get_X_y(data, cols=None):
    if not cols:
        cols = select_features(data)
        
    X = data[cols].values
    y = data['Cover_Type'].values
    
    return X,y

def draw_importance_features(data, model=RandomForestClassifier(), limit=15):
    X,y = get_X_y(data)
    cols = select_features(data)
    
    model.fit(X, y)
    feats = pd.DataFrame(model.feature_importances_, index=data[cols].columns)
    return feats.sort_values(by=[0], ascending=False) [:limit].plot(kind='bar', figsize=(10, 6))

X,y = get_X_y(train)

draw_importance_features(train)

def models():
    yield ExtraTreesClassifier()
    #yield RandomForestClassifier() 
    #yield KNeighborsClassifier() 
    #yield DecisionTreeClassifier() 
    #yield AdaBoostClassifier() 
    #yield BaggingClassifier() 
    #yield GradientBoostingClassifier() 

for model in models():
    n = len(X[0])
    parameters = {'n_estimators': [300], 'max_depth': [75], 'min_samples_leaf': [1]}
    grid = GridSearchCV(model, parameters, cv=3, scoring='accuracy')
    grid.fit(X, y)

    print(model.__class__, grid.best_score_, grid.best_params_)

## Feature



def distance(row, field_one, field_two):
    return np.sqrt( np.power(row[field_one], 2) + np.power(row[field_two], 2) )
    
    
def etl(data):
    data['is_positive_vertical_distance_to_hydrology'] = data['Vertical_Distance_To_Hydrology'] > 0
    data['distance_to_hydrology'] = data.apply(lambda x: distance(x, 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'), axis=1)

    data['hd_from_hydrology_to_fire_points'] = data.apply(lambda x: distance(x, 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Fire_Points'), axis=1)
    #data['hd_from_hydrology_to_road_ways'] = ...
    #data['hd_from_fire_points_to_road_ways'] = ...
    
    data['vd_from_hydrology_to_fire_points'] = data.apply(lambda x: distance(x, 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Fire_Points'), axis=1)
    #data['vd_from_hydrology_to_road_ways'] = ...
    
    #data['evdh'] = data['Elevation'] - data['Vertical_Distance_To_Hydrology']
    #data['ehdh'] = data['Elevation'] - data['Horizontal_Distance_To_Hydrology']
    

etl(train)
draw_importance_features(train)



