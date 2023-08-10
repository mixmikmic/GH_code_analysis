get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from geopy.geocoders import Nominatim
from scipy import cluster
from random import randint
import time

df=pd.read_csv("datasets/yellow_tripdata_2013-01.csv")

#create new column call weekday
timestamp = pd.to_datetime(pd.Series(df['pickup_datetime']))
df['weekday'] = timestamp.dt.weekday_name
df.head()

#drop unnecessary column
df = df.drop(['vendor_id','passenger_count','trip_distance','rate_code',
              'store_and_fwd_flag','payment_type','fare_amount','surcharge','mta_tax',
             'tip_amount','tolls_amount','total_amount','dropoff_datetime',
              'dropoff_longitude','dropoff_latitude'], axis=1)

#get rid off some garbage data
df=df[(df['pickup_latitude'] > 40.492083) & (df['pickup_latitude']<40.944536) &
     (df['pickup_longitude']> -74.267880)& (df['pickup_longitude']< -73.662022)]

df.head()

#regression function for calculating score
def fit_model(X, y):
    model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression(fit_intercept=False))])
    model.fit(X, y)
    return model

def score_model(model, X, y, Xv, yv):
    return tuple([model.score(X, y), model.score(Xv, yv)])

def fit_model_and_score(data, response, validation, val_response):
    model = fit_model(data, response)
    return score_model(model, data, response, validation, val_response)

#convert to address
def convert_to_address(coordinate):
    geolocator = Nominatim()
    location = geolocator.reverse(coordinate)
    return location.address

#get best location for each weekday in a month
def poly_regression(my_month,my_weekday,df):
    #get all the selected weekdays in selected month
    df_select=df[(df['weekday']==my_weekday) & 
                 (pd.to_datetime(df['pickup_datetime']) < pd.datetime(2013,my_month+1,1))&
                (pd.to_datetime(df['pickup_datetime']) > pd.datetime(2013,my_month,1))]
    
    df_select=df_select[:70000]
    #use Kmean to group data by longitude and latitude
    my_cluster=100
    lon=df_select['pickup_longitude'].values
    lat=df_select['pickup_latitude'].values
    coodinate_array=np.array([[lon[i],lat[i]] for i in range(len(lon))])

    kmeans_n = KMeans(n_clusters=my_cluster,n_init=1,random_state=1000)
    kmeans_n.fit(coodinate_array)
    labels = kmeans_n.labels_
    
    # add new column call cluster
    df_select['Cluster']=labels
    
    #prepare for regression
    Cluster_size=df_select.groupby('Cluster').size()
    Cluster_size=np.array([[Cluster_size[i]] for i in range(len(Cluster_size))])
    Cluster_center=kmeans_n.cluster_centers_
    
    #get training data and testing data
    train_size=int(len(Cluster_size)*0.8)
    test_size=int(len(Cluster_size)*0.2)
    train_feature=Cluster_size[:train_size]
    train_response=Cluster_center[:train_size]
    test_feature=Cluster_size[test_size:]
    test_response=Cluster_center[test_size:]
    
    #coefficient of determination (R^2)
    print ("coefficient of determination (R^2):",fit_model_and_score(train_feature, train_response,
                           test_feature, test_response))
    
    #use mean squared error to evaluation model
    MSE_model=Pipeline([('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression(fit_intercept=False))])
    MSE_model.fit(train_feature, train_response)
    X_MSE=(test_feature)
    y_MSE = MSE_model.predict(X_MSE)
    print("MSE: ",mean_squared_error(test_response, y_MSE))
    
    #predict best location
    X=Cluster_size
    y=Cluster_center

    prediction_model=Pipeline([('poly', PolynomialFeatures(degree=3)),
                    ('linear', LinearRegression(fit_intercept=False))])
    prediction_model.fit(X, y)
    X_predict=([max(Cluster_size)])
    y_predict = prediction_model.predict(X_predict)
    print("best location for ",my_weekday, y_predict)
    
    #prepare for visualization
    for data in y_predict:
        visual_x=data[[0]]
        visual_y=data[[1]]
    
    for i in range(len(Cluster_size)):
        if (Cluster_size[i]==Cluster_size.max()):
            max_size_cluster=i
        
    actual_value=kmeans_n.cluster_centers_[max_size_cluster]
    actual_x=actual_value[0]
    actual_y=actual_value[1]
        
    #convert to address
    print("address: ",convert_to_address(str(visual_y[0])+","+str(visual_x[0])))
    
    #visualization for Kmean
    colors = []

    for i in range(my_cluster):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    plt.figure(figsize=(18,9))
    for i in range(my_cluster):
        my_cluster_df=df_select[df_select['Cluster']==i]
        lon_x=my_cluster_df.pickup_longitude.values
        lat_y=my_cluster_df.pickup_latitude.values
        plt.scatter(lon_x,lat_y,alpha=0.2,s=100,c=colors[i])

    plt.axis([visual_x-0.1,visual_x+0.1,visual_y-0.1,visual_y+0.1])
    plt.title("visualization for kmean")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.show()
        
    #scatter plot all the data for selected weekday and prediction(best location in red)
    fig = plt.figure(figsize=(18,9))
    ax = fig.add_subplot(111)
    x_points=lon
    y_points=lat
    ax.scatter(lon,lat,alpha=0.2,s=100)
    ax.scatter(visual_x,visual_y ,c='r',s=100)
    ax.scatter(actual_x,actual_y ,c='y',s=100)
    ax.axis([visual_x-0.05,visual_x+0.05,visual_y-0.05,visual_y+0.05])
    ax.title.set_text("Best pick up location (red point=predicted point, yellow point=actual point)")

dayweek = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
poly_regression(1,dayweek[0],df)

poly_regression(1,dayweek[1],df)

poly_regression(1,dayweek[2],df)

poly_regression(1,dayweek[3],df)

poly_regression(1,dayweek[4],df)

poly_regression(1,dayweek[5],df)

poly_regression(1,dayweek[6],df)



