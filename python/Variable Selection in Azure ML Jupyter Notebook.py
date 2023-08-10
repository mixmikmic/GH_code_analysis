from azureml import Workspace
ws = Workspace(
 workspace_id='b2bbeb56a1d04e1599d2510a06c59d87',
 authorization_token='a3978d933cd84e64ab583a616366d160',
 endpoint='https://studioapi.azureml.net'
)
experiment = ws.experiments['b2bbeb56a1d04e1599d2510a06c59d87.f-id.911630d13cbe4407b9fe408b5bb6ddef']
ds = experiment.get_intermediate_dataset(
 node_id='a0a931cf-9fb3-4cb9-83db-f48211be560c-323',
 port_name='Results dataset',
 data_type_id='GenericCSV'
)
frame = ds.to_dataframe()

mydata = frame
mydata.head()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.feature_selection import RFECV

# create X and y
feature_cols = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
X = mydata[feature_cols]
y = mydata.medv

# initiate the linear model
lm = LinearRegression()

# scale the features
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled_minmax = min_max_scaler.fit_transform(X)
x_scaled_minmax_df = pd.DataFrame(x_scaled_minmax, columns = feature_cols)

# recursive feature elimination with cross validation, using r-squared as metric
rfecv = RFECV(estimator=lm, step=1, cv=5, scoring='r2')
rfecv.fit(x_scaled_minmax_df, y)

# print the optimal number of feature
print("Optimal number of features : %d" % rfecv.n_features_)

# save the selected features
feature_cols_selected = list(np.array(feature_cols)[rfecv.support_])
print("Features selected: " + str(feature_cols_selected))

# plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("R-squared")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#%% fit model with selected features
X_new = mydata[feature_cols_selected]
lm2 = LinearRegression()
lm2.fit(X_new, y)

# print the R-squared
print("The R-squared value is: {0:0.4f} \n".format(lm2.score(X_new, y)))
# save intercept and coefficients
param_df = pd.DataFrame({"Features": ['intercept'] + feature_cols_selected, "Coef": [lm2.intercept_] + list(lm2.coef_)})
cols = param_df.columns.tolist()
cols = cols[-1:]+cols[:-1]
param_df = param_df[cols]
print(param_df)

# assign test data
newX = X_new
newY = y

# join predictions with original data
predicted = lm2.predict(newX)
predicted_df = pd.DataFrame({"predicted": predicted})
mydata_with_pd = newX.join(newY).join(predicted_df)
mydata_with_pd.head()

# check performance metrics
import numpy as np
obs = mydata_with_pd.medv
pred = mydata_with_pd.predicted

mae = np.mean(abs(pred-obs))
rmse = np.sqrt(np.mean((pred-obs)**2))
rae = np.mean(abs(pred-obs))/np.mean(abs(obs-np.mean(obs)))
rse = np.mean((pred-obs)**2)/np.mean((obs-np.mean(obs))**2)

print("Mean Absolute Error: {0:0.6f}".format(mae))
print("Root Mean Squared Error: {0:0.6f}".format(rmse))
print("Relative Absolute Error: {0:0.6f}".format(rae))
print("Relative Squared Error: {0:0.6f}".format(rse))

from azureml import services
@services.publish('b2bbeb56a1d04e1599d2510a06c59d87', 'a3978d933cd84e64ab583a616366d160')
@services.types(crim = float, nox=float, rm=float, dis=float, ptratio=float, lstat=float)
@services.returns(float)
def demoserviceupdate(crim, nox, rm, dis, ptratio, lstat):
    feature_vector = [crim, nox, rm, dis, ptratio,  lstat]
    return lm2.predict(feature_vector)

# information about the web service
print("url: " + demoserviceupdate.service.url + "\n")
print("api_key: " + demoserviceupdate.service.api_key + "\n")
print("help_url: " + demoserviceupdate.service.help_url + "\n")
print("service id: " + demoserviceupdate.service.service_id + "\n")

import urllib2
# If you are using Python 3+, import urllib instead of urllib2

import json 


data =  {

        "Inputs": {

                "input1":
                {
                    "ColumnNames": ["crim", "lstat", "nox", "rm", "ptratio", "dis"],
                    "Values": [ [ "0.00632", "4.98", "0.538", "6.575", "15.3", "4.0900" ], 
                               [ "0.02731", "9.14", "0.469", "6.421", "17.8", "4.9671" ], ]
                },        },
            "GlobalParameters": {
}
    }

body = str.encode(json.dumps(data))

url = 'https://ussouthcentral.services.azureml.net/workspaces/b2bbeb56a1d04e1599d2510a06c59d87/services/8c035f34db59446c86e46f9bdfe0ad2a/execute?api-version=2.0&details=true'
api_key = 'F36t+Klp90HQ0WTcvgHIcbHzyJ+/LbYNGesQ9GSugHb23AVSh/b6V03yiV89aqpbT4PnhyTcJYTnWNOOLTeQSQ==' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib2.Request(url, body, headers) 

try:
    response = urllib2.urlopen(req)

    # If you are using Python 3+, replace urllib2 with urllib.request in the above code:
    # req = urllib.request.Request(url, body, headers) 
    # response = urllib.request.urlopen(req)

    result = response.read()
    print(result) 
except urllib2.HTTPError, error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())

    print(json.loads(error.read()))                 

