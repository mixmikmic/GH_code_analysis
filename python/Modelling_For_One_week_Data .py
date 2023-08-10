#  import those package for modeling 
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')

# Read data file to data frame 
get_ipython().magic("time df_all = pd.read_csv('dublin_2012_week1_distance.csv',dtype={ 'Journey_Pattern_ID': object})")

# Check dataframe first 5 row data 
df_all.head()

# Check last 5 rows 
df_all.tail(3)

# Check all the Journey_Pattern_ID unique value 
df_all['Journey_Pattern_ID'].unique()

#Drop Error Journery Pattern 
df_all.drop(df_all.index[df_all['Journey_Pattern_ID']=='OL77X101'],inplace=True)
df_all.drop(df_all.index[df_all['Journey_Pattern_ID']=='PP071001'],inplace=True)

# Use loc to get the one Journey_Pattern_ID data from the whole data set 
df=df_all.loc[df_all['Journey_Pattern_ID']=='00010001']
df.reset_index(drop=True, inplace=True)

# Check the one Journey_Pattern_ID data size 
df.shape

# Check data set size 
df.shape

# Get those features which are using to train the models 
feature_cols = ['Distance','midweek','HourOfDay']
X = df[feature_cols]
y = df['Trip_Time']
X.columns

# Train the data set with  statsmodels
import statsmodels.formula.api as sm
df_linear = pd.concat([X, y], axis=1)
lm = sm.ols(formula = "Trip_Time ~ Distance+HourOfDay+midweek", data=df_linear).fit()

# Check the model parameters 
lm.params

# Check the summary of the linear regression model 
lm.summary()

# Get the predict of train data set 
lm_predictions = lm.predict(X)

# Plot the pridiction of target feature with the real data 
df.plot(kind='scatter', x='Distance', y='Trip_Time')
plt.plot(X['Distance'], lm_predictions, c='red', linewidth=2)

#plt.savefig('SM_Linear_Reg.png')

# Print the confidence interval of the fitted parameters
lm.conf_int()

# MSE: Mean Squared Error of prediction to real traget feature 
mse=((df_linear.Trip_Time-lm.predict(df_linear))**2).mean()
print("\n Mean Squared Error",mse)

# MAE:  Mean Absolute mean of prediction to real traget feature 
mae = abs(df_linear.Trip_Time-lm.predict(df_linear)).mean()
print("Mean Absolute mean ",mae)

# Import the modules from sklearn package 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Get those features which are using to train the models 
feature_cols = ['Distance','midweek','HourOfDay']
X = df[feature_cols]
y = df['Trip_Time']
X.columns

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
# Using pipeline to use PolynomialFeatures in linear regression to check how to get the best fitting to the data set 

degrees = [2,4,6,10]

#plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X, y)
    scores = cross_val_score(pipeline, X, y,
                             scoring="neg_mean_squared_error", cv=10)

    df.plot(kind='scatter', x='Distance', y='Trip_Time',label="Samples")
    plt.plot(X['Distance'], pipeline.predict(X), c='Blue', label="Model")
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
    plt.show()

# Train the data set with PolynomialFeatures when degree = 2
polynomial_features = PolynomialFeatures(degree=2,include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
pipeline.fit(X, y)

df.plot(kind='scatter', x='Distance', y='Trip_Time',label="Samples")
plt.plot(X['Distance'], pipeline.predict(X), c='Blue', label="Model")

#plt.savefig('Linear_Reg_Poly.png')

# Check the parameter of the linear refression 
pipeline.named_steps['linear_regression'].get_params()

# Check the score of the model 
pipeline.score(X,y)

# MSE: Mean Squared Error
mse=((y-pipeline.predict(X))**2).mean()
print("\n Mean Squared Error",mse)

# MAE:  Mean Absolute Error 
mae = abs(y-pipeline.predict(X)).mean()
print("Mean Absolute Error ",mae)

#Import SVM and assign to a model 
from sklearn import svm
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
clf = svm.SVR()

# Prepare the descriptive features
X = pd.concat([df[['Distance','midweek','HourOfDay']]], axis=1)
y = df.Trip_Time 

#print("Descriptive features:\n", X)
#print("\nTarget feature:\n", y)

# Train the model 
get_ipython().magic('time clf.fit(X, y)')

# Get the predicetion and plot the distance and trip_time of the data and prediction 
y_rbf = clf.predict(X)
plt.scatter(X['Distance'], y, color='darkorange', label='data')
plt.plot(X['Distance'], y_rbf, color='navy', label='RBF model')

# As the plot before couldn't show the result well, we plot the prediction directly
#Plot the distance and trip_time of prediction 
plt.plot(X['Distance'], y_rbf, color='navy', label='RBF model')
plt.show()

# Check the Score of the model 
clf.score(X,y)

# MSE: Mean Squared Error
mse=((y-y_rbf)**2).mean()
print("\n Mean Squared Error",mse)

# MAE:  Mean Absolute Error 
mae = abs(y-y_rbf).mean()
print("Mean Absolute Error ",mae)

# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
# Import MLPRegressor from sklearn.neural_network 
from sklearn.neural_network import MLPRegressor 
mlpreg=MLPRegressor()

# Prepare the descriptive features
X = pd.concat([df[['Distance','midweek','HourOfDay']]], axis=1)
y = df.Trip_Time 

#print("Descriptive features:\n", X)
#print("\nTarget feature:\n", y)

# Train the data with MLP Regressor 
mlpreg.fit(X,y)

# Get the predicetion and plot the distance and trip_time of the real data and prediction 
y_mlpreg = mlpreg.predict(X)
plt.scatter(X['Distance'], y, color='darkorange', label='data')
plt.plot(X['Distance'], y_mlpreg, color='navy', label='ANN model')
#plt.savefig('ANN_result.png')

# Check the model score 
mlpreg.score(X,y)

# MSE: Mean Squared Error   
# Mean Squared Error of linear: 131771.970239 
mse=((y-mlpreg.predict(X))**2).mean()
print("\n Mean Squared Error",mse)

# MAE:  Mean Absolute Error 
# Mean Absolute Error  of linear:  253.761443277
mae = abs(y-mlpreg.predict(X)).mean()
print("Mean Absolute Error ",mae)

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
#from sklearn.multioutput import MultiOutputRegressor

# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#sklearn.ensemble.RandomForestRegressor
max_depth = 30
regr_rf = RandomForestRegressor(max_depth=max_depth,random_state=2)

# Prepare the descriptive features
X = pd.concat([df[['Distance','midweek','HourOfDay']]], axis=1)
y = df.Trip_Time 

#print("Descriptive features:\n", X)
#print("\nTarget feature:\n", y)

regr_rf.fit(X, y)

regr_rf.score(X,y)
#0.94701267864972594

# Get the predicetion 
y_regr_rf = regr_rf.predict(X)
plt.scatter(X['Distance'], y, color='darkorange', label='data')
plt.scatter(X['Distance'], y_regr_rf, color='navy', label='MRF model')
plt.savefig('MRF_result.png')

# MSE: Mean Squared Error   
# Mean Squared Error of linear: 131771.970239 
mse=((y-y_regr_rf)**2).mean()
print("\n Mean Squared Error",mse)

# MAE:  Mean Absolute Error 

# Mean Absolute Error  of linear:  253.761443277
mae = abs(y-y_regr_rf).mean()
print("Mean Absolute Error of MRF ",mae)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Prepare the data set for modeling 
feature_cols = ['Distance','midweek','HourOfDay']
X = df_all[feature_cols]
y = df_all['Trip_Time']
X.columns

# Train the data set with PolynomialFeatures when degree = 2
polynomial_features = PolynomialFeatures(degree=2,include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
get_ipython().magic('time pipeline.fit(X, y)')

# Check model score 
pipeline.score(X,y)

# MSE: Mean Squared Error
mse=((y-pipeline.predict(X))**2).mean()
print("\n Mean Squared Error",mse)

# MAE:  Mean Absolute Error 
mae = abs(y-pipeline.predict(X)).mean()
print("Mean Absolute Error ",mae)

# Add the Journey_Pattern_ID in to train data set 
feature_cols = ['Journey_Pattern_ID','Distance','midweek','HourOfDay']
X = df_all[feature_cols]
y = df_all['Trip_Time']
X.columns

X.shape

# Used the LabelEncoder to convert object column to numberical 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for col in X.columns.values:
    if X[col].dtypes=='object':
        # Using whole data to form an exhaustive list of levels
        data=X[col]
        le.fit(data.values)
        X[col]=le.transform(X[col])

X.head()

# Train the data set with PolynomialFeatures when degree = 2
polynomial_features = PolynomialFeatures(degree=2,include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
get_ipython().magic('time pipeline.fit(X, y)')

#df.plot(kind='scatter', x='Distance', y='Trip_Time',label="Samples")
#plt.plot(X['Distance'], pipeline.predict(X), c='Blue', label="Model")

#plt.savefig('Linear_Reg_Poly.png')

# Check the model's score 
pipeline.score(X,y)

# MSE: Mean Squared Error
mse=((y-pipeline.predict(X))**2).mean()
print("\n Mean Squared Error",mse)

# MAE:  Mean Absolute Error 
mae = abs(y-pipeline.predict(X)).mean()
print("Mean Absolute Error ",mae)

# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#sklearn.ensemble.RandomForestRegressor
max_depth = 30
regr_rf = RandomForestRegressor(max_depth=max_depth,random_state=2)

# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#sklearn.ensemble.RandomForestRegressor
max_depth = 30
regr_rf = RandomForestRegressor(max_depth=max_depth,random_state=2)
# Prepare the descriptive features
X = pd.concat([df_all[['Distance','midweek','HourOfDay']]], axis=1)
y = df_all.Trip_Time 

#print("Descriptive features:\n", X)
#print("\nTarget feature:\n", y)

regr_rf.fit(X,y)

regr_rf.score(X,y)

from sklearn.externals import joblib
joblib.dump(regr_rf,'regr_rf_model_no.sav')

y_regr_rf=regr_rf.predict(X)

#plt.scatter(X['Distance'],y,color='blue', lable='Data')
#plt.plot(X['Distance'],y_regr_rf, color='red', lable='Random Forest')

plt.scatter(X['Distance'], y, color='darkorange', label='data')
plt.savefig('RF_result.png')

plt.scatter(X['Distance'], y_regr_rf, color='navy', label='MRF model')
#plt.get_backend()

# MAE:  Mean Absolute Error 

# Mean Absolute Error  of linear:  253.761443277
mae = abs(y-regr_rf.predict(X)).mean()
print("Mean Absolute Error of RF ",mae)

# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#sklearn.ensemble.RandomForestRegressor
max_depth = 30
regr_rf = RandomForestRegressor(max_depth=max_depth,random_state=2)
# Prepare the descriptive features
X = pd.concat([df_all[['Journey_Pattern_ID','Distance','midweek','HourOfDay']]], axis=1)
y = df_all.Trip_Time 

#print("Descriptive features:\n", X)
#print("\nTarget feature:\n", y)

X.dtypes

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for col in X.columns.values:
    if X[col].dtypes=='object':
        # Using whole data to form an exhaustive list of levels
        data=X[col]
        le.fit(data.values)
        X[col]=le.transform(X[col])

get_ipython().magic('time regr_rf.fit(X,y)')

regr_rf.score(X,y)

# MAE:  Mean Absolute Error 
mae = abs(y-regr_rf.predict(X)).mean()
print("Mean Absolute Error of RF ",mae)

