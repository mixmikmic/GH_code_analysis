import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import norm
from sklearn import linear_model
from sklearn import cross_validation as cval
import sklearn.metrics as metrics
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import xgboost as xgb
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.grid_search import GridSearchCV

#Loading the Boston Housing Dataset
url="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
dataset = pd.read_csv(url, names=names, delim_whitespace = True)

#check the number of cases and features
dataset.shape

#First-hand exploration
dataset.head(5)

#Descriptive statistics
dataset.describe()

#check if data is missing
dataset.apply(lambda x: sum(x.isnull()))
#no missing values

#Plot the diagram depicting correlation of MEDV with various feature variables
corr=dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
k = 14 #you can change the number of variables to display in heatmap in the order of reducing correlation coefficients
cols = corr.nlargest(k, 'MEDV')['MEDV'].index
cm = np.corrcoef(dataset[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Observer the correlation of MEDV with various features and also correlation between features - indicates that 
#MEDV has highest absolute correlation with LSTAT,RM and PTRATIO in that order
#Note that correlation matrix displays Pearson coeefficients (by default) which take into account linear correlation only

sns.pairplot(data=dataset)
plt.show()

#from the pairplots it can be seen that MEDV is linearly related with RM
#However the relationship between MEDV and LSTAT is non-linear and can be made linear by using log transformations, see below
#also observe other scatter plots to investigate more

#separate labels from features
y = dataset['MEDV'].copy()
dataset.drop(labels=['MEDV'], inplace=True, axis=1)

#Split the data into train and test sets
data_train, data_test, labels_train, labels_test = train_test_split(dataset, y, test_size=0.2, random_state=4)

#Apply various regression models

#considering all the 13 features first
features=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
#features=['CRIM', 'ZN',  'CHAS',  'RM','DIS','RAD','TAX','PTRATIO','B','LSTAT'] #Check by removing less important feature - see below
#features=['LSTAT','RM', 'PTRATIO'] #consider top three features having highest correlation with MEDV; check after LSTAT and PTRATIO are log transformed- see below
model = linear_model.LinearRegression()
model = model.fit(data_train[features], labels_train)

pred_train =model.predict(data_train[features]) 
pred_test=model.predict(data_test[features])

#print coefficients
print zip(features, model.coef_)

#print cross-validation score using train data and print R2 scores for training and test data
print "\nCross Validation Score (Train Data) : %s" % "{0:.2%}".format((cval.cross_val_score(model, data_train[features], labels_train, cv=5).mean())) 
print "\nAccuracy: R2 score (Train Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_train,pred_train))
print "Accuracy: R2 score (Test Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_test,pred_test))
print '\nRMSE train: %.3f\nRMSE test: %.3f' % (mean_squared_error(labels_train, pred_train),mean_squared_error(labels_test, pred_test))

R2=metrics.r2_score(labels_train,pred_train)
adjusted_R2 = 1 - (1-R2)*(len(labels_train)-1)/(len(labels_train)-data_train.shape[1]-1)
R2test=metrics.r2_score(labels_test,pred_test)
adjusted_R2test = 1 - (1-R2test)*(len(labels_test)-1)/(len(labels_test)-data_test.shape[1]-1)
print "\nAdjusted R2 score (Train): %s" % "{0:.2%}".format(adjusted_R2)
print "\nAdjusted R2 score (Test): %s" % "{0:.2%}".format(adjusted_R2test)

#Plot scatter of predicted vs. actuals for test set
plt.scatter(pred_test,labels_test)
plt.show()

#Plot Residual vs Predicted scatter plots for Linear Regression
plt.scatter(pred_train, pred_train - labels_train, c='blue', marker='o', label='Training data')
plt.scatter(pred_test,  pred_test - labels_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

#For a good regression model, we would expect that the errors are randomly distributed and the residuals should be randomly scattered around the centerline
#Below scatter diagram indicates presence of certain outliers,and also not so constant variance in errors
#We shall deal with this further below

#QQ plot to assess normality of residuals
res = stats.probplot(residuals, plot=plt)
plt.show()
#The residuals are not normal and there are outliers present

#As we are modeling our data using Linear Regression and we would want to know feature importances let us use OLS method
ols=sm.OLS(labels_train,data_train[features])
result=ols.fit()

result.summary()

#OLS result summary indicates AIC: 2447 (lower the AIC , better is the model)
#High value of condition no. indicates multi-collinearity between features; which can be handled by eliminating certain features
#Features having p-value < 0.05 are more significant
#Test the model by eliminating less important features like INDUS, NOX and AGE (in above cell) - which leads to minor drop in accuracy scores

#Let us try Linear Regression model by exploring some data transformations like log transforming response/feature variables 
#to handle non-linearity, heteroskedasticity (non constant error variance) and non-normality of residuals

#Scatter plot between MEDV and LSTAT
#Try various transformations for LSTAT and MEDV, log transform of LSTAT causes relationship between LSTAT and MEDV to be lnear
#Note that when non-linearity is the major problem, features are log transformed
#if error variance are not constant, that is , if heteroskedasticity is the main problem, response variable is transformed
#if residuals do not adhere to any assumptions both features/ response need to be transformed exploring log, sqrt and other transformations
fig, (axis1,axis2, axis3,axis4) = plt.subplots(1,4,figsize=(15,5))
axis1.scatter(x=dataset.LSTAT, y=y)
axis2.scatter(x=np.log(dataset.LSTAT), y=y)
axis3.scatter(x=np.log(dataset.LSTAT),y=np.log(y))
axis4.scatter(x=np.log(dataset.LSTAT),y=np.sqrt(y))
plt.show()

#Scatter plot between MEDV and PTRATIO
fig, (axis1,axis2, axis3,axis4) = plt.subplots(1,4,figsize=(15,5))
axis1.scatter(x=dataset.PTRATIO, y=y)
axis2.scatter(x=np.log(dataset.PTRATIO), y=y)
axis3.scatter(x=np.log(dataset.PTRATIO),y=np.log(y))
axis4.scatter(x=np.log(dataset.PTRATIO),y=np.sqrt(y))
plt.show()

#Similarly we will log transform PTRATIO

dataset.LSTAT=np.log(dataset.LSTAT)
dataset.PTRATIO=np.log(dataset.PTRATIO)

#Now use RM and transformed LSTAT and PTRATIO (as these have highest absolute correlation with MEDV) as features 
#to test if accuracy scores have improved or not - improved but little
#From different pairplots we noted that there are some non-linear relations between MEDV and features which can be handled to
#some extent by transforming predictor variables

#To inverse these transformations you can use-
dataset.LSTAT=np.exp(dataset.LSTAT)
dataset.PTRATIO=np.exp(dataset.PTRATIO)

labels_train.shape

#Now let us explore PolynomialFeatures which generates a new feature matrix consisting of all polynomial combinations 
#of the features with degree less than or equal to the specified degree
quadratic = PolynomialFeatures(degree=2)
quadratic.fit(data_train[features])
data_train_quad=quadratic.transform(data_train[features])
data_test_quad=quadratic.transform(data_test[features])

pr=linear_model.LinearRegression()
pr.fit(data_train_quad,labels_train)

pred_test_quad=pr.predict(data_test_quad)
pred_train_quad=pr.predict(data_train_quad)

print "\nCross Validation Score (Train Data) : %s" % "{0:.2%}".format((cval.cross_val_score(pr, data_train[features], labels_train, cv=5).mean())) 
print "\nAccuracy: R2 score (Train Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_train,pred_train_quad))
print "Accuracy: R2 score (Test Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_test,pred_test_quad))
print '\nRMSE train: %.3f\nRMSE test: %.3f' % (mean_squared_error(labels_train, pred_train_quad),mean_squared_error(labels_test, pred_test_quad))

R2=metrics.r2_score(labels_train,pred_train_quad)
adjusted_R2 = 1 - (1-R2)*(len(labels_train)-1)/(len(labels_train)-data_train.shape[1]-1)
R2test=metrics.r2_score(labels_test,pred_test_quad)
adjusted_R2test = 1 - (1-R2test)*(len(labels_test)-1)/(len(labels_test)-data_test.shape[1]-1)
print "\nAdjusted R2 score (Train): %s" % "{0:.2%}".format(adjusted_R2)
print "\nAdjusted R2 score (Test): %s" % "{0:.2%}".format(adjusted_R2test)

#Compare the results of Polynomial Regression with those of Linear Regression
#We can notice considerable increase in accuracy and adjusted R2 and drop in RMSE, with CV score remaining more-or-less same

#It should be noted that modeling non-linear transformation by Ploynomial Regression as above is not the best way to
#approach non-linear problems as it can lead to overfitting and over-complex models
#As we've seen above Non-linear relationship between MEDV and LSTAT can be handled by simple log transformation of LSTAT

#For some of the machine learning & dimensionality reduction algorithms like PCA, SVC etc. which are based on distance function,
#it is required that data is scaled or standardised

scaled=preprocessing.StandardScaler()
scaled.fit(data_train[features])
data_train_std=scaled.transform(data_train[features])
data_test_std=scaled.transform(data_test[features])

from sklearn.svm import SVR
svr= SVR(kernel='linear',degree=2)
svr.fit(data_train_std,labels_train)
pred_train=svr.predict(data_train_std)
pred_test=svr.predict(data_test_std)
print "\nCross Validation Score (Train Data) : %s" % "{0:.2%}".format((cval.cross_val_score(svr, data_train_std, labels_train, cv=5).mean())) 
print "\nAccuracy: R2 score (Train Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_train,pred_train))
print "Accuracy: R2 score (Test Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_test,pred_test))
print '\nRMSE train: %.3f\nRMSE test: %.3f' % (mean_squared_error(labels_train, pred_train),mean_squared_error(labels_test, pred_test))

R2=metrics.r2_score(labels_train,pred_train)
adjusted_R2 = 1 - (1-R2)*(len(labels_train)-1)/(len(labels_train)-data_train_std.shape[1]-1)
R2test=metrics.r2_score(labels_test,pred_test)
adjusted_R2test = 1 - (1-R2test)*(len(labels_test)-1)/(len(labels_test)-data_test_std.shape[1]-1)
print "\nAdjusted R2 score (Train): %s" % "{0:.2%}".format(adjusted_R2)
print "\nAdjusted R2 score (Test): %s" % "{0:.2%}".format(adjusted_R2test)

#accuracy and RMSE scores are better when data is standardized

# Random Forests
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=12345)
#rf = RandomForestRegressor(random_state=12345, n_estimators=400,max_depth=7,min_samples_split=2) #parameter tuning using GridSearch - see below
rf.fit(data_train[features], labels_train)
pred_test = rf.predict(data_test[features])
pred_train =rf.predict(data_train[features]) 

#print cross-validation score using train data and print R2 scores for training and test data
print "\nCross Validation Score (Train Data) : %s" % "{0:.2%}".format((cval.cross_val_score(rf, data_train[features], labels_train, cv=5).mean())) 
print "\nAccuracy: R2 score (Train Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_train,pred_train))
print "Accuracy: R2 score (Test Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_test,pred_test))
print '\nRMSE train: %.3f\nRMSE test: %.3f' % (mean_squared_error(labels_train, pred_train),mean_squared_error(labels_test, pred_test))

R2=metrics.r2_score(labels_train,pred_train)
adjusted_R2 = 1 - (1-R2)*(len(labels_train)-1)/(len(labels_train)-data_train.shape[1]-1)
R2test=metrics.r2_score(labels_test,pred_test)
adjusted_R2test = 1 - (1-R2test)*(len(labels_test)-1)/(len(labels_test)-data_test.shape[1]-1)
print "\nAdjusted R2 score (Train): %s" % "{0:.2%}".format(adjusted_R2)
print "\nAdjusted R2 score (Test): %s" % "{0:.2%}".format(adjusted_R2test)

#print feature importances
print "\nFeature Importances: %s" %zip(features, rf.feature_importances_)

#It can be noted that although RMSE on train set is considerably low as compared to linear regression, RMSe on test set is large
#This indicates over-fitting which can be handled by properly tuning the RF, let us use GridSearchCV to tune our RF model

#GridSearch for parameter tuning of Random Forest model
param_grid = { "max_depth" : [4,5,6,7], "min_samples_split" : [2,3,4], "n_estimators":[200,300,400]}
grid_search = GridSearchCV(rf, param_grid, n_jobs=-1, cv=5)
grid_search.fit(data_train[features], labels_train)
print (grid_search.best_params_)
print grid_search.best_score_

#Let us use these parameters to re-evaluate the accuracy scores of random forest model

gbm = xgb.XGBRegressor(seed=777)
#gbm=xgb.XGBRegressor(n_estimators=400,max_depth=7,min_child_weight=3,learning_rate=0.1,gamma=0.05) #parameter tuning using GridSearch - see below
gbm.fit(data_train[features],labels_train)

pred_train =gbm.predict(data_train[features]) 
pred_test=gbm.predict(data_test[features])

#print cross-validation score using train data and print R2 scores for training and test data
print "\nCross Validation Score (Train Data) : %s" % "{0:.2%}".format((cval.cross_val_score(gbm, data_train[features], labels_train, cv=5).mean())) 
print "\nAccuracy: R2 score (Train Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_train,pred_train))
print "Accuracy: R2 score (Test Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_test,pred_test))
print '\nRMSE train: %.3f\nRMSE test: %.3f' % (mean_squared_error(labels_train, pred_train),mean_squared_error(labels_test, pred_test))

R2=metrics.r2_score(labels_train,pred_train)
adjusted_R2 = 1 - (1-R2)*(len(labels_train)-1)/(len(labels_train)-data_train.shape[1]-1)
R2test=metrics.r2_score(labels_test,pred_test)
adjusted_R2test = 1 - (1-R2test)*(len(labels_test)-1)/(len(labels_test)-data_test.shape[1]-1)
print "\nAdjusted R2 score (Train): %s" % "{0:.2%}".format(adjusted_R2)
print "\nAdjusted R2 score (Test): %s" % "{0:.2%}".format(adjusted_R2test)

#It can be noted that although RMSE on train set is considerably low as compared to linear regression, RMSe on test set is large
#This indicates over-fitting which can be handled by properly tuning the XGB, let us use GridSearchCV to tune our XGB model

#GridSearch for parameter tuning of Random Forest model
param_grid = { "max_depth" : [4,5,6,7], "learning_rate" : [0.01,0.05,0.1], "n_estimators":[200,300,400], "gamma":[0.05,0.1],"min_child_weight":[2,3,4]}
grid_search = GridSearchCV(gbm, param_grid, n_jobs=-1, cv=5)
grid_search.fit(data_train[features], labels_train)

print grid_search.best_score_

#Let us use these parameters to re-evaluate the accuracy scores of XGBoost model
#You can further tune this model by using a broader range of parameters and then narrowing down to get the best set of parameters

plt.scatter(pred_test,labels_test)
plt.show()

# plot feature importance
from xgboost import plot_importance
plot_importance(gbm)
plt.show()

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls', 'random_state':15325}
gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(data_train[features], labels_train)
pred_test=gbr.predict(data_test[features])
pred_train=gbr.predict(data_train[features])

#print cross-validation score using train data and print R2 scores for training and test data
print "\nCross Validation Score (Train Data) : %s" % "{0:.2%}".format((cval.cross_val_score(gbr, data_train[features], labels_train, cv=5).mean())) 
print "\nAccuracy: R2 score (Train Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_train,pred_train))
print "Accuracy: R2 score (Test Data) : %s" % "{0:.2%}".format(metrics.r2_score(labels_test,pred_test))
print '\nRMSE train: %.3f\nRMSE test: %.3f' % (mean_squared_error(labels_train, pred_train),mean_squared_error(labels_test, pred_test))

R2=metrics.r2_score(labels_train,pred_train)
adjusted_R2 = 1 - (1-R2)*(len(labels_train)-1)/(len(labels_train)-data_train.shape[1]-1)
R2test=metrics.r2_score(labels_test,pred_test)
adjusted_R2test = 1 - (1-R2test)*(len(labels_test)-1)/(len(labels_test)-data_test.shape[1]-1)
print "\nAdjusted R2 score (Train): %s" % "{0:.2%}".format(adjusted_R2)
print "\nAdjusted R2 score (Test): %s" % "{0:.2%}".format(adjusted_R2test)

#You can tune this further by using GridSearchCV

#We can tune and compare various models for accuracy and RMSE values

