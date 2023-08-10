import csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import datasets, linear_model  
from scipy.stats.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt 
from statsmodels.formula.api import ols
import patsy

data = pd.read_csv('data_state.csv',sep=',') 
data['intercept'] = 1.0
data.head()

data.describe()

data.columns

#independent_factors = ['rental price average','Gini coefficients','w/o health insurance%','POP_ST','POPPCT_URBAN','intercept']
independent_factors = ['rental_price_average','Gini_coefficients','without_health_insurance','POP_ST','POPPCT_URBAN']
independent_factors_l = ['Legal_Aid_Attorneys_per_capita','rental_price_average','Gini_coefficients','without_health_insurance','POP_ST','POPPCT_URBAN']
independent_factors_h = ['unsheltered_homelessness_ratio','rental_price_average','Gini_coefficients','without_health_insurance','POP_ST','POPPCT_URBAN']
X = data[independent_factors]
X_l = data[independent_factors_l]
X_h = data[independent_factors_h]

y_legal = data['Legal_Aid_Attorneys_per_capita'].values
y_homeless = data['unsheltered_homelessness_ratio']
y_ratio = data['ratio']
x_rental = data['rental_price_average'].values
x_gini = data['Gini_coefficients']
x_health = data['without_health_insurance']
x_pop = data['POP_ST']
x_urban = data['POPPCT_URBAN']

a = ['Legal_Aid_Attorneys_per_capita','intercept']
b = ['unsheltered_homelessness_ratio','intercept']
y_legal_intercept = data[a]
y_homeless_intercept = data[b]

fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = legal aid, y = homeless')  
ax1.scatter(y_legal*1000,y_homeless*1000,c = 'c',marker = 'o')
plt.show()
print(pearsonr(y_homeless, y_legal))
#(Pearson’s correlation coefficient,2-tailed p-value) 
#this one varies between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply an exact linear relationship. 
#Positive correlations imply that as x increases, so does y. Negative correlations imply that as x increases, y decreases.

fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = rental price, y = legal aid per capita')  
ax1.scatter(x_rental,y_legal*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_rental, y_legal))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = gini, y = legal aid per capita')  
ax1.scatter(x_gini,y_legal*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_gini, y_legal))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = health, y = legal aid')  
ax1.scatter(x_health,y_legal*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_health, y_legal))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = population, y = legal aid per capita')  
ax1.scatter(x_pop,y_legal*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_pop, y_legal))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = urbanization population, y = legal aid per capita')  
ax1.scatter(x_urban,y_legal*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_urban, y_legal))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = gini(income inequlity), y = homeless per capita')   
ax1.scatter(x_gini,y_homeless*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_gini, y_homeless))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x =  without %health insurance, y = homeless per capita')  
ax1.scatter(x_health,y_homeless*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_health, y_homeless))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = population, y = homeless per capita')  
ax1.scatter(x_pop,y_homeless*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_pop, y_homeless))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = urban, y = homeless per capita')  
ax1.scatter(x_urban,y_homeless*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_urban, y_homeless))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = rental, y = homeless per capita')  
ax1.scatter(x_rental,y_homeless*100,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_rental, y_homeless))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x =rental price , y = homeless/legal aid')  
ax1.scatter(x_rental,y_ratio,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_rental, y_ratio))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x =gini(income inequality) , y = homeless/legal aid')  
ax1.scatter(x_gini,y_ratio,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_gini, y_ratio))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x = without %health insurance , y = homeless/legal aid')  
ax1.scatter(x_health,y_ratio,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_health, y_ratio))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x =population , y = homeless/legal aid')  
ax1.scatter(x_pop,y_ratio,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_pop, y_ratio))

import matplotlib
import matplotlib.pyplot as plt 
 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('The scatter plot of x =urbanization , y = homeless/legal aid')  
ax1.scatter(x_urban,y_ratio,c = 'c',marker = 'o')
plt.show()
print(pearsonr(x_urban, y_ratio))

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(y_legal, y_homeless, y_ratio, color='#ef1234')
_=plt.show()

from sklearn.cross_validation import train_test_split
X_train, X_test= train_test_split(X, test_size=0.2, random_state=0)
X_train_l, X_test_l= train_test_split(X_l, test_size=0.2, random_state=0)
X_train_h, X_test_h= train_test_split(X_h, test_size=0.2, random_state=0)
y_train_homeless, y_test_homeless = train_test_split(y_homeless, test_size=0.2, random_state=0)
y_train_legal, y_test_legal = train_test_split(y_legal, test_size=0.2, random_state=0)
y_train_ratio, y_test_ratio = train_test_split(y_ratio, test_size=0.2, random_state=0)
y_train_legal_intercept,y_test_legal_intercept = train_test_split(y_legal_intercept, test_size=0.2, random_state=0)
y_train_homeless_intercept,y_test_homeless_intercept = train_test_split(y_homeless_intercept, test_size=0.2, random_state=0)

print(len(y_train_homeless))
print(len(y_test_homeless))
print(len(y_train_legal))
print(len(y_test_legal))
print(len(y_train_ratio))
print(len(y_test_ratio))
print(len(X_train))
print(len(X_test))
print(len(X_train_l))
print(len(X_test_l))
print(len(X_train_h))
print(len(X_test_h))

L = data['Legal_Aid_Attorneys_per_capita'].values * 10000
H = data['unsheltered_homelessness_ratio'].values *10000

mean_l = np.mean(L)
mean_h = np.mean(H)

# Total number of values
m = len(L)

# Using the formula to calculate b1 and b2
numer = 0
denom = 0
for i in range(m):
    numer += (L[i] - mean_l) * (H[i] - mean_h)
    denom += (L[i] - mean_l) ** 2
b1 = numer / denom
b0 = mean_h - (b1 * mean_l)

# Print coefficients
print(b1, b0)

max_l = np.max(L)+10
min_l = np.min(L)-10

# Calculating line values x and y
l = np.linspace(min_l, max_l, 1000)
h = b0 + b1 * l

# Ploting Line
plt.plot(l, h, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(L, H, c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

model = linear_model.LinearRegression()
result_2 = model.fit(y_train_legal_intercept, y_train_homeless)

from sklearn.metrics import mean_squared_error
y_predict_homeless = result_2.predict(y_test_legal_intercept)
MSE_1 = mean_squared_error(y_predict_homeless, y_test_homeless)
print (MSE_1)

#OLS for legal, X includes homeless
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
model_legal = sm.OLS(y_train_legal, X_train_h)
results_legal = model_legal.fit()
print(results_legal.summary())

y_pred_legal = results_legal.predict(X_test_h)
MSE_test_legal = mean_squared_error(y_test_legal, y_pred_legal)
print ('The MSE of OLS on the legal aid ratio test is %f.'%(MSE_test_legal))

#OLS for legal, X not includes homeless
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
model_legal = sm.OLS(y_train_legal, X_train)
results_legal = model_legal.fit()
print(results_legal.summary())

y_pred_legal = results_legal.predict(X_test)
MSE_test_legal = mean_squared_error(y_test_legal, y_pred_legal)
print ('The MSE of OLS on the legal aid ratio test is %f.'%(MSE_test_legal))

#OLS for homeless, X includes legal aid
model_homeless = sm.OLS(y_train_homeless, X_train_l)
results_homeless = model_homeless.fit()
print(results_homeless.summary())

y_pred_homeless = results_homeless.predict(X_test_l)
MSE_test_homeless = mean_squared_error(y_test_homeless, y_pred_homeless)
print ('The MSE of OLS on the homeless ratio test is %f.'%(MSE_test_homeless))

#OLS for homeless, X without legal aid
model_homeless = sm.OLS(y_train_homeless, X_train)
results_homeless = model_homeless.fit()
print(results_homeless.summary())

y_pred_homeless = results_homeless.predict(X_test)
MSE_test_homeless = mean_squared_error(y_test_homeless, y_pred_homeless)
print ('without legal aid The MSE of OLS on the homeless ratio test is %f.'%(MSE_test_homeless))

model_ratio = sm.OLS(y_train_ratio, X_train)
results_ratio = model_ratio.fit()
print(results_ratio.summary())

y_pred_ratio = results_ratio.predict(X_test)
MSE_test_ratio = mean_squared_error(y_test_ratio, y_pred_ratio)
print ('The MSE of OLS on the homeless ratio test is %f.'%(MSE_test_ratio))


res_L = ols(formula='Legal_Aid_Attorneys_per_capita ~ unsheltered_homelessness_ratio * rental_price_average + unsheltered_homelessness_ratio * Gini_coefficients + unsheltered_homelessness_ratio * without_health_insurance + unsheltered_homelessness_ratio * POP_ST + unsheltered_homelessness_ratio * POPPCT_URBAN  +rental_price_average* Gini_coefficients+ rental_price_average * without_health_insurance + rental_price_average * POP_ST + rental_price_average* POPPCT_URBAN+Gini_coefficients *without_health_insurance + Gini_coefficients*POP_ST +Gini_coefficients*POPPCT_URBAN + without_health_insurance *POP_ST + without_health_insurance*POPPCT_URBAN + POP_ST*POPPCT_URBAN', data=data).fit()
print(res_L.summary())

res_H = ols(formula='unsheltered_homelessness_ratio ~ Legal_Aid_Attorneys_per_capita * rental_price_average + Legal_Aid_Attorneys_per_capita * Gini_coefficients + Legal_Aid_Attorneys_per_capita *without_health_insurance + Legal_Aid_Attorneys_per_capita * POP_ST +Legal_Aid_Attorneys_per_capita * POPPCT_URBAN + rental_price_average* Gini_coefficients+ rental_price_average * without_health_insurance + rental_price_average * POP_ST + rental_price_average* POPPCT_URBAN+Gini_coefficients *without_health_insurance + Gini_coefficients*POP_ST +Gini_coefficients*POPPCT_URBAN + without_health_insurance *POP_ST + without_health_insurance*POPPCT_URBAN + POP_ST*POPPCT_URBAN', data=data).fit()
print(res_H.summary())

res_H_no_L = ols(formula='unsheltered_homelessness_ratio ~  rental_price_average* Gini_coefficients * without_health_insurance* POP_ST * POPPCT_URBAN', data=data).fit()
print(res_H_no_L.summary())

res_H_no_L = ols(formula='unsheltered_homelessness_ratio ~   Legal_Aid_Attorneys_per_capita *rental_price_average* Gini_coefficients * without_health_insurance* POP_ST * POPPCT_URBAN', data=data).fit()
print(res_H_no_L.summary())

res_R = ols(formula='ratio ~ rental_price_average* Gini_coefficients+ rental_price_average * without_health_insurance + rental_price_average * POP_ST + rental_price_average* POPPCT_URBAN+Gini_coefficients *without_health_insurance + Gini_coefficients*POP_ST +Gini_coefficients*POPPCT_URBAN + without_health_insurance *POP_ST + without_health_insurance*POPPCT_URBAN + POP_ST*POPPCT_URBAN', data=data).fit()
print(res_R.summary())

