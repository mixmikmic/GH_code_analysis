import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import axes3d

get_ipython().run_line_magic('matplotlib', 'inline')

#read the data and separate the predictor and response to numpy arrays
advertising = pd.read_csv('../../data/Advertising.csv', index_col=0)
X = np.array(advertising['TV'])
y = np.array(advertising['sales'])
#calculate the least squares coefficients
b1 = ((X - X.mean()) * (y - y.mean())).sum() / ((X - X.mean())**2).sum()
b0 = y.mean() - b1 * X.mean()
print('Intercept: {:.2f}\nSlope: {:.4f}'.format(b0, b1))
print('Minimum RSS: {:.4f}'.format(((y - b0 - b1 * X)**2).sum()))

#center the X data
X = X - X.mean()
b1 = ((X - X.mean()) * (y - y.mean())).sum() / ((X - X.mean())**2).sum()
b0 = y.mean() - b1 * X.mean()
print('Intercept: {:.2f}\nSlope: {:.4f}'.format(b0, b1))
print('Minimum RSS: {:.4f}'.format(((y - b0 - b1 * X)**2).sum()))
print('Minimum RSS (in $1,000): {:.4f}'.format(((y - b0 - b1 * X)**2).sum()/1000))
#create a mesh for the coefficients
B0, B1 = np.meshgrid(np.linspace(b0-2, b0+2, 100), np.linspace(b1-0.02, b1+0.02, 100), indexing='xy')
#create a blank numpy array with the same shape as the meshed coefficients
rss = np.zeros_like(B0)
#calculate the RSS for each B0, B1 pair in the meshgrid
for (i, j), v in np.ndenumerate(rss):
    rss[i, j] = ((y - B0[i, j] - B1[i, j] * X)**2).sum()/1000

#need to add the subplots separately since you can't change individual projections 
#with the plt.subplots() command
fig = plt.figure(figsize=(15, 6))
    
#left plot
ax1 = fig.add_subplot(1, 2, 1)
cont = ax1.contour(B0, B1, rss, levels=[2.15, 2.2, 2.3, 2.5, 3], cmap=plt.cm.jet)
ax1.scatter(b0, b1, c='r', label=r'$\widehat{\beta}_0$, $\widehat{\beta}_1$ for minimized $RSS$')
ax1.clabel(cont, inline=True, fintsize=10, fmt='%1.1f');
ax1.set(xlabel=r'$\widehat{\beta}_0$', ylabel=r'$\widehat{\beta}_1$');
ax1.legend()

#right plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(B0, B1, rss, alpha=0.5)
#the offset will plot the contours on the plane of the minimum rss value
ax2.contour(B0, B1, rss, offset=rss.min(), levels=[2.15, 2.2, 2.3, 2.5, 3], cmap=plt.cm.jet);
#like offset above, zs will plot the scatter point on the plane of the minimum rss value
ax2.scatter3D(b0, b1, zs=rss.min(), c='r', label=r'$\widehat{\beta}_0$, $\widehat{\beta}_1$ for minimized $RSS$');
ax2.set(xlabel=r'$\widehat{\beta}_0$', ylabel=r'$\widehat{\beta}_1$', zlabel='RSS');
ax2.legend();

#generate 100 random X values
X = np.random.normal(size=100)
#generate 100 Y values using the model Y = 2 + 3X + epsilon
y = 2 + 3 * X + np.random.normal(scale=2.5, size=100)
#generate the true relationship between X and Y
X_true = np.linspace(-5, 5, 100)
y_true = 2 + 3 * X_true

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

#left plot
#find the coefficients for the sample data
b1 = ((X - X.mean()) * (y - y.mean())).sum() / ((X - X.mean())**2).sum()
b0 = y.mean() - b1 * X.mean()
#use the coefficients to predict y
y_pred = b0 + b1 * X_true
#make the plot
ax1.scatter(X, y, s=60, edgecolors='k', facecolors='none')
ax1.plot(X_true, y_true, c='r')
ax1.plot(X_true, y_pred, c='b')
ax1.set(xlim=[-2.5, 2.5], ylim=[-9, 11], xlabel='X', ylabel='Y');

#right plot
ax2.plot(X_true, y_true, c='r', zorder=5)
ax2.plot(X_true, y_pred, c='b')
for i in range(10):
    X = np.random.normal(size=100)
    y = 2 + 3 * X + np.random.normal(scale=2.5, size=100)
    b1 = ((X - X.mean()) * (y - y.mean())).sum() / ((X - X.mean())**2).sum()
    b0 = y.mean() - b1 * X.mean()
    y_pred = b0 + b1 * X_true
    ax2.plot(X_true, y_pred, c='b', alpha=0.2)
ax2.set(xlim=[-2.5, 2.5], ylim=[-9, 11], xlabel='X', ylabel='Y');

#tables 3.1 and 3.2 using statsmodels
import statsmodels.formula.api as sm
TV_ols = sm.ols('sales ~ TV', advertising).fit()
TV_ols.summary().tables[1]

TV_ols.summary().tables[0]

#tables 3.1 and 3.2 using sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = advertising['TV'].values.reshape(-1, 1)
y = advertising['sales'].values

lr = LinearRegression()
lr.fit(X, y)

y_pred = lr.predict(X)
print('Intercept: {:.4f}\nSlope: {:.4f}\nR^2: {:.4f}'       .format(lr.intercept_, lr.coef_[0], r2_score(y, y_pred)))

#table 3.3
#import statsmodels.formula.api as sm <- already imported
radio_ols = sm.ols('sales ~ radio', advertising).fit()
radio_ols.summary().tables[1]

newspaper_ols = sm.ols('sales ~ newspaper', advertising).fit()
newspaper_ols.summary().tables[1]

#table 3.4
advertising_ols = sm.ols('sales ~ TV + radio + newspaper', advertising).fit()
advertising_ols.summary().tables[1]

#table 3.5
advertising.corr()

#table 3.6
advertising_ols.summary().tables[0]

#Create the meshgrid based on min/max values for tv and radio
#print(advertising[['TV', 'radio']].describe())
tv = np.arange(0, 300)
radio = np.arange(0, 50)
radio_grid, tv_grid = np.meshgrid(radio, tv)
sales_grid = np.zeros((tv.shape[0], radio.shape[0]))

#Create the linear model
X = advertising[['TV', 'radio']].values
y = advertising['sales'].values
lr = LinearRegression()
lr.fit(X, y);

#predict sales at each mesh point
for (i, j), v in np.ndenumerate(sales_grid):
    sales_grid[i, j] = (lr.intercept_ + lr.coef_[0] * tv_grid[i, j] +                         lr.coef_[1] * radio_grid[i, j])
    
#make the plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(tv_grid, radio_grid, sales_grid, alpha=0.5)
ax.scatter3D(advertising['TV'], advertising['radio'], advertising['sales'], 
             c='r', zorder=5, s=30);
ax.set(xlabel='TV', ylabel='Radio', zlabel='Sales');

#plot the residuals
sales_pred = lr.predict(X)
for i in range(len(advertising)):
    #make points for x (tv), y (radio), and z (sales)
    tv = [advertising.iloc[i]['TV'], advertising.iloc[i]['TV']]
    radio = [advertising.iloc[i]['radio'], advertising.iloc[i]['radio']]
    sales = [advertising.iloc[i]['sales'], sales_pred[i]]
    #now plot the line from the observed point to the predicted point (on the plane)
    ax.plot(tv, radio, sales, color='black', linewidth=0.5)

credit = pd.read_csv('../../data/Credit.csv', index_col=0)
pp = sns.pairplot(credit)
pp.fig.set_size_inches(12, 12)

female = np.zeros_like(credit['Gender'])
female[credit['Gender'] == 'Female'] = 1
credit['Female'] = female

credit_ols = sm.ols('Balance ~ Female', credit).fit()
credit_ols.summary().tables[1]

asian = np.zeros_like(credit['Ethnicity'])
caucasian = np.zeros_like(asian)
asian[credit['Ethnicity'] == 'Asian'] = 1
caucasian[credit['Ethnicity'] == 'Caucasian'] = 1
credit['Asian'] = asian
credit['Caucasian'] = caucasian

credit_ols = sm.ols('Balance ~ Asian + Caucasian', credit).fit()
credit_ols.summary().tables[1]

credit_ols.summary().tables[0]

advertising['TVxradio'] = advertising['TV'] * advertising['radio']
advertising_ols = sm.ols('sales ~ TV + radio + TVxradio', advertising).fit()
advertising_ols.summary().tables[1]

advertising_ols.summary().tables[0]

auto = pd.read_csv('../../data/Auto.csv', na_values='?')
plt.figure(figsize=(10, 6))
plt.scatter(auto['horsepower'], auto['mpg'], edgecolors='black', facecolors='none', s=20)
sns.regplot(x=auto['horsepower'], y=auto['mpg'], ci=False, 
            line_kws={'color': 'orange'}, scatter=False)
sns.regplot(x=auto['horsepower'], y=auto['mpg'], ci=False, order=2, 
            line_kws={'color': 'green'}, scatter=False)
sns.regplot(x=auto['horsepower'], y=auto['mpg'], ci=False, order=5,
            line_kws={'color': 'blue'}, scatter=False)
plt.legend(['Linear', 'Degree 2', 'Degree 5']);

auto['horsepower2'] = auto['horsepower'] ** 2
auto_ols = sm.ols('mpg ~ horsepower + horsepower2', auto).fit()
auto_ols.summary().tables[1]

#create two linear regression instances
lr1 = LinearRegression()
lr2 = LinearRegression()
#create two X variables, one for linear, one for quadratic
X1 = auto.dropna()['horsepower'].values.reshape(-1, 1)
X2 = auto.dropna()[['horsepower', 'horsepower2']].values
y = auto.dropna()['mpg'].values
#fit each of the models
lr1.fit(X1, y);
lr2.fit(X2, y);
#make predictions with each of the models
y_pred1 = lr1.predict(X1)
y_pred2 = lr2.predict(X2)
#calculate the residuals of each of the models
res1 = y - y_pred1
res2 = y - y_pred2
#make the plots
#left panel
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
sns.regplot(y_pred1, res1, lowess=True, line_kws={'color': 'red', 'lw': 1}, 
            scatter_kws={'s': 20}, ax=ax1)
ax1.hlines(0, 0, 35, linestyle='dotted')
ax1.set(xlabel='Fitted Values', ylabel='Residuals', 
        title='Residual Plot for Linear Fit');
#right panel
sns.regplot(y_pred2, res2, lowess=True, line_kws={'color': 'red', 'lw': 1}, 
            scatter_kws={'s': 20}, ax=ax2)
ax2.hlines(0, 10, 40, linestyle='dotted');
ax2.set(xlabel='Fitted Values', ylabel='Residuals', 
        title='Residual Plot for Quadratic Fit');

credit = pd.read_csv('../../data/Credit.csv', index_col=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(credit['Limit'], credit['Age'], facecolor='none', edgecolor='r')
ax1.set(xlabel='Limit', ylabel='Age')
ax2.scatter(credit['Limit'], credit['Rating'], facecolor='none', edgecolor='r')
ax2.set(xlabel='Limit', ylabel='Rating')

#figure 3.15

#create the scaled predictors
age = (credit['Age'] - credit['Age'].mean()).values
limit = (credit['Limit'] - credit['Limit'].mean()).values
rating = (credit['Rating'] - credit['Rating'].mean()).values
#create the X and y vectors for modeling
X1 = np.array([age, limit]).T
X2 = np.array([rating, limit]).T
y = credit['Balance'].values
#fit a linear regression to each vector pair
lr1 = LinearRegression()
lr2 = LinearRegression()
lr1.fit(X1, y)
lr2.fit(X2, y)
#print the coefficients to create the coefficient grid
print('Regression 1: Intercept={:.4f}, Coefficients={}'       .format(lr1.intercept_, lr1.coef_))
print('Regression 2: Intercept={:.4f}, Coefficients={}'       .format(lr2.intercept_, lr2.coef_))
#make continuous variables for the coefficients for meshing
b_age = np.linspace(lr1.coef_[0]-3, lr1.coef_[0]+3, 100)
b_limit = np.linspace(lr1.coef_[1]-0.02, lr1.coef_[1]+0.02, 100)
b_rating = np.linspace(lr2.coef_[0]-3, lr2.coef_[0]+3, 100)
b_limit2 = np.linspace(lr2.coef_[1]-0.2, lr2.coef_[1]+0.2, 100)
#create the X and Y meshes
X1, Y1 = np.meshgrid(b_limit, b_age)
X2, Y2 = np.meshgrid(b_limit2, b_rating)
#create the Z meshes
Z1 = np.zeros((b_age.size, b_limit.size))
Z2 = np.zeros((b_rating.size, b_limit2.size))
#Calculate the Z's at each point on the X, Y grids
for (i, j), v in np.ndenumerate(Z1):
    Z1[i, j] = ((y - (lr1.intercept_ + X1[i, j] * limit + Y1[i, j]                  * age))**2).sum()/1000000
for (i, j), v in np.ndenumerate(Z2):
    Z2[i, j] = ((y - (lr2.intercept_ + X2[i, j] * limit + Y2[i, j]                  * rating))**2).sum()/1000000
#create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#left plot
cont1 = ax1.contour(X1, Y1, Z1, cmap=plt.cm.jet, levels=[21.25, 21.5, 21.8])
ax1.clabel(cont1, inline=True, fontsize=10, fmt='%1.1f')
ax1.scatter(lr1.coef_[1], lr1.coef_[0], c='r')
ax1.set(xlabel=r'$\beta_{Limit}$', ylabel=r'$\beta_{Age}$')
ax1.legend([r'$\beta_0, \beta_1$ for minimized RSS']);
#right plot
cont2 = ax2.contour(X2, Y2, Z2, cmap=plt.cm.jet, levels=[21.5, 21.8])
ax2.clabel(cont2, inline=True, fontsize=10, fmt='%1.1f')
ax2.scatter(lr2.coef_[1], lr2.coef_[0], c='r')
ax2.set(xlabel=r'$\beta_{Limit}$', ylabel=r'$\beta_{Rating}$')
ax2.legend([r'$\beta_0, \beta_1$ for minimized RSS']);

#table 3.11
credit_ols1 = sm.ols('Balance ~ Age + Limit', credit).fit()
credit_ols2 = sm.ols('Balance ~ Rating + Limit', credit).fit()

credit_ols1.summary().tables[1]

credit_ols2.summary().tables[1]

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

X = np.array([age, limit, rating]).T
vars = ['age', 'limit', 'rating']
for i in range(X.shape[1]):
    print('VIF {}: {}'.format(vars[i], vif(X, i)))



