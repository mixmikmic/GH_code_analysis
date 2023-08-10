# Import the libraries we'll be using
import numpy as np
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
import auquanToolbox.dataloader as dl

# Pull the pricing data for our two stocks and S&P 500
start = '2013-01-01'
end = '2015-12-31'
exchange = 'nasdaq'
base = 'SPX'
m1 = 'AAPL'
m2= 'LRCX'

data = dl.load_data_nologs(exchange, [base,m1,m2], start, end)
bench = data['ADJ CLOSE'][base]
a1= data['ADJ CLOSE'][m1]
a2 = data['ADJ CLOSE'][m2]

# Perform linear regression and print R-squared values
slr12 = regression.linear_model.OLS(a2, sm.add_constant(a1), missing='drop').fit()
slrb1 = regression.linear_model.OLS(a1, sm.add_constant(bench), missing='drop').fit()
slrb2 = regression.linear_model.OLS(a2, sm.add_constant(bench)).fit()
print ("R-squared values of linear regression")
print ("%s and %s: %.2f"%(m1,m2, slr12.rsquared))
print ("%s and %s: %.2f"%(m1,base, slrb1.rsquared))
print ("%s and %s: %.2f"%(m2,base, slrb2.rsquared))

# Add additional data
start = '2013-01-01'
end = '2016-12-31'
data = dl.load_data_nologs(exchange, [base,m1,m2], start, end)
bench = data['ADJ CLOSE'][base]
a1 = data['ADJ CLOSE'][m1]
a2 = data['ADJ CLOSE'][m2]

# Perform linear regression and print R-squared values
slr12 = regression.linear_model.OLS(a2, sm.add_constant(a1)).fit()
slrb1 = regression.linear_model.OLS(a1, sm.add_constant(bench)).fit()
slrb2 = regression.linear_model.OLS(a2, sm.add_constant(bench)).fit()
print ("R-squared values of linear regression")
print ("%s and %s: %.2f"%(m1,m2, slr12.rsquared))
print ("%s and %s: %.2f"%(m1,base, slrb1.rsquared))
print ("%s and %s: %.2f"%(m2,base, slrb2.rsquared))

# Load one year's worth of pricing data for five different assets
start = '2014-01-01'
end = '2016-01-01'
m1='PEP'
m2='MCD'
m3 ='CVS'
m4='DOW'
m5='PG'
data = dl.load_data_nologs(exchange, [m1,m2,m3,m4,m5], start, end)
x1 = data['ADJ CLOSE'][m1]
x2 = data['ADJ CLOSE'][m2]
x3 = data['ADJ CLOSE'][m3]
x4 = data['ADJ CLOSE'][m4]
y = data['ADJ CLOSE'][m5]

# Build a linear model using only x1 to explain y
slr = regression.linear_model.OLS(y, sm.add_constant(x1)).fit()
slr_prediction = slr.params[0] + slr.params[1]*x1

# Run multiple linear regression using x1, x2, x3, x4 to explain y
mlr = regression.linear_model.OLS(y, sm.add_constant(np.column_stack((x1,x2,x3,x4)))).fit()
mlr_prediction = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3 + mlr.params[4]*x4

# Compute adjusted R-squared for the two different models
print ('Using only PEP R-squared:', slr.rsquared_adj)
print ('Using a basket of stocks R-squared:', mlr.rsquared_adj)

# Plot y along with the two different predictions
y.plot(figsize=(15,7))
slr_prediction.plot()
mlr_prediction.plot()
plt.legend(['PG', 'Only PEP', 'Basket of stocks']);
plt.show()

# Load a year and a half of pricing data
start = '2016-01-01'
end = '2016-06-01'
data = dl.load_data_nologs(exchange, [m1,m2,m3,m4,m5], start, end)
x1 = data['ADJ CLOSE'][m1]
x2 = data['ADJ CLOSE'][m2]
x3 = data['ADJ CLOSE'][m3]
x4 = data['ADJ CLOSE'][m4]
y = data['ADJ CLOSE'][m5]

# Extend our model from before to the new time period
slr_prediction2 = slr.params[0] + slr.params[1]*x1
mlr_prediction2 = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3 + mlr.params[4]*x4

# Compute adjusted R-squared over the extended time period
adj = float(len(y) - 1)/(len(y) - 5) # Compute adjustment factor
SST = sum((y - np.mean(y))**2)
SSRs = sum((slr_prediction2 - y)**2)
print ('Using only PEP R-squared:', 1 - adj*SSRs/SST)
SSRm = sum((mlr_prediction2 - y)**2)
print ('Using a basket of stocks R-squared:', 1 - adj*SSRm/SST)

# Plot y along with the two different predictions
y.plot(figsize=(15,7))
slr_prediction2.plot()
mlr_prediction2.plot()
plt.legend(['PG', 'Only PEP', 'Basket']);
plt.show()

# Generate two artificial samples and pool them
sample1 = np.arange(30) + 4*np.random.randn(30)
sample2 = sample1 + np.arange(30)
pool = np.hstack((sample1, sample2))

# Run a regression on the pooled data, with the independent variable being the original indices
model = regression.linear_model.OLS(pool, sm.add_constant(np.hstack((np.arange(30),np.arange(30))))).fit()

# Plot the two samples along with the regression line
plt.scatter(np.arange(30), sample1, color='b')
plt.scatter(np.arange(30), sample2, color='g')
plt.plot(model.params[0] + model.params[1]*np.arange(30), color='r');
plt.show()

# Generate normally distributed errors
randos = [np.random.randn(100) for i in range(100)]
y = np.random.randn(100)
# Generate random walks
randows = [[sum(rando[:i+1]) for i in range(100)] for rando in randos]
yw = [sum(y[:i+1]) for i in range(100)]

plt.figure(figsize=(15,7))
for i in range(100):
    plt.plot(randows[i], alpha=0.5)
plt.show()    

# Compute R-squared of linear regression for each element of randows with yw
rs = [regression.linear_model.OLS(yw, x).fit().rsquared for x in randows]
                    
# Plot and count the random walks that have R-squared with yw > .8
rcount = 0
plt.figure(figsize=(15,7))
for i in range(100):
    if rs[i] > .8:
        rcount += 1
        plt.plot(randows[i], alpha=0.5)
print ('Linearly related walks out of 100:', rcount)

# Plot yw
plt.plot(yw, color='k');
plt.show()

from scipy.stats import pearsonr

# Compute correlation coefficients (Pearson r) and record their p-values
ps = [pearsonr(yw, x)[1] for x in randows]
                    
# Plot and count the random walks that have p-value of correlation with yw < 0.05
pcount = 0
plt.figure(figsize=(15,7))
for i in range(100):
    if ps[i] < .05:
        pcount += 1
        plt.plot(randows[i], alpha=0.5)
print ('Significantly correlated walks out of 100:', pcount)

# Plot yw
plt.plot(yw, color='k');
plt.show()

# Compute R-squared of linear regression for each element of randows with yw
rs = [regression.linear_model.OLS(y, x).fit().rsquared for x in randos]
                    
# Plot and count the random walks that have R-squared with yw > .8
rcount = 0
for i in range(100):
    if rs[i] > .8:
        rcount += 1
        plt.plot(randows[i], alpha=0.5)
print ('Linearly related walks out of 100:', rcount)

ps = [pearsonr(y, x)[1] for x in randos]
                    
# Plot and count the random walks that have p-value of correlation with yw < 0.05
pcount = 0
plt.figure(figsize=(15,7))
for i in range(100):
    if ps[i] < .05:
        pcount += 1
        plt.plot(randows[i], alpha=0.5)
print ('Significantly correlated walks out of 100:', pcount)

# Plot yw
plt.plot(y, color='k');
plt.show()

from statsmodels.tsa.stattools import adfuller

# Compute the p-value of the Dickey-Fuller statistic to test the null hypothesis that yw has a unit root
print (adfuller(yw)[1])

