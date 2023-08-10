sac_csv = '../datasets/sacramento_real_estate_transactions.csv'

import pandas as pd

shd = pd.read_csv(sac_csv)

shd.head()

# Check the dtypes
shd.dtypes

# Zip code is better as a string object (categorical) so I will have to convert it.
shd['zip'] = shd['zip'].astype(str)

# Check out the summary statistics:
shd.describe()

# Looks like we have some values that seem out of place being that there are
# houses with 0 bedrooms,  0 baths, a negative sqr footage and a negative price.  
# There are also some bizarre longitudes/latitudes. A house in Antartica perhaps. 

# Check out the cities. Most cities with very few observations.
shd.city.value_counts()

# Whats the deal with the houses that have 0 bedrooms?
shd[shd['beds'] == 0]

print shd[shd['beds'] == 0].shape

# Given the large value of houses that have 0 beds, 0 baths and 0 square feet 
# I am going to make an assumption that these are plots of land that have yet
# to have anything built on them.
# As a result I will *not* be dropping them.

# what about those houses that are less than 0 dollars?
shd[shd['price'] < 1]

# And the negative square feet?
shd[shd['sq__ft'] < 0]

# Looks like the house with a negative price is also the one with a negative squarefeet.
# It is time to make a choice.  Assume that the data was entered improperly and is meant 
# to be possitive or drop the data.

# Side note, the state is actually labeled wrong as well.

# Let me check if any other values are also not labeled right.
shd[shd['state'] != 'CA']

#Looks like it is just one row, so I am going to drop it.

shd.drop(703, inplace = True)

# Id say we can use 'beds','baths','sq__ft'
# Maybe 'latitude' & 'longitude', but that's more involved.

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

sns.lmplot(x='sq__ft', y='price', data=shd)
plt.show()
sns.lmplot(x='beds', y='price', data=shd)
plt.show()
sns.lmplot(x='baths', y='price', data=shd)
plt.show()

# It looks like Square Footage is a better predictor than Beds or Baths.
# Beds and Baths are discrete measures as opposed to Square feet, which is continuous.  
# Additionally, there is probably some strong correlations between them in that houses 
# with bigger square feet will have more beds and more baths.

# If we dropped all the plots of land that are in the dataset (those with 0 sq ft, 
# beds & baths)  we would see a much stronger trend line in our lm plot.

import numpy as np
import scipy.stats

# Get the optimal Slope and y intercept

def lin_reg(x,y):
    # Using other libraries for standard Deviation and Pearson Correlation Coef.
    # Note that in SLR, the correlation coefficient multiplied by the standard
    # deviation of y divided by standard deviation of x is the optimal slope.
    beta_1 = (scipy.stats.pearsonr(x,y)[0])*(np.std(y)/np.std(x))
    
    # Pearson Co. Coef returns a tuple so it needs to be sliced/indexed
    # the optimal beta is found by: mean(y) - b1 * mean(x)
    beta_0 = np.mean(y)-(beta_1*np.mean(x)) 
    
    #Print the Optimal Values
    print 'The Optimal Y Intercept is ', beta_0
    print 'The Optimal slope is ', beta_1

x = shd['sq__ft'].values
y = shd['price'].values
lin_reg(x,y)

# The intercept should be the relative y value that our data starts around 
# in that from this point out, as x increase so should the y value from this point.

# The intercept is an "offset". Without the intercept our regression line would
# be forced to pass through the origin.

# The slope is the increase in our target (price) variable for a 1-unit increase
# in our predictor variable (sq__ft). So, for every sq__ft increase there is
# an associated increase of ~54 dollars.

# Prediction
# You are a real estate agent with a separate database on house characteristics and locations.
# You want to estimate the most likely price that one of these houses will sell at based
# on the model that we built using this data.

# Inference
# You work for an architecture company that wants to understand what characteristics of a house
# and what areas are associated with perceived value. You have some hypotheses about what
# makes a house valuable but you would like to test these hypotheses.

# predictor: y = 162938.74 + 54.16x
# Creating a list of predicted values
y_pred = []

for x in shd['sq__ft']:
    y = 162938.74 + (54.16*x)
    y_pred.append(y)

# Appending the predicted values to the Sacramento housing dataframe to do DF calcs
shd['Pred'] = y_pred

# Residuals equals the difference between Y-True and Y-Pred
shd['Residuals'] = abs(shd['price']-shd['Pred'])

shd['Residuals'].mean()
# the mean of our residuals is aproximately $96,000, which means that is
# on average how off our prediction is.

# Plot showing out linear forcast
fig = plt.figure(figsize=(20,20))

# change the fontsize of minor ticks label
plot = fig.add_subplot(111)
plot.tick_params(axis='both', which='major', labelsize=20)

# get the axis of that figure
ax = plt.gca()

# plot a scatter plot on it with our data
ax.scatter(x= shd['sq__ft'], y=shd['price'], c='k')
ax.plot(shd['sq__ft'], shd['Pred'], color='r');

# Plot with residuals
fig = plt.figure(figsize=(20,20))

# change the fontsize of minor ticks label
plot = fig.add_subplot(111)
plot.tick_params(axis='both', which='major', labelsize=20)

# get the axis of that figure
ax = plt.gca()

# plot a scatter plot on it with our data
ax.scatter(x= shd['sq__ft'], y=shd['price'], c='k')
ax.plot(shd['sq__ft'], shd['Pred'], color='r');

# iterate over predictions
for _, row in shd.iterrows():
    plt.plot((row['sq__ft'], row['sq__ft']), (row['price'], row['Pred']), 'b-')

# One more plot, Lets look how our Predictions compared to the true values.
sns.lmplot(x='price', y='Pred', data=shd)

# Given our last visual we can see that a lot of points were plotted along the 
# y-intercept (y= 162938.74).  Those were all the houses with a reported
# square footage of 0 feet.  We could probably create a more insightful
# model if we removed those observations from our data.

