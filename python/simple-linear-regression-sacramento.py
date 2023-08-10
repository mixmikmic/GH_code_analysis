sac_csv = './datasets/sacramento_real_estate_transactions.csv'

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

get_ipython().magic('matplotlib inline')

# A:
df = pd.read_csv(sac_csv)
df.head()

df.info()

df['zip']=df['zip'].astype(str)

df.describe()

df.city.value_counts()

df[df['beds']==0].index

# df.drop(703, inplace = True)
df.drop(df[df['beds']<=0].index, inplace=True)
df.drop(df[df['baths']<=0].index, inplace=True)
df.drop(df[df['sq__ft']<=0].index, inplace=True)

# A
df.columns

sns.lmplot(x='sq__ft', y='price', data=df)
sns.lmplot(x='beds', y='price', data=df)
sns.lmplot(x='baths', y='price', data=df)

# A:
sns.heatmap(data=df[['sq__ft','beds','baths','price']].corr())

# A:
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
    return beta_0, beta_1

x = df['sq__ft'].values
y = df['price'].values
lin_reg(x,y)

# A:
# The intercept is the offset value of y where our data starts out
# The slope tells us for every 1 increase in x, the corresponding increase in y

# A:
# Prediction: anyone who needs to forsee future prices of houses
# Inference: anyone who needs to find out the relationship between the house characteristics and its associated value

# A:
# residuals = difference between predicted Y and actual Y
intercept, slope = lin_reg(x,y)
y_predict=[]

for ele in x:
    yhat = intercept + slope * ele
    y_predict.append(yhat)

np_y_predict = np.array(y_predict)
np_y = np.array(y)
np_residuals = abs(np_y - np_y_predict)
np_residuals.mean()

df['Prediction'] = y_predict
fig = plt.figure(figsize=(20,20))
plot = fig.add_subplot(111)
plot.tick_params(axis='both', which='major', labelsize=20)
ax = plt.gca()
ax.scatter(x= df['sq__ft'], y=df['price'], c='k')
ax.plot(df['sq__ft'], y_predict, color='r');
# iterate over predictions
for _, row in df.iterrows():
    plt.plot((row['sq__ft'], row['sq__ft']), (row['price'], row['Prediction']), 'b-')

sns.lmplot(x='price', y='Prediction', data=df)



