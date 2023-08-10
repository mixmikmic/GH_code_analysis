import numpy as np
import pandas as pd
from scipy import stats
import statsmodels
import matplotlib.pyplot as plt
import matplotlib

get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns

train_data = pd.read_csv("data/trainingData.csv")
test_data = pd.read_csv("data/validationData.csv")

train_data.head()

train_data.describe()

# Response variables in our problem are Building, Floor, Latitude, Longitude and Relative Position
(train_data[['FLOOR','BUILDINGID', 'SPACEID','RELATIVEPOSITION','USERID','PHONEID']]
.astype(str)
.describe(include=['object']))

X_train = train_data.iloc[:,:520]
X_test = test_data.iloc[:,:520]

y_train = train_data.iloc[:,520:526]
y_test = test_data.iloc[:,520:526]

X_train.shape

X_train = (X_train
             .replace(to_replace=100,value=np.nan))

# Perform the same transform on Test data
X_test = (X_test
             .replace(to_replace=100,value=np.nan))

X_stack = X_train.stack(dropna=False)
sns.distplot(X_stack.dropna(),kde = False)

X_ap_max = (X_train
           .max(axis = 1,skipna=True)
           .dropna())

fig, ax = plt.subplots(1,1)

sns.distplot(X_ap_max.dropna(), ax = ax,kde=False)
ax.set_xlabel("Highest RSSI per measurement")

print("Skewness of entire RSSI distribution", X_stack.skew())
print("Skewness of max RSSI distribution", X_ap_max.skew())

aps_in_range = (X_train
                 .notnull()
                 .sum(axis = 1))

fig, ax = plt.subplots(1,1)

sns.violinplot(aps_in_range, ax = ax)
ax.set_xlabel("Number of APs in range")

print("Before sample removal:", len(X_train))

y_train = (y_train
          .loc[X_train
              .notnull()
              .any(axis=1),:])

X_train = (X_train
           .loc[X_train
                .notnull()
                .any(axis=1),:])

print("After sample removal:", len(X_train))

1300/520

# Removing columns with all NaN values
all_nan = (X_train
           .isnull()
           .all(axis=0) == False)
filtered_cols = (all_nan[all_nan]
                 .index
                 .values)

print("Before removing predictors with no in-range values", X_train.shape)

X_train = X_train.loc[:,filtered_cols]
X_test = X_test.loc[:,filtered_cols]

print("After removing predictors with no in-range values", X_train.shape)

# Proportion of out of range values
sum(X_stack.isnull() == 0)/len(X_stack)

miss_perc = (X_train
            .isnull()
            .sum(axis=0))

miss_perc *= 100/len(X_train)

sns.distplot(miss_perc,bins = 50,kde=False)

# Skewness of the predictors ignoring out-of-range values
X_skew = X_train.skew()
X_kurtosis = X_train.kurtosis()

g = sns.jointplot(y=X_kurtosis, x=X_skew, stat_func= None)
g.set_axis_labels('Skewness','Kurtosis')

from scipy.stats.distributions import norm

# The grid we'll use for plotting
x_grid = np.linspace(-100, 0, 100)

# Draw points from a bimodal distribution in 1D
np.random.seed(0)
x = np.array(X_train.stack().dropna())

kde_skl = KernelDensity(bandwidth=1.0,rtol=1e-4)
kde_skl.fit(x[:, np.newaxis])
# score_samples() returns the log-likelihood of the samples
log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
plt.plot(x_grid, np.exp(log_pdf), color = 'blue',alpha = 0.5, lw = 3)
#plt.fill(x_grid, pdf_true,ec='gray',fc='gray',alpha=0.4)

z = kde_skl.sample(n_samples=1000000)

[k for k in z if k <=abs_min]

