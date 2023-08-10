get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import pandas as pd

from tobit import *

rs = np.random.RandomState(seed=10)
ns = 100
nf = 10
x, y_orig, coef = make_regression(n_samples=ns, n_features=nf, coef=True, noise=0.0, random_state=rs)
x = pd.DataFrame(x)
y = pd.Series(y_orig)

n_quantiles = 3 # two-thirds of the data is truncated
quantile = 100/float(n_quantiles)
lower = np.percentile(y, quantile)
upper = np.percentile(y, (n_quantiles - 1) * quantile)
left = y < lower
right = y > upper
cens = pd.Series(np.zeros((ns,)))
cens[left] = -1
cens[right] = 1
y = y.clip(upper=upper, lower=lower)
hist = plt.hist(y)

tr = TobitModel()
result = tr.fit(x, y, cens, verbose=False)

fig, ax = plt.subplots()
ind = np.arange(len(coef))
width = 0.25
rects1 = ax.bar(ind, coef, width, color='g', label='True')
rects2 = ax.bar(ind + width, tr.coef_, width, color='r', label='Tobit')
rects3 = ax.bar(ind + (2 * width), tr.ols_coef_, width, color='b', label='OLS')
plt.ylabel("Coefficient")
plt.xlabel("Index of regressor")
plt.title("Tobit vs. OLS on censored data")
leg = plt.legend(loc=(0.22, 0.65))

data_file = 'tobit_data.txt'
df = pd.read_table(data_file, sep=' ')
df.loc[df.gender=='male', 'gender'] = 1
df.loc[df.gender=='female', 'gender'] = 0
df.loc[df.children=='yes', 'children'] = 1
df.loc[df.children=='no', 'children'] = 0
df = df.astype(float)
df.head()

y = df.affairs
x = df.drop(['affairs', 'gender', 'education', 'children'], axis=1)
cens = pd.Series(np.zeros((len(y),)))
cens[y==0] = -1
cens.value_counts()

tr = TobitModel()
tr = tr.fit(x, y, cens, verbose=False)

tr.coef_

