import pandas as pd
import numpy as np
import math
import quandl
import datetime
import pickle

# plotting and plot stying
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
plt.style.use('seaborn')
#sns.set_style("whitegrid", {'axes.grid' : False})
#set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = False
#plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = b"\usepackage{subdepth}, \usepackage{type1cm}"


from sklearn import preprocessing, cross_validation, svm, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

df = quandl.get('WIKI/GOOGL')
df.to_csv('./data/google.csv', sep=';')

df.head()

df = pd.read_csv('./data/google.csv', sep=';', header=0, parse_dates=True, index_col=0)

# building own features, hopefully with more descriptive power
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] *100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] *100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

df.head()

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

# shift data forward to create y
#df['label'] = df['Adj. Close'].shift(-forecast_out)
df['label'] = df['Adj. Close'].shift(1)
df = df.dropna()

plt.plot(df['label'][0:100], label='y ot t+1');
plt.plot(df['Adj. Close'][0:100], label='X ot t');
plt.legend();
plt.show()

df.tail()



df.head()

X = np.array(df.drop(['label'], 1))
#X = X[:-1]
X = preprocessing.scale(X)

y = df['label']
y.dropna(inplace=True)
y = np.array(y)

len(X)

# create random samples
# BAD
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# slice up the actual time series
X_train = X[:3000, :]
y_train = y[:3000]

X_test = X[3000:, :]
y_test = y[3000:]

len(X_test), len(y_test)

X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

y_train.reshape(-1,1)

X_train.reshape(-1, 4)

clf = LinearRegression()

clf.fit(X_train, y_train.reshape(-1, 1))

accuracy = clf.score(X_test, y_test)
accuracy

y_hat = clf.predict(X_test)

plt.plot(y_test, label='Observed')
plt.plot(y_hat, color='orange', linestyle='--', label='Linear Regression Prediction')
plt.legend(loc='upper left')
plt.show()



#rgr_lin = LogisticRegression().fit(X_train, y_train.reshape(-1, 1))
svr_lin = SVR(kernel='linear', C=1e3).fit(X_train, y_train.reshape(-1, 1))
svr_poly = SVR(kernel='poly', C=1e3).fit(X_train, y_train.reshape(-1, 1))
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1).fit(X_train, y_train.reshape(-1, 1))

y_hat_svr_lin = svr_lin.predict(X_test)
y_hat_svr_poly = svr_poly.predict(X_test)
y_hat_svr_rbf = svr_rbf.predict(X_test)



plt.plot(y_test, label='Observed');
plt.plot(y_hat_svr_lin, linestyle='--', label='Linear Kernel');
#plt.plot(y_hat_svr_poly, linestyle='--', label='Plynomial Kernel')
plt.plot(y_hat_svr_rbf, linestyle='--', label='RBF');
plt.legend(loc='upper left');
plt.show()

regr_rf = RandomForestRegressor(n_estimators=50,
                                max_features=3,
                                max_depth=40,
                                ).fit(X_train, y_train.reshape(-1, 1).ravel())

y_hat_regr_rf = regr_rf.predict(X_test)

plt.plot(y_test, label='Observed');
plt.plot(y_hat_svr_lin, linestyle='--', label='Linear Kernel');
plt.plot(y_hat_regr_rf, linestyle='--', label='Random Forest Regression');
plt.legend(loc='upper left');
plt.show()









df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    
df['Adj. Close'].plot()
df['Forecast'].astype(float).plot()
plt.legend(loc=4)
plt.show()



