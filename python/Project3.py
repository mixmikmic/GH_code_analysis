from pandas_datareader import data
import pandas as pd
import datetime

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2016, 12, 1 )

sp500 = data.DataReader("^GSPC", 'yahoo', start, end)

sp500.loc[:, 'avg_price'] = (sp500.loc[:, 'High'] + sp500.loc[:, 'Low'] + sp500.loc[:, 'Close'])/3

sp500.head()

def arithmetic_return_action(data, k = 30, p = 0.05, t = 0.15):
    '''
    This function will calculate arithmetic return of k days and based on the sum return to determine actions
    
    :param df: stock data
    :param k: number of days
    :param p: cutoff rate
    :param t: sum of the return rate to  a action: buy, hold, or sell
    :return: stock data labeled with actions
    '''

    import numpy as np
    df = data.copy()
    df.loc[:, 'temp_return'] = 0
    df.loc[:, 'sum_return']  = 0
    
    
    for i in range(1, k + 1):
        
        df.loc[:, 'temp_return'] = (df.loc[:, 'avg_price'].shift(-i) - df.loc[:, 'Close']) / df.loc[:, 'Close']
        df.loc[np.fabs(df.temp_return) > p, 'sum_return'] += df.loc[(np.fabs(df.temp_return) > p), 'temp_return']

            
    df.loc[df.sum_return >= t, 'action'] = 'buy'    
    df.loc[(df.sum_return < t) & (df.sum_return > -t), 'action'] = 'hold'  
    df.loc[df.sum_return <= -t, 'action'] = 'sell'
    
    return df.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume', 'avg_price', 'sum_return', 'action']]

sp_3m = sp500.loc[:'2016-3-31']

sp_3m_action = arithmetic_return_action(sp_3m)

sp_3m_action.describe()

money_out = sp_3m_action.loc[sp_3m_action.action == 'buy', 'avg_price'].sum() * 100
money_in = sp_3m_action.loc[sp_3m_action.action == 'sell', 'avg_price'].sum() * 100
position = sp_3m_action.loc[sp_3m_action.action == 'buy', 'action'].count() - sp_3m_action.loc[sp_3m_action.action == 'sell', 'action'].count()

avg_cost = (money_out-money_in)/position/100

print('At the end of March 31, 2016, our portfolio position is: {:d} shares, average price bought/sold is: ${:,.2f}'.format(position*100, avg_cost))

sp500.loc['2016-4-28']

profit = (sp500.loc['2016-4-28', 'avg_price'] - avg_cost) * position * 100

print('By the end of April 2016, the portfolio profit is ${:,.2f}'.format(profit))

get_ipython().magic('matplotlib inline')
sp_3m.loc[:, 'avg_price'].plot(title="S&P 500 Price Chart Jan - March 2016")

def label_action(data, k = 30, p = 0.05, t = 0.15):
    '''
    This function will calculate arithmetic return of k days and based on the sum return to label actions
    
    :param df: stock data
    :param k: number of days
    :param p: cutoff rate
    :param t: sum of the return rate to label a action: buy, hold, or sell
    :return: stock data labeled with actions
    '''

    import numpy as np
    df = data.copy()
    df.loc[:, 'temp_return'] = 0
    df.loc[:, 'sum_return']  = 0
    
    
    for i in range(1, k + 1):
        
        df.loc[:, 'temp_return'] = (df.loc[:, 'avg_price'].shift(-i) - df.loc[:, 'Close']) / df.loc[:, 'Close']
        df.loc[np.fabs(df.temp_return) > p, 'sum_return'] += df.loc[(np.fabs(df.temp_return) > p), 'temp_return']

    
    df.loc[df.sum_return >= t, 'action'] = 'buy'    
    df.loc[(df.sum_return < t) & (df.sum_return > -t), 'action'] = 'hold'  
    df.loc[df.sum_return <= -t, 'action'] = 'sell' 
    
    return df

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

df = label_action(sp500)

df.loc[:, 'action'] = label.fit_transform(df.loc[:, 'action'].astype(str))

df = df.loc[:'2016-10-31']

from sklearn import ensemble, metrics
X_train = df.loc[:'2016-09-30',['Open', 'High', 'Low', 'Close', 'Volume', 'avg_price']]
y_train = df.loc[:'2016-09-30','action']
X_test = df.loc['2016-10-01':,['Open', 'High', 'Low', 'Close', 'Volume', 'avg_price']]
y_test = df.loc['2016-10-01':, 'action']

params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

confusion = metrics.confusion_matrix(y_test, y_predict)

print(clf.score(X_test, y_test))

from matplotlib import pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, '{:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        else: 
            plt.text(j, i, '{:.0f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


np.set_printoptions(precision=2)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1,2]),
                      title='Confusion matrix, without normalization')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1,2]), normalize=True,
                      title='Normalized confusion matrix')

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

confusion = metrics.confusion_matrix(y_test, y_predict)

print(clf.score(X_test, y_test))

np.set_printoptions(precision=2)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1,2]),
                      title='Confusion matrix, without normalization')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1,2]), normalize=True,
                      title='Normalized confusion matrix')

it_stock = pd.read_csv('/Users/Maxwell/PycharmProjects/Github/FinancialProgramClass/companylist_it.csv').loc[:, 'Symbol']
bank_stock = pd.read_csv('/Users/Maxwell/PycharmProjects/Github/FinancialProgramClass/companylist_bank.csv').loc[:, 'Symbol']

stock = pd.DataFrame()

for ticker in it_stock:
    
    df_temp = data.DataReader(ticker, 'yahoo', start, end)
    df_temp.loc[:, 'Ticker'] = str(ticker)
    df_temp.loc[:, 'Industry'] = 'IT'
    df_temp.loc[:, 'avg_price']= (df_temp.loc[:, 'High'] + df_temp.loc[:, 'Low'] + df_temp.loc[:, 'Close'])/3
    df_temp = label_action(df_temp)
    stock = pd.concat([stock,df_temp])

for ticker in bank_stock:
    
    df_temp = data.DataReader(ticker, 'yahoo', start, end)
    df_temp.loc[:, 'Ticker'] = str(ticker)
    df_temp.loc[:, 'Industry'] = 'Bank'
    df_temp.loc[:, 'avg_price']= (df_temp.loc[:, 'High'] + df_temp.loc[:, 'Low'] + df_temp.loc[:, 'Close'])/3
    df_temp = label_action(df_temp)
    stock = pd.concat([stock,df_temp])

stock.head()

label_action = LabelEncoder()
#label_ticker = LabelEncoder()
#label_industry = LabelEncoder()

df = stock.copy()

df.loc[:, 'action'] = label_action.fit_transform(df.loc[:, 'action'].astype(str))
#df.loc[:, 'Ticker'] = label_ticker.fit_transform(df.loc[:, 'Ticker'].astype(str))
#df.loc[:, 'Industry'] = label_industry.fit_transform(df.loc[:, 'Industry'].astype(str))

df = df.loc[:'2016-10-31']


X_train = df.loc[:'2016-09-30',['Open', 'High', 'Low', 'Close', 'Volume', 'avg_price']]
y_train = df.loc[:'2016-09-30','action']
X_test = df.loc['2016-10-01':,['Open', 'High', 'Low', 'Close', 'Volume', 'avg_price']]
y_test = df.loc['2016-10-01':, 'action']

params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

df.loc['2016-10-01':, 'predict_action_gb'] = label_action.inverse_transform(y_predict)

confusion = metrics.confusion_matrix(y_test, y_predict)

print(clf.score(X_test, y_test))

np.set_printoptions(precision=2)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1,2]),
                      title='Confusion matrix, without normalization')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1,2]), normalize=True,
                      title='Normalized confusion matrix')

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

confusion = metrics.confusion_matrix(y_test, y_predict)

print(clf.score(X_test, y_test))

np.set_printoptions(precision=2)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1,2]),
                      title='Confusion matrix, without normalization')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1,2]), normalize=True,
                      title='Normalized confusion matrix')

portfolio = df.loc['2016-10-01':].copy()

portfolio.loc[:, 'action'] = label_action.inverse_transform(df.loc['2016-10-01':, 'action'])
#portfolio.loc[:, 'Ticker'] = label_ticker.inverse_transform(df.loc['2016-10-01':, 'Ticker'])
#portfolio.loc[:, 'Industry'] = label_industry.inverse_transform(df.loc['2016-10-01':, 'Industry'])

portfolio.loc[portfolio.predict_action_gb == 'buy', 'position']= 100
portfolio.loc[portfolio.predict_action_gb == 'sell', 'position'] = -100
portfolio.loc[portfolio.predict_action_gb == 'hold', 'position'] = 0

portfolio.loc[:, 'value'] = portfolio.loc[:, 'position'] * portfolio.loc[:, 'avg_price'] 


df_new = portfolio.groupby(['Ticker', 'Industry']).sum()

df_new

pnl = df_new.copy()[['position', 'value']]

last_day = stock.loc['2016-12-1', ['Ticker', 'avg_price']]

last_day.reset_index(drop=True, inplace=True)

pnl.reset_index(inplace=True)

pnl = pd.merge(pnl, last_day, on='Ticker')

pnl.loc[:, 'p&l'] = pnl.loc[:, 'position'] * pnl.loc[:, 'avg_price'] - pnl.loc[:, 'value']

result = pnl.groupby(['Ticker', 'Industry']).sum()

result.loc[:, 'p&l'].plot(kind='bar', figsize=(20,18))
plt.title("Portfolio P&L Chart")

print('If we hold the position until 12-1-2016, the portfolio P&L will be ${:,.0f}'.format(result.loc[:,'p&l'].sum()))



