get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')

import os
import pandas as pd
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def plot_selected(df, columns, start_index, end_index):

    plot_data(df.ix[start_index:end_index,columns], title="Stock Data")
    
    
def plot_data(df, title):
    ax = df.plot(title=title,fontsize=12,figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def show_supervised_linear_regression():
    
    df = pd.read_csv(symbol_to_path('SPY'), index_col='Date', 
                parse_dates=True, 
                usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], 
                na_values=['nan'])
    
    # sort data chronologically
    df = df.sort_index(ascending=True, axis=0)
    
    # print df.head(), "\n"
    # print df.describe()
    
    # add new column to view Adj Close 5 days later
    df['Adj_Close_5_Days_Later'] = df['Adj Close']
    df['Adj_Close_5_Days_Later'] = df['Adj_Close_5_Days_Later'].shift(-5)
    # print df.head(6)
    
    # reduce data by date
    # df_smaller_set = df['20150101':'20160101']
    
    # Slice and plot
    # plot_selected(df, ['Adj Close'], '2015-01-01', '2016-01-01')
    
    # Get the features and labels from the stock dataset
    # X = df.iloc[:,:-1]
    # y = df.iloc[:, -1]
    # Split the data into training/testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    ##### NOTE:  Can't use gridsearchCV.train_test_split because it doesn't do roll-forward cross validation. #####
    # print "\n", "df.shape: ", df.shape, "\n"
    X_train = df.iloc[0:1000,:-1]
    y_train = df.iloc[0:1000, -1]
    X_test = df.iloc[1000:1253,:-1]
    y_test = df.iloc[1000:1253, -1]
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X_train, y_train)
    
    # Query
    regr.predict(X_test)
    
    # Plot outputs
    print "\n"
    plt.figure(figsize=(12,10))
    plt.title("Real-world results vs machine learning predictions")
    plt.xlabel("Prediction")
    plt.ylabel("Real-world result")
    plt.scatter(regr.predict(X_test), y_test, color='blue')
    # plt.plot(regr.predict(X_test), regr.predict(X_test), color='blue', linewidth=1)
    plt.show()
    
    # The coefficients
    # print "Coefficients (formatted): "
    # print ("Open: {0:.4f}".format(round(regr.coef_[0],4)))
    # print ("High: {0:.4f}".format(round(regr.coef_[1],4)))
    # print ("Low: {0:.4f}".format(round(regr.coef_[2],4)))
    # print ("Close: {0:.4f}".format(round(regr.coef_[3],4)))
    # print ("Volume: {0:.9f}".format(round(regr.coef_[4],9)))
    # print ("Adj Close: {0:.9f}".format(round(regr.coef_[5],9))), "\n"
    
    # Explained variance score: 1 is perfect
    # Score
    print "Score on training data"
    print "regr.score(X_train, y_train): ", regr.score(X_train, y_train), "\n"
    
    print "Score on testing (unseen) data"
    print('regr.score(X_test, y_test): %.2f' % regr.score(X_test, y_test))
    # The mean square error
    print "Mean squared error: ", mean_squared_error(y_test, regr.predict(X_test)), "\n"
    # print("Residual sum of squares: %.2f"
          # % np.mean((regr.predict(X_test) - y_test) ** 2))
    
    # print "Prediction - regr.predict(X_test): "
    # print regr.predict(X_test)[0]
    # print regr.predict(X_test)[1]
    # print regr.predict(X_test)[2]
    # print regr.predict(X_test)[3]
    # print regr.predict(X_test)[4], "\n"
    # print "Actual (y_test): "
    # print y_test.head()

if __name__ == "__main__":
    show_supervised_linear_regression()



