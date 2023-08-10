"""
A random forest classifier aimed at determining whether a stock will be higher or lower after some given amount of days.
Replication of Khaidem, Saha, & Roy Dey (2016)

Documentation on function:
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as make_forest
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import numpy as np
import tqdm

'''
### Outline ###
We have a bunch of columns of different length target values
We drop all target values except the ones we want to analyze (or else when we remove NA we will remove too much data)
We then input the data and features in to the first .fit parameter, and the labels in the second
'''
criterion="gini"
num_features = 6
n_estimators = 65
prediction_window = 1
oob_score = True
full_data = pd.read_csv('data_preprocessed.csv')
train_labels = ["Close_detrend","Volume","EWMA", "SO","WR","RSI"]

# drop all target columns not to be analyzed
#headers = full_data.columns.values
#headers = headers[13:] # should return just the headers of the target values
#headers = headers[headers!='Target({})'.format(prediction_window)]
#selected_data = full_data.drop(headers, axis=1)
selected_data = full_data.dropna(axis=0, how='any') # using the subset parameter might allow us to skip dropping other targets?



### Drop useless labels ###
selected_data.drop(["Unnamed: 0"], axis = 1, inplace = True)
selected_data.drop(["Date"], axis = 1, inplace = True)
selected_data.drop(["Open","High","Low"], axis = 1, inplace = True)
#selected_data.drop(["Symbol","Open","High","Low"], axis = 1, inplace = True)

def split_x_y(df,train_labels,prediction_window):
    x = df[train_labels].as_matrix()
    y = df['Target({})'.format(prediction_window)].as_matrix()
    
    return x,y
    
def train_on_df(x,y,train_frac):
    msk = np.random.rand(len(x)) < train_frac
    
    train_x = x[msk]
    train_y = y[msk]
    
    test_x = x[~msk]
    test_y = y[~msk]
    
    Random_Forest = make_forest(n_estimators=n_estimators, max_features=num_features, bootstrap=True, oob_score=oob_score, verbose=0,criterion=criterion,n_jobs=-1)
    Random_Forest.fit(train_x, train_y)
        
    
    test_accurency = Random_Forest.score(test_x, test_y)
    return Random_Forest, test_accurency

stock_forests = {}
import tqdm
num_symboles = len(selected_data['Symbol'].unique())-1
for idx,stock in tqdm.tqmd(enumerate(selected_data['Symbol'].unique())):
    symbole_df = selected_data[selected_data["Symbol"]==stock]

    x1,y1 = split_x_y(symbole_df, train_labels,1)
    x30,y30 = split_x_y(symbole_df, train_labels,30)


    forest1, accurency1 = train_on_df(x1,y1,0.8)
    forest30, accurency30 = train_on_df(x30,y30,0.8)

    stock_forests[stock] = {1:{"acc":accurency1,
                                "forest":forest1},
                            30:{"acc":accurency30,
                                "forest":forest30}
                            }

    df_stock = pd.DataFrame()
    df_stock["Close"] = symbole_df["Close"]
    df_stock["Close_detrend"] = symbole_df["Close_detrend"]
    df_stock["Target(1)"] = symbole_df["Target(1)"]
    df_stock["Target(30)"] = symbole_df["Target(30)"]
    df_stock["Prediction(1)"] = forest1.predict(symbole_df[train_labels].as_matrix())
    df_stock["Prediction(30)"] = forest30.predict(symbole_df[train_labels].as_matrix())
    df_stock.to_csv("results/result_{}.csv".format(stock))
    print("Done {}/{}".format(idx,num_symboles))

f_all = open("results/_ALL.csv","w")
f_all.write("Symbole,accPrediction(1),accPrediction(30)\n")
for symbole, vals in stock_forests.items():
    f_all.write("{},{},{}\n".format(symbole,vals[1]["acc"],vals[30]["acc"]))

x1,y1 = split_x_y(selected_data, train_labels,1)
x30,y30 = split_x_y(selected_data, train_labels,30)

complete_forest1, complete_acc1 = train_on_df(x1,y1,0.8)
complete_forest30, complete_acc30 = train_on_df(x30,y30,0.8)

print("\tacc1: {}%".format(str(round(complete_acc1*100,2))))
print("\tacc30: {}%".format(str(round(complete_acc30*100,2))))

to_compare = ["AAPL", "CAT", "BA", "SBUX"]
complete_against ={}
for stock in to_compare:
    symbole_df = selected_data[selected_data["Symbol"]==stock]

    x1,y1 = split_x_y(symbole_df, train_labels,1)
    x30,y30 = split_x_y(symbole_df, train_labels,30)

    acc1 = complete_forest1.score(x1,y1)
    acc30 = complete_forest30.score(x30,y30)
    print("For Stock {} against complete:".format(stock))
    print("\tacc1: {}%".format(str(round(acc1*100,2))))
    print("\tacc30: {}%".format(str(round(acc30*100,2))))

to_compare = ["AAPL", "CAT", "BA", "SBUX"]
for model in to_compare:
    for test_stock in to_compare:
        if test_stock == model:
            continue
        symbole_df = selected_data[selected_data["Symbol"]==test_stock]

        x1,y1 = split_x_y(symbole_df, train_labels,1)
        x30,y30 = split_x_y(symbole_df, train_labels,30)
        
        acc1 = stock_forests[model][1]["forest"].score(x1,y1)
        acc30 = stock_forests[model][30]["forest"].score(x30,y30)
        print("For {} in {}-Model".format(test_stock,model))
        print("\tacc1: {}%".format(str(round(acc1*100,2))))
        print("\tacc30: {}%".format(str(round(acc30*100,2))))

#import pickle
#f = open("complete_forest_v1.pick","wb")
#s = pickle.dump(complete_forest,f)

