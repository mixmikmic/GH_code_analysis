from config import datafolder
from functions import *
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn
# increase the size of graphs
matplotlib.rcParams['savefig.dpi'] *= 1.5

import warnings
warnings.filterwarnings("ignore")

# choosing a dataset of price relative, a filter and its parameters 
dataname = "DJIA"  # among SP500, TSE, DJIA
dataset = pd.read_csv(datafolder+dataname+".csv") #price relative
f = KCA # filter you want
params = {"window": 5} # parameter you want

# build dataframe of price relative predictions
prediction = f(dataset,params).divide(to_absolute(dataset)) # build predictions

# adjusting the data to measure the performance
adjusted_dataset, adjusted_prediction = adjust_data(dataset, prediction, horizon=1) # adjust the matrix of true and predicted price relatives 

# measure of the performance
regression_report(adjusted_dataset, adjusted_prediction, output="all")

f = MA 
parameter_range = range(1,30)
parameter_name = "window"

for dataname in ["DJIA","TSE","SP500"] :
    dataset = pd.read_csv(datafolder+dataname+".csv") #price relative data
    MAE = []
    DPA = []
    for p in parameter_range:
        params[parameter_name] = p
        prediction = f(dataset,params).divide(to_absolute(dataset)) # build predictions
        adjusted_dataset, adjusted_prediction = adjust_data(dataset, prediction, horizon=1) # adjust the matrix of true and predicted price relatives 
        report = regression_report(adjusted_dataset, adjusted_prediction, output="average")
        
        MAE.append(report["MAE"])
        DPA.append(report["DPA"])

    plt.figure(1)
    plt.plot(parameter_range, MAE, label = dataname)
    
    plt.figure(2)
    plt.plot(parameter_range, DPA, label = dataname)

    
plt.figure(1)
plt.xlabel(parameter_name)
plt.ylabel("Mean Absolute Error")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))

plt.figure(2)
plt.xlabel(parameter_name)
plt.ylabel("Direction Prediction Accuracy")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))

plt.show()





