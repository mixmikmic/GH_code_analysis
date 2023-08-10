get_ipython().magic('matplotlib notebook')

import sys, os, pdb
import uuid, json, time
import pandas as pd

# import predictions algorithms
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

os.chdir(os.getcwd() + '/ml_api')
# import main stocks predictor / data preprocessing file
import lib.stocks as st
import lib.visualizer as vzr

DATE_TRAIN_START = '2015-01-01'
DATE_TEST_START = '2016-06-01'
DATE_END = '2016-09-01'

WINDOWS = [5]
HORIZONS = [7]

TICKERS_TRAIN = ['AMZN', 'AAPL', 'CSCO', 'NVDA', 'GOOG']
TICKERS_PREDICT = ['NVDA', 'GOOG', 'AMZN']

# create a directory with a unique ID
TRIAL_ID = uuid.uuid1()
DIRECTORY = 'trials/%s'%TRIAL_ID
os.makedirs(DIRECTORY)

print "Loading data for %s..."%', '.join(TICKERS_TRAIN)

data_files = st.loadMergedData(
    WINDOWS, HORIZONS, TICKERS_TRAIN, TICKERS_PREDICT,
    DATE_TRAIN_START, DATE_END, DIRECTORY
)

print "A new trial started with ID: %s\n"%TRIAL_ID
print "The data files generated are:"
print data_files

classifiers = [
    ('GradientBoosted', MultiOutputRegressor(GradientBoostingRegressor())),
    # ('AdaBoost', MultiOutputRegressor(AdaBoostRegressor()))
]

from IPython.display import display

# - combine the results of each classifier along with its w + h into a response object
all_results = {}

# - train each of the models on the data and save the highest performing
#         model as a pickle file
for h, w, file_path in data_files:
    # Start measuing time
    time_start = time.time()
    
    # load data
    finance = pd.read_csv(file_path, encoding='utf-8', header=0)
    finance = finance.set_index(finance.columns[0])
    finance.index.name = 'Date'
    
    # perform preprocessing
    X_train, y_train, X_test, y_test =         st.prepareDataForClassification(finance, DATE_TEST_START, TICKERS_PREDICT, h, w)

    results = {}

    print "Starting an iteration with a horizon of {} and a window of {}...".format(h, w)

    for i, clf_ in enumerate(classifiers):
        print "Training and testing the %s model..."%(clf_[0])
        
        # perform k-fold cross validation
        results['cross_validation_%s'%clf_[0]] =             st.performCV(X_train, y_train, 10, clf_[1], clf_[0])
        
        # perform predictions with testing data and record result
        preds, results['accuracy_%s'%clf_[0]] =             st.trainPredictStocks(X_train, y_train, X_test, y_test, clf_[1], DIRECTORY)

        print "\nBelow is a sample of of the results:\n"
        display(preds.sample(10).sort_index().reindex_axis(sorted(preds.columns), axis=1))
            
        # plot results
        vzr.visualize_predictions(preds, title='Testing Data Results')

    results['window'] = w
    results['horizon'] = h

    # Stop time counter
    time_end = time.time()
    results['time_lapsed'] = time_end - time_start

    all_results['H%s_W%s'%(h, w)] = results

print json.dumps(all_results, indent=4)



