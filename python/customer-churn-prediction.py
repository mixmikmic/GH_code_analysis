# Let's import Graphlab Create and a few other libraries
import graphlab as gl
import graphlab.aggregate
import datetime
import time

#Data can come directly from a SQL database, for this webinar, we will load from a local copy
data = gl.SFrame("https://static.turi.com/datasets/churn-prediction/online_retail.csv")
data

data = data.remove_columns(['InvoiceNo', 'Description'])
data

import dateutil
from dateutil import parser
def string_time_to_datetime(x):
    import datetime
    import pytz
    return dateutil.parser.parse(x)

data['InvoiceDate'] = data['InvoiceDate'].apply(string_time_to_datetime)

(train, valid) = gl.churn_predictor.random_split(data, user_id = 'CustomerID', fraction = 0.9, seed = 12)
train_trial = gl.TimeSeries(train, index = 'InvoiceDate')
valid_trial = gl.TimeSeries(valid, index = 'InvoiceDate')

userdata = gl.SFrame("https://static.turi.com/datasets/churn-prediction/online_retail_side_data_extended.csv")
userdata

print "Start date : %s" % train_trial.min_time
print "End date   : %s" % train_trial.max_time

# Period of inactivity that defines churn -- meaning that if a user stops purchasing
# items for 7 days, we'll consider them as having churned.
churn_period_trial = datetime.timedelta(days = 30) 

# Different beginning of months
churn_boundary_aug = datetime.datetime(year = 2011, month = 8, day = 1) 
churn_boundary_sep = datetime.datetime(year = 2011, month = 9, day = 1) 
churn_boundary_oct = datetime.datetime(year = 2011, month = 10, day = 1) 

model = gl.churn_predictor.create(train_trial,
                                  user_data = userdata,
                                  user_id='CustomerID',
                                  churn_period = churn_period_trial,
                                  time_boundaries = [churn_boundary_aug, churn_boundary_sep, churn_boundary_oct])

# Evaluate this model in October
evaluation_time = churn_boundary_oct

metrics = model.evaluate(valid_trial, evaluation_time, user_data = userdata)

print(metrics)

# metrics['precision_recall_curve'].show()

# Make predictions in the future.

predictions_trial = model.predict(valid_trial, user_data = userdata)
predictions_trial.print_rows()

predictions_trial.sort('probability', ascending=False).print_rows(20,max_column_width=20)

predictions_trial.sort('probability', ascending=False)[200:300] .print_rows(20,max_column_width=20)

model.trained_model

model.trained_model.get_feature_importance()

