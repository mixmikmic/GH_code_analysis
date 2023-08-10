import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.cross_validation import train_test_split

def eval_metric(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse = make_scorer(eval_metric, greater_is_better=False)

train = pd.read_csv('../data/Train_UWu5bXk.csv')
test = pd.read_csv('../data/Test_u94Q5KV.csv')

## mapping
item_sales_map = train.groupby('Item_Identifier').Item_Outlet_Sales.mean().to_dict()
item_sales_median = train.groupby('Item_Identifier').Item_Outlet_Sales.median().to_dict()

X = train[train.columns.drop('Item_Outlet_Sales')]
y = train.Item_Outlet_Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

def get_predictions(item_ids, mapping, factor=1.):
    return np.array([factor * mapping[item_id] for item_id in item_ids])    

train_predictions = get_predictions(X_train.Item_Identifier, item_sales_map, factor=.9)
test_predictions = get_predictions(X_test.Item_Identifier, item_sales_map, factor=.9)

print 'RMSE on the training examples %f ' %(eval_metric(y_train, train_predictions))
print 'RMSE on the test examples %f ' %(eval_metric(y_test, test_predictions))

plt.hist(train_predictions, alpha=0.7, color='b')
plt.hist(y_train, color='r', alpha=.5);

factors = np.linspace(0.6, 1.1, 20)

errors = []
for factor in factors:
    predictions = get_predictions(train.Item_Identifier, item_sales_median, factor=factor)
    errors.append(eval_metric(train.Item_Outlet_Sales, predictions))

plt.plot(factors, errors)
plt.xlabel('Factors')
plt.ylabel('RMSE')
plt.title('Multiplicative Factor vs Error (Median)');

errors = []
for factor in factors:
    predictions = get_predictions(train.Item_Identifier, item_sales_map, factor=factor)
    errors.append(eval_metric(train.Item_Outlet_Sales, predictions))

plt.plot(factors, errors)
plt.xlabel('Factors')
plt.ylabel('RMSE')
plt.title('Multiplicative Factor vs Error ( Mean )');

mapping = train.groupby(['Item_Identifier', 'Outlet_Type']).Item_Outlet_Sales.mean().to_dict()

def get_predictions(item_ids, outlet_types, mapping, item_map, factor=1.):
    predictions = []
    for item_id, outlet_type in zip(item_ids, outlet_types):
        if (item_id, outlet_type) in mapping:
            predictions.append(factor * mapping[(item_id, outlet_type)])
        else:
            predictions.append(item_map[item_id])
    
    return np.array(predictions)

errors = []
for factor in factors:
    predictions = get_predictions(train.Item_Identifier, train.Outlet_Type, mapping, item_sales_map, factor=factor)
    errors.append(eval_metric(train.Item_Outlet_Sales, predictions))

plt.plot(factors, errors)
plt.xlabel('Factors')
plt.ylabel('RMSE')
plt.title('Multiplicative Factor vs Error ( Mean )');

predictions = get_predictions(test.Item_Identifier, test.Outlet_Type, mapping, item_sales_map, factor=0.99473684)

plt.hist(predictions, bins=100, alpha=0.5);

submission = pd.read_csv('../data/SampleSubmission_TmnO39y.csv'); 
submission.loc[:, 'Item_Outlet_Sales'] = predictions

submission.to_csv('../submissions/base_submission.csv', index=False);



