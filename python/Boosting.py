get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble

from IPython.display import Image
import pydotplus  # NOTE: May Require conda install graphviz

# import data
header_url = 'https://gist.githubusercontent.com/jeff-boykin/b5c536467c30d66ab97cd1f5c9a3497d/raw/5233c792af49c9b78f20c35d5cd729e1307a7df7/field_names.txt'
data_url = 'https://gist.githubusercontent.com/jeff-boykin/b5c536467c30d66ab97cd1f5c9a3497d/raw/5233c792af49c9b78f20c35d5cd729e1307a7df7/breast-cancer.csv'

header_list = pd.read_csv(header_url, header=None, squeeze=True).tolist();
data = pd.read_csv(data_url, header=None, names=header_list, index_col='ID')

# divide data into predictors (X) and predictand (y), and subdivide into test/train sets.
Y = data['diagnosis'].replace({'M': 1, 'B': 0}) # convert to binary outcome
X = data.iloc[:, data.columns != 'diagnosis']

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.4)

# fit a classifier
clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
clf.fit(X_train, Y_train)

# visualize the classifier
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, filled=True, rounded=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  

# evaluate performance of single decision tree
print(metrics.classification_report(y_true=Y_test, y_pred=clf.predict(X_test)))

# Adaptive Boosting from scratch

def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    
    # Initialize uniform weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    for i in range(M):
        # Fit a classifier with the specified weights
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        
        # Encode misclassifications as binary 0/1
        miss = [int(x) for x in (pred_train_i != Y_train)]
        
        # Encode misclassifications as 1/-1 to update weights (equivalent to above)
        miss2 = [x if x==1 else -1 for x in miss]
        
        # Compute weighted classification error
        err_m = np.dot(w,miss) / sum(w)
        
        # Compute new weights, alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_i])]
    
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    return pred_train, pred_test

_, y_hat = adaboost_clf(Y_train, X_train, Y_test, X_test, M=5, clf=tree.DecisionTreeClassifier(min_samples_leaf=5))
print(metrics.classification_report(y_true=Y_test, y_pred=y_hat))

# Finally, let's use the fully-implimented version from scikit-learn
clf = ensemble.AdaBoostClassifier(learning_rate=1, n_estimators=50)
clf.fit(X_train, Y_train)
print(metrics.classification_report(y_true=Y_test, y_pred=clf.predict(X_test)))

df = pd.read_csv('Gradient_boosting_example_data.csv', index_col='PersonID')
features = ['LikesGardening', 'PlaysVideoGames', 'LikesHats']
predictand = 'Age'
X = df[features]
y = df[predictand]

print(df)

# decision tree depth-1 (stump) implimentation from first principles
for feature in X.columns:
    feature_means = df.groupby([feature])[predictand].mean()
    prediction_colname = 'stump_prediction_{}'.format(feature)
    residual_colname = 'stump_residual_{}'.format(feature)
    for index, value in enumerate(df[feature]):
        index += 1 # zero indexing correction
        df.loc[index, prediction_colname] = feature_means[value]
        df.loc[index, residual_colname] = df.loc[index, predictand] - df.loc[index, prediction_colname]

print(df)

# which stump produces the lowest residuals?
df.filter(regex='residual').apply(lambda resid: np.sqrt(np.sum((resid) ** 2))).sort_values()

# second tree on the residuals of the first
predictand = 'stump_residual_LikesGardening'
for feature in X.columns:
    feature_means = df.groupby([feature])[predictand].mean()
    prediction_colname = 'second_prediction_{}'.format(feature)
    residual_colname = 'second_residual_{}'.format(feature)
    for index, value in enumerate(df[feature]):
        index += 1 # zero indexing correction
        df.loc[index, prediction_colname] = feature_means[value]
        df.loc[index, residual_colname] = df.loc[index, predictand] - df.loc[index, prediction_colname]

# which stump produces the lowest residuals?
df.filter(regex='second_residual').apply(lambda resid: np.sqrt(np.sum((resid) ** 2))).sort_values()

df['boosted_prediction'] = df.stump_prediction_LikesGardening + df.second_prediction_PlaysVideoGames
df['boosted_residuals'] = df.Age - df.boosted_prediction
summary = df[['Age', 'stump_prediction_LikesGardening', 'boosted_prediction']].apply(round)

print('mean square errors on stump prediction: {:.2f}'.format(metrics.mean_squared_error(df.stump_prediction_LikesGardening, df.Age)))
print('mean square errors on boosted prediction: {:.2f}'.format(metrics.mean_squared_error(df.boosted_prediction, df.Age)))

# helper functions
def compute_gradient(prediction1, prediction2, target, loss_function):
    loss_function_gradient = loss_function(prediction2, target) - loss_function(prediction1, target)
    return loss_function_gradient

def rmse(y_hat, y_true, normalize=False):
    '''RMSE represents the sample standard deviation of the differences between predicted and observed values'''
    y_hat = _convert_to_1D_np_array(y_hat)
    y_true = _convert_to_1D_np_array(y_true)
    normalization = y_true.size if normalize else 1
    return np.sqrt(np.sum((y_hat - y_true) ** 2) / normalization)

def sse(y_hat, y_true):
    return np.sum((y_hat - y_true) ** 2)

def mean_absolute_deviation(y_hat, y_true):
    '''mean absolute deviation is approximately 0.798 x standard deviation for normally distributed random variables'''
    y_hat = _convert_to_1D_np_array(y_hat)
    y_true = _convert_to_1D_np_array(y_true)
    return np.mean(np.abs(y_hat - y_true))

def median_absolute_deviation(y_hat, y_true):
    '''median absolute deviation is approximately 0.675 x standard deviation for normally distributed random variables'''
    y_hat = _convert_to_1D_np_array(y_hat)
    y_true = _convert_to_1D_np_array(y_true)
    return np.median(np.abs(y_hat - y_true))

def symmetric_mean_absolute_percent_error(y_hat, y_true):
    '''SMAPE is bound between 0 and 200%'''
    y_hat = _convert_to_1D_np_array(y_hat)
    y_true = _convert_to_1D_np_array(y_true)
    return np.sum(np.abs(y_hat - y_true) / ((np.abs(y_true) + np.abs(y_hat)) / 2.0)) * (100.0 / y_true.size)

def standardize(y):
    y = _convert_to_1D_np_array(y)
    return (y - np.mean(y)) / np.std(y)

def mean_absolute_scaled_error(y_hat, y_true):
    '''MASE from Hyndman and Koehler'''
    y_hat = _convert_to_1D_np_array(y_hat)
    y_true = _convert_to_1D_np_array(y_true)
    denom_sum = np.sum(abs(y_true[1:] - y_true[:-1]))
    error = abs(y_true - y_hat)
    denom = denom_sum / (float(len(y_hat)) - 1)
    return np.mean(error / denom)

def _convert_to_1D_np_array(y):
    if isinstance(y, (int, float, list)):
        return np.array(y)
    elif isinstance(y, pd.Series):
        return np.array(y)
    elif isinstance(y, pd.DataFrame):
        if y.shape[1] > 1:
            raise Exception('y must be 1-dimensional')
        else:
            return np.array(y.ix[:, 0])
    elif isinstance(y, np.ndarray):
        if y.ndim > 1:
            raise Exception('y must be 1-dimensional')
        else:
            return y
    else:
        raise Exception('y must be of type int, float, long or a 1-dimensional array-like object')

ytrue = [13, 25]
yhat1 = [35, 35]
improvement = -1
yhat2 = [item + improvement for item in yhat1]

gradient_descent_1 = compute_gradient(yhat1[0], yhat2[0], ytrue[0], loss_function=sse)
gradient_descent_2 = compute_gradient(yhat1[1], yhat2[1], ytrue[1], loss_function=sse)
print(gradient_descent_1, gradient_descent_2)

gradient_descent_1 = compute_gradient(yhat1[0], yhat2[0], ytrue[0], median_absolute_deviation)
gradient_descent_2 = compute_gradient(yhat1[1], yhat2[1], ytrue[1], median_absolute_deviation)
print(gradient_descent_1, gradient_descent_2)



