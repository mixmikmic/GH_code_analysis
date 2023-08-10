import sys;
import subprocess;
import numpy as np
import pandas as pd

from get_labels import get_labels
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split

import keras.backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import LSTM

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

labels = get_labels();
labels_array = np.array([x for x in labels]);
labels_reshaped = labels_array.reshape(1851243, 1, 1070);

train_x = joblib.load("/mnt/cleaned_tfidf_reduced_420_morning");
train_x_reshaped = train_x.reshape(1851243,1,1000);

x_train, x_test, y_train, y_test = train_test_split(train_x_reshaped, labels_reshaped, test_size=0.20, random_state=1024)

#Our custom loss function
def multiclass_loss(y_true, y_pred):
    EPS = 1e-5
    y_pred = K.clip(y_pred, EPS, 1 - EPS)
    return -K.mean((1 - y_true) * K.log(1 - y_pred) + y_true * K.log(y_pred))

model = load_model('khot_LSTM_1353.h5', custom_objects={"multiclass_loss":multiclass_loss})

del labels
del labels_array
del labels_reshaped

del train_x
del train_x_reshaped

del x_train
del y_train

def get_preds_array():
    all_test = [];
    all_preds = [];
    all_preds_proba = [];
    
    for idx, test_val in enumerate(x_test):
        y_test_val = y_test[idx];
        
        k = len(y_test_val[y_test_val == 1])
        
        pred_val = model.predict(test_val.reshape(1,1,1000))[0][0]
        topk = pred_val.argsort()[-1 * k:][::-1]

        pred_arr = np.zeros(y_test_val.shape);
        pred_arr[0,topk] = 1;
        
        all_test.extend(y_test_val[0]);
        all_preds.extend(pred_arr[0]);
        all_preds_proba.extend(pred_val); # commend out this line to disable probabilty predictions
        
        if idx % 500 == 0:
            sys.stdout.write('\rOn ' + str(idx) + ' / ' + str(len(x_test)));
            
    return (all_preds, all_test, all_preds_proba);

(all_preds, all_test, all_preds_proba) = get_preds_array();

#joblib.dump((all_preds, all_test), 'predictions_test.pkl')

preds_arr = np.array(all_preds);
#del all_preds

test_arr = np.array(all_test);
#del all_test

preds_proba_arr = np.array(all_preds_proba)
#del (all_preds_proba)

print(np.sum(preds_arr));
print(np.sum(test_arr));

def tp_rate(predictions, actuals, get_rate=True):
    sums = predictions + actuals;
    all2s = sums[sums == 2];
    if get_rate:
        #return len(all2s) / float(sum(sums));
        return len(all2s) / float(len(sums));
    else:
        return len(all2s);

def fp_rate(predictions, actuals, get_rate=True):
    act_not = np.logical_not(actuals).astype(int);
    return tp_rate(predictions, act_not, get_rate);

def fn_rate(predictions, actuals, get_rate=True):
    pred_not = np.logical_not(predictions).astype(int);
    return tp_rate(pred_not, actuals, get_rate);

def tn_rate(predictions, actuals, get_rate=True):
    pred_not = np.logical_not(predictions).astype(int);
    act_not = np.logical_not(actuals).astype(int);
    return tp_rate(pred_not, act_not, get_rate);

def accuracy(predictions, actuals):
    tp_val = tp_rate(predictions, actuals, False);
    tn_val = tn_rate(predictions, actuals, False);
    
    return (tp_val + tn_val) / float(len(predictions));

def precision(predictions, actuals, get_rate=True):
    tp = tp_rate(predictions, actuals, get_rate);
    fp = fp_rate(predictions, actuals, get_rate);
    return (float(tp) / (tp + fp));

def recall(predictions, actuals, get_rate=True):
    tp = tp_rate(predictions, actuals, get_rate);
    fn = fn_rate(predictions, actuals, get_rate);
    
    return (tp / float(tp + fn));

def confusion_array(predictions, actuals, get_rate=True):
    fp = fp_rate(predictions, actuals, get_rate);
    tp = tp_rate(predictions, actuals, get_rate);
    fn = fn_rate(predictions, actuals, get_rate);
    tn = tn_rate(predictions, actuals, get_rate);
    
    conf = np.array([[tp, fp], [fn, tn]]);

    conf_pdf = pd.DataFrame(conf);
    conf_pdf.columns = ['Condition True', 'Condition False'];
    conf_pdf = conf_pdf.set_index(np.array(['Predicted True', 'Predicted False']))
    
    return conf_pdf;

conf_vals = confusion_array(preds_arr, test_arr, False)
conf_arr = confusion_array(preds_arr, test_arr)
acc_val = accuracy(preds_arr, test_arr);
precision_val = precision(preds_arr, test_arr);
recall_val = recall(preds_arr, test_arr);

print ("Confusion matrix:")
conf_vals

print ("Confusion rate matrix:")
conf_arr

print ("Our values: \n\t accuracy: %f \n\t prcision: %f \n\t recall: %f" % (acc_val, precision_val, recall_val))

sklearn_accuracy = metrics.accuracy_score(test_arr, preds_arr)
sklearn_precision = metrics.precision_score(test_arr, preds_arr)
sklearn_recall = metrics.recall_score(test_arr, preds_arr)
sklearn_f1 = metrics.f1_score(test_arr, preds_arr)
sklearn_roc = metrics.roc_auc_score(test_arr, preds_proba_arr)
sklearn_conf_matrix = metrics.confusion_matrix(test_arr, preds_arr)

print ("Sklearn accuracy:", sklearn_accuracy)
print ("Sklearn precision:", sklearn_precision)
print ("Sklearn recall:", sklearn_recall)
print ("Sklearn f1 score:", sklearn_f1)
print ("sklearn roc:", sklearn_roc)
print ("sklearn confusion matrix", sklearn_conf_matrix)

def plot_roc(y_test, predictions):
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC')
    plt.legend(loc="lower right")
    plt.show()

plot_roc(test_arr, preds_proba_arr)

