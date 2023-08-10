from sklearn.model_selection import train_test_split

import xgboost as xgb
import pandas as pd
import numpy as np

import pickle
import random

pd.set_option("max_columns", 999)

np.random.seed(1)

path_to_data = "/Users/clifford-laptop/Documents/space2vec/data/engineered-data.pkl"

data = pickle.load(open(path_to_data, 'rb'))

targets = [
    "OBJECT_TYPE",
]

ids = [
    "ID",
]

continuous = [
    "AMP",
    "A_IMAGE",
    "A_REF",
    "B_IMAGE",
    "B_REF",
    "COLMEDS",
    "DIFFSUMRN",
    "ELLIPTICITY",
    "FLUX_RATIO",
    "GAUSS",
    "GFLUX",
    "L1",
    "LACOSMIC",
    "MAG",
    "MAGDIFF",
    "MAG_FROM_LIMIT",
    "MAG_REF",
    "MAG_REF_ERR",
    "MASKFRAC",
    "MIN_DISTANCE_TO_EDGE_IN_NEW",
    "NN_DIST_RENORM",
    "SCALE",
    "SNR",
    "SPREADERR_MODEL",
    "SPREAD_MODEL",
]

categorical = [
    "BAND",
    "CCDID",
    "FLAGS",
]

ordinal = [
    "N2SIG3",
    "N2SIG3SHIFT",
    "N2SIG5",
    "N2SIG5SHIFT",
    "N3SIG3",
    "N3SIG3SHIFT",
    "N3SIG5",
    "N3SIG5SHIFT",
    "NUMNEGRN",
]

booleans = [
    "MAGLIM",
]

data = pd.get_dummies(
    data, 
    prefix = categorical, 
    prefix_sep = '_',
    dummy_na = True, 
    columns = categorical, 
    sparse = False, 
    drop_first = False
)

target = data[targets]
inputs = data.drop(columns = ids + targets)

x_train, x_valid, y_train, y_valid = train_test_split(
    inputs, 
    target, 
    test_size = 0.2, 
    random_state = 42,
    stratify = target.as_matrix()
)

params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'silent': 1,
    'objective': 'binary:logistic',
    'scale_pos_weight': 0.5,
    'n_estimators': 40,
    "gamma": 0,
    "min_child_weight": 1,
    "max_delta_step": 0, 
    "subsample": 0.9, 
    "colsample_bytree": 0.8, 
    "colsample_bylevel": 0.9, 
    "reg_alpha": 0, 
    "reg_lambda": 1, 
    "scale_pos_weight": 1, 
    "base_score": 0.5,  
    "seed": 23, 
    "nthread": 4
}

bst = xgb.XGBClassifier(**params)

bst.fit(
    x_train, 
    y_train, 
    eval_set = [(x_train, y_train), (x_valid, y_valid)], 
    eval_metric = ['auc'], 
    verbose = True
)

def metrics(outputs, labels, threshold=0.5):
    predictions = outputs >= (1 - threshold)
    true_positive_indices = (predictions == 0) * (labels == 0)
    false_positive_indices = (predictions == 0) * (labels == 1)
    true_negative_indices = (predictions == 1) * (labels == 1)
    false_negative_indices = (predictions == 1) * (labels == 0)

    true_positive_count = true_positive_indices.sum()
    false_positive_count = false_positive_indices.sum()
    true_negative_count = true_negative_indices.sum()
    false_negative_count = false_negative_indices.sum()
   
    return {
        # Missed detection rate
        'MDR': false_negative_count / (true_positive_count + false_negative_count),
        # True positive rate
        'FPR': false_positive_count / (true_negative_count + false_positive_count)
    }

def get_metrics(outputs, labels, with_acc=True):
    
    all_metrics = {}
    
    # FPR and MDR 0.4
    temp = metrics(outputs, labels, threshold=0.4)
    all_metrics["FALSE_POSITIVE_RATE_4"] = temp["FPR"]
    all_metrics["MISSED_DETECTION_RATE_4"] = temp["MDR"]
    
    # FPR and MDR 0.5
    temp = metrics(outputs, labels, threshold=0.5)
    all_metrics["FALSE_POSITIVE_RATE_5"] = temp["FPR"]
    all_metrics["MISSED_DETECTION_RATE_5"] = temp["MDR"]
    
    # FPR and MDR 0.6
    temp = metrics(outputs, labels, threshold=0.6)
    all_metrics["FALSE_POSITIVE_RATE_6"] = temp["FPR"]
    all_metrics["MISSED_DETECTION_RATE_6"] = temp["MDR"]
    
    # Summed FPR and MDR
    all_metrics["FALSE_POSITIVE_RATE"] = all_metrics["FALSE_POSITIVE_RATE_4"] + all_metrics["FALSE_POSITIVE_RATE_5"] + all_metrics["FALSE_POSITIVE_RATE_6"] 
    all_metrics["MISSED_DETECTION_RATE"] = all_metrics["MISSED_DETECTION_RATE_4"] + all_metrics["MISSED_DETECTION_RATE_5"] + all_metrics["MISSED_DETECTION_RATE_6"]
    
    # The true sum
    all_metrics["PIPPIN_METRIC"] = all_metrics["FALSE_POSITIVE_RATE"] + all_metrics["MISSED_DETECTION_RATE"]
    
    # Accuracy
    if with_acc:
        predictions = np.around(outputs).astype(int)
        all_metrics["ACCURACY"] = (predictions == labels).sum() / len(labels)
    
    return all_metrics

y_predictions = bst.predict_proba(x_valid)[:, 1:]

all_metrics = get_metrics(y_predictions, y_valid)

print("FPR (0.4): " + str(all_metrics["FALSE_POSITIVE_RATE_4"][0]))
print("FPR (0.5): " + str(all_metrics["FALSE_POSITIVE_RATE_5"][0]))
print("FPR (0.6): " + str(all_metrics["FALSE_POSITIVE_RATE_6"][0]))
print("")
print("MDR (0.4): " + str(all_metrics["MISSED_DETECTION_RATE_4"][0]))
print("MDR (0.5): " + str(all_metrics["MISSED_DETECTION_RATE_5"][0]))
print("MDR (0.6): " + str(all_metrics["MISSED_DETECTION_RATE_6"][0]))
print("")
print("SUMMED FPR: " + str(all_metrics["FALSE_POSITIVE_RATE"][0]))
print("SUMMED MDR: " + str(all_metrics["MISSED_DETECTION_RATE"][0]))
print("TOTAL SUM: " + str(all_metrics["PIPPIN_METRIC"][0]))
print("")
print("ACCURACY: " + str(all_metrics["ACCURACY"][0]))



