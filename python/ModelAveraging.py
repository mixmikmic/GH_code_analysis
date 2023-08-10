import numpy as np
import pandas as pd
import os, sys

basepath = os.path.expanduser('~/Desktop/src/African_Soil_Property_Prediction/')
sys.path.append(os.path.join(basepath, 'src'))

from sklearn.externals import joblib

from collections import defaultdict

np.random.seed(5)

from helper import utils
from models import eval_metric, cross_validation, find_weights, models_definition

train = pd.read_csv(os.path.join(basepath, 'data/raw/training.csv'))
sample_sub = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv'))

y_Ca, y_P, y_Sand, y_SOC, y_pH = utils.define_target_variables(train)

class Averaging:
    labels = ['Ca', 'P', 'Sand', 'SOC', 'pH']
        
    def __init__(self, dataset_name, cv=False):
        trains, tests     = self.load_files(dataset_name)
        self.cv           = cv
        self.dataset_name = dataset_name
        
        if cv:
            
            params = {
            'test_size': 0.2,
            'random_state': 3
            }

            self.itrain,  self.itest  = cross_validation.split_dataset(len(trains[0]), **params)
            self.ytrains, self.ytests = utils.get_Ys(y_Ca, y_P, y_Sand, y_SOC, y_pH, itrain, itest)
    
    def load_files(self, dataset_name):
        self.trains, self.tests = utils.load_datasets(dataset_name, self.labels)
        return self.trains, self.tests
    
    def predict(self):
        self.preds = defaultdict(list)
        
        for i in range(len(self.labels)):
            if self.cv:
                self.Xtr, self.Xte  = utils.get_Xs(self.trains[i], self.itrain, self.itest)
            else:
                self.Xte = self.tests[i]
                
            model_names = os.listdir(path=os.path.join(basepath, 'data/processed/%s/%s/models'%(self.dataset_name, self.labels[i])))
            for j in range(len(model_names)):
                model = joblib.load(os.path.join(basepath, 'data/processed/%s/%s/models/%s/%s'%(self.dataset_name, self.labels[i], model_names[j], model_names[j])))
                self.preds[self.labels[i]].append(model.predict(self.Xte))  

        return self.preds

avg1    = Averaging('dataset_5', cv=False)
preds1  = avg1.predict()

avg2    = Averaging('dataset_6', cv=False)
preds2  = avg2.predict()

final_preds = defaultdict(list)

for k, v in preds1.items():
    for p in v:
        final_preds[k].append(p)
    
for k, v in preds2.items():
    for p in v:
        final_preds[k].append(p)

predictions_all_targets = [final_preds['Ca'],
                           final_preds['P'],
                           final_preds['Sand'],
                           final_preds['SOC'],
                           final_preds['pH']
                          ]

balanced_preds = []
weights = []

for i in range(5):
    weight, balanced_pred = find_weights.find(avg1.ytests[i], predictions_all_targets[i])
    
    weights.append(weight)
    print('MCRMSE for index:%d is: %f'%(i+1, eval_metric.mcrmse([avg1.ytests[i]], [balanced_pred])))
    balanced_preds.append(balanced_pred)

# print(len(balanced_preds[0]))
print('\n=================================')
print('MCRMSE for all of the targets: ', eval_metric.mcrmse(y_tests, balanced_preds))

joblib.dump(weights, os.path.join(basepath, 'data/interim/weights/weights'))

weights = joblib.load(os.path.join(basepath, 'data/interim/weights/weights'))

predictions_Ca_stacked   = find_weights.stack_predictions(final_preds['Ca'])
predictions_P_stacked    = find_weights.stack_predictions(final_preds['P'])
predictions_Sand_stacked = find_weights.stack_predictions(final_preds['Sand'])
predictions_SOC_stacked  = find_weights.stack_predictions(final_preds['SOC'])
predictions_pH_stacked   = find_weights.stack_predictions(final_preds['pH'])

final_preds_Ca   = find_weights.balance_predictions(y_Ca, predictions_Ca_stacked, weights[0])
final_preds_P    = find_weights.balance_predictions(y_Ca, predictions_P_stacked, weights[1])
final_preds_Sand = find_weights.balance_predictions(y_Ca, predictions_Sand_stacked, weights[2])
final_preds_SOC  = find_weights.balance_predictions(y_Ca, predictions_SOC_stacked, weights[3])
final_preds_pH   = find_weights.balance_predictions(y_Ca, predictions_pH_stacked, weights[4])

sample_sub['Ca']   = final_preds_Ca
sample_sub['P']    = final_preds_P
sample_sub['Sand'] = final_preds_Sand
sample_sub['SOC']  = final_preds_SOC
sample_sub['pH']   = final_preds_pH

sample_sub.to_csv(os.path.join(basepath, 'submissions/2_datasets_clustered.csv'), index=False)



