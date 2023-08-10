RESULTS_PATH = '../results/'

import numpy as np
import inputs
import classification
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# task_list = [inputs.binary, inputs.six_transient, inputs.seven_transient, inputs.seven_class, inputs.eight_class]
# min_obs_list = [5,10]
# num_features_list = [31, 27, 21]
# oversample_list = [True, False]
# model_list = [classification.svc, classification.rf, classification.mlp]
# scaler_list = [StandardScaler, MinMaxScaler]

task_list = [inputs.eight_class]
min_obs_list = [10]
num_features_list = [27]
oversample_list = [False]
model_list = [classification.rf]
scaler_list = [StandardScaler]

for combination in itertools.product(task_list, min_obs_list, num_features_list, oversample_list, model_list, scaler_list):
    task, min_obs, num_features, oversample, model, scaler = combination
    print('STARTING TASK: ', task.__name__, min_obs, num_features, oversample, model.__name__, scaler.__name__)
    # Obtain inputs
    X_train, X_test, y_train, y_test = task(min_obs, num_features, oversample=oversample)
    # Scale inputs
    scaler = scaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Perform Classification
    clf = model(X_train, y_train, X_test, y_test, min_obs, num_features, oversample, task, scaler)
    y_pred = clf.predict(X_test)
    ID_test = task(min_obs, num_features, oversample=oversample, remove_ids=False)[1][:,0]
    incorrect = np.where(y_pred != y_test)
    correct = np.where(y_pred == y_test)
    print('Finished Task\n')

num_objects = 4

np.random.seed(42)

dict_correct = { 'task': task.__name__ }

for target in np.unique(y_pred):
    correct_target_indexes = np.where(y_pred[correct] == target)[0]
    num_target_objects = correct_target_indexes.shape[0]
    rand_indexes = np.random.choice(num_target_objects, num_objects, replace=False)
    dict_correct[target] = ID_test[correct][correct_target_indexes][rand_indexes].tolist()
    print(target)
    print(dict_correct[target])
with open('correct.txt','w') as f:
    f.write(str(dict_correct))

np.random.seed(42)

dict_incorrect = { 'task': task.__name__ }

for target in np.unique(y_pred):
    incorrect_target_indexes = (np.where(y_test[incorrect] == target))[0]
    num_target_objects = incorrect_target_indexes.shape[0]
    rand_indexes = np.random.choice(num_target_objects, num_objects, replace=False)
    dict_incorrect[target] = ID_test[incorrect][incorrect_target_indexes][rand_indexes].tolist()
    print(target)
    print(dict_incorrect[target])
with open('incorrect.txt','w') as f:
    f.write(str(dict_incorrect))

# ID_test[correct][:15], y_test[correct][:15], y_pred[correct][:15]

# ID_test[incorrect][:15], y_test[incorrect][:15], y_pred[incorrect][:15]

