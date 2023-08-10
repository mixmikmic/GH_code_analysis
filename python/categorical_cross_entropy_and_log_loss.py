from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import log_loss
import random
import numpy as np

np.random.seed(777)

NO_OF_OUTPUT = 100

output_label = [random.randint(0, 3) for i in range(NO_OF_OUTPUT)]
pred_label = [[random.uniform(0, 1.0) for j in range(4)] for i in range(NO_OF_OUTPUT)]

y_true_cat = to_categorical(output_label)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return list(e_x / e_x.sum())

y_pred_cat = [softmax(v) for v in pred_label]

y_true = K.constant(y_true_cat)
y_pred = K.constant(y_pred_cat)

K.eval(K.categorical_crossentropy(target=y_true, output=y_pred)).mean()

log_loss(y_true_cat, y_pred_cat)

def cross_entropy(predictions, targets):
    N = targets.shape[0]
    res = -np.sum(targets*np.log(predictions))/N
    return res

x = cross_entropy(np.array(y_pred_cat), y_true_cat)

print(x)

output_label = [random.randint(0, 1) for i in range(NO_OF_OUTPUT)]
pred_label = [random.uniform(0, 1.0) for i in range(NO_OF_OUTPUT)]

output_label[0:5], pred_label[0:5]

log_loss(output_label, pred_label)

def log_loss_computation(predictions, targets):
    res = 0
    for pred, targ in zip(predictions, targets):
        res += - targ * np.log(pred) - (1- targ) * np.log(1-pred)
    return res/ len(targets)

log_loss_computation(pred_label, output_label)



