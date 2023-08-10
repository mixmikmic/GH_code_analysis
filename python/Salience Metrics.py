import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def targets_to_binary(target_matrix):
    binary_matrix = np.zeros(target_matrix.shape)
    binary_matrix[target_matrix == 1.0] = 1.0
    return binary_matrix

def peak_mask(prediction_matrix, binary=False):
    peaks = scipy.signal.argrelmax(prediction_matrix, axis=0)
    mask = np.zeros(prediction_matrix.shape)
    if binary:
        mask[peaks] = 1
    else:
        mask[peaks] = prediction_matrix[peaks]
    return mask

def mask_prediction(prediction_matrix, binary_matrix):
    return prediction_matrix * binary_matrix

def energy_recall(prediction_matrix, target_matrix):
    binary_matrix = targets_to_binary(target_matrix)
    mask = mask_prediction(prediction_matrix, binary_matrix)
    return np.sum(mask.flatten()) / np.sum(binary_matrix.flatten())

def energy_false_alarm(prediction_matrix, target_matrix):
    binary_matrix = 1 - targets_to_binary(target_matrix)
    mask = mask_prediction(prediction_matrix, binary_matrix)
    return np.sum(mask.flatten()) / np.sum((binary_matrix).flatten())

def mean_amplitude_error(prediction_matrix, target_matrix):
    binary_matrix = targets_to_binary(target_matrix)
    mask = mask_prediction(prediction_matrix, binary_matrix)
    n_positive = np.sum(binary_matrix.flatten())
    return np.sum(binary_matrix.flatten() - mask.flatten())/float(n_positive)

def peak_recall(prediction_matrix, target_matrix):
    binary_matrix = targets_to_binary(target_matrix)
    prediction_peak_mask = peak_mask(prediction_matrix, binary=True)
    mask = mask_prediction(prediction_peak_mask, binary_matrix)
    return np.sum(mask.flatten()) / np.sum(binary_matrix.flatten())

def mean_frequency_error(prediction_matrix, target_matrix):
    binary_matrix = targets_to_binary(target_matrix)
    n_positive = np.sum(binary_matrix.flatten())
    prediction_peak_mask = peak_mask(prediction_matrix, binary=True)
    mask = mask_prediction(prediction_peak_mask, target_matrix)
    return np.sum(mask.flatten()) / float(n_positive)

y_pred = np.load('/home/rmb456/repos/multif0/pred_output_example.npy')
y_true = np.load('/home/rmb456/repos/multif0/true_output_example.npy')

e_recall = energy_recall(y_pred, y_true) #average energy of ground truth positives
e_fa = energy_false_alarm(y_pred, y_true) #average energy of ground truth negatives
mae = mean_amplitude_error(y_pred, y_true)
pr = peak_recall(y_pred, y_true)
mfe = mean_frequency_error(y_pred, y_true)
print([e_recall, e_fa, mae, pr, mfe])

plt.figure(figsize=(15, 7))
plt.imshow(mask, origin='lower')
plt.axis('auto')
plt.colorbar()



