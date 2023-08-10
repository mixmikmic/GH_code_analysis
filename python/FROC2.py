import cv2
import numpy as np
import os 
import dicom
import data
import copy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from scipy import ndimage
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from scipy import ndimage
from sklearn import metrics,metrics
from scipy.ndimage.measurements import label
get_ipython().magic('matplotlib inline')

def getPara(predict, true, threshold):
    (TP, FP, TN, FN, class_lable) = perf_measure(true, predict, threshold)
    if((TP + FN)== 0):
        TPR = 0
    else:
        TPR = np.float(TP)/(TP + FN)
    
    class_lable = class_lable.astype(bool).reshape(264,132)
    true = predict.astype(bool).reshape(264,132)
    
    predict2 = remove_small_objects(class_lable, 64,in_place=False)
    true2 = remove_small_objects(true, 64,in_place=False)
    labeled_array1, num_features1 = label(predict2)
    labeled_array2, num_features2 = label(true2)
    FP_num = num_features1 - num_features2
    return TPR, FP_num

def perf_measure(y_actual, predict, threshold):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    predict = transfer_prob(predict, threshold)
    for i in range(len(predict)): 
        if y_actual[i]==predict[i]==1:
           TP += 1
    for i in range(len(predict)): 
        if y_actual[i]==1 and y_actual[i]!=predict[i]:
           FP += 1
    for i in range(len(predict)): 
        if y_actual[i]==predict[i]==0:
           TN += 1
    for i in range(len(predict)): 
        if y_actual[i]==0 and y_actual[i]!=predict[i]:
           FN += 1

    return(TP, FP, TN, FN, predict)

def transfer_prob(y_score, threshold):
    y_result = []
    for i in range(len(y_score)):
        if y_score[i] >= threshold:
            y_result.append(1)
        else:
            y_result.append(0)
    return np.asarray(y_result)


TPR_list= np.load('./resolution/TPR_list_reso.npy') 
FP_num_list = np.load('./resolution/FP_num_list_reso.npy')

#TPR_list1= np.load('TPR_list_pure_good.npy') 
#FP_num_list1 = np.load('FP_num_list_pure_good.npy')

#TPR_list2= np.load('TPR_list_160.npy') 
#FP_num_list2 = np.load('FP_num_list_160.npy')

# false_positive_rate, true_positive_rate, threshol = metrics.roc_curve(y_true.reshape(y_true.shape[0]*y_true.shape[1]), y_score.reshape(y_score.shape[0]*y_score.shape[1]))
# thresholds = []
# count = 1
# for i in range(threshol.shape[0]):
#     if(threshol[i] > 0.01 and count%100 == 0):
#         thresholds.append(threshol[i])
#     count += 1
# thresholds = np.asarray(thresholds)
plt.gca().set_color_cycle(['red', 'green', 'blue'])
plt.title('Free response Receiver Operating Characteristic')
plt.plot((FP_num_list), (TPR_list), marker = 'o')
# plt.plot((FP_num_list1), (TPR_list1), marker = 'o')
# plt.plot((FP_num_list2), (TPR_list2), marker = 'o')

plt.ylim([ 0.3,0.85])
plt.xlim([-0.1,6])
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Negative Numbers')
plt.legend(['Filter size: 160', 'Filter size: 64', 'Filter size: 160'], loc='lower right')
plt.show()

plt.imshow(y_score[18].reshape(264,132),'gray')

getPara(y_score[18], y_true[18], 0.43136)

getPara(y_score[18], y_true[18], 0.40587)

(TP, FP, TN, FN, class_lable) = perf_measure(y_true[18], y_score[18], 0.43136)
class_lable = class_lable.astype(bool).reshape(264,132)
true =  y_true[18].astype(bool).reshape(264,132)

predict2 = remove_small_objects(class_lable, 64,in_place=False)
true2 = remove_small_objects(true, 64,in_place=False)
labeled_array1, num_features1 = label(predict2)
labeled_array2, num_features2 = label(true2)
FP_num = num_features1 - num_features2

num_features1

plt.imshow(predict2,'gray')

(TP, FP, TN, FN, class_lable) = perf_measure(y_true[18], y_score[18], 0.40587)
class_lable = class_lable.astype(bool).reshape(264,132)
true =  y_true[18].astype(bool).reshape(264,132)

predict2 = remove_small_objects(class_lable, 64,in_place=False)
true2 = remove_small_objects(true, 64,in_place=False)
labeled_array1, num_features1 = label(predict2)
labeled_array2, num_features2 = label(true2)
FP_num = num_features1 - num_features2

num_features1

y_score = np.load('./resolution/predicted_prob.npy')
y_true = np.load('./resolution/answer_image.npy')
reso = np.load('./resolution/resolution.npy')

y_score = y_score[0:y_score.shape[0],1]
y_true = y_true[0:y_true.shape[0],1]

scores = []
trues = []
next_start = 0
for i in range(len(reso)):
    ysize = (reso[i][0] - 56)
    xsize = (reso[i][0]*0.5 - 28)
    scores.append(y_score[next_start: next_start + np.int(ysize * xsize )])
    trues.append(y_true[next_start: next_start + np.int(ysize * xsize )])
    next_start = np.int( next_start + ysize * xsize )
    

for i in range(len(reso)):
    ysize = (reso[i][0] - 56)
    xsize = (reso[i][0]*0.5 - 28)
    print(ysize,xsize)

import cv2
import numpy as np
import os
import dicom
import data
import copy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from scipy import ndimage
from keras.optimizers import SGD
from keras.utils import np_utils
from skimage.morphology import remove_small_objects
from scipy import ndimage
from sklearn import metrics, metrics
from scipy.ndimage.measurements import label


def getPara(predict, true, threshold, resolution):
    (TP, FP, TN, FN, class_lable) = perf_measure(true, predict, threshold)
    if((TP + FN) == 0):
        TPR = 0
    else:
        TPR = np.float(TP) / (TP + FN)

    class_lable = class_lable.astype(
        bool).reshape(resolution[0], resolution[1])
    true = true.astype(bool).reshape(resolution[0], resolution[1])

    predict2 = remove_small_objects(class_lable, 160, in_place=False)
    labeled_array1, num_features1 = label(predict2)
    labeled_array2, num_features2 = label(true)
    FP_num = num_features1 - num_features2
    if FP_num < 0:
        FP_num = 0
    return TPR, FP_num


def perf_measure(y_actual, predict, threshold):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    predict = transfer_prob(predict, threshold)
    for i in range(len(predict)):
        if y_actual[i] == predict[i] == 1:
            TP += 1
    for i in range(len(predict)):
        if y_actual[i] == 0 and y_actual[i] != predict[i]:
            FP += 1
    for i in range(len(predict)):
        if y_actual[i] == predict[i] == 0:
            TN += 1
    for i in range(len(predict)):
        if y_actual[i] == 1 and y_actual[i] != predict[i]:
            FN += 1

    return(TP, FP, TN, FN, predict)


def transfer_prob(y_score, threshold):
    y_result = []
    for i in range(len(y_score)):
        if y_score[i] >= threshold:
            y_result.append(1)
        else:
            y_result.append(0)
    return np.asarray(y_result)


y_score = np.load('./resolution/predicted_prob.npy')
y_true = np.load('./resolution/answer_image.npy')
reso = np.load('./resolution/resolution.npy')

y_score = y_score[0:y_score.shape[0], 1]
y_true = y_true[0:y_true.shape[0], 1]

scores = []
trues = []
next_start = 0
for i in range(len(reso)):
    ysize = (reso[i][0] - 56)
    xsize = (reso[i][0] * 0.5 - 28)
    reso[i][0] = ysize
    reso[i][1] = xsize
    scores.append(y_score[next_start: next_start + np.int(ysize * xsize)])
    trues.append(y_true[next_start: next_start + np.int(ysize * xsize)])
    next_start = np.int(next_start + ysize * xsize)
    #print(next_start,ysize,xsize)

                                                                                                                                                                                             
y_score = np.asarray(scores)
y_true = np.asarray(trues)
#y_true = y_true.astype(np.int)

set_size = 22

count = 1


thresholds = []
tmp = 0
for m in range(1, 10, 1):
    tmp += 1
    # print(m/np.float(100))
    thresholds.append(m / np.float(100))
for i in range(1, 10, 1):
    thresholds.append(i / np.float(10))
for m in range(90, 100, 1):
    thresholds.append(m / np.float(100))
tmp = 0
for m in range(900, 1000, 1):
    tmp += 1
    if(tmp % 10 == 0):
        thresholds.append(m / np.float(1000))
thresholds = sorted(thresholds, reverse=True)
thresholds = np.asarray(thresholds)


TPR_list = []
FP_num_list = []
delete = [6, 7, 12, 19, 21]

#delete = [6,7,8,9,10,11,12,13,19,20,21]
for t in range(1, thresholds.size):
    tpr_sum = 0
    fp_sum = 0
    for i in range(set_size):
        if i not in delete:
            TPR, FP_num = getPara(y_score[i], y_true[i].astype(np.int),  thresholds[t], reso[i])

getPara(y_score[i], y_true[i].astype(np.int),  thresholds[t], reso[i])

predict = y_score[i]
true = y_true[i]
threshold = 0.5

TP = 0
FP = 0
TN = 0
FN = 0
predict = transfer_prob(predict, threshold)
y_actual = true

for i in range(len(predict)):
    if y_actual[i] == predict[i] == 1:
        TP += 1
for i in range(len(predict)):
    if y_actual[i] == 0 and y_actual[i] != predict[i]:
        FP += 1
for i in range(len(predict)):
    if y_actual[i] == predict[i] == 0:
        TN += 1
for i in range(len(predict)):
    if y_actual[i] == 1 and y_actual[i] != predict[i]:
        FN += 1

for i in range(len(predict)):
    if y_actual[i] == predict[i] == 1:
        TP += 1

y_true[i].astype(np.int)

reso



TPR_list1= np.load('TPR_list_160.npy') 
FP_num_list1 = np.load('FP_num_list_160.npy')

#TPR_list2= np.load('TPR_list1.npy') 
#FP_num_list2 = np.load('FP_num_list1.npy')

false_positive_rate, true_positive_rate, threshol = metrics.roc_curve(y_true.reshape(y_true.shape[0]*y_true.shape[1]), y_score.reshape(y_score.shape[0]*y_score.shape[1]))
thresholds = []
count = 1
for i in range(threshol.shape[0]):
    if(threshol[i] > 0.01 and count%100 == 0):
        thresholds.append(threshol[i])
    count += 1
thresholds = np.asarray(thresholds)
   


TPR_list= np.load('./resolution/TPR_list_reso.npy') 
FP_num_list = np.load('./resolution/FP_num_list_reso.npy')

TPR_list

FP_num_list


TPR_list= np.load('./resolution/TPR_list_600.npy') 
FP_num_list = np.load('./resolution/FP_num_list_600.npy')
TPR_list1= np.load('./resolution/TPR_list_500.npy')
FP_num_list1 = np.load('./resolution/FP_num_list_500.npy')

TPR_list2= np.load('./resolution/TPR_list_490.npy') 
FP_num_list2 = np.load('./resolution/FP_num_list_490.npy')

TPR_list3= np.load('./resolution/TPR_list_400.npy') 
FP_num_list3 = np.load('./resolution/FP_num_list_400.npy')

TPR_list4= np.load('./resolution/TPR_list_noreso.npy') 
FP_num_list4 = np.load('./resolution/FP_num_list_noreso.npy')

TPR_list5= np.load('./resolution/TPR_list_special.npy') 
FP_num_list5 = np.load('./resolution/FP_num_list_spec.npy')

plt.gca().set_color_cycle(['red', 'green', 'blue','orange','purple','yellow'])
plt.title('Free response Receiver Operating Characteristic')
plt.plot((FP_num_list), (TPR_list), marker = 'o')
plt.plot((FP_num_list1), (TPR_list1), marker = 'o') 
plt.plot((FP_num_list2), (TPR_list2), marker = 'o') 
plt.plot((FP_num_list3), (TPR_list3), marker = 'o') 
plt.plot((FP_num_list4), (TPR_list4), marker = 'o') 

plt.ylim([ 0.3,0.9]) 
plt.xlim([-0.1,3])
plt.grid()
plt.legend(loc='lower right') 
plt.ylabel('True Positive Rate')
plt.xlabel('False Negative Numbers')
plt.legend([ 'Filter size: 600', 'Filter size: 500', 'Filter size: un reso_500','Filter size: 400','unstable resolution'], loc='lower right')
plt.show()


TPR_list4= np.load('./resolution/TPR_list_special.npy') 
FP_num_list4 = np.load('./resolution/FP_num_list_special.npy')

plt.gca().set_color_cycle(['red', 'green', 'blue','orange','purple'])
plt.title('Free response Receiver Operating Characteristic')

plt.plot((FP_num_list4), (TPR_list4), marker = 'o') 

plt.ylim([ 0.3,0.95]) 
plt.xlim([-0.1,3])
plt.grid()
plt.legend(loc='lower right') 
plt.ylabel('True Positive Rate')
plt.xlabel('False Negative Numbers')
plt.legend([ 'Filter size: 600', 'Filter size: 500', 'Filter size: un reso_500','Filter size: 400','unstable resolution'], loc='lower right')
plt.show()



