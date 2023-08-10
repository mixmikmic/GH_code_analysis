import itertools
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd

data = pd.io.parsers.read_table(
    filepath_or_buffer='/home/if/ChallengeAll/dataset/F_measure_103.txt',
    header=None,
    )
data.dropna(how='any', inplace=True)# how : {‘any’, ‘all’} any : if any NA values are present,drop that label


accuracy_data = pd.io.parsers.read_table(
    filepath_or_buffer='/home/if/ChallengeAll/dataset/accuracy.txt',
    header=None,
    )
data.tail(0)

def percent2decimal(x):      # 百分数转换为小数
     return float(x.strip('%'))/100
accuracy = np.zeros(103)

sum = 0
for i in range(len(data)):   
    accuracy = percent2decimal(accuracy_data[0][i]) 
    #print accuary
    sum += accuracy
print ("average accuracy: %f"% float(sum/103))

result = np.loadtxt('/home/if/ChallengeAll/dataset/F_measure_103.txt')

num = 0
sum_precision = 0
sum_recall = 0
sum_F1_score = 0

for i in range(len(data)):
    #print data[i].sum()
    T_p = data[i][i]
    recall = float(T_p)/float(result[i].sum())    
    sum_recall += recall
    
    if data[i].sum() == 0 :
        precision = 0
    else:
        num+=1 
        precision = float(T_p)/float(data[i].sum())      
    sum_precision += precision
    if (recall == 0) or (precision ==0):
        F1_score = 0
    else:
        F1_score = 2*precision*recall/(precision + recall)
    print F1_score
    sum_F1_score += F1_score
   
print( 'F1_score:%f' % float(sum_F1_score/103)) 
print 2*sum_precision*sum_recall/(sum_precision+sum_recall)/103
#recall =  sum_recall/103
#F1_score =  2*precision*recall/(precision + recall)

from sklearn.metrics import confusion_matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
cnf_matrix = confusion_matrix(y_true, y_pred)

confusion_matrix = np.loadtxt('/home/if/ChallengeAll/dataset/mnist_conf.txt')

classesname = [0,1,2,3,4,5,6,7,8,9]

plot_confusion_matrix(confusion_matrix, normalize=False, classes = classesname, title='Confusion matrix' )
plt.show()



print result.sum()

True_positive = np.diag(np.diag(result))


print True_positive.sum()
  
for i in range(len(result)):
    for classes in result:
        print classes.sum()  # 每一类所有的值
    
print result.shape



