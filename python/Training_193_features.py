import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import sklearn as sk
import tensorflow as tf
from sklearn.svm import NuSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

def load193(parent_dir, file_title, train_folds, dev_folds, test_folds):
    train_set = []
    dev_set = []
    test_set = []
    
    for i in train_folds:
        ds_filename = parent_dir + file_title + str(i)+".csv"
        df = pd.read_csv(ds_filename, index_col = None)
        train_set.append(df)
        
    for i in dev_folds:
        ds_filename = parent_dir + file_title + str(i)+".csv"
        df = pd.read_csv(ds_filename, index_col = None)
        dev_set.append(df)
        
    for i in test_folds:
        ds_filename = parent_dir + file_title + str(i)+".csv"
        df = pd.read_csv(ds_filename, index_col = None)
        test_set.append(df)
        
    print("done!")    
    return  pd.concat(train_set, ignore_index=True), pd.concat(dev_set, ignore_index=True), pd.concat(test_set, ignore_index=True)  #dev_set, test_set

parent_dir = "./UrbanSound8K/audio/"
file_title = "features193"
train_folds = np.array(range(1,9)) #first 8 folds as training set
dev_folds = np.array([9]) #9th fold as dev set
test_folds = np.array([10]) #10th fold as test set

train_pd, dev_pd, test_pd = load193(parent_dir, file_title, train_folds, dev_folds, test_folds)

print(train_pd.shape)
print(dev_pd.shape)
print(test_pd.shape)
train_pd.head()


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

train_x, train_y = train_pd.iloc[:, 0:193].values, train_pd.iloc[:, 193].values
dev_x, dev_y = dev_pd.iloc[:, 0:193].values, dev_pd.iloc[:, 193].values
test_x, test_y = test_pd.iloc[:, 0:193].values, test_pd.iloc[:, 193].values

#confusion matrix plot
label_id_name = {0: 'air_conditioner',
 1: 'car_horn',
 2: 'children_playing',
 3: 'dog_bark',
 4: 'drilling',
 5: 'engine_idling',
 6: 'gun_shot',
 7: 'jackhammer',
 8: 'siren',
 9: 'street_music'}

def plt_confusion(title, prediction, label, label_id_name):
    classe_names = label_id_name.values()
    matrix = confusion_matrix(label, prediction)

    plt.figure(figsize=[10,10])
    plt.imshow(matrix, cmap='hot', interpolation='nearest',  vmin=0, vmax=200)
    plt.colorbar()
    plt.title(title, fontsize=18)
    plt.ylabel('True Value', fontsize=18)
    plt.xlabel('Prediction', fontsize=18)
    plt.grid(b=False)
    plt.yticks(range(10), classe_names, fontsize=14)
    plt.xticks(range(10), classe_names, fontsize=14, rotation='vertical')

    plt.show()

forest = OneVsRestClassifier(RandomForestClassifier(n_estimators = 500, max_depth=20, min_samples_leaf=30))

start_time = time.time()
forestmodel = forest.fit(train_x, train_y)
end_time = time.time()
print("seconds:", end_time-start_time)

rf_prediction = forestmodel.predict(dev_x)

rf_accuracy = np.sum(rf_prediction == dev_y)/dev_x.shape[0]
print("Random Forest Accuracy: ", rf_accuracy)

plt_confusion("Random Forest Confustion", dev_y, rf_prediction, label_id_name)

svm = OneVsRestClassifier(NuSVC(nu=.08, kernel='poly', decision_function_shape='ovr'))

start_time = time.time()
svmmodel = svm.fit(train_x, train_y)
end_time = time.time()
print("seconds:", end_time-start_time)

svm_prediction = svmmodel.predict(dev_x)

svm_accuracy = np.sum(svm_prediction == dev_y)/dev_x.shape[0]
print("SVM Accuracy: ", svm_accuracy)

plt_confusion("SVM Confusion", dev_y, svm_prediction, label_id_name)

training_epochs = 5000
n_dim = train_x.shape[1]
n_classes = 10
n_hidden_units_one = 280 
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.1

train_y_one_hot = one_hot_encode(train_y)
dev_y_one_hot = one_hot_encode(dev_y)
test_y_one_hot = one_hot_encode(test_y)


print(dev_y_one_hot[751])
print(dev_y[751])
print(dev_y_one_hot.shape)

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.global_variables_initializer()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1])) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

start_time = time.time()

cost_history = np.empty(shape=[1],dtype=float)
dev_accuracy_history = np.zeros(1, dtype=float)
train_accuracy_history = np.zeros(1, dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):            
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y_one_hot})
        cost_history = np.append(cost_history,cost)
    
        if((epoch+1) % 200 == 0):
            y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: dev_x})
            y_true = sess.run(tf.argmax(dev_y_one_hot,1))
            dev_accu= np.sum(y_pred == y_true)/dev_x.shape[0]
            dev_accuracy_history = np.append(dev_accuracy_history, dev_accu)
            
            yo_pred = sess.run(tf.argmax(y_,1),feed_dict={X: train_x})
            yo_true = sess.run(tf.argmax(train_y_one_hot,1))
            train_accu= np.sum(yo_pred == yo_true)/train_x.shape[0]
            train_accuracy_history = np.append(train_accuracy_history, train_accu)
            
            
end_time = time.time()
print("seconds:", end_time-start_time)

ffn_accuracy = np.sum(y_pred == y_true)/dev_x.shape[0]
print("Feedforward Network Accuracy: ", ffn_accuracy)

ffn_accuracy1 = np.sum(yo_pred == yo_true)/train_x.shape[0]
print("Feedforward Network Training Accuracy: ", ffn_accuracy1)

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print("F-Score:", round(f,3))
print(p, r, f, s)

fig = plt.figure(figsize=(10,8))
plt.plot(dev_accuracy_history, "r--", label = "Dev Accuracy")
plt.plot(train_accuracy_history, "g", label ='Training Accuracy')
plt.ylabel("Accuracy")
plt.xlabel("Epochs *200")
plt.legend()
plt.show()

plt_confusion("FNN Confusion", y_true, y_pred, label_id_name)

