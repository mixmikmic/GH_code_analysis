get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import xgboost
from scipy.stats import kurtosis,skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
import os
import seaborn as sn
from sklearn.model_selection import train_test_split
from time import time
import tensorflow as tf
from keras.utils import *

TARGET_LABELS = {
    1: "Working at Computer",
    2: "Standing Up, Walking and Going updown stairs",
    3: "Standing",
    4: "Walking",
    5: "Going UpDown Stairs",
    6: "Walking and Talking with Someone",
    7: "Talking while Standing",        
}

#to calculate the mod of a vector
def absoluter(q):
    return np.sqrt((q[0]**2 + q[1]**2 + q[2]**2))

data = {1.0:[],2.0:[],3.0:[],4.0:[],5.0:[],6.0:[],7.0:[]}
a = []
path = "/Volumes/Adithya/Adithya/ML/DATA SETS/Activity Recognition from Single Chest-Mounted Accelerometer/"

def data_extract(string):
    global data,a
    with open(path + string + ".csv",'rU') as f:
        c = csv.reader(f)
        for l in c:
            temp  = list(map(float , l))
            if(temp[4]>0.0):
                a.append((temp[1:]))                        #the first element is serial number of the data
    return        

for i in xrange(1,16):
    data_extract(str(i))

g = np.array([0.0, 0.0, 0.0])
c = np.array(a)
b = c

#normalisation
b = c[:,:-1]
lab = c[:,-1]
m = np.mean(b,axis =0)
v = np.var(b,axis = 0)
b = (b-m)/v
lab = lab.reshape(-1,1)
b = np.hstack((b,lab))


arr=np.zeros((len(b),12))
labels=b[:,-1]

for i in range (0,len(b)):
    g = 0.9*g + 0.1*b[i,:-1]  #low pass filter
    v = b[i,:-1]
    v = v - g   #high pass filter

    for j in xrange(3):
        arr[i,j]=b[i,j]
    arr[i,3]=absoluter(b[i,:-1])
    j=4

    for k in xrange(3):
        arr[i,j+k]=g[k]
    arr[i,7]=absoluter(g)
    j=8

    for k in xrange(3):
        arr[i,j+k]=v[k]
    arr[i,11]=absoluter(v)

arr = np.hstack((arr,labels.reshape(-1,1)))

#the method must be changed since it is a dictionary now 
def attr_extract():
    global c
    attr_mean = np.mean(arr,axis=0)
    attr_var = np.var(arr,axis=0)
    attr_std = np.std(arr,axis=0)
    return
   

#arr = np.array(b)
arr.shape

start=time()
try:
    del attributes
except NameError:
    attributes = np.zeros((1,21))   #93 originals     #removed fft too
    #attributes = np.array(attributes)
act = []
#act = np.array(act)
l = 0
i=0
t = time()
while i < len(arr):
    print i
    if i+51>len(arr):
        break
    if(arr[i,-1] != arr[i+51,-1]):
        i+=1
        continue
    else:
        attr_mean = np.mean(arr[i:i+52,:-1],axis = 0)
        attr_var = np.var(arr[i:i+52,:-1],axis=0)
        attr_min = np.amin(arr[i:i+52,:-1],axis = 0)
        attr_max = np.amax(arr[i:i+52,:-1],axis = 0)  
        attr_skew = skew(arr[i:i+52,:-1],axis=0)
        attr_kurtosis = kurtosis(arr[i:i+52,:-1],axis=0)
        
        attr_coeff = []
        
        for j in [0]:  #[0,4,8]
            attr_coeff.append(np.corrcoef(arr[i:i+52,j+0],arr[i:i+52,j+1])[1,0])
            attr_coeff.append(np.corrcoef(arr[i:i+52,j+1],arr[i:i+52,j+2])[1,0])
            attr_coeff.append(np.corrcoef(arr[i:i+52,j+2],arr[i:i+52,j+0])[1,0])
        attr_coeff = np.array(attr_coeff)
        
        attr = np.hstack((attr_mean,attr_var))
        attr = np.hstack((attr,attr_min))
        attr = np.hstack((attr,attr_max))
        attr = np.hstack((attr,attr_skew))
        attr = np.hstack((attr,attr_kurtosis))
        attr = np.hstack((attr,attr_coeff))
        
        act.append(arr[i,-1])
    
        attributes = np.vstack((attributes,attr))
        del attr
        i+=26
print time() - start        
start = time()
attributes = np.delete(attributes,0,0)

xyz = act
act = np.array(act)
act = act.reshape(-1,1)

f = pd.read_pickle("/Users/adithya8.0.0/Desktop/Dk/att.pkl")
f = np.array(f)
np.savetxt("/Users/adithya8.0.0/Desktop/Dk/att.txt",f)

X_train, X_test, y_train, y_test = train_test_split(attributes, act, test_size=0.25,random_state = 0)
rf = RandomForestClassifier(n_estimators=int(np.sqrt(X_train.shape[1]))+1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
labels = list(TARGET_LABELS.values())
sn.heatmap(conf_mat, xticklabels=labels, yticklabels=labels)

test_score = rf.score(X_test, y_test)
print("Test score: %f"%test_score)

X_train, X_test, y_train, y_test = train_test_split(attributes, act, test_size=0.25,random_state = 0)

attributes.shape

X_train.shape

X_test.shape

attributes.shape

len(act)

from collections import Counter
c=Counter(xyz)
print c

for x in conf_mat[np.arange(len(conf_mat)),np.arange(len(conf_mat))]*1.0/np.sum(conf_mat,axis=0):
    print x

conf_mat[np.arange(len(conf_mat)),np.arange(len(conf_mat))]

conf_mat

v=c.values()
plt.bar(np.arange(1,8),v)

def efficiency(feats):
    num_est=5
    columns_list=["Mean x","Mean y","Mean z",
              "Dev x","Dev y","Dev z",
              "Corr x-y","Corr x-z","Corr y-z",
              "Min x","Min y","Min z",
             "Max x","Max y","Max z",
             "Skew x","Skew y","Skew z",
             "Kurtosis x", "Kurtosis y","Kurtosis z"]

    print "For Random Forest:"
    y_t = y_train
    #X_train, X_test, y_t, y_test = train_test_split(feats, y_train, test_size=0.25, random_state=0)
    rf=RandomForestClassifier(n_estimators=100,max_features=6,criterion='entropy')
    rf.fit(X_train,y_t)
    
    t1=time()
    print rf.score(X_test,y_test)
    print "Time Taken : " + str(time()-t1)
    
    y_pred = rf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    importances=rf.feature_importances_
    idx=np.argsort(importances)
    for _n_ in np.array(columns_list)[idx]:
        print _n_
    print
    sn.heatmap(conf_mat)
    
    print "For Decision Trees:"
    d_tree=DecisionTreeClassifier()
    d_tree.fit(X_train,y_t)
    t1=time()
    print d_tree.score(X_test,y_test)
    print "Time Taken : " + str(time()-t1)
    print
    
    print "For SVM:"
    clf=svm.SVC()
    clf.fit(X_train,y_t)
    
    t1=time()
    print clf.score(X_test,y_test)
    print "Time Taken : " + str(time()-t1)
    print
    
    
    print "For Naive bayes:"
    gnb=GaussianNB()
    gnb.fit(X_train,y_t)
    t1=time()
    print gnb.score(X_test,y_test)
    print "Time Taken : " + str(time()-t1)
    print 

    print "For GMM:"
    gmm_=GaussianMixture()
    gmm_.fit(X_train,y_t)
    t1=time()
    res_gmm=gmm_.predict(X_test)
    t2=time()
    print res_gmm,y_test
    print(sum(res_gmm==y_test)*1.0/len(res_gmm))
    print "Time Taken : " + str(t2-t1)
    print 

    print "For Adaboost:"
    ada=AdaBoostClassifier(n_estimators=100,learning_rate=0.001)
    ada.fit(X_train,y_t)
    t1=time()
    print ada.score(X_test,y_test)
    print "Time Taken : " + str(time()-t1)
    print 


    print "For XGboost:"
    xg=xgboost.XGBClassifier(n_estimators=num_est)
    xg.fit(X_train,y_t)

    t1=time()
    print xg.score(X_test,y_test)
    print "Time Taken : " + str(time()-t1)
    print





    

efficiency(attributes)

X_train, X_test, y_train, y_test = train_test_split(attributes, act, test_size=0.25,random_state = 042)

num_features=21 #Change for each of things, put it in a loop
num_activities=7 #For daily sports and activities

x=tf.placeholder(tf.float32,shape=[None,num_features])
y=tf.placeholder(tf.float32,shape=[None,num_activities])

w1=tf.Variable(tf.random_uniform(shape=(num_features,100)))
b1=tf.Variable(tf.zeros(shape=(100,)))
w2=tf.Variable(tf.random_uniform(shape=(100,200)))
b2=tf.Variable(tf.zeros(shape=(200,)))
w3=tf.Variable(tf.random_uniform(shape=(200,7)))
b3=tf.Variable(tf.zeros(shape=(7,)))

h1=tf.add(tf.matmul(x,w1),b1)
h1_act=tf.nn.sigmoid(h1)
h2=tf.add(tf.matmul(h1_act,w2),b2)
h2_act=tf.nn.sigmoid(h2)
out=tf.add(tf.matmul(h2_act,w3),b3)


init=tf.global_variables_initializer()
y_t_1,y_test_1=np_utils.to_categorical(y_train),np_utils.to_categorical(y_test)
y_t_1=y_t_1[:,1:]
y_test_1=y_test_1[:,1:]
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=y_t_1))
opzr=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for epochs in xrange(6000):
    _,c=sess.run([opzr,cost],feed_dict={x: X_train, y:y_t_1})
    if epochs%100==0:
        print "Epoch no : "+str(epochs) + "; Loss : "+str(c) 
res=tf.equal(tf.argmax(out,1),tf.argmax(y_test_1,1))
accuracy = tf.reduce_mean(tf.cast(res, "float"))


print "Accuracy is :  ",accuracy.eval(session=sess,feed_dict={x:X_test,y:y_test_1})

print "For Extra Trees:"
for feats in [attributes]:
    X_train, X_test, y_train, y_test = train_test_split(attributes, act, test_size=0.25,random_state = 42)
    rf=ExtraTreesClassifier(n_estimators=100,max_features=10,criterion='entropy')
    rf.fit(X_train,y_train)
    t1=time()
    print rf.score(X_test,y_test)
    print "Time Taken : " + str(time()-t1)
    print (classification_report(y_test,rf.predict(X_test),digits=4))
    



