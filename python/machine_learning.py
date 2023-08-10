



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
import re
sns.set()

df=pd.read_csv('ntcir12.csv',sep='\t', index_col='Unnamed: 0')

tags=['person','chair', 'book', 'tvmonitor', 'laptop', 'bottle','cup', 'car','diningtable', 'cell phone',
             'keyboard', 'bowl', 'mouse', 'clock','toilet', 'sink', 'remote', 'suitcase', 'pottedplant','refrigerator',
             'knife', 'handbag', 'vase', 'aeroplane', 'cat','bed', 'sofa', 'backpack', 'tie', 'spoon', 'toothbrush',
             'traffic light', 'bicycle', 'train', 'bird', 'microwave', 'bench','fork', 'oven', 'motorbike', 'donut',
             'wine glass', 'pizza','apple', 'scissors', 'umbrella', 'cake', 'bus', 'truck','banana', 'parking meter',
             'sandwich', 'sports ball', 'broccoli','carrot', 'orange', 'teddy bear', 'dog', 'snowboard','skateboard', 'boat',
             'surfboard', 'frisbee', 'skis', 'hot dog','bear', 'elephant', 'toaster', 'stop sign', 'hair drier', 'kite',
             'sheep', 'zebra', 'tennis racket', 'baseball bat', 'fire hydrant','horse', 'cow', 'giraffe', 'baseball glove']

df.index = pd.to_datetime(df.index)

df.head(5)

# import modules 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix

#import different Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#import other features
from sklearn.metrics import accuracy_score

# will use KNN as the initial classifier
acc_neigh=[]
X1=df[tags]
y1=df['activity']
for neighbor in xrange(1,12): #iterate using different neighbours from 1 to 10
    acc= []
    for i in xrange(10): #iterate 10 times for every neighbour
        x_train1, x_test1, y_train1, y_test1 = train_test_split(X1,y1,test_size=0.3) 
        knn= KNeighborsClassifier(n_neighbors=neighbor)
        knn.fit(x_train1,y_train1)
        acc.append(knn.score(x_test1,y_test1)) # accuracy score
    acc_neigh.append(np.mean(acc)) # append mean value for every neighbour
    print 'neighbor {} processed'.format(neighbor)
   

# plot the results
scores = acc_neigh
plt.plot(np.arange(1,12),scores)
plt.xlabel('Number of neighbours',fontsize= 18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.ylabel('SCORE',fontsize= 18)
plt.show()         

PRC = 0.3
acc_r=np.zeros((10,5))
X=df[tags]
y=df['activity']
for i in xrange(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)
    nn2 = KNeighborsClassifier(n_neighbors=2)
    nn5 = KNeighborsClassifier(n_neighbors=5)
    svc = SVC()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=100)
    
    nn2.fit(X_train,y_train)
    nn5.fit(X_train,y_train)
    svc.fit(X_train,y_train)
    dt.fit(X_train,y_train)
    rf.fit(X_train,y_train)   
    
    y_pred_nn2=nn2.predict(X_test)
    y_pred_nn5=nn5.predict(X_test)
    y_pred_svc=svc.predict(X_test)
    y_pred_dt=dt.predict(X_test)
    y_pred_rf=rf.predict(X_test)
    
    acc_r[i][0] = accuracy_score(y_pred_nn2, y_test)
    acc_r[i][1] = accuracy_score(y_pred_nn5, y_test)
    acc_r[i][2] = accuracy_score(y_pred_svc, y_test)
    acc_r[i][3] = accuracy_score(y_pred_dt, y_test)
    acc_r[i][4] = accuracy_score(y_pred_rf, y_test)

plt.boxplot(acc_r);
for i in xrange(5):
    xderiv = (i+1)*np.ones(acc_r[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
    plt.plot(xderiv,acc_r[:,i],'ro',alpha=0.3)
    
ax = plt.gca()
ax.set_xticklabels(['2-NN','5-NN','SVM','D. Tree', 'R. Forest'],fontsize= 18)
plt.yticks(fontsize= 18)
plt.ylabel('SCORE',fontsize= 18)
plt.show()

PRC = 0.3
X=df[tags]
y=df['activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)   
y_pred_rf=rf.predict(X_test)
print accuracy_score(y_test,y_pred_rf)

print classification_report(y_test,y_pred_rf)

y_train[y_train == 'biking'].count()

a = classification_report(y_test,y_pred_rf)
a= a.split('\n')
precision=[]
x_act = sorted(y.unique())
for i,v in enumerate(a):
    if i>=2:      
        if (v != ''):
            if 'avg' not in v:                
                value= float(v[33:39])
                precision.append(value)                

activity_score={}
for i,v in enumerate(x_act):
    activity_score[v]=precision[i]
    
list= sorted(activity_score.items(), key=lambda x:x[1],reverse=True)
#print list
x_val = [x[0] for x in list]
y_val = [x[1] for x in list]

x_num= np.arange(len(x_val))
plt.xticks(x_num,x_val,rotation= 90,fontsize= 18)
plt.bar(x_num,y_val)
plt.xlabel('activity',fontsize = 18)
plt.ylim([0.3,0.85])
plt.ylabel('precision',fontsize=18)
plt.yticks(fontsize= 18)
plt.show()

activity_occurences={}
for val in x_val:
    activity_occurences[val]=y_train[y_train==val].count()
    
list1= sorted(activity_occurences.items(), key=lambda x:x[1],reverse=True)
print list1
x_val = [x[0] for x in list1]
y_val = [x[1] for x in list1]

x_num= np.arange(len(x_val))
plt.xticks(x_num,x_val,rotation= 90,fontsize= 18)
plt.bar(x_num,y_val,log=False)
plt.xlabel('activity',fontsize = 18)
plt.ylabel('# apearences in the training data',fontsize=18)
plt.yticks(fontsize= 18)
plt.show()    

values = np.zeros((len(list),2))
for i in range(len(list)):
    tag = list[i][0]
    valor = list[i][1]
    values[i,1]=activity_occurences[tag]
    values[i,0] = valor
    
sns.regplot(values[:,1], values[:,0],order=1)
plt.xlabel('# appearences in the training data',fontsize = 18)
plt.ylabel('precision',fontsize=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.show()     

def plot_confusion_matrix(test, pred,normalize=False):
    cm=confusion_matrix(test,pred)
    
    if normalize: 
        cm = cm.astype('float')/cm.sum(axis= 1)
        plt.imshow(cm,cmap=plt.cm.jet, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Predicted label',fontsize= 18)
        plt.ylabel('True label',fontsize= 18)
        x_act= np.sort(y.unique())   
        plt.xticks(np.arange(0,len(x_act)),x_act,rotation=90,fontsize= 18)
        plt.yticks(np.arange(0,len(x_act)),x_act,rotation=0,fontsize= 18)
        plt.grid('off')
        plt.show()

    else:
        plt.imshow(cm,cmap=plt.cm.jet, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Predicted label',fontsize= 18)
        plt.ylabel('True label',fontsize= 18)
        x_act= np.sort(y.unique())   
        plt.xticks(np.arange(0,len(x_act)),x_act,rotation=90,fontsize= 18)
        plt.yticks(np.arange(0,len(x_act)),x_act,rotation=0,fontsize= 18)
        plt.grid('off')
        plt.show()

    
    

a=plot_confusion_matrix(y_test,y_pred_rf,normalize=True)

print "classification accuracy:", accuracy_score(y_test, y_pred_rf)
print "classification accuracy:", rf.score(X_test, y_test)

mask =  df['location'] == 'Home'
mask = mask.astype(int)

df1=df.copy()

df1['Home'] = mask

mask = df['day_of_week'] > 4
mask = mask.astype(int)

df1['Weekend'] = mask

tags1=['person','chair', 'book', 'tvmonitor', 'laptop', 'bottle','cup', 'car','diningtable', 'cell phone',
             'keyboard', 'bowl', 'mouse', 'clock','toilet', 'sink', 'remote', 'suitcase', 'pottedplant','refrigerator',
             'knife', 'handbag', 'vase', 'aeroplane', 'cat','bed', 'sofa', 'backpack', 'tie', 'spoon', 'toothbrush',
             'traffic light', 'bicycle', 'train', 'bird', 'microwave', 'bench','fork', 'oven', 'motorbike', 'donut',
             'wine glass', 'pizza','apple', 'scissors', 'umbrella', 'cake', 'bus', 'truck','banana', 'parking meter',
             'sandwich', 'sports ball', 'broccoli','carrot', 'orange', 'teddy bear', 'dog', 'snowboard','skateboard', 'boat',
             'surfboard', 'frisbee', 'skis', 'hot dog','bear', 'elephant', 'toaster', 'stop sign', 'hair drier', 'kite',
             'sheep', 'zebra', 'tennis racket', 'baseball bat', 'fire hydrant','horse', 'cow', 'giraffe', 'baseball glove','Home','Weekend']

y = df1['activity']
X = df1[tags]

PRC = 0.3
acc_r=np.zeros((10,2))
for i in xrange(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100)
    knn.fit(X_train,y_train)
    rf.fit(X_train,y_train)   
    y_pred_knn= knn.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    acc_r[i][0] = accuracy_score(y_test,y_pred_knn)
    acc_r[i][1] = accuracy_score(y_test,y_pred_rf)
    
plt.boxplot(acc_r)
for i in xrange(2):
    xderiv = (i+1)*np.ones(acc_r[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
    plt.plot(xderiv,acc_r[:,i],'ro',alpha=0.3)
    
ax = plt.gca()
ax.set_xticklabels(['5-NN','RF-100 est'],fontsize= 18)
plt.yticks(fontsize= 18)
plt.ylim([0, 0.9])
plt.ylabel('SCORE',fontsize= 18)
plt.show()

print classification_report(y_test,y_pred_rf)

y_train[y_train == 'biking'].count()

a = classification_report(y_test,y_pred_rf)
a= a.split('\n')
precision=[]
x_act = sorted(y.unique())
for i,v in enumerate(a):
    if i>=2:      
        if (v != ''):
            if 'avg' not in v:                
                value= float(v[33:39])
                precision.append(value)                

activity_score={}
for i,v in enumerate(x_act):
    activity_score[v]=precision[i]
    
list= sorted(activity_score.items(), key=lambda x:x[1],reverse=True)
#print list
x_val = [x[0] for x in list]
y_val = [x[1] for x in list]

x_num= np.arange(len(x_val))
plt.xticks(x_num,x_val,rotation= 90,fontsize= 18)
plt.bar(x_num,y_val)
plt.xlabel('activity',fontsize = 18)
plt.ylim([0.3,0.9])
plt.ylabel('precision',fontsize=18)
plt.yticks(fontsize= 18)
plt.show()

activity_occurences={}
for val in x_val:
    activity_occurences[val]=y_train[y_train==val].count()
    
list1= sorted(activity_occurences.items(), key=lambda x:x[1],reverse=True)
print list1
x_val = [x[0] for x in list1]
y_val = [x[1] for x in list1]

x_num= np.arange(len(x_val))
plt.xticks(x_num,x_val,rotation= 90,fontsize= 18)
plt.bar(x_num,y_val,log=False)
plt.xlabel('activity',fontsize = 18)
plt.ylabel('# apearences in the training data',fontsize=18)
plt.yticks(fontsize= 18)
plt.show()    

values = np.zeros((len(list),2))
for i in range(len(list)):
    tag = list[i][0]
    valor = list[i][1]
    values[i,1]=activity_occurences[tag]
    values[i,0] = valor
    
sns.regplot(values[:,1], values[:,0],order=1)
plt.xlabel('# appearences in the training data',fontsize = 18)
plt.ylabel('precision',fontsize=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.show()     

plot_confusion_matrix(y_test, y_pred_rf,normalize=True)

tags_rgb = ['R_bin1', 'R_bin2', 'R_bin3', 'R_bin4', 'R_bin5', 'R_bin6', 'R_bin7', 'R_bin8','G_bin1', 'G_bin2', 'G_bin3', 'G_bin4', 'G_bin5', 'G_bin6', 'G_bin7', 'G_bin8','B_bin1', 'B_bin2', 'B_bin3', 'B_bin4', 'B_bin5', 'B_bin6', 'B_bin7', 'B_bin8']

tags2 = tags1 + tags_rgb

#FULL DETAIL DATA: probability 
y = df1['activity']
X = df1[tags2]

PRC = 0.3
acc_r=np.zeros((10,2))
for i in xrange(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100)
    knn.fit(X_train,y_train)
    rf.fit(X_train,y_train)   
    y_pred_knn= knn.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    acc_r[i][0] = accuracy_score(y_test,y_pred_knn)
    acc_r[i][1] = accuracy_score(y_test,y_pred_rf)
    
plt.boxplot(acc_r)
for i in xrange(2):
    xderiv = (i+1)*np.ones(acc_r[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
    plt.plot(xderiv,acc_r[:,i],'ro',alpha=0.3)
    
ax = plt.gca()
ax.set_xticklabels(['5-NN','RF-100 est'],fontsize= 18)
plt.yticks(fontsize= 18)
plt.ylim([0, 0.9])
plt.ylabel('SCORE',fontsize= 18)
plt.show()

plot_confusion_matrix(y_test, y_pred_rf,normalize=True)



import warnings
warnings.filterwarnings('ignore')

#asigning a value to each of the activities from 0 to 1 (normalized)
dic_act={}
val=np.linspace(0,1,20)
for i,v in enumerate(y.unique()):
    dic_act[v]=val[i]

dic_act

df1.head()

def get_previous_act(): # function to get the previous activity
    data=df1.activity.values
    val= []
    X = df1[tags2]
    
    for item in range(len(data)): #For each element in the dataframe:
        if item == 0: # We predict the activity for the first element
            result1 = rf.predict(X.ix[item,:])
            valor_previ = np.nan
            val.append(np.nan)
            
        else:
            result2 = rf.predict(X.ix[item,:])
            if result2!=result1: #check if previous activity is different
                val.append(dic_act[result1[0]])
                valor_previ = dic_act[result1[0]]
                result1 = result2
            else:
                val.append(valor_previ)  #If it's not different, we save the previous value
                
    return val

dummy= get_previous_act()

df2=df1.copy()
df2['prev_activity'] = pd.Series(dummy, index=df.index)

tags=['prev_activity']

tags3= tags2+tags

df2 = df2.dropna(how='any',subset=["activity"],axis=0)

df2.head()

#FULL DETAIL DATA: probability 
y = df2[1:]['activity'] #remove the 1st data point
X = df2[1:][tags3] #remove the 1st data point

PRC = 0.3
acc_r=np.zeros((10,2))
for i in xrange(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100)
    knn.fit(X_train,y_train)
    rf.fit(X_train,y_train)   
    y_pred_knn= knn.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    acc_r[i][0] = accuracy_score(y_test,y_pred_knn)
    acc_r[i][1] = accuracy_score(y_test,y_pred_rf)

plt.boxplot(acc_r)
for i in xrange(2):
    xderiv = (i+1)*np.ones(acc_r[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
    plt.plot(xderiv,acc_r[:,i],'ro',alpha=0.3)
    
ax = plt.gca()
ax.set_xticklabels(['5-NN','RF-100 est'],fontsize= 18)
plt.yticks(fontsize= 18)
plt.ylim([0, 0.9])
plt.ylabel('SCORE',fontsize= 18)
plt.show()

a=plot_confusion_matrix(y_test, y_pred_rf,normalize=True)

print classification_report(y_test,y_pred_rf)

a = classification_report(y_test,y_pred_rf)
a= a.split('\n')
precision=[]
x_act = sorted(y.unique())
for i,v in enumerate(a):
    if i>=2:      
        if (v != ''):
            if 'avg' not in v:   
                
                value= float(v[33:39])
                
                precision.append(value)                

activity_score={}
for i,v in enumerate(x_act):
    activity_score[v]=precision[i]
    
list= sorted(activity_score.items(), key=lambda x:x[1],reverse=True)
#print list
x_val = [x[0] for x in list]
y_val = [x[1] for x in list]

x_num= np.arange(len(x_val))
plt.xticks(x_num,x_val,rotation= 90,fontsize= 18)
plt.bar(x_num,y_val)
plt.xlabel('activity',fontsize = 18)
plt.ylim([0.5,1.0])
plt.ylabel('precision',fontsize=18)
plt.yticks(fontsize= 18)
plt.show()

activity_occurences={}
for val in x_val:
    activity_occurences[val]=y_train[y_train==val].count()
   
list1= sorted(activity_occurences.items(), key=lambda x:x[1],reverse=True)

x_val = [x[0] for x in list1]
y_val = [x[1] for x in list1]

x_num= np.arange(len(x_val))
plt.xticks(x_num,x_val,rotation= 90,fontsize= 18)
plt.bar(x_num,y_val,log=False)
plt.xlabel('activity',fontsize = 18)
plt.ylabel('# apearences in the training data',fontsize=18)
plt.yticks(fontsize= 18)
plt.show()    

values = np.zeros((len(list),2))
for i in range(len(list)):
    tag = list[i][0]
    valor = list[i][1]
    values[i,1]=activity_occurences[tag]
    values[i,0] = valor
    
sns.regplot(values[:,1], values[:,0],order=1)
plt.xlabel('# appearences in the training data',fontsize = 18)
plt.ylabel('precision',fontsize=18)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
#plt.ylim([0.4,1])
plt.show()     

#train and test from diffent users
#TRAIN data is made up using user 1 and 2 information 
df_train = df2[df2['user'] !='u2']
df_train = df_train[df_train['user'] !='u3']
y_train = df_train[1:]['activity']
X_train = df_train[1:][tags3]
# TEST data is made up using user 3 information only
df_test =df2[df2['user']=='u2']
test_data= df_test[tags3]
y_data =df_test['activity']
#####

#print unique activity labels for train and test data
print np.sort(y.unique()) 
print np.sort(y_data.unique())

df_train.tail(2)

knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100)
knn.fit(X_train,y_train)
rf.fit(X_train,y_train)   
y_pred_knn= knn.predict(test_data)
y_pred_rf = rf.predict(test_data)

print classification_report(y_data,y_pred_rf)

plot_confusion_matrix(y_data,y_pred_rf,normalize=True)





a = classification_report(y_data,y_pred_rf)
a= a.split('\n')
precision=[]
x_act = sorted(y_data.unique())
for i,v in enumerate(a):
    if i>=2:      
        if (v != ''):
            if 'avg' not in v:   
                value= float(v[33:39])
                precision.append(value) 

activity_score={}
for i,v in enumerate(x_act):
    activity_score[v]=precision[i]
    
list= sorted(activity_score.items(), key=lambda x:x[1],reverse=True)
#print list
x_val = [x[0] for x in list]
y_val = [x[1] for x in list]

x_num= np.arange(len(x_val))
plt.xticks(x_num,x_val,rotation= 90,fontsize= 18)
plt.bar(x_num,y_val)
plt.xlabel('activity',fontsize = 18)
plt.ylim([0.0,1.0])
plt.ylabel('precision',fontsize=18)
plt.yticks(fontsize= 18)
plt.show()

print "classification accuracy:", accuracy_score(y_data, y_pred_rf)



#TEST MODEL FOR EVERY USER INDIVIDUALLY

df_user1=df2[df2['user']=='u1']
df_user2=df2[df2['user']=='u2']
df_user3=df2[df2['user']=='u3']
y1 = df_user1[1:]['activity']
X1 = df_user1[1:][tags3]
y2 = df_user2[1:]['activity']
X2 = df_user2[1:][tags3]
y3 = df_user3[1:]['activity']
X3 = df_user3[1:][tags3]

PRC = 0.3
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=PRC)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=PRC)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=PRC)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train1,y_train1)   
y_pred_rf1 = rf.predict(X_test1)
print "classification accuracy:", accuracy_score(y_test1, y_pred_rf1)
rf.fit(X_train2,y_train2)   
y_pred_rf2 = rf.predict(X_test2)
print "classification accuracy:", accuracy_score(y_test2, y_pred_rf2)
rf.fit(X_train1,y_train1)   
y_pred_rf3 = rf.predict(X_test3)
print "classification accuracy:", accuracy_score(y_test3, y_pred_rf3)





