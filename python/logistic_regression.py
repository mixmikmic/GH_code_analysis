import itertools
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import f1_score
import multiprocessing as mp
import os

# Test different feature vectors for each sequence
d = {'A':0,'G':1,'C':2,'T':3}

# Construct k-mer dictionary
kmer_to_ind = {}
ind_to_kmer = {}
k_ind = 0
for k in range(1,6):
    for kmer in [''.join(i) for i in itertools.product('ACGT', repeat = k)]:
        kmer_to_ind[kmer] = k_ind
        ind_to_kmer[k_ind] = kmer
        k_ind += 1

# Feature mapping 1: char to int
def seq_to_int(s):
    return map(lambda x: d[x], s)

# Feature mapping 2: k-mer counting for k = 1, 2, 3, 4, 5
def kmer_count(s):
    v = np.zeros(len(kmer_to_ind))
    for k in range(1,6):
        for kmer in [s[i:i+k] for i in range(len(s)-k+1)]:
            v[kmer_to_ind[kmer]] += 1
    return v

def load_features(feature_extractor,filename,num_steps=200):
    # Read in sequence
    f = open(filename)
    X = []
    start_time = time.time()
    for i,line in enumerate(f):
        if i % 1000 == 0:
            print time.time()-start_time, ' s'
        s = line.split()[0]
        a = int(len(s)/2-num_steps/2)
        b = int(len(s)/2+num_steps/2)
        X.append(feature_extractor(s[a:b]))
    return np.array(X)
  
fileprefix = '../jz-rnn-tensorflow/data/deepsea_multitask/deepsea'
feature_extractor = kmer_count

X_test = load_features(feature_extractor,fileprefix + '.data.test.txt')
X_train = load_features(feature_extractor,fileprefix + '.data.train.txt')
Y_test = np.loadtxt(fileprefix + '.labels.test.txt')
Y_train = np.loadtxt(fileprefix + '.labels.train.txt')

pickle.dump((X_test,X_train,Y_test,Y_train),file('data_200.pickle','wb'))
# X_test,X_train,Y_test,Y_train = pickle.load(file('data.pickle','rb'))

# logistic regression model

def logistic_reg(inputs):
    X_test,X_train,Y_test,Y_train,i = inputs
    logreg = linear_model.LogisticRegression(C=1e6)
    logreg.fit(X_train, Y_train)
    Yhat_train = logreg.predict(X_train)
    Yhat_test = logreg.predict(X_test)
    f1_train = f1_score(Y_train,Yhat_train)
    f1_test = f1_score(Y_test,Yhat_test)
    f = open('logistic_regression_results/'+str(i))
    f.write('%.3f\t%.3f\n'%(f1_train,f1_test))
    f.close()
    return f1_train,f1_test

num_tasks = np.shape(Y_train)[1]

all_f1_train = []
all_f1_test = []
start_time = time.time()
for i in range(num_tasks):
    inputs = (X_test,X_train,Y_test[:,i],Y_train[:,i])
    f1_train,f1_test = logistic_reg(inputs)
    all_f1_train.append(f1_train)
    all_f1_test.append(f1_test)
    f = open('logistic_regression_results','a')
    f.write('Task ' + str(i) + ':\tf1 train = %.3f\tf1 test = %.3f\n'%(f1_train,f1_test))
    f.close()
    print 'Task ' + str(i) + ':\tf1 train = %.3f\tf1 test = %.3f'%(f1_train,f1_test) 
    print time.time()-start_time

num_tasks = np.shape(Y_train)[1]
all_inputs = [(X_test,X_train,Y_test[:,i],Y_train[:,i],i) for i in range(num_tasks)]
pool=mp.Pool(processes=40)
pool.map(logistic_reg,all_inputs)

# Read in logistic regression results as a K-by-5 matrix where K = num tasks
# 5 entries are:
#   1. task ID
#   2. number of non-zero training examples
#   3. number of non-zero testing examples
#   4. F1 score for training (-1 if no non-zero labels)
#   5. F1 score for testing (-1 if no non-zero labels)

def logistic_regression_analysis_matrix(dirname):
    X = np.zeros([919,5])
    for i in range(919):
        f = open(dirname+str(i))
        for line in f:
            z = line.split()
        X[i,0] = i
        X[i,1] = float(z[0])
        X[i,2] = float(z[1])
        if len(z) > 3:
            X[i,3] = float(z[2])
            X[i,4] = float(z[3])
        else:
            X[i,3] = -1
            X[i,4] = -1
    return X

def get_best_tasks(X):
    # Sort tasks by F1 score (descending)
    a = sorted([(val,i,X[i,1]/float(80000)) for i,val in enumerate(X[:,4])])
    a.reverse()
    # Grab tasks with best scores
    good_inds = np.array([i[1] for i in a])
    return a, good_inds

X = logistic_regression_analysis_matrix('logistic_regression_100_results/')
a, good_inds = get_best_tasks(X)

get_ipython().magic('matplotlib inline')

# Histograms of total # of 1's in labels
plt.hist(X[:,1],bins=20)
plt.title('histogram of # of non-zero training examples')
plt.figure()
plt.hist(X[:,2],bins=20)
plt.title('histogram of # of non-zero testing examples')

# Plot of F1 scores in sorted order
plt.figure()
plt.plot(np.sort(X[:,4]))
plt.title('F1 scores on testing set (sorted)')

# Copy over validation errors from training model on AWS
import os
key = '54.153.39.112'
os.system('scp -i ../cs224d.pem ubuntu@'+key
          +':deep_learning_genomics_nlp/jz-rnn-tensorflow/weights/*'
          +' ../jz-rnn-tensorflow/aws_weights/weights2')

# Load the F1 matrix from the validation errors file
# f = open('../jz-rnn-tensorflow/aws_weights/weights1/validation_errors')
f = open('../jz-rnn-tensorflow/aws_weights/validation_errors')

F1 = []
train_loss = []
valid_loss = []
for line in f:
    z = line.split('\t')
    F1.append([float(i) for i in z[3].translate(None,'[').translate(None,']').translate(None,'\n').split(',')])
    train_loss.append(float(z[0]))
    valid_loss.append(float(z[1]))
F1 = np.array(F1)
train_loss = np.array(train_loss)
valid_loss = np.array(valid_loss)
print np.shape(F1)

# Print out the max F1 for each of logistic regressions best-performing tasks
n = 919#179
test = []
save_tasks = []
print 'Epoch\tDeep RNN F1\tLog Reg F1\tnumber of positive training examples'
for j,(i,k) in enumerate([(np.max(F1[:,good_inds[i]]),np.argmax(F1[:,good_inds[i]])) 
                          for i in range(n)]): 
    if a[j][2]*80000 <= 800:
#         print '%d\t%3f\t%3f\t%d'%(k,i,a[j][0],a[j][2]*80000)
        test.append([i,a[j][0]])
        save_tasks.append(a[j][1])
test = np.array(test)
save_tasks = np.array(save_tasks)
print '='*80
print """The RNN beats the logistic regression model on %.1f%% of the %d tasks 
    with at least 10%% positive training examples"""%(100*np.sum(test[:,0]>test[:,1])/float(len(test)),len(test))
print 'On average, the RNN F1 score is %.3f times greater.'%(np.sum(test[:,0]-test[:,1])/len(test))

# Compare the performance of the RNN at the most recent epoch to that of the LR model 
for ii in range(len(F1)):
    test2 = []
    for j,i in enumerate([F1[ii,good_inds[i]] for i in range(n)]):
        if a[j][2]*80000 <= 800:
            test2.append([i,a[j][0]])
    test2 = np.array(test2)

    print '%d\t%.3f\t%.3f'%(ii,np.sum(test2[:,0]>test[:,1])/float(len(test2)),
                            (np.sum(test2[:,0]-test2[:,1])/len(test2)))
    
    if ii > 42: break

# Plot training and validation losses for num_steps = 100
plt.figure(figsize=(6,4))
plt.plot(train_loss,label='Training')
plt.plot(valid_loss,'r',label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.savefig('/data/deep_learning/project_deliverables/loss_100.png', format='png', dpi=900)

# Plot everything using a bar plot
import colorsys
def bars_curves_set(curves_set,labels,method_names):
    N_sets = len(curves_set)
    HSV_tuples = [(x*1.0/N_sets, 0.6, 0.7) for x in range(N_sets)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    N = len(curves_set[0])
    ind = np.arange(N)
    width = 0.35       
    fig, ax = plt.subplots(figsize=(20,8))
    plt.grid(which='minor',zorder=0)
    plt.grid(which='major',zorder=0)
    for i in range(N_sets):
        rects = ax.bar(ind-0.2+i*0.2,curves_set[i],width,color=RGB_tuples[i], 
                    ecolor='k',align='center',label=labels[i],zorder=3)
    ax.set_ylabel('F1 score',size=15)
    xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] -0.2 for patch in rects]
    plt.xticks(xticks_pos, method_names, ha='right', rotation=90)
    plt.yticks(np.arange(0,1.1*max([max(i) for i in curves_set]),0.05))
    plt.xlim([np.min(xticks_pos)-0.5,np.max(xticks_pos)+0.5])
    plt.xlabel('Tasks sorted by logistic regression performance',size=15)
    plt.legend(fontsize=15)
    plt.savefig('/data/deep_learning/project_deliverables/F1_scores.png', format='png', dpi=200)

k = 20
curve_set = [test[:k,0],test2[:k,0],test[:k,1]]
bars_curves_set(curve_set,
            labels = ['RNN (max over all epochs)','RNN (best epoch)','Logistic Regression'],
            method_names = np.array([str(i) for i in save_tasks[:k]]))

np.where(test[:,1] == 0)

# Sort indices where logistic regression fails
iz = np.where(test[:,1] == 0)[0][0]
ii = np.array([j[1] for j in sorted([(a,i) for i,a in enumerate(test2[iz:,0])],reverse=True)])

iz_ind = range(iz,iz+len(ii))

plt.figure(figsize=(8,4))
plt.plot(test[:iz,0],label='RNN (max over all epochs)',c='r')
temp1 = test[iz:,0]
plt.plot(iz_ind,temp1[ii],c='r')
plt.plot(test2[:iz,0],label='RNN (best epoch)',c='g')
temp2 = test2[iz:,0]
plt.plot(iz_ind,temp2[ii],c='g')
plt.plot(test[:iz,1],label='Logistic Regression',c='b')
temp3 = test2[iz:,1]
plt.plot(iz_ind,temp3[ii],c='b')
plt.xlabel('Tasks sorted by logistic regression performance')
plt.ylabel('F1 score')
plt.legend()
plt.savefig('/data/deep_learning/project_deliverables/F1_scores_498.png', format='png', dpi=200)

X = logistic_regression_analysis_matrix('logistic_regression_200_results/')
a, good_inds = get_best_tasks(X)

# Plot of F1 scores in sorted order
plt.figure()
plt.plot(np.sort(X[:,4]))
plt.title('F1 scores on testing set (sorted)')

# ip = '54.67.103.213'
# os.system('scp -i ../cs224d_200.pem ubuntu@'+key
#           +':deep_learning_genomics_nlp/jz-rnn-tensorflow/weights/validation_errors'
#           +' ../jz-rnn-tensorflow/aws_weights_200/')

f = open('../validation_errors')

F1 = []
train_loss = []
valid_loss = []
for line in f:
    z = line.split('\t')
    F1.append([float(i) for i in z[3].translate(None,'[').translate(None,']').translate(None,'\n').split(',')])
    train_loss.append(float(z[0]))
    valid_loss.append(float(z[1]))
F1 = np.array(F1)
train_loss = np.array(train_loss)
valid_loss = np.array(valid_loss)
print np.shape(F1)

# Print out the max F1 for each of logistic regressions best-performing tasks
n = 200
test = []
for j,(i,k) in enumerate([(np.max(F1[:,good_inds[i]]),np.argmax(F1[:,good_inds[i]])) 
                          for i in range(n)]): 
    print '%d\t%3f\t%3f'%(k,i,a[j][0]), a[j][2]*80000
    if a[j][2]*80000 > 800:
        test.append([i,a[j][0]])
test = np.array(test)
print '='*80
print """The RNN beats the logistic regression model on %.1f%% of the %d tasks 
    with at least 10%% positive training examples"""%(100*np.sum(test[:,0]>test[:,1])/float(len(test)),len(test))
print 'On average, the RNN F1 score is %.3f times greater.'%(np.sum(test[:,0]/test[:,1])/len(test))

# Compare the performance of the RNN at each epoch to that of the LR model 
for j in range(len(F1)):
    print 'Line '+str(j)
    for i in range(10):
        print '%3f\t%3f'%(F1[j,good_inds[i]],a[i][0])
    print '-'*80

# Plot training and validation losses for num_steps = 100
plt.figure(figsize=(10,6))
plt.plot(train_loss[1:],label='Training')
plt.plot(valid_loss[1:],'r',label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.savefig('/data/deep_learning/project_deliverables/loss_200.eps', format='eps', dpi=900)

