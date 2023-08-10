from IPython.display import Image
Image(filename='/home/mrafi123/Downloads/handwrittendigits.png') 

##mnist_loader : Return the MNIST data as a tuple containing 
##the training data,the validation data, and the test data.

import mnist_loader
training_inputs,training_results, validation_inputs,validation_results, test_inputs,test_results = mnist_loader.load_data_dr()

## Import the python libraries to be used for this experiment
import datetime
import gc
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

training_inputs = np.array(training_inputs)
training_results = np.array(training_results)   
test_inputs = np.array (test_inputs)
test_results = np.array(test_results)
print (training_inputs.shape)
print (training_results.shape)
print (test_inputs.shape)
print (test_results.shape)

print ('Size of Training Data in MB',training_inputs.nbytes/1000000)
print ('Size of Training Labels in MB',training_results.nbytes/1000000)
print ('Size of Test Data in MB',test_inputs.nbytes/1000000)
print ('Size of Test Labels in MB',test_results.nbytes/1000000)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

n = 20  # how many digits we will display
plt.figure(figsize=(20, 6))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_inputs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

num=len(test_inputs)
a = datetime.datetime.now()   
clf = KNeighborsClassifier()

#training knn
clf = clf.fit(training_inputs, training_results)
b = datetime.datetime.now()
c=b-a
print("Time required for Training in Seconds ",c.seconds)

#predicting
a = datetime.datetime.now()      
y_pred = clf.predict(test_inputs[:num])
b = datetime.datetime.now()
c=b-a
print("Time required for Predicting in Seconds ",c.seconds)

#checking the predicted labels against the original labels and printing output
r=0
w=0
for i in range(num):
    if np.array_equal(y_pred[i],test_results[i]):
        r+=1
    else:
        w+=1
print ("tested ", num, " digits")
print ("correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%")
print ("got correctly ", float(r)*100/(r+w), "%")

def recognizePCA(train, trainlab, test, labels, num=None):

    if num is None:
        num=len(test)
    train4pca = np.array(train)
    test4pca = np.array(test)   

    n_components = 25

    #Apply pca
    a = datetime.datetime.now()
    print ("Size of training data set before reduction in Mega Bytes ",train4pca.nbytes/1000000)
    print ("Size of test data set before reduction in Mega Bytes ",test4pca.nbytes/1000000)

    pca = RandomizedPCA(n_components=n_components).fit(train4pca)
    xtrain = pca.transform(train4pca)
    xtest = pca.transform(test4pca)
    print ("Size of training data set after reduction to 25 Components in Mega Bytes ",xtrain.nbytes/1000000)
    print ("Size of test data set after reduction to 25 Components in Mega Bytes ",xtest.nbytes/1000000)
    
    print ("Compression acheived for training %",(1-(xtrain.nbytes/train4pca.nbytes))*100)
    print ("Compression acheived for test %",(1-(xtest.nbytes/test4pca.nbytes))*100)

    b = datetime.datetime.now()
    c=b-a
    print("Time required for PCA in seconds",c.seconds)
    
    a = datetime.datetime.now()   
    clf = KNeighborsClassifier()

    #fitting knn    
    clf = clf.fit(xtrain, trainlab)
    b = datetime.datetime.now()
    c=b-a
    print("Time required for Training in seconds",c.seconds)

    a = datetime.datetime.now()   
    #predicting
    y_pred = clf.predict(xtest[:num])
    #print ('y_pred ',np.shape(y_pred))
    b = datetime.datetime.now()
    c=b-a
    print("Time required for Predicting in seconds",c.seconds)



    #checking the predicted labels against the original labels and printing output
    r=0
    w=0
    for i in range(num):
        if np.array_equal(y_pred[i],labels[i]):
            r+=1
        else:
            w+=1
    print ("tested ", num, " digits")
    print ("correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%")
    print ("got correctly ", float(r)*100/(r+w), "%")

recognizePCA(training_inputs, training_results, test_inputs, test_results)

n_components = 3

#Apply pca
pca = RandomizedPCA(n_components=n_components).fit(training_inputs)
xtrain = pca.transform(training_inputs)
xtest = pca.transform(test_inputs)
xtest = xtest/255
    

### hence select a sample of few images
xtest_sample = xtest[:2000,:]
test_results_sample = test_results[0:2000]





import matplotlib.pyplot as plt
get_ipython().magic('matplotlib')
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#for i in range(len(test_results_sample)): #plot each point + it's index as text above
ax.scatter(xtest_sample[:,0],xtest_sample[:,1],xtest_sample[:,2],c=test_results_sample,s=10)

for x,y,z,i in zip(xtest_sample[:,0],xtest_sample[:,1],xtest_sample[:,2],test_results_sample):
    ax.text(x,y,z,i)

ax.set_xlabel(' PC#1')
ax.set_ylabel(' PC#2')
ax.set_zlabel(' PC#3')

plt.show()

get_ipython().magic('matplotlib inline')

fig=plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
fig.suptitle('MNIST Images - Data Representation in 2D',fontsize=34, fontweight='bold')

plt.xlabel('Principal Component#1', fontsize=25)
plt.ylabel('Principal Component#2', fontsize=25)

plt.scatter(xtest_sample[:,0],xtest_sample[:,1],c=test_results_sample,s=100)
for label, x, y in zip(test_results_sample, xtest_sample[:,0], xtest_sample[:,1]):
    ax.annotate(label,xy=(x,y),textcoords='data', size=16)
ax.legend()
plt.grid()
plt.show()



