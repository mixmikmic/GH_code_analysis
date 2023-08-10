from numpy import *
import numpy as np
import cPickle
import scipy.io as io
from random import randrange
from scipy.cluster.vq import whiten  # this package allows us to apply whitenning  
from matplotlib import pyplot as plt
from os.path import join
from sklearn.cluster import KMeans
from sklearn import metrics
import cPickle as pickle
from os.path import join
import matplotlib
from sklearn.feature_extraction import image
from sklearn import preprocessing
from ipywidgets import FloatProgress
from IPython.display import display

class NaiveBayes:
    def __init__(self, NLABELS, training_set,
                 testing_set, training_labels, test_labels):
            self.NLABELS = NLABELS
            self.train_set = training_set
            self.test_set = testing_set
            self.train_labels = training_labels
            self.test_labels = test_labels
    
    def compute_means_var(self):
        means = {}
        variances = {}
        for lbl in range(self.NLABELS):
            subtrain = self.train_set[self.train_labels==lbl]
            mean = subtrain.mean(axis=0)   
            means[lbl]=mean                 
            var = sum((subtrain[n] - mean)**2 
                      for n in range(subtrain.shape[0]))/subtrain.shape[0]
            variances[lbl]=var
        return means, variances
        
    def compute_priors(self):
        priors = {}
        priors = np.zeros([self.NLABELS,1])
        for lbl in range(self.NLABELS):
            priors[lbl]=self.train_labels[self.train_labels==lbl].shape[0]
        priors = priors/priors.sum()
        return priors
    
    def computePosteriors(self, image, m, v, p):
        posteriors = np.zeros([self.NLABELS,1])
        for lbl in range(self.NLABELS):
                mean = m[lbl]
                sigma2 = v[lbl]
                non_null = sigma2!=0
                scale = 0.5*np.log(2*sigma2[non_null]*math.pi)
                expterm = -0.5*np.divide(np.square(image[non_null]-mean[non_null])
                                         ,sigma2[non_null])
                llh = (expterm-scale).sum()
                post = llh + np.log(p[lbl]) 
                posteriors[lbl]=post
        return posteriors
    
    def run_naive_bayes(self):
        f = FloatProgress(min=0, max=NLABELS)
        display(f)

        means, variances = self.compute_means_var()
        priors =self.compute_priors()
        total=0.0
        correct=0.0
        confusion = np.zeros([self.NLABELS,self.NLABELS])
        dataset= self.test_set
        dataset_label = self.test_labels
        for i in range(len(dataset)):
            f.value+=1
            posts = self.computePosteriors( dataset[i],  means, variances,priors )
            hyp=np.argmax(posts)  
            ref=dataset_label[i]
            if hyp==ref:
                correct+=1
            confusion[hyp][ref]+=1
            total+=1
        print "Correctly classified images : "+str(correct)+" / "+str(total)+ " -> "+ str(correct*100/total) 
    


NUM_CLUSTERS =50
NLABELS = 10
# Get labels
with open(join('cifar-10-batches-py','test_batch'),'rb') as f:
    data_2 = pickle.load(f)
labels2 = data_2['labels']

# Load features
features = pickle.load(open("features/hard-k-150/raw-data/projecteatures-hard-300-16.obj", "rb"))

# Run Naive Bayes
train_set = empty((7000, 4*NUM_CLUSTERS))
test_set = empty((3000, 4*NUM_CLUSTERS))
train_labels = empty((7000,))
test_labels = empty((3000,))

train_set[0:7000,:] = features[0:7000,:]
test_set[0:3000,:] = features[7000:10000,:]
train_labels=np.array(labels2[:7000])
test_labels =np.array(labels2[7000:])
naive_bayes = NaiveBayes(NLABELS,train_set,test_set,train_labels,test_labels )
naive_bayes.run_naive_bayes()





