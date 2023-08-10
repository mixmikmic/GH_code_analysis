import sys
sys.path.append('../')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
from python.utility import *
from python.validation import *
from sklearn.metrics import jaccard_similarity_score, f1_score
from sklearn.metrics.pairwise import paired_euclidean_distances

# stores the what we got from segmentation
x_folder = "../data/Data Annotation/Predicted/Test/"

# stores the labelled data folder
y_folder = "../data/Data Annotation/GroundTruth/Test/" 

x =  folderToImages(x_folder)
y =  folderToImages(y_folder)

# Overlap ratio measures 
jaccard_similarity_score(y[0], x[0])

#sk learn f1-score
f1_score(y[0], x[0], average='macro')

#sklearn f1-score 
f1_score(y[0], x[0], average='micro')

#sklearn f1-score
f1_score(y[0], x[0], average='weighted')

#sklearn elcudian distance
#paired_euclidean_distances(y[0],x[0])

obj = Validation()
obj.fit(x[0],y[0])
obj.get_f_score()

obj.plot()



