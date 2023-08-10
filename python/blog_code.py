import os 
import pandas as pd 
import numpy as np 

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
import keras 
from keras.applications import VGG19, InceptionV3, Xception, ResNet50

from keras.applications.imagenet_utils import preprocess_input as preprocess_type1
from keras.applications.inception_v3 import preprocess_input as preprocess_type2

from cub_util import CUB200

NUM_CLASSES = 200
DATA_DIR = os.path.expanduser(os.path.join("~", "data", "blog")) 
CUB_DIR = os.path.join(DATA_DIR, "CUB_200_2011", "images")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

# If this fails, the CUB-200 dataset is not in the rigt place... :) 
assert os.path.exists(CUB_DIR)

X, _ = CUB200(CUB_DIR, image_size=(100, 100)).load_dataset(num_per_class=5)

n = X.shape[0]
rnd_birds = np.vstack([np.hstack([X[np.random.choice(n)] for i in range(10)])
                       for j in range(10)])
plt.figure(figsize=(8, 8))
plt.imshow(rnd_birds / 255)
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.title("100 random birds...", fontsize=30)

def load_features_compute_once(model, im_size, preprocess, save_path):
        
    if os.path.exists(save_path):
        data = pd.read_csv(save_path, compression='gzip', header=0, index_col=0)
        X = data.values 
        y = data.index.values
    else:
        X, y = CUB200(CUB_DIR, image_size=im_size).load_dataset()
        X = model(include_top=False, weights="imagenet", pooling='avg').predict(preprocess(X))
        pd.DataFrame(X, index=y).to_csv(save_path, compression='gzip', header=True, index=True)

    return X, y

X_resnet, y = load_features_compute_once(ResNet50, (244, 244), preprocess_type1, 
                                         os.path.join(FEATURES_DIR, "CUB200_resnet"))

X_resnet.shape

clf = LinearSVC()
results = cross_val_score(clf, X_resnet, y, cv=3, n_jobs=-1)

print(results)
print("Overall accuracy: {:.3}".format(np.mean(results) * 100.))

X_resnet, y = load_features_compute_once(ResNet50, (244, 244), preprocess_type1, 
                                         os.path.join(FEATURES_DIR, "CUB200_resnet"))

X_vgg, _ = load_features_compute_once(VGG19, (244, 244), preprocess_type1, 
                                         os.path.join(FEATURES_DIR, "CUB200_VGG19"))

X_incept, _ = load_features_compute_once(InceptionV3, (299, 299), preprocess_type2, 
                                         os.path.join(FEATURES_DIR, "CUB200_inception"))

X_xcept, _ = load_features_compute_once(Xception, (299, 299), preprocess_type2, 
                                         os.path.join(FEATURES_DIR, "CUB200_xception"))

X_all = np.hstack([X_vgg, X_resnet, X_incept, X_xcept])
inx = np.cumsum([0] + [X_vgg.shape[1], X_resnet.shape[1], X_incept.shape[1], X_xcept.shape[1]])

y = LabelEncoder().fit_transform(y)

base_classifier = LogisticRegression
meta_classifier = LinearSVC

pipes = [make_pipeline(ColumnSelector(cols=list(range(inx[i], inx[i+1]))), base_classifier())
         for i in range(4)]

stacking_classifier = StackingClassifier(classifiers=pipes, meta_classifier=meta_classifier(), 
                                         use_probas=True, average_probas=True, verbose=1)

results = cross_val_score(stacking_classifier, X_all, y, cv=3, n_jobs=-1)

print(results)
print("Overall accuracy: {:.3}".format(np.mean(results) * 100.))



