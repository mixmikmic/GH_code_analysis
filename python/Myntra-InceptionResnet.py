import numpy as np
import pandas as pd
from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

arch = inceptionresnet_2
sz=320
bs = 32
path='data/hackerearth-myntra'

trn_tfms,val_tfms = tfms_from_model(arch,sz,crop_type=CropType.NO)

data = ImageClassifierData.from_csv(path='data/hackerearth-myntra/',folder='train',csv_fname='myntra-last.csv',tfms=(trn_tfms,val_tfms),bs=bs,num_workers=2)

learn = ConvLearner.pretrained(arch,data,precompute=True,ps=0.4)

lrf = learn.lr_find()

learn.sched.plot()

learn.fit(0.0005,5)

learn.fit(0.0005,5)

learn.fit(0.0001,5)

learn.fit(0.0001,5)







from sklearn.metrics import confusion_matrix
def plot_confusion_matrix_mine(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(12,12))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

log_preds,y = learn.TTA()

probs = np.mean(np.exp(log_preds),0)

accuracy_np(probs, y)

y[10]=9

predictions = np.argmax(probs,axis=1)

cm = confusion_matrix(y, predictions)

plot_confusion_matrix_mine(cm,data.classes,normalize=False)

plot_confusion_matrix_mine(cm,data.classes,normalize=True)

learn = ConvLearner.pretrained(arch,data,precompute=True,ps=0.2)

lrf = learn.lr_find()

learn.sched.plot()

learn.fit(0.005,5)

learn.fit(0.005,2)

learn.fit(0.005,2)

learn.fit(0.0001,5)

learn.fit(0.0005,2)

learn.fit(0.0005,2)

learn.fit(0.0001,2)

learn.fit(0.0001,5)

log_preds,y = learn.TTA()

probs = np.mean(np.exp(log_preds),0)

accuracy_np(probs, y)

y[10]=9

predictions = np.argmax(probs,axis=1)

cm = confusion_matrix(y, predictions)

plot_confusion_matrix_mine(cm,data.classes,normalize=False)

plot_confusion_matrix_mine(cm,data.classes,normalize=True)

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax
x,y=next(iter(data.val_dl))

show_img(data.val_ds.denorm(to_np(x))[3]);







