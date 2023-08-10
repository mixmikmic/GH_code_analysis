import numpy as np
import sklearn.datasets, sklearn.linear_model, sklearn.neighbors
import sklearn.manifold, sklearn.cluster
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os, time
import scipy.io.wavfile, scipy.signal
import cv2
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (18.0, 10.0)

from jslog import js_key_update
# This code logs keystrokes IN THIS JUPYTER NOTEBOOK WINDOW ONLY (not any other activity)
# Log file is ../jupyter_keylog.csv

get_ipython().run_cell_magic('javascript', '', 'function push_key(e,t,n){var o=keys.push([e,t,n]);o>500&&(kernel.execute("js_key_update(["+keys+"])"),keys=[])}var keys=[],tstart=window.performance.now(),last_down_t=0,key_states={},kernel=IPython.notebook.kernel;document.onkeydown=function(e){var t=window.performance.now()-tstart;key_states[e.which]=[t,last_down_t],last_down_t=t},document.onkeyup=function(e){var t=window.performance.now()-tstart,n=key_states[e.which];if(void 0!=n){var o=n[0],s=n[1];if(0!=s){var a=t-o,r=o-s;push_key(e.which,a,r),delete n[e.which]}}};\nIPython.OutputArea.auto_scroll_threshold = 9999;')

digits = sklearn.datasets.load_digits()
digit_data = digits.data


selection = np.random.randint(0,200,(10,))

digit_seq = [digit_data[s].reshape(8,8) for s in selection]
plt.imshow(np.hstack(digit_seq), cmap="gray", interpolation="nearest")
for i, d in enumerate(selection):    
    plt.text(4+8*i,10,"%s"%digits.target[d])
plt.axis("off")
plt.title("Some random digits from the downscaled MNIST set")
plt.figure()

# apply principal component analysis
pca = sklearn.decomposition.PCA(n_components=2).fit(digit_data)
digits_2d = pca.transform(digit_data)

# plot each digit with a different color (these are the true labels)
plt.scatter(digits_2d[:,0], digits_2d[:,1], c=digits.target, cmap='jet', s=60)
plt.title("A 2D plot of the digits, colored by true label")
# show a few random draws from the examples, and their labels
plt.figure()

## now cluster the data
kmeans = sklearn.cluster.KMeans(n_clusters=10)
kmeans_target = kmeans.fit_predict(digits.data)
plt.scatter(digits_2d[:,0], digits_2d[:,1], c=kmeans_target, cmap='jet', s=60)
plt.title("Points colored by cluster inferred")

# plot some items in the same cluster
# (which should be the same digit or similar!)
def plot_same_target(target):
    plt.figure()
    selection = np.where(kmeans_target==target)[0][0:20]
    digit_seq = [digit_data[s].reshape(8,8) for s in selection]
    plt.imshow(np.hstack(digit_seq), cmap="gray", interpolation="nearest")
    for i, d in enumerate(selection):    
        plt.text(4+8*i,10,"%s"%digits.target[d])
    plt.axis("off")
    plt.title("Images from cluster %d" % target)
    
for i in range(10):    
    plot_same_target(i)    

## now cluster the data, but do it with too few and too many clusters

for clusters in [3,20]:
    plt.figure()
    kmeans = sklearn.cluster.KMeans(n_clusters=clusters)
    kmeans_target = kmeans.fit_predict(digits.data)
    plt.scatter(digits_2d[:,0], digits_2d[:,1], c=kmeans_target, cmap='jet')
    plt.title("%d clusters is not good" % clusters)
    # plot some items in the same cluster
    # (which should be the same digit or similar!)
    def plot_same_target(target):
        plt.figure()
        selection = np.where(kmeans_target==target)[0][0:20]
        digit_seq = [digit_data[s].reshape(8,8) for s in selection]
        plt.imshow(np.hstack(digit_seq), cmap="gray", interpolation="nearest")
        for i, d in enumerate(selection):    
            plt.text(4+8*i,10,"%s"%digits.target[d])
        plt.axis("off")

    for i in range(clusters):
        plot_same_target(i)    


def color_histogram(img, n):
    """Return the color histogram of the 2D color image img, which should have dtype np.uint8
    n specfies the number of bins **per channel**. The histogram is computed in YUV space. """
    # compute 3 channel colour histogram using openCV
    # we convert to YCC space to make the histogram better spaced
    chroma_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 
    # compute histogram and reduce to a flat array
    return cv2.calcHist([chroma_img.astype(np.uint8)], channels=[0,1,2], mask=None, histSize=[n,n,n], ranges=[0,256,0,256,0,256]).ravel()
    

## Solution

