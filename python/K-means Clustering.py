from time import sleep

import numpy as np
from sklearn.datasets import make_blobs

from ipywidgets import *

from bqplot import OrdinalColorScale, CATEGORY10
import bqplot.pyplot as plt

n_slider = IntSlider(description='points', value=150, min=20, max=300, step=10)
k_slider = IntSlider(description='K', value=3, min=2, max=10)
cluster_std_slider = FloatSlider(description='cluster std', value=.8, min=.5, max=3)

iter_label_tmpl = 'Iterations: {}'
iter_label = Label(value=iter_label_tmpl.format(''))
iter_label.layout.width = '300px'

fig = plt.figure(title='K-means Clustering', animation_duration=1000)
fig.layout.width = '800px'
fig.layout.height = '700px'

plt.scales(scales={'color': OrdinalColorScale(colors=CATEGORY10)})

axes_options = {'x': {'label': 'X1'}, 'y': {'label': 'X2'}, 'color': {'visible': False}}

# scatter of 2D features
points_scat = plt.scatter([], [], color=[], stroke='black', axes_options=axes_options)

# scatter of centroids
centroid_scat = plt.scatter([], [], color=[], stroke_width=3, stroke='black',
                            default_size=400, axes_options=axes_options)

go_btn = Button(description='GO', button_style='success', layout=Layout(width='50px'))

def clear():
    # clear all
    with points_scat.hold_sync():
        points_scat.x = []
        points_scat.color = []
    
    with centroid_scat.hold_sync():
        centroid_scat.x = []
        centroid_scat.color = []
    
    iter_label.value = iter_label_tmpl.format('')
    
def start_animation():
    go_btn.disabled = True
    clear()
    
    # get the values of parameters from sliders
    n = n_slider.value
    K = k_slider.value
    
    # 2D features made from K blobs
    X, _ = make_blobs(n_samples=n, centers=K, cluster_std=cluster_std_slider.value)
    np.random.shuffle(X)
    
    # plot the points on a scatter chart
    with points_scat.hold_sync():
        points_scat.x = X[:, 0]
        points_scat.y = X[:, 1]
    
    # randomly pick K data points to be centroids
    centroids = X[:K]
    
    i = 0
    
    # try for 20 iterations
    # FIXME: algo simetimes converges to a local minimum!!
    while i < 20:
        iter_label.value = iter_label_tmpl.format(i + 1)
        
        with centroid_scat.hold_sync():
            centroid_scat.x = centroids[:, 0]
            centroid_scat.y = centroids[:, 1]
            centroid_scat.color = np.arange(K)
            
        # assign clusters to points based on the closest centroid
        clusters = np.argmin(np.linalg.norm(X.reshape(n, 1, 2) - centroids, axis=2), axis=1)
        
        # color code the points by their clusters
        points_scat.color = clusters

        # compute new centroids from the clusters
        new_centroids = np.array([X[clusters == k].mean(axis=0) for k in range(K)])
        
        if np.all(centroids == new_centroids):
            # if centroids don't change we are done
            break
        else: 
            # update the centroids and repeat
            centroids = new_centroids
            i = i + 1
            sleep(1)
            
    go_btn.disabled = False

go_btn.on_click(lambda btn: start_animation())

controls_layout = VBox([n_slider, k_slider, cluster_std_slider, go_btn, iter_label])
controls_layout.layout.margin = '60px 0px 0px 0px'

HBox([VBox([fig]), controls_layout])

