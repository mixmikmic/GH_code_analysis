get_ipython().magic('matplotlib inline')
from ipywidgets import interact, fixed

import skimage
from skimage import data, filters, io
import numpy as np

i = data.coffee()
io.Image(i)

def edit_image(image, sigma=0.1, r=1.0, g=1.0, b=1.0):
    new_image = filters.gaussian_filter(image, sigma=sigma, multichannel=True)
    new_image = skimage.img_as_float(new_image)
    new_image[:,:,0] = (255*r)*new_image[:,:,0]
    new_image[:,:,1] = (255*g)*new_image[:,:,1]
    new_image[:,:,2] = (255*b)*new_image[:,:,2]
    new_image = np.floor(new_image).astype('uint8')
    return io.Image(new_image)

new_i = edit_image(i, 4.0, r=0.8, b=0.7);
new_i

lims = (0.0,1.0,0.01)
interact(edit_image, image=fixed(i), sigma=(0.0,5.0,0.1), r=lims, g=lims, b=lims);

from IPython.display import display

from sympy import Symbol, Eq, factor, init_printing
init_printing(use_latex='mathjax')

x = Symbol('x')

def factorit(n):
    display(Eq(x**n-1, factor(x**n-1)))

factorit(15)

interact(factorit, n=(2,40));

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def normal_2d(n, mux, muy, sigmax, sigmay, corr):
    mean = [mux, muy]
    cov = [[sigmax**2, corr*sigmax*sigmay],[corr*sigmax*sigmay,sigmay**2]]
    d = np.random.multivariate_normal(mean, cov, n)
    return d[:,0], d[:,1]

x, y = normal_2d(100, 0.5, 0.0, 4.0, 2.0, 0.8)

plt.scatter(x, y, s=100, alpha=0.6);

def plot_normal_2d(n, mux, muy, sigmax, sigmay, corr):
    x, y = normal_2d(n, mux, muy, sigmax, sigmay, corr)
    plt.scatter(x, y, s=100, alpha=0.6)
    plt.axis([-10.0,10.0,-10.0,10.0])

interact(plot_normal_2d, n=(10,100,10), mux=(-5.0,5.0,1), muy=(-5.0,5.0,1),
         sigmax=(0.01,5.0,0.01), sigmay=(0.01,5.0,0.01), corr=(-0.99,0.99,0.01));

def f(x):
    print(x)

interact(f, x=True);

interact(f, x=(0,10,2));

interact(f, x='Spencer');

interact(f, x=dict(this=f, that=tuple, other=str));

import ipywidgets as w

w.VBox([w.HBox([w.Button(description='Click'), w.FloatRangeSlider(), w.Text()]), 
      w.HBox([w.Button(description='Press'), w.FloatText(), w.Button(description='Button'),
            w.FloatProgress(value=40)]), 
      w.HBox([w.ToggleButton(description='Toggle'), w.IntSlider(description='Foobar'),
            w.Dropdown(options=['foo', 'bar'])]),
     ])



