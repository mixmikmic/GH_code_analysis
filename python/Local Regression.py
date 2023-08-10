import numpy as np

from ipywidgets import *
from bqplot import LinearScale
import bqplot.pyplot as plt

from plotting_utils import padded_val

import warnings
warnings.simplefilter('ignore')

def gaussian_kernel(x_train, x_test, bw=1.):
    z = (x_train - x_test[:, np.newaxis]) / bw
    return np.exp(-.5 * z ** 2)

def local_regression(x_train, y_train, x_test,
                     kernel=gaussian_kernel, bw=1., order=0):
    '''
    computes local regression with weights coming from the kernel function
    '''
    # compute the weights using the kernel function
    w = kernel(x_train, x_test, bw=bw)
    
    if order == 0: # weighted average
        return np.dot(w, y_train) / np.sum(w, axis=1)
    else: # weighted polyfit
        y_test = np.empty_like(x_test)
        for i, x0 in enumerate(x_test):
            y_test[i] = np.polyval(np.polyfit(x_train, y_train, w=w[i], deg=order), x0)
        return y_test

# generate some train/test data
x_train = np.linspace(-5, 5, 100)
y_train = x_train ** 2 + np.random.randn(100) * 5
x_test = np.linspace(-10, 10, 200)

x0, x1 = np.min(x_train), np.max(x_train)
y0, y1 = np.min(y_train), np.max(y_train)
    
_, _, ymin, ymax = [padded_val(x) for x in (x0, x1, y0, y1)]

axes_options = {'x': {'label': 'X'},
                'y': {'tick_format': '0.0f', 'label': 'Y'}}

reg_fig = plt.figure(animation_duration=1000)
reg_fig.layout.width = '800px'
reg_fig.layout.height = '550px'

plt.scales(scales={'x': LinearScale(min=-8, max=8),
                   'y': LinearScale(min=ymin, max=ymax)})

scatter = plt.scatter(x_train, y_train, axes_options=axes_options,
                      colors=['red'], enable_move=True, stroke='black',
                      interactions = {'click': 'add'})

reg_line = plt.plot(x_test, [], 'g', stroke_width=5, 
                    opacities=[.6], interpolation='basis')

zero_line = plt.hline(level=0, colors=['white'], stroke_width=.6)

kernel_fig = plt.figure(animation_duration=1000, title='Gaussian Kernel')
kernel_fig.layout.width = '600px'
kernel_fig.layout.height = '400px'
plt.scales(scales={'y': LinearScale(min=0, max=1)})
axes_options = {'x': {'num_ticks': 8, 'label': 'X'}, 
                'y': {'num_ticks': 8, 'tick_format': '.1f'}}
kernel_line = plt.plot(x_train, [], 'm', axes_options=axes_options, interpolation='basis')

# widgets for hyper params
bw_slider = FloatSlider(description='Band Width', 
                        min=.1, max=10, step=.1, value=3,
                        continuous_update=False,
                        readout_format='.1f',
                        layout=Layout(width='290px'))

order_slider = IntSlider(description='Order',
                         min=0, max=10, step=1, value=0,
                         continuous_update=False,
                         layout=Layout(width='300px'))

reset_button = Button(description='Reset Points', button_style='success')
reset_button.layout.margin = '0px 0px 0px 50px'

sliders_layout = VBox([bw_slider, order_slider])
sliders_layout.layout.margin = '60px 0px 0px 0px'

def update_reg_line(change):
    bw = bw_slider.value
    order = order_slider.value
    reg_fig.title = 'Local regression(bw={}, polynomial_order={})'.format(bw, order)
    try:
        reg_line.y = local_regression(scatter.x,
                                      scatter.y,
                                      x_test,
                                      bw=bw, 
                                      order=order)
    except Exception as e:
        print(e)

def reset_points(*args):
    with scatter.hold_sync():
        # hold_sync will send trait updates 
        # (x and y here) to front end in one trip
        scatter.x = x_train
        scatter.y = y_train

reset_button.on_click(lambda btn: reset_points())

# event handlers for widget traits
for w in [bw_slider, order_slider]:
    w.observe(update_reg_line, 'value')

scatter.observe(update_reg_line, 'y')

def update_kernel_plot(*args):
    new_bw_value = bw_slider.value
    kernel_line.y = gaussian_kernel(x_train, np.array([0]), bw=bw_slider.value).squeeze()
    
bw_slider.observe(update_kernel_plot, 'value')

update_reg_line(None)
update_kernel_plot(None)
VBox([HBox([sliders_layout, kernel_fig]), 
      VBox([reg_fig, reset_button])])

