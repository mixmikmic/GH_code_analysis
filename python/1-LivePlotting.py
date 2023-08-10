# . env/bin/activate
# pip install -r requirements.txt 

# bokeh-server

import numpy as np

#from bokeh.embed import notebook_div
#div = notebook_div(plot)

# Or, using the same page's cryptic hints :
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

output_notebook(hide_banner=False)  # If you don't like banners, set to False

N=1000
x=np.random.random(size=N)*100
y=np.random.random(size=N)*100
radii=np.random.random(size=N)*5.0
c=[ "#%02x%02x%02x" % (r,g,150) for r,g in zip(np.floor(50+2*x), np.floor(30+2*y)) ]

plot = figure()
plot.circle(x,y, radius=radii, fill_color=c, fill_alpha=0.6, line_color=None)

show(plot)

import theano
a = theano.shared(3.)
a.name = 'a'
x = theano.tensor.scalar('data')
cost = abs(x ** 2. - x ** a) * 12.
cost.name = 'cost'

from fuel.streams import DataStream
from fuel.datasets import IterableDataset
data_stream = DataStream(IterableDataset(
  np.random.rand(150).astype(theano.config.floatX)
))
np.random.rand(10).astype(theano.config.floatX)

from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring

import blocks.extras.extensions.plot
reload(blocks.extras.extensions.plot)
from blocks.extras.extensions.plot import Plot

# Needs 'url' in order to stream (creating a session on the bokeh-server) - and "default" is a special case
output_notebook(url="default", hide_banner=True)  # If you like banners, the default is False

#plotter = Plot('Plotting example', channels=[['cost'], ['a']], after_batch=True)
plotter = Plot('Plotting example', channels=[['cost','a']], after_batch=True)

main_loop = MainLoop(
     model=None, data_stream=data_stream,
     algorithm=GradientDescent(cost=cost,
                               params=[a],
                               step_rule=Scale(learning_rate=0.01)
     ),
     extensions=[
        FinishAfter(after_n_epochs=1),
        TrainingDataMonitoring([cost, a], after_batch=True),
        #TrainingDataMonitoring([cost], after_batch=True),
        plotter,
     ])  

main_loop.run()

