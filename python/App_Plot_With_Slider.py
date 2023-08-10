import warnings
warnings.filterwarnings('ignore')

import holoviews as hv
hv.notebook_extension('bokeh')

import param

import numpy as np
print("HV version", hv.__version__ )

def simulate_random_walk( Nsteps=20, Nwalks=1000, max_step_length=2.):
    '''compute Nwalks=1000 realizations of
    sampling the distance from the origin after every one of Nsteps=20 steps,
    with stepsize drawn from a uniform distribution
    '''
    rwalk      = np.random.uniform(-max_step_length, max_step_length, size=(Nsteps+1, Nwalks))
    rwalk[0,:] = 0.
    rwalk      = np.cumsum(rwalk,axis=0)
    return rwalk

# since we want to deploy an app, we cannot use cellmagic %%opts
#       pass opts as an argument therefore
# to use events to drive the display, define and return an hv.DynamicMap and an associated stream

def random_walk_plot( rwalk, path_opts = dict( height=200, width=550, shared_axes=False, yticks=3),
                             hist_opts = dict( height=200, width=380, shared_axes=False, xticks=3, yaxis=None, invert_axes=True),
                             useStream = False
                    ):
    '''Generate a DynamicMap of the Distribution of the distance from the starting point at each step size
       Optionally allow the DynamicMap to be controlled by a stream
    '''
    N_steps_plus_one, N_walks = rwalk.shape
    n_bins     = 20
    steps      = np.linspace(0.,float(N_steps_plus_one-1), N_steps_plus_one )

    # infer axis bounds
    D            = np.ceil(np.abs(rwalk).max())           # max distance from the origin
    W            = np.ceil(np.abs(rwalk[1,:]).max())      # max step length
    data_per_bin = np.ceil(float(N_walks)/float(n_bins))  # average number of data in each histogram bin
                                                          #    we will use [0,3.5*data_per_bin] for the Frequency range
    def ith_path(ns):
        path = hv.Path((steps[0:ns+1],rwalk[0:ns+1]),
                       label='%d Random walks, uniformly distributed step size'%N_walks)\
               .opts(style=dict(alpha=0.02, color='darkred'))
        return path.redim(x=('x','Number of Steps'),y=('y','Distance Moved')).redim.range(x=(0,N_steps_plus_one-1), y=(-D,D))

    def ith_histogram(ns):
        hist = hv.Histogram( *np.histogram(rwalk[ns], n_bins, density=False))               .opts(style=dict(fill_color='darkred', alpha=0.4))
        return hist.redim.range(Frequency=(0, 3.5*data_per_bin), x=(-D,D)).relabel('Histogram of Distance Moved at given Step')
    
    def ith_plot(ns):
        h = ith_path     (ns).opts( plot=path_opts ) +            ith_histogram(ns).opts( plot=hist_opts )
        return h

    if useStream:
        Step = hv.streams.Stream.define('StepNumber', ns = param.Integer(default=1,bounds=(1,rwalk.shape[0]-1), doc='a step number'))
        step=Step()
        return hv.DynamicMap(ith_plot, streams=[step]), step
    else:
        return hv.DynamicMap(ith_plot, kdims=[hv.Dimension(('ns','Number of Steps'), range=(1,N_steps_plus_one-1),step=1, type=int)]), None

plot_code =    '\n# -------------------------------------------------------------------------------------------\n' +In[2]        + '\n# -------------------------------------------------------------------------------------------\n' +In[3]        + '\n'

rwalk            = simulate_random_walk(Nsteps=20, Nwalks=3000, max_step_length=2.)
rwalk_plot,step  = random_walk_plot( rwalk )
rwalk_plot  # could specify the step to be shown  with  `rwalk_plot[2]` or `rwalk_plot.select(ns=3)`

# check that we can control the display with the step stream:
from IPython.core.display import HTML
display(HTML('''
<div style=\"color:blue;text-align:center;font_size:20px;\">
Distribution of the Distance from the Starting Point<br>
As additional steps are taken, the distribution evolves from uniform to normal
</div>
'''
))

rwalk_plot,step  = random_walk_plot( rwalk , useStream=True)
rwalk_plot

from time import sleep
for i in range( 2, rwalk.shape[0]):
    sleep(0.4); rwalk_plot.event(ns=i)
print('Rwalk plot currently displays step number {ns}'.format(ns=step.ns))
sleep(1);rwalk_plot.event(ns=1)
print('Rwalk plot has now been reset to step number {ns}'.format(ns=step.ns))

# Now let us try this as an app, with the display in the notebook
renderer = hv.renderer('bokeh')

rwalk        = simulate_random_walk(Nsteps=20, Nwalks=3000, max_step_length=2.)
rwalk_plot,_ = random_walk_plot( rwalk, useStream=False )

renderer.app(rwalk_plot, show=True, new_window=False)

plot_code_header = '''# 1D Random Walk with uniformly distributed step size
import holoviews as hv
import numpy as np
import param
# -------------------------------------------------------------------------------------------
''' 

plot_code_instantiation = '''
# -------------------------------------------------------------------------------------------
plot,_    = random_walk_plot( simulate_random_walk(Nsteps=20, Nwalks=3000, max_step_length=2.) )

import holoviews.plotting.bokeh

renderer  = hv.renderer('bokeh')
doc       = renderer.server_doc( plot )
doc.title = 'Random Walk Example 1'
'''
# -------------------------------------------------------------------------------------------
with open( 'randomwalk_app_V1.py','w') as out:
    out.write( plot_code_header )
    out.write( plot_code )
    out.write( plot_code_instantiation )

from bokeh.models import Slider, Button
from bokeh.io import curdoc
from bokeh.layouts import layout

def update_doc( doc ):
    # --------------------------------------------------------------------
    # we will use the same plot as before
    Nsteps        = 20
    rwalk_plot,ns = random_walk_plot( simulate_random_walk(Nsteps=Nsteps, Nwalks=3000, max_step_length=2.), useStream=True )
    rendered_plot = hv.renderer('bokeh').get_plot( rwalk_plot )

    def gen_update_plot():
        def update_plot():
            update_plot.cur_step += 1
            if update_plot.cur_step > Nsteps: update_plot.cur_step = 1
            ns.event(ns = update_plot.cur_step )
        update_plot.cur_step = 1
        return update_plot

    update_plot = gen_update_plot()
    
    # --------------------------------------------------------------------
    # add a button to start/stop the animation
    
    play_button_labels = ['► Play', '❚❚ Pause']
    play_button = Button(label=play_button_labels[0], width=60)

    def gen_animate_plot():
        def animate_plot():
            if play_button.label == play_button_labels[0]: # not playing
                play_button.label = play_button_labels[1]
                doc.add_periodic_callback( update_plot, animate_plot.update_rate )
            else:
                play_button.label = play_button_labels[0]
                doc.remove_periodic_callback( update_plot )
        animate_plot.update_rate = 400
        return animate_plot

    animate_plot = gen_animate_plot()

    play_button.on_click( animate_plot )

    # --------------------------------------------------------------------
    # add a button to take a step
    next_button = Button(label='Next', width=60)
    next_button.on_click( update_plot )

    # --------------------------------------------------------------------
    # add a slider to control the update rate
    
    rate_slider = Slider( start=200, end=1000, value=400, step=10, title='Update Rate in msec')
    
    def set_update_rate( attrname, old, new):
        animate_plot.update_rate = new
        if play_button.label == play_button_labels[1]:
            doc.remove_periodic_callback( update_plot )
            doc.add_periodic_callback( update_plot, animate_plot.update_rate )
           
    rate_slider.on_change('value', set_update_rate )

    # -----------------------------------------------------------------
    # layout the display
    view = layout([ [rendered_plot.state],[rate_slider, play_button, next_button ]], sizing_mode = 'fixed')

    doc.add_root( view )
    
    return doc

gui_code = '\n# -------------------------------------------------------------------------------------------\n' +In[11] + '\n'

# To display this in the notebook, we need
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.io import show as show_app

handler = FunctionHandler( update_doc )
app     = Application(handler)

show_app(app, notebook_url='localhost:8888')

# To deploy this as an app, we again print out the code
# Note the layout will be slightly different: we will not fixt the overlapping buttons in the display above...
# -------------------------------------------------------------------------------------------
with open( 'randomwalk_app_V2.py','w') as out:
    out.write( plot_code_header )
    out.write( plot_code )
    out.write( gui_code )
    out.write( '''
renderer = hv.renderer('bokeh')
doc = update_doc( curdoc() )
''')



