# This is an executable cell. Click anywhere within the grey area (code text) and press Shift+Enter to run.
import matplotlib.pyplot as plt
import ipywidgets as widgets
from intsim import intsim
get_ipython().magic('matplotlib inline')

# This is an executable cell. Click anywhere within the grey area (code text) and press Shift+Enter to run.
def view_image(ant,myres,EW=False,PointSource=True):
    intsim(ant,freq=1e9,myres=myres,EW=EW,PointSource=PointSource)
    plt.show()

# This is an executable cell. Click anywhere within the grey area (code text) and press Shift+Enter to run.
widgets.interact_manual(view_image,ant=widgets.IntSlider(min=2, max=50, step=1),myres=widgets.IntSlider(min=1,max=100,step=1));

# This is an executable cell. Click anywhere within the grey area (code text) and press Shift+Enter to run.
widgets.interact_manual(view_image,ant=widgets.IntSlider(min=2, max=50, step=1),myres=widgets.IntSlider(min=1,max=100,step=1));





















