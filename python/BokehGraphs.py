# Import modules
import numpy as np
import pandas as pd
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot

# DISABLE_BOKEH
try:
    DISABLE_BOKEH
except NameError:
    DISABLE_BOKEH = False
    print("DISABLE_BOKEH not specified, set to "+str(DISABLE_BOKEH))
    
# directory
try:
    directory
except NameError:
    directory = "F:\\PA_UC\\"
    print("Directory not specified, set to "+directory)

# stub
try:
    stub
except NameError:
    stub = 1
    print("Stub not specified, set to "+str(stub))
    
# data
try:
    data
except NameError:
    print("No data available, running ImportData:")
    get_ipython().magic('run ./ImportData.ipynb')
    print("-----")
    
# Interparticle distance
try:
    data["Dist"]
except KeyError:
    print("No data on the inter-particle distance available, running InterparticleDistance:")
    get_ipython().magic('run ./InterparticleDistance.ipynb')
    print("-----")

# Prepare Bokeh
output_notebook()

# Set datasource
source = ColumnDataSource(data)

def BokehPlotDistribution():
    pDist = figure(
        tools="crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,tap,save,hover", 
        active_drag="lasso_select", 
        active_scroll="wheel_zoom", 
        title="Distribution", 
        x_axis_label="Horizontal position x / mm", 
        y_axis_label="Vertical position y / mm")

    pDist.circle(
        'X', 
        'Y', 
        source=source, 
        size=3, 
        line_color='#005B82', 
        fill_color='#005B82', 
        fill_alpha=1
    )
    
    return pDist

def BokehPlotMorphology():
    pDA = figure(
        tools="crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,tap,save,hover", 
        active_drag="lasso_select", 
        active_scroll="wheel_zoom", 
        title="Morphology", 
        x_axis_label="Mean particle diameter / um", 
        y_axis_label="Aspect ratio")

    pDA.circle(
        'd', 
        'A', 
        source=source, 
        size=3, 
        line_color="#005B82", 
        fill_color="#005B82", 
        fill_alpha=1
    )
    
    return pDA

def BokehPlotChemistry(chX='UM', chY='OK'):
    # Check first if columns exists
    if not chX in data.columns:
        print("X column ("+chX+") not found! UM used instead.")
        chX = "UM"
    
    if not chY in data.columns:
        print("Y column ("+chY+") not found! OK used instead.")
        chY = "OK"
        
    pChem = figure(
        tools="crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,tap,save,hover", 
        active_drag="lasso_select", 
        active_scroll="wheel_zoom", 
        title="Chemical composition", 
        x_axis_label="Weight fraction "+chX, 
        y_axis_label="Weight fraction "+chY)

    pChem.circle(
        chX, 
        chY, 
        source=source, 
        size=3, 
        line_color='#005B82', 
        fill_color='#005B82', 
        fill_alpha=1
    )
    return pChem

def BokehPlotHistogram(bins=50):
    # Prepare data
    hist, edges = np.histogram(data["d"], density=True, bins=bins)

    # Graph
    pHist = figure(
        tools="crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,tap,save,hover", 
        active_drag="lasso_select", 
        active_scroll="wheel_zoom", 
        title="Particle size distribution", 
        x_axis_label="Particle diameter d / um", 
        y_axis_label="Number of particles")

    pHist.quad(
        top=hist, 
        bottom=0, 
        left=edges[:-1], 
        right=edges[1:],
        fill_color="#005B82", 
        fill_alpha=0.25,
        line_color="#005B82"
    )
    
    pHist.line(
        [np.median(data["d"]), np.median(data["d"])],
        [0, max(hist)],
        line_width=1,
        line_color="red"
    )

    return pHist

def BokehPlotDistance(bins=50):
    # Prepare data
    histNN, edgesNN = np.histogram(data["Dist"], density=True, bins=bins)

    # Graph
    pNN = figure(
        tools="crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,tap,save,hover", 
        active_drag="lasso_select", 
        active_scroll="wheel_zoom", 
        title="Inter-particle distance", 
        x_axis_label="Distance to neirest neighbor / um", 
        y_axis_label="Number of particles")

    pNN.quad(
        top=histNN, 
        bottom=0, 
        left=edgesNN[:-1], 
        right=edgesNN[1:],
        fill_color="#005B82", 
        fill_alpha=0.25,
        line_color="#005B82"
    )
    
    #xx = np.median(data["Dist"])
    xx = 75
    pNN.line(
        [xx, xx],
        [0, max(histNN)],
        line_width=1,
        line_color="red"
    )

    return pNN

# Prepare a grid with the graphs
pAll = gridplot(
    [
        [BokehPlotDistribution(), BokehPlotMorphology()], 
        [BokehPlotChemistry()],
        [BokehPlotHistogram(), BokehPlotDistance()]
    ],
    plot_width=490,
    plot_height=490
)

# Finally, plot the graphs (only when enabled)
if(DISABLE_BOKEH==False):
    fig = show(pAll, notebook_handle=True)



