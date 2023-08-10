# Basic usage of cosimtracpro as a python lib.
# if you installed the package you can import the module directly.
# if you cloned/downloaded the packages you must add it to the system path
import os
try:
    import cosimtracpro
except ImportError:
    import sys
    sys.path.append('..')
    import cosimtracpro
    
# first go to the download section in repository and download and  unzip scram61.zip
# then import needed functions and parse it into a container:
from cosimtracpro.fileparsers import validateinputfiles, parsefiles, buildchanlist, chanlist2json
# validate files given as inputs:
validatedfiles = validateinputfiles([os.path.join('..','scram61','ciclo5','SUMMARY')])
# parsing results:
parsingresults = parsefiles(validatedfiles)
# build information container
chancontainer = buildchanlist(parsingresults)
# Note: the chanlist is a ordered dict from collections.
# to transfor in a normal list of obeject 
chanlist = list(chancontainer.values())

# Visual Exploration:
# pivottablejs:
chanlistdict = []
for element in chancontainer.values():
        values = dict()
        values['code'] = element.code
        values['label'] = element.label
        values['num'] = element.num
        values['numwithreflectors'] = element.numwithreflectors
        values['chanlocation.quadrant'] = element.chanlocation.quadrant
        values['chanlocation.cart'] = element.chanlocation.cart
        values['chanlocation.table'] = element.chanlocation.table
        values['chanlocation.polar'] = element.chanlocation.polar
        values['fuelgroup'] = element.fuelgroup
        values['fueltype'] = element.fueltype
        values['apl2d'] = element.apl2d
        values['flo2d'] = element.flo2d
        values['lk12d'] = element.lk12d
        values['lk22d'] = element.lk22d
        values['lk32d'] = element.lk32d
        values['win2d'] = element.win2d
        values['exp2d'] = element.exp2d
        values['rpf2d'] = element.rpf2d
        # values['rpf3d'] = element.rpf3d
        chanlistdict.append(values)
import pandas as pd
from IPython.display import display, HTML
df = pd.DataFrame(chanlistdict)
display(df)

from pivottablejs import pivot_ui
pivot_ui(df)

# Visual Exploration:
# Scatter maps:
# first let give someformat to the data:
heatchanlist_x = [chan.chanlocation.cart[0] for chan in chanlist]
heatchanlist_y = [chan.chanlocation.cart[1] for chan in chanlist]
heatchanlist_z = [chan.rpf2d for chan in chanlist]
heatchanlist_text = [chan.fuelgroup for chan in chanlist]
#heatchanlist_z = [chan.rpf2d for chan in chanlist]
annotations = [{'text':str(chan.chanlocation.quadrant)} for chan in chanlist]
import plotly
plotly.offline.init_notebook_mode()
import plotly.plotly as py
from plotly.graph_objs import *

data = Data([
    Scatter(
        x=heatchanlist_x,
        y=heatchanlist_y,
        text=heatchanlist_text,
        marker=Marker(
            size=15,
            # cmax=39,
            # cmin=0,
            symbol='hexagon',
            color=heatchanlist_z,
            colorbar=ColorBar(
                title='Relative Power'
            ),
            colorscale = [[0, '#0000FF'],[0.5, '#FFFF00'] ,[1, '#FF0000']]
        ),
        mode='markers')
])

fig = Figure(data=data)
fig['layout'].update(width = 750, height = 750)

plotly.offline.iplot(fig)
# plotly.offline.plot(fig)

# to short data one can use the following:
import operator
chanlist.sort(key=operator.attrgetter('chanlocation.quadrant','fuelgroup','apl2d'))
# sorted(chanlist,key=operator.attrgetter('chanlocation.quadrant'))
## To be continued

