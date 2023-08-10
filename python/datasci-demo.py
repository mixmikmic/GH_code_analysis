# Import one of these two.
#import plotly.plotly as py        # plotly.plotly handles all connections to Plotly servers.
import plotly.offline as pyo      # plotly.offline handles offline plotting requests.

import plotly.graph_objs as go    # plotly.graph_objs holds components used for construcing plots.

pyo.init_notebook_mode()          # initialize offline ipython plotting.

import pandas as pd        # for creating dataframes.
from math import log10     # for log transforming values.

# Load our Ks & Kn values from raw data file using panda's 'read_table' function.
vals = pd.read_table('mutation-data.txt',      # input filepath
                     sep='\t',                 # value separator
                     header=None,              # no header
                     usecols=[0,1],            # use only first two columns
                     names=["Ks", "Kn"],       # name our columns for easy access
                     skip_blank_lines=True,    # ignore any empty lines
                     comment='#')              # ignore any content prefaced with a '#'
vals.head()

vals.shape

# Create a 'trace' for each of our datasets of interest.
ks_trace = go.Histogram(name="Ks",           # give the dataset a descriptive name.
                        x=vals['Ks'],        # select our 'Ks' column from the dataframe as our X-values.
                        histnorm='percent')  # normalize by percentage.

kn_trace = go.Histogram(name="Kn", 
                        x=vals['Kn'], 
                        histnorm='percent', 
                        visible=False)

logks_trace = go.Histogram(name='log10(Ks)', 
                           x=[log10(k) for k in vals['Ks'] if k != 0],  # log-transform values.
                           histnorm='percent', 
                           visible=False)

logkn_trace = go.Histogram(name='log10(Kn)', 
                           x=[log10(k) for k in vals['Kn'] if k != 0], 
                           histnorm='percent', 
                           visible=False)

# Compile all traces into a list.
data = [ks_trace, kn_trace, logks_trace, logkn_trace]

# Define our plot's Layout()
layout = go.Layout(
    title = 'Mutation Ratios of Syntenic Gene Pairs<br>Maize & Sorghum',   # give our plot a title.
    xaxis = {'title': 'Ratio'},                                            # give our x-axis a label.
    yaxis = {'title': 'Percentage of Gene Pairs'},                         # give our y-axis a label.
    updatemenus=list([                                                     # create a dropdown menu.
        dict(
            x=-0.05,          # define our dropdown menu coordinates.
            y=1,
            yanchor='top',    # anchor coordinates at top.
            buttons=list([
                dict(
                    args=['visible', [True, False, False, False]],  # set an argument & the values for each trace.
                    label='Ks',                                     # set the display name on our dropdown.
                    method='restyle'                                # use the 'restyle' function (updates data).
                ),
                dict(
                    args=['visible', [False, True, False, False]],
                    label='Kn',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, True, False]],
                    label='log(Ks)',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, False, True]],
                    label='log(Kn)',
                    method='restyle'
                )
            ])
        )
    ]),
)

fig = go.Figure(data=data, layout=layout)  # compile our data & layout into a figure.
pyo.iplot(fig, filename='datasci-demo')    # using the offline module, create an inline plot.



