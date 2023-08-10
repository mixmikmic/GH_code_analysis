import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os

dirPath = os.path.realpath('.')
fileName = 'assets/coolingExample.xlsx'
filePath = os.path.join(dirPath, fileName)
df = pd.read_excel(filePath,header=0)
cols = df.columns

# Create a trace
trace = go.Scatter(
    x = df[cols[0]],
    y = df[cols[1]]
)

data = [trace]

# Edit the layout
layout = dict(title='Temperature vs. Time',
              xaxis=dict(title='Time'),
              yaxis=dict(title='Temperature (C)'),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Thermal Data')

