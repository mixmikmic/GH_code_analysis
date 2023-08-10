import plotly

import plotly.plotly as py
from plotly import graph_objs as go

trace0 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17]
)
trace1 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = Data([trace0, trace1])

py.plot(data, filename = 'basic-line')

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 

import numpy as np
x = np.random.randn(2000)
y = np.random.randn(2000)

iplot([Histogram2dContour(x=x,
                          y=y,
                          contours = Contours(coloring = 'heatmap')), 
      Scatter(x= x,
              y=y,
              mode = 'markers',
              marker = Marker(color = 'white', size = 3, opacity = 0.3))],
            show_link = False,
            )

import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Scatter(x = [1,2,3],
                   y = [4, 5, 6],
                   marker ={'color' :'purple'},
                   mode = 'text',
                   text = ['one', 'two','three'],
                   name = '2nd Online Graph')
data  = go.Data([trace1])
layout = go.Layout(title = '2nd Plot', 
                  xaxis = {'title':'x'},
                  yaxis = {'title': 'y'})
figure = go.Figure(data = data, layout = layout)
py.iplot(figure, filename = '2ndplot')

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/yankev/test/master/life-expectancy-per-GDP-2007.csv')

americas = df[(df.continent=='Americas')]
europe = df[(df.continent=='Europe')]

trace_comp = go.Scatter(x = americas.gdp_percap,
                       y = americas.life_exp,
                       mode = 'markers',
                       marker = dict(size = 12, line = dict(width =1), color = 'navy'),
                       name = 'Americas',
                       text = americas.country)

trace_comp1 = go.Scatter(x=europe.gdp_percap,
                         y=europe.life_exp,
                         mode='markers',
                         marker=dict(size=12,
                                     line=dict(width=1),
                                     color="red"),
                         name='Europe',
                         text=europe.country,)
data_comp = [trace_comp, trace_comp1]
layout = go.Layout(title = 'Life Expectancy v. Per Capita GDP, 2007',
                  hovermode = 'closest',
                   xaxis=dict(title='GDP per capita (2000 dollars)',
                              ticklen=5,
                              zeroline=False,
                              gridwidth=2,),
                   yaxis=dict(title='Life Expectancy (years)',
                              ticklen=5,
                              gridwidth=2),)
fig = go.Figure(data = data_comp, layout = layout)
py.iplot(fig, filename='life-expectancy-per-GDP-2007')



