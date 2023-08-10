import numpy as np
from plotly.graph_objs import Layout, Surface, Data, Figure, Contour
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

numpts = 50
x = y = np.linspace(0, numpts-1, numpts-1)
xg, yg = np.meshgrid(x, y)   

def f(x, y, a=1/2):
    '''A Cobb-Douglas production function'''
    return (x**a)*(y**(1-a))

zz = f(xg, yg)

surface1 = Surface(x=x, y=y, z=zz, showscale=False)
layout = Layout(title='Cobb-Douglas Production', autosize=True, 
            height=600, width=600)

fig = Figure(data=Data([surface1]), layout=layout)

iplot(fig)

contour1 = Contour(x=x, y=y, z=zz, showscale=False,
            contours=dict(start=0, end=40, size=2)  )

layout = Layout(title='Cobb-Douglas Production', autosize=True,   height=600,
   width=600)
data = Data([contour1])
fig = Figure(data=data, layout=layout)

iplot(fig)

contour1 = Contour(x=x, y=y, z=zz, showscale=False,
            contours=dict(
            coloring='lines'
        ))

layout = Layout(title='Cobb-Douglas Production', autosize=True,   height=600,
   width=600)
data = Data([contour1])
fig = Figure(data=data, layout=layout)

iplot(fig)

