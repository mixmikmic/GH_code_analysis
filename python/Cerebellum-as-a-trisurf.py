import numpy as np
import matplotlib.cm as cm

vertices=np.loadtxt('Data/cerebellum-v.txt', dtype=np.float32)
faces=np.loadtxt('Data/cerebellum-f.txt', dtype=np.int32)

print vertices.shape, faces.shape, faces[0,:]

x,y,z=vertices.T

cmap=cm.Oranges#matplotlib  colormap
colormap=[cmap(k*0.05)[:3] for k in range(16)]# define a list of color codes to be passed to create_trisurf

from plotly.figure_factory import create_trisurf

fig = create_trisurf(x=x,       
                     y=y, 
                     z=z, 
                     plot_edges=False,
                     colormap=colormap[3:14],
                     simplices=faces,
                     title="Cerebellum",
                     show_colorbar=False
                    )

fig['layout']['scene'].update(camera=dict(eye=dict( x= 1, y=-1, z=0.9)))
fig['layout']['scene'].aspectratio=dict(x=0.85, y=1., z=0.8)
fig['layout'].update(width=600, height=600, font=dict(family='Balto'))

fig['data'][0].update(lighting=dict(ambient=0.18,
                                    diffuse=1,
                                    fresnel=0.1,
                                    specular=1,
                                    roughness=0.05,
                                    facenormalsepsilon=1e-8,
                                    vertexnormalsepsilon=1e-15)
                                   )
fig['data'][0].update(lightposition=dict(x=100,
                                         y=200,
                                         z=0
                                        )
                      )

import plotly.plotly as py
py.sign_in('empet', 'api_key')
py.iplot(fig, filename='Cerebellum')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()

