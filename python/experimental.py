a = 12

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
from windIO.Plant import WTLayout
import numpy as np

wtl = WTLayout('hornsrev.yml')
farm_name = 'Horns Rev'

import plotly
from plotly.offline import init_notebook_mode
init_notebook_mode() # run at the start of every ipython notebook to use plotly.offline
                     # this injects the plotly.js source files into the notebook
print(plotly.__version__) # requires version >= 1.9.0

wtl.plot_location((32,'U'),layout={'title':farm_name+' location'})

wtl.plot_layout(layout={'title':'Lillgrund layout'})

get_ipython().run_cell_magic('bash', '', "\nwtl.V80['name']")

get_ipython().run_cell_magic('bash', '', 'ls')

import utm

lat, lon = np.array([utm.to_latlon(x, y, 33, 'U') for x, y in wtl.positions]).T
wt = {'type': 'scattergeo',
      'lon': lon,
      'lat': lat,
      'text': wtl.wt_names,
      'name': 'wt'}

wt = {'type': 'scattergeo',
      'lon': lon,
      'lat': lat,
      'text': wtl.wt_names,
      'mode': 'markers',
      'name': 'wt'}

contries = {'type': 'scattergeo',
      'lon': [lon[0]],
      'lat': [lat[0]],
      'mode': 'markers',            
      'marker': {
        'size': 8,
        'opacity': 0.8,
        'symbol': 'square',
        'line': {
                'width': 1,
                'color': 'rgb(102,102,102)'
            },            
      },
      'geo': 'geo2',
}

layout = {
    'geo':{
        'resolution': 50,
        'showland': True,
        'showlakes': True,
        'showsubunits': True,
        'showcountries': True,
        'projection': {
          'type': 'Mercator',
        },
        'lonaxis': {
            'showgrid': True,
            'gridwidth': 0.5,
            'range': [ lon.min() - 0.50, lon.max() + 0.50],
            'dtick': 5
        },
        'lataxis': {
            'showgrid': True,
            'gridwidth': 0.5,
            'range': [ lat.min() - 0.50, lat.max() + 0.50],
            'dtick': 5
        }
      },
    'geo2':{
        'scope': 'europe',
        'showland': True,
        'showlakes': True,
        'showsubunits': True,
        'showcountries': True},      
      'title': 'Lillgrund location',
      'width': 800,
      'height': 800
    }

#mm = Scatter(x=[mm['position'][0] for mm in wtl['metmasts']],
#             y=[mm['position'][1] for mm in wtl['metmasts']],
#             text=[mm['name'] for mm in wtl['metmasts']],
#             mode='markers',
#             name='metmasts')
         
iplot(Figure(data=[wt, contries], layout=layout))







wtl.plot_layout()

1000*2702/(500*2664.)



off = 20
for tobj, prop in zip(['layout', 'metmasts', 'transformers'], 
                      ['ob',       'Dg',       'sk']):
    for nm, vm in wtl[tobj].items():
        pos = vm['position']
        pl.plot(pos[0], pos[1], prop)
        pl.text(pos[0]+off, pos[1]+off, vm['name'], fontsize=fontsize)

wtl['turbines']['']

from matplotlib.patches import Polygon, Circle, FancyArrowPatch
import numpy as np

def plot_turbine(pos, turbine_type, figure, fontsize=14, legend=False):
    color = 'gray'
    HH = turbine_type['hub_height']
    TB = HH/10.
    RD = turbine_type['rotor_diameter']

    fact = np.array([TB, HH])
    A = np.array([pos, 0.])
    B = A + [1., 0.]
    C = B + [-0.25, 0.95]
    D = C + [-0.5, 0.]
    tower = Polygon(np.array([A,B,C,D]) * fact, color=color)

    E = A + [-0.25, 0.95]
    F = E + [2.25, 0.0]
    G = F + [0., 0.1]
    H = E + [0., 0.05]
    I = H + [-0.5, 0.]
    J = E + [-0.5, 0.]
    nacelle = Polygon(np.array([E, F, G, H, I, J]) * fact, color=color)


    K = (I + J)/2.0
    hub = Circle(K * fact, radius=(I[1]-J[1])/2.0 * fact[1], color=color)

    L = (J+E)/2. + [0., -0.5*RD/HH]
    M = (I+H)/2. + [0., +0.5*RD/HH*np.sin(-np.pi/2.0+2*np.pi/3.)]
    blade1 = Polygon(np.array([J,E,L]) * fact, color=color)
    blade2 = Polygon(np.array([I,H,M]) * fact, color=color)

    # Plotting the patches
    ax1 = fig1.add_subplot(111, aspect='equal')
    for p in [tower, nacelle, hub, blade1, blade2]:
        ax1.add_patch(p)

    # Plotting the control points
    #for name in 'ABCDEFGHIJKLM':
    #    p = eval(name)*fact
    #    pl.plot(p[0], p[1], 'ob', alpha=0.5)
    #    pl.text(p[0], p[1], name, fontsize=fontsize)

    if legend:
        # Plotting the rotor diam
        ax1.add_patch(FancyArrowPatch((L + [-1., 0.])*fact, (L + [-1., 0.])*fact + [0., RD], arrowstyle='|-|', mutation_scale=5.0,
                                     linewidth=2.))
        N = K + [-1.4, 0.25]
        pl.text(N[0]*fact[0], N[1]*fact[1], 'Rotor Diameter = {} m'.format(RD), fontsize=fontsize, rotation=90)

        # Plotting the hub height
        ax1.add_patch(FancyArrowPatch((A + [2.5, 0.])*fact, (A + [2.5, 0.])*fact + [0., HH], arrowstyle='|-|', mutation_scale=5.0,
                                     linewidth=2.))
        M = A + [2.8, 0.7]
        pl.text(M[0]*fact[0], M[1]*fact[1], 'Hub Height = {} m'.format(HH), fontsize=fontsize, rotation=90)


    

fig1 = pl.figure(figsize=[10, 10]);
plot_turbine(0.0, wtl['turbine_types']['SWP2.3MW'], figure=fig1, fontsize=15, legend=True, )
pl.axis('equal')


fig1 = pl.figure(figsize=[15, 7]);
for wt in wtl.wt_list:
    if 'H' in wt['name']:
        rel_pos = np.array(wtl['turbines']['D-01']['position']) - wt['position']
        dist = np.sqrt(rel_pos[0]**2. + rel_pos[1]**2.)
        plot_turbine(dist, wtl['turbine_types'][wt['type']], fig1)
pl.axis('equal')

fs = 15
turbine = wtl['turbine_types']['SWP2.3MW']
ct = np.array(turbine['c_t_curve'])
pl.plot(ct[:,0], ct[:,1])
pl.xlabel('Wind Speed [m/s]', fontsize=fs)
pl.ylabel('C_T [-]', fontsize=fs)

fs = 15
pc = np.array(turbine['power_curve'])
pl.plot(pc[:,0], pc[:,1])
pl.xlabel('Wind Speed [m/s]', fontsize=fs)
pl.ylabel('Power [kW]', fontsize=fs)

from IPython.core.display import HTML
def css_styling():
    styles = open('/Users/pe/pire/numericalmoocstyle.css', 'r').read()
    return HTML(styles)
css_styling()

