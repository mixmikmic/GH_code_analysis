import warnings
warnings.filterwarnings('ignore')

import holoviews as hv
hv.notebook_extension('bokeh','matplotlib')

import param

import numpy as np
import pandas as pd
print("HV version", hv.__version__ )

def background( func=lambda x,y: np.sin(x*y), size=(500,500)):
    ''' Given the ODE y'=f(x,y)
             bg,vec,xaxis_line,yaxis_line = background()
        returns a grayscale image of the slopes y',
                a vector field representation of the slopes, and
                a set of axis lines for -5<x<5, -5<y<5
    '''
    
    # compute the data
    vals = np.linspace( -5, 5, num=150 )
    X, Y = np.meshgrid(vals, vals)

    clines  = func(X,Y)                    # f(x,y)
    theta   = np.arctan2( clines, 1)       # angle of the slope y' at x,y

    # Obtain the vector field (subsample the grid)
    h,w=size
    vf_opts = dict(size_index=3, height=h, width=w, xticks=9, yticks=9)
    vec_field = hv.VectorField((vals[::3],vals[::3],theta[::3,::3],0*clines[::3,::3]+1) ).opts(plot=vf_opts)
    
    # overlay the vector field with a gray scale image:
    #     this is tricky: we need RGB to be able to set alpha
    #        and the data must be normalized to [0,1]
    #   to work correctly the RGB constructor RGB((x,y, r,g,b,a), vdims=[]) must set
    #        the vdim names 'R','G','B','A'
    def norm(arr):
        arr = (arr-arr.min())
        return arr/arr.max()

    normXY    = norm(clines)
    img_field = hv.RGB( (vals, vals, normXY, normXY, normXY, 0*clines+0.1), vdims=['R','G','B','A'] )                .opts(style=dict(width=600, height=600, cmap='gray'), plot=dict(shared_axes=False))

    # finally, we add the axes as VLine, HLine and return an array of the plot Elements
    hv_opts = dict( color='green', alpha=0.5)
    return [img_field,vec_field, hv.HLine(0).opts(style=hv_opts),
                hv.VLine(0).opts(style=hv_opts)]

#Test it:
hv.Overlay(background(size=(400,400))).relabel("y' = sin(x y)" ) +hv.Overlay(background(func=lambda x,y: x*np.sin(5*y), size=(400,400))).relabel("y' = x sin(5 y)" )

def euler_step(x,y,h,func):
    '''x <- x +h, y_<- y + h f(x,y)'''
    hs = h * func(x,y)
    x = x + h
    y = y + hs
    return x,y

def euler_table(x0,y0,n,h,func):
    '''compute up to n euler steps with step size h for  y' = f(x,y) starting from (x0,y0)
    return the results in an hv.Table
    '''
    xl = [x0]; yl=[y0]
    for i in range(n):
        x0,y0 = euler_step(x0,y0,h,func)
        xl.append(x0);yl.append(y0)
        if np.abs(x0) > 5. or np.abs(y0) > 5. : break   # we ran off the grid
    return hv.Table(pd.DataFrame(dict(x=xl,y=yl)), kdims=['x'],vdims=['y'])

def euler_curve(x0,y0,n,h,func):
    '''compute up to n euler steps with step size h for  y' = f(x,y) starting from (x0,y0)
    return the results in an hv.Curve
    '''
    return euler_table(x0,y0,n,h,func).to.curve( label= 'h=%6.3f'%h)

def append_euler_plots( l, start, func, n=10000, h=[.5,.2,.01,.0011] ):
    for hi in h: l.append( euler_curve(*start, n, hi, func) )
    return l

# Test the computations and the resulting displays:
# --------------------------------------------------------------
funcs = dict(
    sin    = [ lambda x,y: np.sin(x*y),          "y' = sin(x y)"      ],
    sinsin = [ lambda x,y: np.sin(x)*np.sin(y),  "y' = sin(x) sin(y)" ],
    cos    = [ lambda x,y: np.cos(x),            "y' = cos(x)"        ],
    exp    = [ lambda x,y: np.exp(-x/( x**2 + 1)), "y' = exp(x /(x**2 + 1))"  ],
    x2sin  = [ lambda x,y: x**2*np.sin(5*y),     "y' = x**2 sin(5 y)"  ],
    xsin   = [ lambda x,y: x   *np.sin(5*y),     "y' = x sin(5 y)"     ],
    xtan   = [ lambda x,y: x   *np.tan(y),       "y' = x tan(y)"       ],
    xcosh  = [ lambda x,y: x/np.cosh(y),         "y' = x / cosh(y)"    ],
)

func_sel = 'sin'
f1       = funcs[func_sel][0]
f1_doc   = funcs[func_sel][1]
l1 = background(f1)
append_euler_plots(l1, (-5,np.pi/4.75),f1 )

func_sel = 'sinsin'
f2       = funcs[func_sel][0]
f2_doc   = funcs[func_sel][1]
l2 = background(f2)
append_euler_plots(l2, (-5, np.pi/4.75),f2 )
append_euler_plots(l2, (-5,-np.pi/4.75),f2 )


# We need to call redim in case some curve overshot the grid
pos_opts = dict(legend_position='right', toolbar='above')
hv.Overlay(l1).redim.range(x=(-5,5),y=(-5,5)).opts(plot=pos_opts).relabel(f1_doc) +hv.Overlay(l2).redim.range(x=(-5,5),y=(-5,5)).opts(plot=pos_opts).relabel(f2_doc)

from holoviews.streams import PointerXY, SingleTap, DoubleTap

func_sel = 'xtan'
f1       = funcs[func_sel][0]
f1_doc   = funcs[func_sel][1]
l1         = background(f1)
pt         = PointerXY(x=-5,y=3)
tap        = SingleTap(transient=True)
double_tap = DoubleTap(rename={'x': 'x2', 'y': 'y2'}, transient=True)

pos_opts = dict(legend_position='right', toolbar='above')

# prepopulate the list with 8 Curves to allow their display (WHY IS THIS NECESSARY????)
# removing the next two lines stops react_to_tap updating the display,
# even though it updates the list

append_euler_plots(l1, (-5, np.pi/4.75),f1 )
append_euler_plots(l1, (-5,-np.pi/4.75),f1 )


def react_to_tap(x,y,x2,y2):
    if not None in [x2,y2]:
        del l1[4:]             # temorary hack: want a reset button later
    elif not None in [x,y]:
        append_euler_plots( l1, (x,y), f1 )
        #l1.append( hv.Points([(x,y,5)]).opts(style(color='red')))
    return hv.Overlay(l1).redim.range(x=(-5,5),y=(-5,5)).opts(plot=pos_opts).relabel( f1_doc )

hv.DynamicMap( react_to_tap, streams=[ tap, double_tap ]).opts(plot=pos_opts).relabel( f1_doc )

# Check this:
print('Number of curves prior to tap: ', len(l1)-4 )
#hv.Overlay(l1).redim.range(x=(-5,5),y=(-5,5)).opts(plot=pos_opts).relabel(func_doc)
hh=react_to_tap(-5,2, None, None)
print('Number of curves after tap: ', len(l1)-4 )
hh



