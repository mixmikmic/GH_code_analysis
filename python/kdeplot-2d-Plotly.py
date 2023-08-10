import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

xl = pd.ExcelFile("Data/CSCEng.xls")
dfc = xl.parse("Sheet1")
dfc.columns

xl = pd.ExcelFile("Data/SystEng.xls")
dfi = xl.parse("Sheet1")
dfi.columns

cubehelix_cs=[[0.0, '#fcf9f7'],
 [0.16666666666666666, '#edcfc9'],
 [0.3333333333333333, '#daa2ac'],
 [0.5, '#bc7897'],
 [0.6666666666666666, '#925684'],
 [0.8333333333333333, '#5f3868'],
 [1.0, '#2d1e3e']]

def kde_scipy( vals1, vals2, (a,b), (c,d), N ):
    
    #vals1, vals2 are the values of two variables (columns)
    #(a,b) interval for vals1; usually larger than (np.min(vals1), np.max(vals1))
    #(c,d) -"-          vals2 
    
    x=np.linspace(a,b,N)
    y=np.linspace(c,d,N)
    X,Y=np.meshgrid(x,y)
    positions = np.vstack([Y.ravel(), X.ravel()])

    values = np.vstack([vals1, vals2])
    kernel = st.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    return [x, y, Z]

import plotly.plotly as py
from plotly.graph_objs import *   

def make_kdeplot(varX, varY, (a,b), (c,d), N, colorsc, title):
    #varX, varY are lists, 1d numpy.array(s), or dataframe columns, storing the values of two variables
   
    x, y, Z = kde_scipy(varY, varX, (a,b), (c,d), N )
    
    data = Data([
       Contour(
           z=Z, 
           x=x,
           y=y,
           colorscale=colorsc,
           #reversescale=True,
           opacity=0.9,    
           contours=Contours(
               showlines=False)      
        ),        
     ])

    layout = Layout(
        title= title,  
        font= Font(family='Georgia, serif',  color='#635F5D'),
        showlegend=False,
        autosize=False,
        width=650,
        height=650,
        xaxis=XAxis(
            range=[a,b],
            showgrid=False,
            nticks=7
        ),
        yaxis=YAxis(
            range=[c,d],
            showgrid=False,
            nticks=7
        ),
        margin=Margin(
            l=40,
            r=40,
            b=85,
            t=100,
        ),
    )
     
    return Figure( data=data, layout=layout )

N=200
a,b=(5,11)
fig=make_kdeplot(dfc['multiannual'], dfc['bachelor-th'], (a,b), (a,b), 
                 N, cubehelix_cs,'kde plot of two sets of data' )

py.sign_in('empet', 'my_api_key')
py.iplot(fig, filename='kde-2D-CSCE')

a, b=(4,12)
fig=make_kdeplot(dfi['multiannual'], dfi['bachelor-th'], (a,b), (a,b),
                 N, cubehelix_cs, 'kde plot of two sets of data')
py.iplot(fig, filename='kde-2D-SE')

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris.csv')
virginica = df.loc[df.Name == "Iris-virginica"]
a, b=(5,8.5)
c,d=(2,4)
N=100
fig=make_kdeplot(virginica.SepalLength, virginica.SepalWidth, (a,b), (c,d),
    N, cubehelix_cs, 'kde plot of joint distribution for virginica SepalLength and SepalWidth')
py.iplot(fig,  filename='virginica-sepal-length-vs-width')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()

