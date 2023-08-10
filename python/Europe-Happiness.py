from IPython.display import Image
Image(url='https://afremov.com/image.php?type=P&id=18696')

import numpy as np
import pandas as pd
import matplotlib.cm as cm

df=pd.read_excel('Europe-Happiness.xlsx')
df

N=len(df)

score=df['Score'].values
country=list(df.Country)
country[13]='U Kingdom'
country[14]='Czech Rep'
country[38]='Bosn Herzeg'
world_rk=list(df['World-Rank'])

import plotly.plotly as py
from plotly.graph_objs import *

def map_z2color(val, colormap, vmin, vmax):
    #map the normalized value val to a corresponding color in the mpl colormap
    
    if vmin>=vmax:
        raise ValueError('incorrect relation between vmin and vmax')
    t=(val-vmin)/float((vmax-vmin))#normalize val
    C=map(np.uint8, np.array(colormap(t)[:3])*255)
    #convert to a Plotly color code:
    return 'rgb'+str((C[0], C[1], C[2]))
       
    

def set_layout(title, plot_size):# set plot layout
    axis=dict(showline=False, 
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title='' 
             )

    return Layout(title=title,
                  font=Font(size=12), 
                  xaxis=XAxis(axis, range=[-3.5, 4]),
                  yaxis=YAxis(axis, range=[-4.5, 3.5]),
                  showlegend=False,
                  width=plot_size,
                  height=plot_size,
                  hovermode='closest',
                  annotations=[]        
                 )  

def set_annotation(x, y, anno_text,  angle, fontsize=11): # annotations
    return Annotation(x= x,  
                      y= y,       
                      text= anno_text,      
                      textangle=angle, # angle in degrees 
                      font= Font(size=fontsize),  
                      showarrow=False     
                     ) 

bar_height=[score[k]/4.75 for k in range(N)]

theta=[np.pi/2+(2*k+1)*np.pi/72  for k in range(N)]# angular position of base bar centers

xi=[np.cos(theta[k]) for k in range(N)]# starting bar position
yi=[np.sin(theta[k]) for k in range(N)]


xf=[(bar_height[k]+1)*np.cos(theta[k]) for k in range(N)]#end bar position
yf=[(bar_height[k]+1)*np.sin(theta[k]) for k in range(N)]

xm=[(xi[k]+xf[k])*0.5   for k in range(N)]#mid bar position for inserting hovering text
ym=[(yi[k]+yf[k])*0.5   for k in range(N)]

xpos_t=[(bar_height[k]+1.32)*np.cos(theta[k]) for k in range(N)]#text position
ypos_t=[(bar_height[k]+1.32)*np.sin(theta[k]) for k in range(N)]

import matplotlib.cm as cm
cmap=cm.viridis

vmin=score[-1]
vmax=score[0]
bar_colors=[map_z2color(score[k], cmap, score[-1], score[0]) for k in range(N)]

text=[country[k]+'<br>Score: '+'{:0.3f}'.format(score[k])+'<br>World rank: ' 
      +'{:d}'.format(world_rk[k])  for  k in range(len(score))]

trace=Scatter(x=xm,
              y=ym,
              name='',
              mode='markers' ,
              marker=dict(size=0.05, color=bar_colors[15]),
              text=text,
              hoverinfo='text')
tracet=Scatter(x=xf,
               y=yf,
               name='',
               mode='markers' ,
               marker=dict(size=0.05, color=bar_colors[15]), 
               text=text,
               hoverinfo='text')

traceB=[Scatter(x=[xi[k], xf[k], None], # Circular bars are drawn as lines of width 9
                y=[yi[k], yf[k], None], 
                mode='lines', 
                line=dict(color=bar_colors[k], width=9.6),
                hoverinfo='none')  
        for k in range(N)]

title="Europe Happiness Score and Global Ranking"+"<br>Data: World Happiness Report, 2016"+"<a href='http://worldhappiness.report/wp-content/uploads/sites/2/2016/03/HR-V1_web.pdf'> [1]</a>"
layout=set_layout('', 1000)

annotext_angle=[(-(180*np.arctan(yi[k]/xi[k])/np.pi)) for k in range(N)]# angles in degrees, computed following 
                                                                        #Plotly reversed trigonometric rule

for k in range(N):
    layout['annotations']+=[set_annotation(xpos_t[k], ypos_t[k], country[k],  annotext_angle[k],  fontsize=10)]
layout['annotations']+=[set_annotation(0.5, 0.1, title, 0 ) ]    

fig=Figure(data=Data(traceB+[trace, tracet]), layout=layout)
py.sign_in('empet', 'api_key')
py.iplot(fig, filename='Europe-Happiness')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()

