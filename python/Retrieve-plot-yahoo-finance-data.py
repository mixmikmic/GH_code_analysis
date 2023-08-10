from datetime import datetime
import numpy as np
import pandas as pd
from pandas.io.data import DataReader

index_symbol = ['^GSPC', '^DJI','^NDX', '^NYA', '000001.SS', '^N225','^HSI', '^STI' ]
explicit_name=['S&P 500', 'DOW JONES', 'NASDAQ', 'NYSE', 'SHANGHAI', 'NIKKEI', 'HKSE', 'SES']

d = {}
for symb in index_symbol:
    d[symb] = DataReader(symb, "yahoo", start=datetime( 2015, 8, 7 ), end=datetime( 2015, 8, 26 ))
panel = pd.Panel(d)# Panel of data
df = panel.minor_xs('Adj Close') #df is a dataframe containing adjusted closing price for each index
print df

import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *

fig = tls.make_subplots(
    rows=4,
    cols=2,
    shared_xaxes=False,
    subplot_titles=('S&P 500', 'SHANGHAI',  'DOW JONES', 'NIKKEI', 'NASDAQ','HKSE', 'NYSE', 'SES')
)

fig['layout']['annotations'][2]

def make_trace(y, sbplt):# this function creates the trace for each subplot
    # y stock index value
    # stock index name
    # sbplt - subplot
    return Scatter(
        x=df.index,  
        y=y,            
        name='',     
        
        line=Line(
        color= '#2c7fb8',
        width=0.75
    ),
         
        xaxis='x{}'.format(sbplt),                    
        yaxis='y{}'.format(sbplt)     
    )

sbplt=1
for symb in index_symbol[:4]: #make traces for left subplots (first column)
    fig['data'] +=[make_trace(df[symb], sbplt)]
    sbplt+=2              
sbplt=2

for symb in  index_symbol[4:]: #make traces for the second  column
    fig['data']+=[make_trace(df[symb], sbplt)]
    sbplt+=2              

axis_style = dict(
    zeroline=False,     
    showgrid=True, 
    gridwidth=1,
    gridcolor='#FFFFFF')  

def make_XAxis():
    xaxis = XAxis(zeroline=False,  
                  nticks=4,
                 )   
    return xaxis


def make_YAxis():
    yaxis = YAxis()  
    yaxis.update(axis_style)                     
    return yaxis

title = 'Stock market indices (left USA, right ASIA)'
fig['layout'].update(title=title,                                 
                     font= Font(size=12))  
                    

fig['layout'].update(
    showlegend=False, 
    hovermode='closest',
    autosize=False,   
    height=800,       
    width=700,       
    margin=Margin(
        t=100,    
        b=100,     
        r=25,      
        l=70       
    ),
    plot_bgcolor='#EFECEA',  
    
)

subpts=range(1,9)# list of subplots, 

for sbplt in subpts:
    fig['layout'].update({'xaxis{}'.format(sbplt): make_XAxis()})
    fig['layout'].update({'yaxis{}'.format(sbplt): make_YAxis()})

anno_text="Data source:<a href='http://finance.yahoo.com/stock-center/'> [1]</a>. Access via <a href='http://pandas.pydata.org/pandas-docs/stable/remote_data.html'> [2] </a>"

fig['layout']['annotations']+=[
    Annotation(
            showarrow=False, 
            text=anno_text,  
            xref='paper',     
            yref='paper',     
            x=0,  
            y=-0.15,  
            xanchor='left',   
            yanchor='bottom',  
            font=Font(
            size=11 )
            )
] 
for sbplt in subpts: #change the default font size for  subplots title
    fig['layout']['annotations'][sbplt-1]['font']= {'size': 12}

fig['layout']['annotations'][0]# check the update

py.sign_in("empet", "my_api_key")
py.iplot(fig, filename='stock-indices-7-26') 

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()

