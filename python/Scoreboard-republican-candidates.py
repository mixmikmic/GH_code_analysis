newyorktimes_cs=[[0.0, '#8B0000'],
 [0.06666666666666667, '#9E051B'],
 [0.13333333333333333, '#B0122C'],
 [0.2, '#C0223B'],
 [0.26666666666666666, '#CF3447'],
 [0.3333333333333333, '#DB4551'],
 [0.4, '#E75758'],
 [0.4666666666666667, '#F06A5E'],
 [0.5333333333333333, '#F87D64'],
 [0.6, '#FE906A'],
 [0.6666666666666666, '#FFA474'],
 [0.7333333333333333, '#FFB880'],
 [0.8, '#FFCB91'],
 [0.8666666666666667, '#FFDEA7'],
 [0.9333333333333333, '#FFEEC1'],
 [1.0, '#FFFFE0']]
 

tab_vals14=[[1,2,3,4,5,6,6,8,9,9,9,12,13,13,13,13],
          [1,7,5,12,5,4,12,7,2,3,12,7,7,12,7,12],
          [4,7,2,1,10,5,6,7,9,12,3,14,12,11,15,16],
          [2,9,4,1,3,8,10,11,6,5,6,14,14,12,14,13],
          [1,3,4,14,8,2,13,12,7,6,9,16,5,10,12,15]]

candidates=['Bush', 'Rubio', 'Walker', 'Trump', 'Kasich', 'Cruz', 'Fiorina', 'Huckabee', 'Paul']+ ['Christie', 'Carson', 'Santorum', 'Perry', 'Jindal', 'Graham', 'Pataki']

factors=['Prediction Market', 'NationalEndorsements', 'Iowa Polls']+['New Hampshire Polls', 'Money Raised']

import plotly.plotly as py  
from plotly.graph_objs import *

data14=Data([Heatmap(z=tab_vals14,
                   y=factors,
                   x=candidates,
                   colorscale=newyorktimes_cs,
                   showscale=False
                   )])

width = 900 
height =450

anno_text="Data source:<a href='http://www.nytimes.com/interactive/2015/08/06/upshot/2016-republican-presidential-candidates-dashboard.html'> [1]</a>"

title = "A scoreboard for republican candidates as of August 14, 2015"               

layout = Layout(
    title=' ',  
    font=Font(
        family='Balto, sans-serif',
        size=12,
        color='rgb(68,68,68)'
    ),
    showlegend=False,
    xaxis=XAxis(
        title='',  
        showgrid=True,
        side='top'
    ),
    yaxis=YAxis(
        title='',
        autorange='reversed',
        showgrid=True,   
        autotick=False,  
        dtick=1          
    ),
    autosize=False, 
    height=height,   
    width=width,
    margin=Margin(
        l=135,
        r=40,
        b=85,
        t=170
    )
)
   
annotations = Annotations([
    Annotation(
            showarrow=False, 
            text=anno_text,  
            xref='paper',     
            yref='paper',     
            x=0,  
            y=-0.1,  
            xanchor='left',   
            yanchor='bottom',  
            font=Font(
            size=11 )
            )]) 

fig=Figure(data=data14, layout=layout)        
fig['layout'].update(
title=title,
annotations=annotations
)    

py.sign_in('empet', 'my_api_key')
py.iplot(fig,filename='Heatmap-republican-candidates-14')

tab_vals17=[[1,2,3,4,5,6,7,7,9,9,11,11,13,13,13,13],
  [1,7,5,12,5,4,7,12,2,12,3, 7,7,12,7,12],
   [4,7,2,1,10,5,7, 6, 9,3, 12, 14,12,11,15,16],
  [2,9,4,1,3,8,11, 10, 6,6, 5, 14,14,12,14,13],
  [1,3,4,14,8,2,12, 13, 7,9, 6,16,5,10,11,15]]

candidates17=['Bush', 'Rubio', 'Walker', 'Trump', 'Kasich', 'Cruz',  'Huckabee', 'Fiorina','Paul']+  ['Carson',  'Christie', 'Santorum', 'Perry', 'Jindal', 'Graham', 'Pataki']

fig['data'].update(Data([Heatmap(z=tab_vals17,
                   y=factors,
                   x=candidates17,
                   colorscale=newyorktimes_cs,
                   showscale=False
                   )]))

for i, row in enumerate(tab_vals17):
    for j, val in enumerate(row):
        annotations.append(
            Annotation(
                text=str(val), 
                x=candidates[j], y=factors[i],
                xref='x1', yref='y1',
                font=dict(color='white' if tab_vals17[i][j]<12 else 'rgb(150,150,150)'),
                showarrow=False))

fig['layout'].update(
title="A scoreboard for republican candidates as of August 17, 2015 <br> Annotated heatmap",
annotations=annotations
)    

py.iplot(fig,filename='Annotated heatmap-republican-candidates-17')

from IPython.core.display import HTML
def  css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()

