import pandas as pd
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly as ps
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot 
init_notebook_mode(connected=True) 

# Create a dictionary object with the parameters top be set
myData = dict(type ='choropleth',
           locations = ['AZ','NY','TX','CA'],
           locationmode = 'USA-states',
           colorscale = 'Greens',
           text=['text1','text2','text3','text4'],
           z = [1.0,2.0,4.0,5.0],
           colorbar = {'title':'Color bar Title Here'})

# Create a layout object
myLayout =dict(geo={'scope':'usa'})

myChoroMap = go.Figure(data=[myData],layout=myLayout)

iplot(myChoroMap)

plot(myChoroMap)



exportData = pd.read_csv('./Geographical Plotting/2011_US_AGRI_Exports')

exportData.head()

myExportData = dict(type ='choropleth',
           locations = exportData['code'],
           locationmode = 'USA-states',
           colorscale = 'YlOrRd_r',
           text=exportData['text'],
           z = exportData['total exports'],
            marker = dict(line=dict(color='rgb(255,255,255)',width=2)),
           colorbar = {'title':'Millions USD'})

myLayout =dict( geo=dict(scope='usa',showlakes=True,lakecolor='rgb(85,173,240)'),
                title='Exportd Per Us State')

myChoroMap = go.Figure(data=[myExportData],layout=myLayout)

iplot(myChoroMap)



electionData = pd.read_csv('./Geographical Plotting/2012_Election_Data')

electionData.head()

myData = dict(type ='choropleth',
           locations = electionData['State Abv'],
           locationmode = 'USA-states',
           colorscale = 'Greens',
           text=electionData['VEP Highest Office'],
           z = electionData['VAP Highest Office'],
           colorbar = {'title':'Color bar Title Here'})

myLayout =dict(geo={'scope':'usa'})

myChoroMap = go.Figure(data=[myData],layout=myLayout)

iplot(myChoroMap)



