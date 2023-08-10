batch = False
if batch== True :
    a=pd.read_csv('batch.csv')
    for i in range(a.count()):
        a=input('Chromosome : ')
        b=input('Start : ')
        c=input('End : ')
        d=input('Cell Line : ')
       
a=input('Chromosome : ')
b=input('Start : ')
c=input('End : ')
d=input('Cell Line : ')

import requests
url = 'https://jqhp2vtj5k.execute-api.ap-southeast-2.amazonaws.com/prod/submit'
data = '''{
  "genome": {
                "assembly":"GRCh37"
  },
  "chromosome": "1",
  "proximal": 12776100,
  "distal": 12776420,
  "fasta": "GTCTGCTGGCTCTGAACCATGTCCTAAATGGTTTCCACTGCGCACAGCTTCCTCTCAGCCCGCTCTGAGCTGGAAGCAGCATGTGGGACCTGGCCCTGATCTTCCTCGCAGCAGCCTGAGTGTTCTCACTAGGGGTCACTCTGTGGGTCATTTGCAGCCATTTTTTCACTGTGCACATCCCTGCAGCGGTTGGCCACCCTGTGAAACTGAGAGTCCTCCATTGCATCTTCCAGCTGCTGTTGACATGGGTGAGTTTTGTGCTTTATGTGTCCCCTCCAGCTGACCATTAAGGAAGGCGGCAGGAAAAATCACACACCGGAA",
  "cellLine": {
    "eid": "E001"
  }
}'''
response = requests.post(url, data=data)

response.content

job = response.json()

job

id=a['data']['JobID']

url = 'https://jqhp2vtj5k.execute-api.ap-southeast-2.amazonaws.com/prod/results/' + id +'/targets'

response = requests.get(url)

x=response.json()
x

import pandas as pd
df=pd.DataFrame(x)

df['data']

from pandas.io.json import json_normalize
result = pd.DataFrame(json_normalize(response.json(), 'data'))

result.drop(['wucrispr','histones','sgrnascorer','position','transcription'], axis = 1, inplace = True)

result.columns = ['activity','score', 'location','1','2','3','0','sequence','strand']

df = result[['location','strand','sequence','activity','1','2','3','0','score']]

df.head(10)

import plotly.plotly as py
import plotly.graph_objs as go
import plotly

# Create random data with numpy
import numpy as np
plotly.tools.set_credentials_file(username='stv.nouri', api_key='0P3ksqgxyPBxoTSmwEPl')
random_x = df['location'].loc[(df['strand'] == '+') & ((df['location']%2)!= 0)]
random_x1 = df['location'].loc[(df['strand'] == '-') & ((df['location']%2)!= 0)]
random_x2 = df['location'].loc[(df['strand'] == '+') & ((df['location']%2)== 0)]
random_x3 = df['location'].loc[(df['strand'] == '-') & ((df['location']%2)== 0)]

random_y0 = df['score'].loc[(df['strand'] == '+') & ((df['location']%2)!= 0)] + 5
random_y1 = df['score'].loc[(df['strand'] == '-') & ((df['location']%2)!= 0)] +10
random_y2 = df['score'].loc[(df['strand'] == '+') & ((df['location']%2)== 0)] +15
random_y3 = df['score'].loc[(df['strand'] == '-') & ((df['location']%2)== 0)] 

# Create traces
trace0 = go.Scatter(
    x = random_x,
    y = random_y0,
    mode = 'markers',
    name = '+ Low',
    marker = dict(
        size = 10,
        symbol='triangle-right',
        color='black'
        )
)
trace1 = go.Scatter(
    x = random_x1,
    y = random_y1,
    mode = 'markers',
    name = '- Low',
     marker = dict(
        size = 10,
        symbol='triangle-left',
        color='black'
        )
)

trace2 = go.Scatter(
    x = random_x2,
    y = random_y2,
    mode = 'markers',
    name = '+ high',
     marker = dict(
        size = 10,
        symbol='triangle-right', 
        color='green'
        )
)
trace3 = go.Scatter(
    x = random_x3,
    y = random_y3,
    mode = 'markers',
    name = '- High',
     marker = dict(
        size = 10,
        symbol='triangle-left',
         color='green'
        )
)

data = [trace0, trace1, trace2,trace3]
py.iplot(data, filename='scatter-mode')

result #the Final table reprentation of work

result.sort_values(by='score', ascending=False).style.bar(subset=['score'], align='mid',color='lightgreen',width=100)



