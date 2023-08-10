import pandas as pd
import matplotlib.pyplot as plt
player_data = pd.read_csv('/Users/zoeolson1/player_data.csv')
player_data.head(2)

df = pd.DataFrame(player_data, columns = ['att_hd_goal', 'att_lf_goal', 'att_rf_goal'])
df2 = df.fillna(0)
df2.head(2)

df2.columns = ['header', 'left-foot', 'right-foot']
df2.loc['Total']= df2.sum()
df2

goal_types = df2.T

goal_types.head(3)

goal_types = pd.DataFrame(goal_types['Total'])

goal_types.head(3)

colors = ["#e49300", "#0263f3", "#dbff62", "#ff004e","#a14400"]

plt.pie(goal_types['Total'], labels = goal_types.index, shadow= False, colors = colors, explode = (0, 0, 0), startangle = 90, autopct='%1.1f%%',)

plt.axis('equal')

plt.tight_layout()
plt.show()

ng = pd.DataFrame(player_data, columns = ['nationality', 'id', 'pl_goals', 'appearances', 'blocked_scoring_att'])
nat_goals = ng.fillna(0)
nat_goals.head(3)

nat_goals2 = (nat_goals.groupby(['nationality'],as_index=False).id.count())
nat_goals2.head()


nat_goals3 = (nat_goals.groupby(['nationality'],as_index=False).pl_goals.sum())
nat_goals3.head()

merged_goals = pd.merge(left=nat_goals2,right=nat_goals3, left_on='nationality', right_on='nationality')
merged_goals

final_data = merged_goals.loc[merged_goals['id'] >= 10]

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
merged_goals.plot.bar()

import plotly.graph_objs as go
import plotly.plotly as py

py.sign_in('zoe1114', 'gqr5grvyef')

trace1 = go.Bar(
    x=final_data['nationality'],
    y=final_data['id'],
    name='Players',
    marker=dict(
        color='rgb(55, 83, 109)'
    )
)
trace2 = go.Bar(
    x=final_data['nationality'],
    y=final_data['pl_goals'],
    name='Goals',
    marker=dict(
        color='rgb(26, 118, 255)'
    )
)
data = [trace1, trace2]
layout = go.Layout(
    title='# of Players vs. Goals Made',
    xaxis=dict(
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='style-bar')

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

nd = pd.read_csv('/Users/zoeolson1/player_data3.csv')
nd.head(2)
nd['nationality'] = str(nd['nationality'])


words =' '.join(nd['nationality'])
print "amount of players for analysis: ", (len(words.split(",")))

cloud = WordCloud(font_path='System/Library/Fonts/Noteworthy.ttc', stopwords=STOPWORDS,
                      background_color='white',
                      width=500, height=500).generate(words)

plt.imshow(cloud)
plt.axis("off")
plt.show()
plt.close()

import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

py.sign_in('zoe1114', 'gqr5grvyef')

nd = pd.read_csv('/Users/zoeolson1/player_data3.csv', parse_dates = True)
nd['dob'] = pd.to_datetime(nd['dob'])
nd['year2'] = nd['dob'].dt.year

layout = go.Layout(title='Year Born vs. Number of Appearances')

# Create traces
trace0 = go.Scatter(
    x = nd['year2'],
    y = nd['appearances'],
    mode = 'lines+markers',
    name = 'lines+markers'
)
#trace1 = go.Scatter(
 #   x = nd['dob'],
#    y = nd['wins'],
   # mode = 'lines+markers',
#    name = 'lines+markers'
#)
#trace2 = go.Scatter(
#    x = random_x,
#    y = random_y2,
#    mode = 'markers',
#    name = 'markers'
#)
data = [trace0]

# Plot and embed in ipython notebook!
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='line-mode')

import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd


df = pd.read_csv('/Users/zoeolson1/player_data3.csv')

dt = pd.DataFrame({'count' : df.groupby(['nationality', 'Latitude', 'Longitude'])['id'].count()}).reset_index()

dt['text'] = dt['nationality'] + '<br>Count ' + (dt['count']).astype(str)
limits = [(0,1),(2,6),(7,12),(13,17),(18,24),(25,30)]
colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)","rgb(0,255, 255)", "rgb(255,255,51)"]
cities = []
                                    
                                      
for i in range(len(limits)):
    lim = limits[i]
    df_sub = dt[lim[0]:lim[1]]
    city = dict(
        type = 'scattergeo',
        locationmode = 'Africa',
        lon = df_sub['Latitude'],
        lat = df_sub['Longitude'],
        text = df_sub['text'],
        marker = dict(
            size = df_sub['count']*20,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1]) )
    cities.append(city)
                                      
layout = dict(
        title = '2014 English Primer League Player Orgins<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope='Europe',
            projection=dict( type='Mercator' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=cities, layout=layout )
py.iplot( fig, validate=False, filename='d3-bubble-map-populations' )



