import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode()

df = pd.read_clipboard()

df.head()

df.VALUE = df.VALUE.str.replace('%','')
df.VALUE = df.VALUE.astype(float)

df.sort_values(by='VALUE', ascending=False).head(10)

# Go here to choose other colors: http://colorbrewer2.org/  # of classes=6, sequential, rgb
scl = [[0.0, 'rgb(241,238,246)'],[0.2, 'rgb(212,185,218)'],[0.4, 'rgb(201,148,199)'],            [0.6, 'rgb(223,101,176)'],[0.8, 'rgb(221,28,119)'],[1.0, 'rgb(152,0,67)']]

df['text'] = "Defect %"

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df['STATECD'],
        z = df['VALUE'].astype(float),  # Ensure it is a numeric type
        locationmode = 'USA-states',
        text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',  # white lines
                width = 2
            )
        ),
        colorbar = dict(
            title = "Defect %"
        )
    ) ]

layout = dict(
        title = 'Hard Drive Failures',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',  # white color lakes
        ),
        width=800,
        height=550
    )
   
fig = dict( data=data, layout=layout )

iplot(fig, show_link=False )

def state_choropleth(df, title='State Choropleth'):
    """df = dataframe, title=your desired title"""
    
    # Go here to choose other colors: http://colorbrewer2.org/  # of classes=6, sequential, rgb
    scl = [[0.0, 'rgb(241,238,246)'],[0.2, 'rgb(212,185,218)'],[0.4, 'rgb(201,148,199)'],                [0.6, 'rgb(223,101,176)'],[0.8, 'rgb(221,28,119)'],[1.0, 'rgb(152,0,67)']]

    df['text'] = "Defect %"

    data = [ dict(
            type='choropleth',
            colorscale = scl,
            autocolorscale = False,
            locations = df['STATECD'],
            z = df['VALUE'].astype(float),  # Ensure it is a numeric type
            locationmode = 'USA-states',
            text = df['text'],
            marker = dict(
                line = dict (
                    color = 'rgb(255,255,255)',  # white lines
                    width = 2
                )
            ),
            colorbar = dict(
                title = "Defect %"
            )
        ) ]

    layout = dict(
            title = 'Hard Drive Failures',
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showlakes = True,
                lakecolor = 'rgb(255, 255, 255)',  # white color lakes
            ),
            width=800,
            height=550
        )
       
    fig = dict( data=data, layout=layout )

    iplot(fig, show_link=False )

df = pd.read_clipboard()
df.VALUE = df.VALUE.str.replace('%','')

state_choropleth(df, 'Hard Drive Failures')

