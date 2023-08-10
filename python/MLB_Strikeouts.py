import pandas as pd

data = "../data/strikeouts.csv"
pd.read_csv(data).head()

from altair import *

# Create an empty LayeredChart with our data
chart = LayeredChart(data).transform_data(
    calculate=[Formula(field='so_per_game', expr='datum.so / datum.g')]
).configure_cell(
    height=300,
    width=700
)

# Add-in team-by-team points
chart += Chart().mark_circle(
    color='gray',
    opacity=0.1,
).encode(
    x=X('year:T', timeUnit='year', axis=Axis(title=' ')),
    y=Y('so_per_game', axis=Axis(title='Strikeouts Per Game')),
    detail='histcode:N',
)

chart.display()

# Add a rolling-mean as a line
chart += Chart().mark_line().encode(
    x='year:T',
    y='mean(so_per_game)',
)

# Add rolling-mean as a circle
chart += Chart().mark_circle().encode(
    x='year:T',
    y='mean(so_per_game)',
)

chart.display()

