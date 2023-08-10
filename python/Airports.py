import pandas as pd

# Download data and extract the columns we'll use

# full_data = pd.read_csv('http://ourairports.com/data/airports.csv')
# data = full_data[['type', 'latitude_deg', 'longitude_deg']]
# data.to_csv('../data/airports.csv', index=False)

data_url = '../data/airports.csv'
pd.read_csv(data_url).head()

from altair import Chart, X, Y, Axis, Scale

Chart(data_url).mark_circle(
    size=1,
    opacity=0.2
).encode(
    x=X('longitude_deg:Q', axis=Axis(title=' ')),
    y=Y('latitude_deg:Q', axis=Axis(title=' '),
        scale=Scale(domain=(-60, 80))),
    color='type:N',
).configure_cell(
    width=800,
    height=350
).configure_axis(
    grid=False,
    axisWidth=0,
    tickWidth=0,
    labels=False,
)

