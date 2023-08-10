# Data Manipulation
import psycopg2
import os
import pandas as pd
import numpy as np
import datetime, time

# Exploratory Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh
import graphlab as gl
gl.canvas.set_target('ipynb')
sns.set_style('white')

# Interactive Graphics

# To communicate with Plotly's server, sign in with credentials file
import plotly.plotly as py

# Useful Python/Plotly tools
import plotly.tools as tls

# Graph objects to piece together plots
from plotly.graph_objs import *

get_ipython().magic('matplotlib inline')

# Setup connection to Pecan Street Dataport
try:
    conn = psycopg2.connect("dbname='postgres' user='2c53Epq3kSyQ' host='dataport.pecanstreet.org' port='5434' password=''")
except:
    "Error: Check there aren't any open connections in notebook or pgAdmin"

conn.close()

# psql -h dataport.pecanstreet.org -p 5434 -U 2c53Epq3kSyQ -W '' postgres

cursor = conn.cursor()

cursor.execute("SELECT dataid, localhour, SUM(use) FROM university.electricity_egauge_hours GROUP BY dataid, localhour")
for row in cursor:
    print row
    if row == None:
        break

electricity_df = pd.read_sql("SELECT localhour, SUM(use) AS usage, SUM(air1) AS cooling, SUM(furnace1) AS heating,                              SUM(car1) AS electric_vehicle                              FROM university.electricity_egauge_hours                              WHERE dataid = 114 AND use > 0                               AND localhour BETWEEN '2013-10-16 00:00:00'::timestamp AND                              '2016-02-26 08:00:00'::timestamp                              GROUP BY dataid, localhour                              ORDER BY localhour", conn)

electricity_df['localhour'] = electricity_df.localhour.apply(pd.to_datetime)

electricity_df.set_index('localhour', inplace=True)

electricity_df.fillna(value=0.0, inplace=True)

# Min: 2013-10-16 00:00:00
# Max: 2016-02-26 08:00:00
# Count: 20,721
electricity_df.tail()

electricity_df_nocar = electricity_df[['usage', 'cooling', 'heating']]

electricity_df_car = electricity_df[['usage','electric_vehicle']]

electricity_df_nocar.plot(figsize=(18,9), title="Pecan Street Household 114 Hourly Energy Consumption")
sns.despine();

electricity_df_car.plot(figsize=(18,9), title="Pecan Street Household 114 Hourly Energy Consumption")
sns.despine();

# Geohash: 30.292432 -97.699662 Austin, TX

from bokeh.io import output_file, output_notebook
from bokeh.plotting import show
from bokeh.models import GMapPlot, GMapOptions, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool

map_options = GMapOptions(lat=30.292432, lng=-97.699662, map_type="roadmap", zoom=11)

plot = GMapPlot(x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options, title="Austin, TX")

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

output_notebook()
# output_file("gmap_plot.html")

show(plot);

weather_df = pd.read_sql("SELECT localhour, temperature                              FROM university.weather                              WHERE localhour BETWEEN '2013-10-16 00:00:00'::timestamp AND                              '2016-02-26 08:00:00'::timestamp                              AND latitude = 30.292432                              ORDER BY localhour", conn)

weather_df['localhour'] = weather_df.localhour.apply(pd.to_datetime)

weather_df.set_index('localhour', inplace=True)

weather_df.fillna(value=0.0, inplace=True)

# Count: 20,673
weather_df.count()

print "Austin had a temperate range of {} degrees between 2013 and 2016".format(weather_df.temperature.max() - weather_df.temperature.min())
weather_df.plot(figsize=(16,8), title='Austin, TX Hourly Weather')
plt.ylabel('Temperature (F)')
plt.axhline(weather_df.temperature.max(), c='r')
plt.axhline(weather_df.temperature.min(), c='blue')
sns.despine();

tls.set_credentials_file(stream_ids=[
        "bcjrhlt0lz",
        "pkkb5aq85l",
        "c5ygaf48l0",
        "l3nh9ls79j"
    ])

stream_ids = tls.get_credentials_file()['stream_ids']
stream_ids

# help(Stream)

electricity = Stream(
    token="c5ygaf48l0",  # (!) link stream id to 'token' key
    maxpoints=500      # (!) keep a max of 80 pts on screen
)
weather = Stream(
    token="l3nh9ls79j",  # (!) link stream id to 'token' key
    maxpoints=500      # (!) keep a max of 80 pts on screen
)

trace1 = Scatter(x=[], y=[], mode='lines+markers', stream=electricity, name='electricity usage')
trace2 = Scatter(x=[], y=[], mode='lines+markers', stream=weather, name='weather')
data = Data([trace1,trace2])

# Add title to layout object
layout = Layout(title='Pecan Street Sensor Data')

# Make a figure object
fig = Figure(data=data, layout=layout)

# (@) Send fig to Plotly, initialize streaming plot, open new tab
py.iplot(fig, filename='Pecan Street Streaming Electricity Usage')

# (@) Make instance of the Stream link object, 
#     with same stream id as Stream id object
s1 = py.Stream("c5ygaf48l0")
s2 = py.Stream("l3nh9ls79j")

# (@) Open the stream
s1.open()
s2.open()

i = 0    # a counter
k = 5    # some shape parameter
k2 = 10
N = 200  # number of points to be plotted

# Delay start of stream by 5 sec (time to switch tabs)


while i<N:
    i += 1   # add to counter

    # Current time on x-axis, random numbers on y-axis
    x = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    y = (np.cos(k*i/50.)*np.cos(i/50.)+np.random.randn(1))[0]
    
    x2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    y2 = (np.cos(k2*i/50.)*np.cos(i/50.)+np.random.randn(1))[0]
    
    # (-) Both x and y are numbers (i.e. not lists nor arrays)

    # (@) write to Plotly stream!
    s1.write(dict(x=x, y=y))
    s2.write(dict(x=x2, y=y2))

    # (!) Write numbers to stream to append current data on plot,
    #     write lists to overwrite existing data on plot (more in 7.2).

    time.sleep(0.08)  # (!) plot a point every 80 ms, for smoother plotting

# (@) Close the stream when done plotting
s1.close()
s2.close()

datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

# tls.embed('streaming-demos','12')

class PlotlyStream(object):
    
    def __init__(self, electricity_data, weather_data, stream_tokens):
        self.electricity = electricity_data
        self.weather = weather_data
        self.tokens = stream_tokens
    
    def create_plot(self, chart_title, x_title, y_title, maxpoints):
        """Method to generate Plotly plot in notebook for rendering streaming data"""
        
        e_stream = Stream(token= self.tokens[0], maxpoints= maxpoints)
        trace1 = Scatter(x=[], y=[], mode='lines+markers', stream = e_stream, name='Usage')
        
#         w_stream = Stream(token= self.tokens[1], maxpoints= maxpoints)
#         trace2 = Scatter(x=[], y=[], mode='lines+markers', stream = w_stream, yaxis='y2', name='Temp')
        
#         data = Data([trace1, trace2])
        data = Data([trace1])
        
        # Initialize layout object
        layout = Layout(title= chart_title, 
                        showlegend=True,
                        xaxis= dict(title= x_title,
                                    autorange=True,
                                    range= [self.electricity.index.min(),self.electricity.index.max()],
                                    ticks='outside',
                                    type='date'
                                  ),
                        yaxis= dict(title = y_title,
                                    autorange=True,
                                    range=[self.electricity.min(),self.electricity.index.max()],
                                  ),
#                         yaxis2 = dict(title = y2_title,
#                                       range=[self.weather.min(), self.weather.max()],
#                                       overlaying='y',
#                                       side='right'
#                                      ),
                        hovermode='closest'
                       )
        
        # Create figure object
        fig = Figure(data=data, layout=layout)
        
        # (@) Send fig to Plotly, initialize streaming plot, open new tab
        return py.iplot(fig, filename='Pecan Street Streaming Electricity Usage')
    
    def plot_stream(self, plot_freq=0.2, start_delay=0.1):
        """Method to write data to Plotly servers to render on graph"""
        
        s1 = py.Stream(self.tokens[0])
#         s2 = py.Stream(self.tokens[1])
        
        s1.open()
#         s2.open()
        
        counter = 0
        N = 1000
        
        # Create small delay before plotting begins
#         time.sleep(start_delay)
        
        electricity = self.electricity.iterrows()
#         weather = self.weather.iterrows()
        
        while counter < N:
            counter += 1
            
            timestamp1, usage = electricity.next()
#             timestamp2, temperature = weather.next()
            
            # .strftime('%Y-%m-%d %H.%f')
            
#             times = []
#             usages = []
#             temperatures = []
            
            x1 = timestamp1.strftime('%Y-%m-%d %H')
            y1 = usage
            
#             x2 = timestamp2.strftime('%Y-%m-%d %H:%M:%S.%f')
#             y2 = temperature
            
            s1.write(dict(x=x1, y=y1))
#             time.sleep(plot_freq)
#             s2.write(dict(x=x2, y=y2))
            time.sleep(plot_freq)
        
        s1.close()
#         s2.close()

PecanStreet = PlotlyStream(electricity_df_usage, weather_df, ["pkkb5aq85l","c5ygaf48l0"])

PecanStreet.create_plot("Pecan Street Household 114 Electricity Usage", 
                        "Time (Hours)", 
                        "KwH (Kilowats per Hour)",
                        100)

PecanStreet.plot_stream()

electricity_df_usage = electricity_df[electricity_df.columns[:2]]

electricity_df_usage.values

electricity_sf = gl.SFrame(data=electricity_df_usage)
electricity_sf.tail()

electricity_sf.show(view='Line Chart')

#### Bayesian Changepoints https://dato.com/learn/userguide/anomaly_detection/bayesian_changepoints.html

model = gl.anomaly_detection.bayesian_changepoints.create(electricity_sf, feature='usage', lag=7)

model.summary()

scores = model['scores']

scores.head()

sketch = scores['changepoint_score'].sketch_summary()
threshold = sketch.quantile(0.995)
changepoints = scores[scores['changepoint_score'] > threshold]
changepoints.print_rows(4, max_row_width=100, max_column_width=30)

electricity_df['changepoint_score'] = scores['changepoint_score']

electricity_df.changepoint_score = electricity_df.changepoint_score.astype('float64')

def plot_changepoints(electricity_df, weather_df, quantile=0.995):
    """Function to plot changepoints on time series"""
    
    # Referenced: http://matplotlib.org/examples/api/two_scales.html
    fig, ax1 = plt.subplots(figsize=(24,10))
    ax1.plot(electricity_df.index, electricity_df.usage)
    plt.plot(electricity_df.index, electricity_df.electric_vehicle)
    ax1.set_xlabel('Time (h)')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('KwH (Kilowats Per Hour)', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    changepoint_ts = electricity_df.index[electricity_df.changepoint_score > electricity_df.changepoint_score.quantile(quantile)]
    for changepoint in changepoint_ts:
        plt.axvline(changepoint, c='r')
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(weather_df.index, weather_df.temperature, 'm--')
    ax2.set_ylabel('Temperature (F)', color='m')
    for ytick in ax2.get_yticklabels():
        ytick.set_color('m')
    plt.show() 

plot_changepoints(electricity_df['2014-08-18'], weather_df['2014-08-18']);

plot_changepoints(electricity_df['2015'], weather_df['2015']);

plot_changepoints(electricity_df['2016'], weather_df['2016']);

weather_sf = gl.SFrame(data=weather_df)

weather_sf.head()

weather_ts = gl.TimeSeries(data=weather_sf, index='localhour')

weather_ts.head()

weather_sf.show(view='Line Chart')



