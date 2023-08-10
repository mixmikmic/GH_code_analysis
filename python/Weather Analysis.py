import numpy as np
import pandas as pd
import re
import string
import warnings

from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, ColumnDataSource, FuncTickFormatter
from bokeh.palettes import Category10, gray
from bokeh.plotting import Figure, show, figure
from datetime import datetime, timedelta
from sklearn import tree
from scipy.stats import describe

# load data
alert_data = pd.read_csv('../data/my_mta_data_for_analysis.csv', encoding='latin1', parse_dates =['time'])
weather_data = pd.read_csv('../data/central_park_weather.csv', encoding='latin1', parse_dates = ['DATE'])

# load general variables
subway_lines = 'ABCDEFGJLMNQRSWZ1234567'
subway_colors = {
    'A': '#0039A6',
    'B': '#FF6319',
    'C': '#0039A6',
    'D': '#FF6319',
    'E': '#0039A6',
    'F': '#FF6319',
    'G': '#6CBE45',
    'J': '#996633',
    'L': '#A7A9AC',
    'M': '#FF6319',
    'N': '#FCCC0A',
    'Q': '#FCCC0A',
    'R': '#FCCC0A',
    'S': '#808183',
    'Z': '#996633',
    'W': '#FCCC0A',
    '1': '#EE352E',
    '2': '#EE352E',
    '3': '#EE352E',
    '4': '#00933C',
    '5': '#00933C',
    '6': '#00933C',
    '7': '#B933AD'
}

# load visualization tools & settings
output_notebook()
warnings.filterwarnings('ignore')

# filtering, but including end disruption alerts
alert_data_ex = alert_data[
    (~alert_data['estimated'].isin(['planned service change','non service','retraction'])) 
    & (~alert_data['update'])
]

# Find subway lines mentioned
regex = re.compile('[%s]' % re.escape(string.punctuation))
def extract_subway_lines(txt, subway_lines=subway_lines):
    cleaned_txt = regex.sub(' ', txt)
    split_txt = cleaned_txt.split()
    if 'All' in split_txt:
        return [s for s in subway_lines]
    else:
        return [s for s in split_txt if s in subway_lines]
    
alert_data['lines'] = alert_data['title'].apply(
    lambda x: extract_subway_lines(x)
)

# only get daily weather and fields 
daily_weather = weather_data[weather_data['REPORTTPYE'] == 'SOD']
daily_weather = daily_weather[
    ['DATE','DAILYMaximumDryBulbTemp','DAILYMinimumDryBulbTemp','DAILYPeakWindSpeed','DAILYSnowfall', 'DAILYPrecip']
]
daily_weather['day'] = daily_weather['DATE'].apply(lambda x: x.date())

daily_weather.head()
daily_weather['DAILYPrecip'] = daily_weather['DAILYPrecip'].str.replace('T','0.01')
daily_weather['DAILYPrecip'] = pd.to_numeric(daily_weather['DAILYPrecip'])

daily_weather['DAILYSnowfall'] = daily_weather['DAILYSnowfall'].str.replace('T','0.01')
daily_weather['DAILYSnowfall'] = pd.to_numeric(daily_weather['DAILYSnowfall'])

daily_alerts = pd.DataFrame(
    alert_data_ex['time'].apply(
        lambda x: x.date()
    ).value_counts()).reset_index()

daily_alerts_and_weather = daily_alerts.merge(
    daily_weather,
    left_on='index',
    right_on='day'
)

daily_alerts_and_weather.head()

daily_alerts_and_weather['log_alerts'] =  np.log(daily_alerts_and_weather['time'])
daily_alerts_and_weather['log_precip'] =  np.log(daily_alerts_and_weather['DAILYPrecip'])
precip_days = daily_alerts_and_weather[daily_alerts_and_weather['DAILYPrecip'] > .01]


plot = figure(
    plot_width=800, 
    plot_height=500, 
    title='Delays vs Precipitation', 
    y_minor_ticks=2)
plot.scatter(
    x=precip_days['DAILYPrecip'],
    y=precip_days['time'],
    alpha= .3
)

print("R2: " + str(precip_days[['DAILYPrecip','time']].corr()['time'][0]))


# run regression for plot line
reg = np.polyfit(precip_days['DAILYPrecip'], precip_days['time'], 1)
r_x, r_y = zip(*((0.1 * i, 0.1 * i * reg[0] + reg[1]) for i in range(30)))
plot.line(x= r_x, y=r_y, line_width=3, line_color='red')
print("regression outputs: " + ', '.join([str(r) for r in reg]))
show(plot)

# snowfall only
snow_days = daily_alerts_and_weather[daily_alerts_and_weather['DAILYSnowfall'] > .01]


plot = figure(
    plot_width=800, 
    plot_height=500, 
    title='Delays vs Snowfall', 
    y_minor_ticks=2)
plot.scatter(
    x=snow_days['DAILYSnowfall'],
    y=snow_days['time'],
    alpha= .3
)

print("R2: " + str(snow_days[['DAILYSnowfall','time']].corr()['time'][0]))


# run regression for plot line
reg = np.polyfit(snow_days['DAILYSnowfall'], snow_days['time'], 1)
r_x, r_y = zip(*((0.1 * i, 0.1 * i * reg[0] + reg[1]) for i in range(250)))
plot.line(x= r_x, y=r_y, line_width=3, line_color='red')
print("regression outputs: " + ', '.join([str(r) for r in reg]))
show(plot)

# temperature
cold_dry_weather = daily_alerts_and_weather[
    (daily_alerts_and_weather['DAILYMaximumDryBulbTemp'] < 70) &
    (daily_alerts_and_weather['DAILYPrecip'] < .02)
]
plot = figure(
    plot_width=800, 
    plot_height=500, 
    title='Delays in Cold Weather', 
    y_minor_ticks=2)
plot.scatter(
    x=cold_dry_weather['DAILYMinimumDryBulbTemp'],
    y=cold_dry_weather['time'],
    alpha= .3
)

print("R2: " + str(cold_dry_weather[['DAILYMinimumDryBulbTemp','time']].corr()['time'][0]))


# run regression for plot line
reg = np.polyfit(cold_dry_weather['DAILYMinimumDryBulbTemp'], cold_dry_weather['time'], 1)
r_x, r_y = zip(*((i, i * reg[0] + reg[1]) for i in range(70)))
plot.line(x= r_x, y=r_y, line_width=3, line_color='red')
print("regression outputs: " + ', '.join([str(r) for r in reg]))
show(plot)

# temperature
hot_dry_weather = daily_alerts_and_weather[
    (daily_alerts_and_weather['DAILYMaximumDryBulbTemp'] > 90) &
    (daily_alerts_and_weather['DAILYPrecip'] < .02)
]
plot = figure(
    plot_width=800, 
    plot_height=500, 
    title='Delays in Cold Weather', 
    y_minor_ticks=2)
plot.scatter(
    x=hot_dry_weather['DAILYMaximumDryBulbTemp'],
    y=hot_dry_weather['time'],
    alpha= .3
)

print("R2: " + str(hot_dry_weather[['DAILYMaximumDryBulbTemp','time']].corr()['time'][0]))


# run regression for plot line
reg = np.polyfit(hot_dry_weather['DAILYMaximumDryBulbTemp'], hot_dry_weather['time'], 1)
r_x, r_y = zip(*((i, i * reg[0] + reg[1]) for i in range(90,100)))
plot.line(x= r_x, y=r_y, line_width=3, line_color='red')
print("regression outputs: " + ', '.join([str(r) for r in reg]))
show(plot)



