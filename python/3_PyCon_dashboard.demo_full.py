# Data handling
import pandas as pd

# Interactivity
from IPython.display import display, Javascript, Markdown
from ipywidgets import interactive, Select, HBox, VBox

# Plotting (Plot.ly offline mode)
import plotly
from plotly import graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=False)

# Data and parameters

CAPITALS = {'Canberra': 70351,
            'Sydney': 66062,
            'Darwin': 14015,
            'Melbourne': 86338,
            'Adelaide': 23090,
            'Perth': 9225,
            'Brisbane': 40913,
            'Hobart': 94029}
CAPITAL_NAMES = sorted(CAPITALS.keys())

SEASONS = ['Winter 2015', 'Spring 2015', 'Summer 2015-2016', 'Autumn 2016']

OBS_DF = pd.read_csv('/Users/myliu/local/notebooks/PyCon/PyCon_MinT_obs_2015-2016.csv')
FCST_DF = pd.read_csv('/Users/myliu/local/notebooks/PyCon/PyCon_MinT_fcst_2015-2016.csv')

# widgets with styling

SEASON_WIDGET = Select(options=SEASONS,
                       description='Season',
                       width='250px',
                       height='80px')

CITY_WIDGET = Select(options=CAPITAL_NAMES,
                     description='City',
                     width='250px',
                     height='142px')

WIDGET_STATE = {'city': CAPITAL_NAMES[0], 'season': SEASONS[0]}

def update_city(change):
    new_city = change['new']
    WIDGET_STATE['city'] = new_city

def update_season(change):
    new_season = change['new']
    WIDGET_STATE['season'] = new_season

CITY_WIDGET.observe(update_city, names='value')
SEASON_WIDGET.observe(update_season, names='value')

display(VBox([CITY_WIDGET, SEASON_WIDGET]))

_MIN_T = -10
_MAX_T = 40

def plot_scatter(**kwargs):
    city = WIDGET_STATE['city']
    season = WIDGET_STATE['season']
    
    obs_selection = OBS_DF.query('station_number == {} and season == "{}"'
                                 .format(CAPITALS[city], season))
    fcst_selection = FCST_DF.query('station_number == {} and season == "{}"'
                                 .format(CAPITALS[city], season))
    
    y = obs_selection['MinT']
    
    data = []
    for lead_day in fcst_selection['lead_day'].unique():
        x = fcst_selection.query('lead_day == {}'.format(lead_day))['MinT']
        
        visible = True if (lead_day == 1) else "legendonly"
        
        trace = go.Scatter(x=x,
                           y=y,
                           mode='markers',
                           name="Lead Day {}".format(lead_day),
                           visible=visible)
        data.append(trace)
    
    ideal = go.Scatter(x=[_MIN_T, _MAX_T],
                       y=[_MIN_T, _MAX_T],
                       mode='lines',
                       showlegend=False,
                       hoverinfo='none'
                       )
    data.append(ideal)
    
    layout = go.Layout(
        width=800,
        height=600,
        font={'size': 18},
        title="MinT forecasts vs observations<br>{}, {}".format(city, season),
        xaxis={'title': 'Forecasts (degrees Celsius)',
               'range': [_MIN_T, _MAX_T],
               'gridcolor': '#BBBBBB', 'gridwidth': 1},
        yaxis={'title': 'Observations (degrees Celsius)',
               'range': [_MIN_T, _MAX_T],
               'gridcolor': '#BBBBBB', 'gridwidth': 1},
    )

    figure = go.Figure(data=data, layout=layout)
    iplot(figure, show_link=False)

display(interactive(plot_scatter, __manual=True))

_MIN_ERR = 0
_MAX_ERR = 12

def plot_errs(**kwargs):
    city = WIDGET_STATE['city']
    season = WIDGET_STATE['season']
    
    fcst_selection = FCST_DF.query('station_number == {} and season == "{}"'
                                   .format(CAPITALS[city], season))
    
    data = []
    for lead_day in fcst_selection['lead_day'].unique():
        x = fcst_selection.query('lead_day == {}'.format(lead_day))['abs_err']
        
        visible = True if (lead_day == 1) else "legendonly"
        
        trace = go.Histogram(x=x,
                             name="Lead Day {}".format(lead_day),
                             histnorm="probability",
                             opacity=0.2,
                             autobinx=False,
                             xbins={'start': _MIN_ERR,
                                    'end': _MAX_ERR,
                                    'size': 0.5},
                             visible=visible)
        data.append(trace)
    
    layout = go.Layout(
        width=800,
        height=600,
        font={'size': 18},
        title="Absolute errors in MinT forecasts<br>{}, {}".format(city, season),
        xaxis={'title': 'Absolute Error (degrees Celsius)',
               'range': [_MIN_ERR, _MAX_ERR],
               'gridcolor': '#BBBBBB', 'gridwidth': 1},
        yaxis={'title': 'Frequency (PDF)',
               'gridcolor': '#BBBBBB', 'gridwidth': 1},
        barmode='overlay'
    )

    figure = go.Figure(data=data, layout=layout)
    iplot(figure, show_link=False)

display(interactive(plot_errs, __manual=True))

# some Javascript magic for auto-updating plots

def update_all(**kwargs):
    display(Javascript("$('.widget-button').click()"))
    
widget_interactive = interactive(update_all, city=CITY_WIDGET, season=SEASON_WIDGET, __manual=False)

display(Javascript("$('.widget-button').hide()"))
display(Javascript("$('.widget-button').click()"))



