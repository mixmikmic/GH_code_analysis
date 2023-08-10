import pandas as pd
import numpy as np

# Read newest CIOCS data
ciocs = pd.read_csv('https://docs.google.com/spreadsheets/d/1pE5kECEnegc6qbkntT-WnuJSP5UhAEHvaSdBy4KrwFo/export?format=csv',
            header=5,
            na_values=['NA',''],
            parse_dates=[10,11],
            names=['City','Week','Tests','Reagent','Condoms','Age','Men','Origin','Country','Skin','StartDate','EndDate','Note'])
ciocs.head()

ciocs.dtypes #Checking how pandas reads the different columns

ciocs.describe()

ciocs['Tests'].describe()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ggplot import *
get_ipython().magic('matplotlib inline')

ggplot(aes(x='City'), data=ciocs) + geom_bar(stat = 'identity', y='Tests')

ggplot(aes(x='StartDate', y='Tests', colour='City'), data=ciocs) +    geom_line() +    stat_smooth() +    scale_color_brewer(type='qual', palette='Paired') # Qual:'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 
                                                      # 'Set1', 'Set2', 'Set3'

ggplot(aes(x='Tests', y='Reagent'), data=ciocs) +    geom_point() +    stat_smooth(method='lm',colour='#00aeef',fill='#00aeef') +    theme_seaborn()

ggplot(aes(x='Tests', y='Reagent'), data=ciocs) +    geom_point() +    stat_smooth(span=0.6,colour='#00aeef',fill='#00aeef') +    theme_bw()

ggplot(aes(x='Tests', y='Reagent'), data=ciocs) +    geom_point() +    stat_smooth(method='lm',colour='#00aeef',fill='#00aeef') # Using linear smooth

from bokeh.plotting import *
get_ipython().magic('matplotlib inline')
output_notebook()

figure(
    title='Number of Tests',        # Plot title
    title_text_font='Courier New',  # Title font
    title_text_color='#5d6263',     # Title font colour
    plot_width=1000,                # Plot width
    plot_height=600,                # Plot height
    background_fill='#f6f6f6',      # Background colour
    border_fill='#f6f6f6',          # Border background
    v_symmetry=True,                
    h_symmetry=True,                
    outline_line_color='#f6f6f6',   # Plot area border colour
    x_axis_type = 'datetime',       # For timeseries only
    #tools='pan,box_zoom,previewsave,resize,select,reset' # Available: pan,wheel_zoom,box_zoom,previewsave,resize,select,reset
)
hold()
line(
    ciocs['StartDate'],             # x
    ciocs['Tests'],                 # y
    color='#00aeef',                # Line colour
    line_width=3,                   # Line width in px
    legend='Tests',                 # Legend label
)
legend().label_text_font='Courier New'
legend().label_text_color='#5d6263'
legend().border_line_color='#f6f6f6'
yaxis().axis_line_color = None
xaxis().axis_line_color = '#d4d4d4'
axis().major_label_text_font="Courier New"
axis().major_label_text_font_size="12pt"
xgrid().grid_line_color = None
ygrid().grid_line_color = "#d4d4d4"
ygrid().grid_line_width = 1
show()

ggplot(aes(x='Week', y='Reagent', colour='City'), data=ciocs) + geom_line() + stat_smooth()

ggplot(aes(x='Week', y='Reagent', colour='City'), data=ciocs) +    stat_smooth(se=False, size=8) +     geom_point(alpha=0.8) # Drop the confidence interval and make the line itself broad and make the dots transparent

ggplot(aes(x='Week', y='Reagent'), data=ciocs) +    geom_jitter() +    stat_smooth(se=False, size=8, colour='#00aeef') +    facet_wrap('City')

figure()
hold()
line(ciocs['StartDate'], ciocs['Reagent'], color='#00447c', legend='Reagent Tests', x_axis_type = 'datetime', 
     tools='pan,box_zoom,previewsave,resize,select,reset')
show()

ggplot(aes(x='Week', y='Condoms', colour='City'), data=ciocs) +    geom_line() +    stat_smooth()

figure()
hold()
line(ciocs['StartDate'], ciocs['Condoms'], color='#cf5c42', line_width=3, legend='Condoms Distributed', x_axis_type = 'datetime', 
     tools='pan,box_zoom,previewsave,resize,select,reset')
show()

from IPython.core.display import HTML
styles = open("../css/custom.css", "r").read()
HTML(styles)



