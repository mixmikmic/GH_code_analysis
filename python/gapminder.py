import pandas as pd
import numpy as np
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter
from bokeh.io import output_notebook
output_notebook()

data = pd.read_csv("gapminder.csv", index_col='Year', thousands =',')
data.head()

data.loc[2010].population.head()

p = figure(height =250, x_axis_type='log',x_range=(100,100000), y_range=(0,100))   #dimensions of chart
p.diamond(x=data.loc[2010].income, y = data.loc[2010].life, color='firebrick',size=6)  #what data and how should it be represented
p.xaxis[0].formatter = NumeralTickFormatter(format="0$,")  #naming of x axis
p.yaxis[0].formatter = NumeralTickFormatter(format="0,")   #labels on y-axis
show(p)

from bokeh.palettes import Spectral6
from bokeh.models import HoverTool
from bokeh.models import LinearInterpolator, CategoricalColorMapper
from bokeh.models import ColumnDataSource



size_mapper = LinearInterpolator(     #this is to give size for each data point according to their population
    x=[data.population.min(), data.population.max()],
    y = [5,50]
)

#to see changes done every year
from ipywidgets import interact
from bokeh.io import push_notebook


source = ColumnDataSource(dict(
    x= data.loc[2010].income,  #x-axis of data
    y= data.loc[2010].life,    #y-axis of data
    country = data.loc[2010].Country,    #country
    population = data.loc[2010].population,
    region = data.loc[2010].region,
    income = data.loc[2010].income
    ))


def update (year):
    new_data = dict(
        x= data.loc[year].income,
        y=data.loc[year].life,
        country = data.loc[year].Country,
        region = data.loc[year].region,
        population = data.loc[year].population,
    )
    source.data = new_data #updating the source data with the newdata i.e data of each year for sliding tool
    p.title.text = str(year)  #updating the title 
    push_notebook() #push this into chart



#to give color to each type of data point
color_mapper = CategoricalColorMapper(
   factors = list(data.region.unique()),  #this tells the compiler to color the continents
  palette = Spectral6,)


hover = HoverTool(tooltips = [("Country","@country"),("Income","@x"), ("Life","@y")], #when u hover mouse on data points
                  show_arrow=False)


PLOT_OPTS = dict(     #the dimensions of figure is given
    height =250, 
    x_axis_type='log',
    x_range=[data.income.min(), data.income.max()],
    y_range=(15,90)
)


p = figure( #how do u want the overall dimensions of fig
    title = str('2010 income vs life expectancy'),toolbar_location='above',  #title should always be in string format
    title_location = 'above',
    tools=[hover],
    **PLOT_OPTS)


p.circle(
    x='x',y='y', #these have been wriiten before and is being called for the sake of hovering to work and is defined in update function
    size={'field':'population', 'transform': size_mapper},  #we cant use the size of data point as population as the china population is one billion and all the pixels gets filled and hence we use a mapper and give the rangee of X and y axis
    color = {'field':'region','transform':color_mapper}, #this will color all the regions defined by color_mapper
    legend='region', #a legend of which color is what continent
    source=source,  #what is the data source
    alpha=0.6) #how much of transparency of data pojint

p.xaxis[0].formatter = NumeralTickFormatter(format='0$,') #xaxis labels
p.legend.border_line_color = None  #to remove the border
p.legend.location = (0,-70)  #this is going to take legend out of the chart box
p.right.append(p.legend[0]) #this is going to place the legend to the right
show(p, notebook_handle=True) #notebook_handle will take the consideration of viewing each year that we defined in update

#to see changes done every year, creating a interact sliding bar
from ipywidgets import interact
from bokeh.io import push_notebook

def update (year):
    new_data = dict(
        x= data.loc[year].income,
        y=data.loc[year].life,
        country = data.loc[year].Country,
        region = data.loc[year].region,
        population = data.loc[year].population,
    )
    source.data = new_data
    p.title.text = str(year)
    push_notebook()

interact(update, year=(1800,2014,1))

interact(update, year=(1800,2014,1)) #create a bar from year 1800 to 2014 



