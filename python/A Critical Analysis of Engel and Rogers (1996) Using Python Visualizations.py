import pandas as pd
import numpy as np

# Import Data
us_data_url = 'http://www.ssc.wisc.edu/~cengel/Data/Border/USA.xls'
us_price_data =   pd.read_excel(us_data_url,
                na_values=np.nan).stack(dropna=False).reset_index()
us_l2m_price_data =   pd.read_excel(us_data_url,
                na_values=np.nan).shift(2).stack(dropna=False).reset_index()
# l2m stands for lagged by two months

can_data_url = 'http://www.ssc.wisc.edu/~cengel/Data/Border/CAN.xls'
can_price_data =   pd.read_excel(can_data_url,
                na_values=np.nan).stack(dropna=False).reset_index()
can_l2m_price_data =   pd.read_excel(can_data_url,
                na_values=np.nan).shift(2).stack(dropna=False).reset_index()

# Process US Data
# Create common index to merge price and lagged price series
us_price_data['join_index'] = us_price_data['level_0'] +  us_price_data['level_1']
us_l2m_price_data['join_index'] = us_l2m_price_data['level_0'] +  us_l2m_price_data['level_1']
us_price_data = us_price_data.merge(us_l2m_price_data[['join_index', 0]],
                                    how='left', on='join_index')

# Add country column
us_price_data['country'] = 'US'

# Split date into two columns
us_price_data['year'], us_price_data['month'] =  zip(*us_price_data['level_0'].map(lambda x: x.split(':')))

# Split city and good code into two columns
us_price_data['city_code'], us_price_data['good_code'] =  zip(*us_price_data['level_1'].map(lambda x: (x[:2], x[2:])))

# Process Canadian Data
# Create common index to merge price and lagged price series
can_price_data['join_index'] = can_price_data['level_0'] +  can_price_data['level_1']
can_l2m_price_data['join_index'] = can_l2m_price_data['level_0'] +  can_l2m_price_data['level_1']
can_price_data = can_price_data.merge(can_l2m_price_data[['join_index', 0]],
                                      how='left', on='join_index')

# Add country column
can_price_data['country'] = 'Canada'

# Split date into a month and a year column
can_price_data['year'], can_price_data['month'] =  zip(*can_price_data['level_0'].map(lambda x: x.split(':')))

# Split city and good code into two columns
# Explanation: Each series is labeled with a letter(s) and a number.
# The letter designates the city.  Two letters are used for U.S. cities
# (e.g., CH for Chicago), and only one letter for Canadian cities.
# The number corresponds to one of the 14 goods, listed in the same order
# we have them in the paper.  "Good 0" is the city's overall CPI, also used
# in the paper.  Thus, LA2 is "Food away from home" for Los Angeles.
# Source: https://www.ssc.wisc.edu/~cengel/Data/Border/BorderData.htm
can_price_data['city_code'], can_price_data['good_code'] =  zip(*can_price_data['level_1'].map(lambda x: (x[:1], x[1:])))

# Merging and cleaning up the dataframe
price_data = pd.concat([us_price_data, can_price_data])
price_data = price_data.drop(['level_1', 'join_index'], axis=1)

# Reformat date column
price_data['level_0'] = pd.to_datetime(price_data['level_0'].str.replace(':',
                                                                         '-'))

# Rename columns
price_data.columns = ['date', 'price', 'pricel2m', 'country', 'year', 'month',
                      'city_code', 'good_code']

# Replace negative values by np.nan
price_data.loc[price_data['price'] < 0, 'price'] = np.nan
price_data.loc[price_data['pricel2m'] < 0, 'pricel2m'] = np.nan

# Reorganize columns
price_data = price_data[['date', 'year', 'month', 'country', 'city_code',
                        'good_code', 'price', 'pricel2m']]

# Reset index
price_data = price_data.reset_index(drop=True)

# Create dictionaries containing good descriptions and city names

goods_descriptions = {"0": "City CPI",
                      "1": "Food at home",
                      "2": "Food away from home",
                      "3": "Alcoholic beverages",
                      "4": "Shelter",
                      "5": "Fuel and other utilities",
                      "6": "Household furnishings & operations",
                      "7": "Men's and boy's apparel",
                      "8": "Women's and girl's apparel",
                      "9": "Footwear",
                      "10": "Private transportation",
                      "11": "Public transportation",
                      "12": "Medical care",
                      "13": "Personal care",
                      "14": "Entertainment"}

city_names = {"CH": "Chicago",
              "LA": "Los Angeles",
              "NY": "New York City",
              "PH": "Philadelphia",
              "DA": "Dallas",
              "DT": "Detroit",
              "HS": "Houston",
              "PI": "Pittsburgh",
              "SF": "San Francisco",
              "BA": "Baltimore",
              "BO": "Boston",
              "MI": "Miami",
              "ST": "St. Louis",
              "WA": "Washington, DC",
              "Q": "Quebec",
              "M": "Montreal",
              "O": "Ottawa",
              "T": "Toronto",
              "W": "Winnipeg",
              "R": "Regina",
              "E": "Edmonton",
              "C": "Calgary",
              "V": "Vancouver"}

# Inverse mappings
inv_goods_descriptions = {v: k for k, v in goods_descriptions.items()}
inv_city_names = {v: k for k, v in city_names.items()}

price_data['good_description'] = price_data['good_code'].map(goods_descriptions)
price_data['city_name'] = price_data['city_code'].map(city_names)

price_data.good_code.unique()

price_data.info()

price_data.describe()

price_data.sample(n=15)

# Start date
price_data.date.head(1)

# End date
price_data.date.tail(1)

from bokeh.plotting import figure, show, output_notebook, gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import all_palettes

output_notebook()

TOOLS = "crosshair,pan,wheel_zoom,reset,tap,save"

colors = all_palettes['Category20'][len(goods_descriptions)]

grid = []
grid_width = 3
plot_list = []

for city_code in city_names:
    hover = HoverTool(tooltips=[
        ("index", "$index"),
        ("good type", "@good"),
        ("(x,y)", "($x, $y)"),
    ])

    p = figure(x_axis_type="datetime", tools=[TOOLS, hover], plot_width=300,
               plot_height=300)
    p.title.text = city_names[city_code]
    p.title.align = 'center'

    for good_code in goods_descriptions:
        condition = (price_data['city_code'] == city_code) &          (price_data['good_code'] == good_code)
        source = ColumnDataSource(data=dict(
            x=price_data['date'][condition],
            y=price_data['price'][condition],
            good=price_data['good_description'][condition]))

        p.scatter(x='x', y='y', color=colors[int(good_code)], source=source)

    # Append the plot to a list to create the grid
    if len(plot_list) < grid_width:
        plot_list.append(p)
    else:
        grid.append(plot_list)
        plot_list = []
        plot_list.append(p)

# Append remaining plots
if plot_list:
    grid.append(plot_list)

p = gridplot(grid)

show(p)

def mean_init_price_index(price_type, city_code, good_code):
    index = price_data[(price_data['city_code'] == city_code) &
                       (price_data['good_code'] == good_code) &
                       price_data['year'].isin(['1980', '1981'])][
                           price_type].mean()
    return index


def data_normalization(df, col_to_normalize, city_names, goods_descriptions):
    for city_code in city_names:
        for good_code in goods_descriptions:
            condition = (df['city_code'] == city_code) &              (df['good_code'] == good_code)
            df.loc[condition, col_to_normalize + 'n'] =                 df[col_to_normalize][condition] /                 mean_init_price_index(col_to_normalize, city_code, good_code)

data_normalization(price_data, 'price', city_names, goods_descriptions)
data_normalization(price_data, 'pricel2m', city_names, goods_descriptions)

def data_interpolation(df, city_names, goods_descriptions):
    for city_code in city_names:
        for good_code in goods_descriptions:
            condition = (df['city_code'] == city_code) &                  (df['good_code'] == good_code)
            df.loc[condition, 'pricen'] =                 df.loc[condition,
                       ['date', 'pricen']
                       ].set_index('date').interpolate(method='cubic').values

data_interpolation(price_data, city_names, goods_descriptions)

from ipywidgets import interact
import flexx
from bokeh.models import Legend


def city_plot_update(city):
    p_cities = figure(x_axis_type="datetime", tools=TOOLS, plot_width=800,
                      plot_height=600, toolbar_location="above",
                      title='Evolution of Prices by Cities')

    colors = all_palettes['Category20'][len(goods_descriptions)]

    lines = []
    legend_it = []

    for good_code in goods_descriptions:
        condition = (price_data['city_code'] == inv_city_names[city]) &          (price_data['good_code'] == good_code)
        temp_line = p_cities.line(x=price_data['date'][condition],
                                  y=price_data['pricen'][condition],
                                  color=colors[int(good_code)])
        lines.append(temp_line)
        legend_it.append((goods_descriptions[good_code], [temp_line]))

    legend = Legend(items=legend_it, location=(0, 100))
    legend.click_policy = "hide"

    p_cities.add_layout(legend, 'right')
    p_cities.title.text_font_size = '12pt'
    p_cities.yaxis.axis_label = 'Normalized Price Index'
    p_cities.xaxis.axis_label = 'Year'

    show(p_cities)

interact(city_plot_update, city=city_names.values())

from bokeh.palettes import magma


def good_plot_update(good):
    p_goods = figure(x_axis_type="datetime", tools=TOOLS, plot_width=800,
                     plot_height=600, toolbar_location="above",
                     title='Evolution of Prices by Cities')

    lines = []
    legend_it = []
    colors = magma(len(city_names))

    for (i, city) in enumerate(city_names):
        condition = (price_data['city_code'] == city) &          (price_data['good_code'] == inv_goods_descriptions[good])
        temp_line = p_goods.line(x=price_data['date'][condition],
                                 y=price_data['pricen'][condition],
                                 color=colors[i])
        lines.append(temp_line)
        legend_it.append((city_names[city], [temp_line]))

    legend = Legend(items=legend_it, location=(0, 25))
    legend.click_policy = "hide"

    p_goods.add_layout(legend, 'right')

    show(p_goods)

interact(good_plot_update, good=goods_descriptions.values())

def countries_plot_update(good):
    p_countries = figure(x_axis_type="datetime", tools=TOOLS, plot_width=600,
                         plot_height=600, toolbar_location="above",
                         title='Evolution of Prices by Countries')

    lines = []

    for (i, city) in enumerate(city_names):
        condition = (price_data['city_code'] == city) &          (price_data['good_code'] == inv_goods_descriptions[good])
        if len(city) == 2:
            temp_line = p_countries.line(x=price_data['date'][condition],
                                         y=price_data['pricen'][condition],
                                         color='blue',
                                         legend='US Cities')
        else:
            temp_line = p_countries.line(x=price_data['date'][condition],
                                         y=price_data['pricen'][condition],
                                         color='red',
                                         legend='Canadian Cities')
        lines.append(temp_line)

    p_countries.legend.location = 'bottom_right'
    p_countries.title.text_font_size = '12pt'
    p_countries.yaxis.axis_label = 'Normalized Price Index'
    p_countries.xaxis.axis_label = 'Year'

    show(p_countries)

interact(countries_plot_update, good=goods_descriptions.values())

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=800,
           plot_height=600, toolbar_location="above",
           title='Evolution of Shelter Prices')

lines = []
legend_it = []
colors = magma(len(city_names))

for (i, city) in enumerate(city_names):
    condition = (price_data['city_code'] == city) &      (price_data['good_code'] == inv_goods_descriptions['Shelter'])
    temp_line = p.line(x=price_data['date'][condition],
                       y=price_data['pricen'][condition],
                       color=colors[i])
    lines.append(temp_line)
    legend_it.append((city_names[city], [temp_line]))

legend = Legend(items=legend_it, location=(0, 25))
legend.click_policy = "hide"

p.add_layout(legend, 'right')

r = p.circle([pd.Timestamp('1982-06-01'),
              pd.Timestamp('1990-06-01')], [1.125, 1.9])

glyph = r.glyph
glyph.size = 110
glyph.fill_alpha = 0.2
glyph.line_color = "firebrick"
glyph.line_dash = [6, 3]
glyph.line_width = 2

show(p)

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=600,
           plot_height=600, toolbar_location="above",
           title='Evolution of Private Transportation Prices')

lines = []

for (i, city) in enumerate(city_names):
    condition = (price_data['city_code'] == city) &      (price_data['good_code'] ==
      inv_goods_descriptions['Private transportation'])
    if len(city) == 2:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='blue',
                           legend='US Cities')
    else:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='red',
                           legend='Canadian Cities')
    lines.append(temp_line)

p.legend.location = 'bottom_right'
p.title.text_font_size = '12pt'
p.yaxis.axis_label = 'Normalized Price Index'
p.xaxis.axis_label = 'Year'

r = p.circle([pd.Timestamp('1986-03-01')], [1.15])

glyph = r.glyph
glyph.size = 110
glyph.fill_alpha = 0.2
glyph.line_color = "firebrick"
glyph.line_dash = [6, 3]
glyph.line_width = 2

show(p)

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=600,
           plot_height=600, toolbar_location="above",
           title='Evolution of Alcoholic Beverages Prices')

lines = []

for (i, city) in enumerate(city_names):
    condition = (price_data['city_code'] == city) &      (price_data['good_code'] == inv_goods_descriptions['Alcoholic beverages'])
    if len(city) == 2:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='blue',
                           legend='US Cities')
    else:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='red',
                           legend='Canadian Cities')
    lines.append(temp_line)

p.legend.location = 'bottom_right'
p.title.text_font_size = '12pt'
p.yaxis.axis_label = 'Normalized Price Index'
p.xaxis.axis_label = 'Year'

r = p.circle([pd.Timestamp('1991-03-01')], [1.55])

glyph = r.glyph
glyph.size = 140
glyph.fill_alpha = 0.2
glyph.line_color = "firebrick"
glyph.line_dash = [6, 3]
glyph.line_width = 2

show(p)

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=800,
           plot_height=600, toolbar_location="above",
           title='Evolution of Fuel & Utilities Prices')

lines = []
legend_it = []
fuel_cities = ['Chicago', 'Los Angeles', 'Dallas', 'San Francisco',
               'Baltimore', 'Boston', 'St. Louis']
colors = all_palettes['Category20'][len(fuel_cities)]

for (i, city) in enumerate(fuel_cities):
    condition = (price_data['city_code'] == inv_city_names[city]) &      (price_data['good_code'] ==
      inv_goods_descriptions['Fuel and other utilities'])
    temp_line = p.line(x=price_data['date'][condition],
                       y=price_data['pricen'][condition],
                       color=colors[i])
    lines.append(temp_line)
    legend_it.append((city, [temp_line]))

legend = Legend(items=legend_it, location=(0, 200))
legend.click_policy = "hide"

p.add_layout(legend, 'right')

show(p)

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=600,
           plot_height=600, toolbar_location="above",
           title="Evolution of Medical Care Prices")

lines = []

for (i, city) in enumerate(city_names):
    condition = (price_data['city_code'] == city) &      (price_data['good_code'] == inv_goods_descriptions['Medical care'])
    if len(city) == 2:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='blue',
                           legend='US Cities')
    else:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='red',
                           legend='Canadian Cities')
    lines.append(temp_line)

p.legend.location = 'bottom_right'
p.title.text_font_size = '12pt'
p.yaxis.axis_label = 'Normalized Price Index'
p.xaxis.axis_label = 'Year'

show(p)

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=600,
           plot_height=600, toolbar_location="above",
           title="Evolution of Men's Apparel Prices")

lines = []

for (i, city) in enumerate(city_names):
    condition = (price_data['city_code'] == city) &      (price_data['good_code'] ==
      inv_goods_descriptions["Men's and boy's apparel"])
    if len(city) == 2:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='blue',
                           legend='US Cities')
    else:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='red',
                           legend='Canadian Cities')
    lines.append(temp_line)

p.legend.location = 'bottom_right'
p.title.text_font_size = '12pt'
p.yaxis.axis_label = 'Normalized Price Index'
p.xaxis.axis_label = 'Year'

show(p)

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=600,
           plot_height=600, toolbar_location="above",
           title="Evolution of Women's Apparel Prices")

lines = []

for (i, city) in enumerate(city_names):
    condition = (price_data['city_code'] == city) &      (price_data['good_code'] ==
      inv_goods_descriptions["Women's and girl's apparel"])
    if len(city) == 2:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='blue',
                           legend='US Cities')
    else:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='red',
                           legend='Canadian Cities')
    lines.append(temp_line)

p.legend.location = 'bottom_right'
p.title.text_font_size = '12pt'
p.yaxis.axis_label = 'Normalized Price Index'
p.xaxis.axis_label = 'Year'

show(p)

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=600,
                 plot_height=600, toolbar_location="above",
                 title="Evolution of Footwear Prices")

lines = []

for (i, city) in enumerate(city_names):
    condition = (price_data['city_code'] == city) &      (price_data['good_code'] == inv_goods_descriptions['Footwear'])
    if len(city) == 2:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='blue',
                           legend='US Cities')
    else:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='red', 
                           legend='Canadian Cities')
    lines.append(temp_line)

p.legend.location = 'bottom_right'
p.title.text_font_size = '12pt'
p.yaxis.axis_label = 'Normalized Price Index'
p.xaxis.axis_label = 'Year'

show(p)

p = figure(x_axis_type="datetime", plot_width=600,
           plot_height=600, toolbar_location="right",
           title="Comparison of Women's Apparel Prices between New York and Philadelphia")

ny_ph = ['NY', 'PH']

for i, city in enumerate(ny_ph):
    condition = (price_data['city_code'] == city) &      (price_data['good_code'] == inv_goods_descriptions['Footwear'])

    p.line(x=price_data['date'][condition],
                               y=price_data['pricen'][condition],
                               color=all_palettes['Category10'][10][i],
                               legend=city_names[city])

p.legend.location = 'bottom_right'

show(p)

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=600,
           plot_height=600, toolbar_location="above",
           title='Evolution of City CPI')

lines = []

for (i, city) in enumerate(city_names):
    condition = (price_data['city_code'] == city) &      (price_data['good_code'] == inv_goods_descriptions['City CPI'])
    if len(city) == 2:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='blue',
                           legend='US Cities')
    else:
        temp_line = p.line(x=price_data['date'][condition],
                           y=price_data['pricen'][condition],
                           color='red',
                           legend='Canadian Cities')
    lines.append(temp_line)

p.legend.location = 'bottom_right'
p.title.text_font_size = '12pt'
p.yaxis.axis_label = 'Normalized Price Index'
p.xaxis.axis_label = 'Year'

r = p.circle([pd.Timestamp('1991-01-01')], [1.7])

glyph = r.glyph
glyph.size = 110
glyph.fill_alpha = 0.2
glyph.line_color = "firebrick"
glyph.line_dash = [6, 3]
glyph.line_width = 2

show(p)

