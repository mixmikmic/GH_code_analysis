import pandas as pd
from datetime import datetime
from yahoo_finance import Share
import collections
import numpy as np
from bokeh.charts import TimeSeries
from bokeh.io import output_notebook
from bokeh.models import HoverTool
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
output_notebook()

df = pd.DataFrame.from_csv('News_on_US.csv', index_col = None)

df1 = df.dropna()
df1.reset_index(inplace = True, drop = False)

df2 = df1.copy()
df2.Time = df2.Time.apply(lambda x: datetime.strptime(x, '%A, %d %B, %Y'))
df2.Time = df2.Time.apply(lambda x: x.date().strftime('%Y-%m-%d'))
df2.Title = df2.Title.apply(lambda x: repr(x))

df2.head()

date_lb = sorted(df2.Time)[0]
date_ub = sorted(df2.Time)[-1]
print 'Date Range: %s to %s' % (date_lb, date_ub)
date_range_df = pd.date_range(start = date_lb, end = date_ub, freq = 'D')
date_range_lis = date_range_df.date.tolist()
total_date_lis = [str(x) for x in date_range_lis]

temp_dict = dict(list(df2.groupby(df2.Time)))

news_dict = {}
for date in total_date_lis:
    if date in temp_dict.keys():
        news_dict[date] = '|||'.join(temp_dict[date].Title.tolist())
    else:
        news_dict[date] = 'No News Today'

news_dict.items()[:5]

od = collections.OrderedDict(sorted(news_dict.items()))

od.items()[:5]

news_lis = []
for k,v in od.items():
    news_lis.append(v)

news_lis[:5]

DIA = Share('DIA')
DIA_history = DIA.get_historical(start_date = date_lb, end_date = date_ub)
DIA_df = pd.DataFrame([[x['Date'], x['Close']] for x in DIA_history[0:]])
DIA_df1 = DIA_df.copy()
DIA_df1.columns = ['Date', 'Price']

DIA_df2 = DIA_df1.copy()
DIA_df2.Date = DIA_df2.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

print DIA_df2.head()
print len(DIA_df2)
print DIA_df2.loc[0, 'Date']
print DIA_df2.loc[250, 'Date']
print DIA_df2.tail()

merged_price = pd.merge(pd.DataFrame({'Date': date_range_df}), DIA_df2, on = 'Date', how = 'left')
merged_price.head()

merged_price1 = merged_price.copy()
merged_price1.Price = merged_price1.Price.astype(float).interpolate(method = 'linear')
merged_price1.head()

merged_price2 = merged_price1.set_index('Date')
merged_price2.index

merged_price3 = pd.DataFrame(data = {'Price': merged_price2.Price}, 
                    index = pd.date_range(start = date_lb, end = date_ub, freq = 'D'))
print merged_price3.index
merged_price3.head()

vals_list_of_list = merged_price3.values.T.tolist()

ts_list_of_list = []
for i in range(0,len(merged_price3.columns)):
    ts_list_of_list.append(merged_price3.index)

_tools_to_show = 'box_zoom,pan,save,hover,resize,reset,tap,wheel_zoom' 

p = figure(width=1200, height=900, x_axis_type="datetime", tools=_tools_to_show)

p.multi_line(ts_list_of_list, vals_list_of_list, line_color=['#3399cc'])

result_lis = []
for (name, series) in merged_price3.iteritems():
    result_lis.append((name, series))

name_for_display = np.tile('DIA', [len(merged_price3.index),1])

source = ColumnDataSource({'x': merged_price3.index, 'y': series.values, 'series_name': name_for_display, 
                           'News': news_lis, 'fonts': np.tile('<b>bold</b>', [len(merged_price3.index),1])})

p.scatter('x', 'y', source = source, fill_alpha=0, line_alpha=0.3, line_color="#cad3c5", size = 10)

hover = p.select(dict(type=HoverTool))

hover.tooltips = [("Series", "@series_name"), ("News", "@News"),  ("Value", "$y{0.00%}"),]

hover.mode = 'mouse'

show(p)



