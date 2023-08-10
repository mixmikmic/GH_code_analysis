import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import load_workbook, Workbook
get_ipython().run_line_magic('matplotlib', 'inline')

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.io import output_notebook
from bokeh.models import HoverTool, Label, LabelSet, Span

TOOLS = " crosshair, pan, wheel_zoom, zoom_in, zoom_out, box_zoom, undo, redo, reset, tap, save, box_select, poly_select, lasso_select"
output_notebook()

df = pd.read_excel(r"C:\EY\Thought Leadership\MA Correlation\correlation_test_data.xlsx" , sheetname='Sheet1')

#df.columns.tolist()

dataset = df[['Company',
 'Company Ticker',
 'market_cap',
 'Sector',
 'Est.Periods',
 'avg_TSR',
 'avg_ROA',
 'avg_ROE',
 'avg_EPS',
 'avg_dividend_ratio',
 'Rev_grow.Avg',
 'ebit_grow.avg',
 'avg_cost_growh',
 'Acquisitions',
 'ma_per_year',
 'Acq value',
 'Divestment',
 'div_per_year',
 'div_value',
 'Div as % deal value',
 'Deal intensity',
 'avg_deal_size',
 'avg_deal_size_mktcap',
 'domestic',
 'cross_border',
 'domestic_percentage',
 'in_sector',
 'out_sector',
 'sector_percentage',
 'assets_growth',
 'ppe_percent',
 'ppe_growth',
 'sga_growth',
 'sga_percent',
 'friendly',
 'hostile'
 ]]

#dataset.head()

#dataset.dtypes

# Adding calculated columns
dataset.is_copy=False
dataset.loc[:, 'ppe_vs_asset'] = dataset.loc[:,'ppe_growth']/dataset.loc[:,'assets_growth']
dataset.loc[:, 'rev_vs_cost'] = dataset.loc[:, 'Rev_grow.Avg']/dataset.loc[:, 'avg_cost_growh']
dataset.loc[:, 'rev_vs_ebit'] = dataset.loc[:, 'Rev_grow.Avg']/dataset.loc[:, 'ebit_grow.avg']
dataset.loc[:, 'ebit_vs_cost'] = dataset.loc[:, 'ebit_grow.avg']/dataset.loc[:, 'avg_cost_growh']
dataset.loc[:, 'ebit_vs_sga'] = dataset.loc[:, 'ebit_grow.avg']/dataset.loc[:, 'sga_growth']
# Replacing negative and positive infinite numbers
dataset = dataset.replace(to_replace=[-np.inf, np.inf], value=np.nan)

#dataset.describe()

dataset = dataset.fillna(value=0)

dataset.shape

#Filtering out inconsistent data points
dataset = dataset[~(((dataset['div_value']==0) & (dataset['Divestment']!=0)) | ((dataset['div_value']!=0) & (dataset['Divestment']==0)))] 
dataset = dataset[~(((dataset['Acq value']==0) & (dataset['Acquisitions']!=0)) | ((dataset['Acq value']!=0) & (dataset['Acquisitions']==0)))]
dataset = dataset[~(((dataset['div_per_year']==0) & (dataset['Divestment']!=0)) | ((dataset['div_per_year']!=0) & (dataset['Divestment']==0)))] 
dataset = dataset[~(((dataset['ma_per_year']==0) & (dataset['Acquisitions']!=0)) | ((dataset['ma_per_year']!=0) & (dataset['Acquisitions']==0)))] 
dataset = dataset[~(((dataset['ma_per_year']==0) & (dataset['Deal intensity']!=0)) | ((dataset['ma_per_year']!=0) & (dataset['Deal intensity']==0)))]

#Truncating outliers and only keep the data points within its two standard deviations.
dataset = dataset[np.abs(dataset['avg_TSR']-dataset['avg_TSR'].mean())<=(2*dataset['avg_TSR'].std())]
dataset = dataset[np.abs(dataset['avg_ROA']-dataset['avg_ROA'].mean())<=(2*dataset['avg_ROA'].std())]
dataset = dataset[np.abs(dataset['avg_ROE']-dataset['avg_ROE'].mean())<=(2*dataset['avg_ROE'].std())]
dataset = dataset[np.abs(dataset['avg_cost_growh']-dataset['avg_cost_growh'].mean())<=(2*dataset['avg_cost_growh'].std())]
dataset = dataset[np.abs(dataset['avg_deal_size_mktcap']-dataset['avg_deal_size_mktcap'].mean())<=(2*dataset['avg_deal_size_mktcap'].std())]
dataset = dataset[np.abs(dataset['Deal intensity']-dataset['Deal intensity'].mean())<=(2*dataset['Deal intensity'].std())]
dataset = dataset[np.abs(dataset['Rev_grow.Avg']-dataset['Rev_grow.Avg'].mean())<=(2*dataset['Rev_grow.Avg'].std())]
dataset = dataset[np.abs(dataset['ebit_grow.avg']-dataset['ebit_grow.avg'].mean())<=(2*dataset['ebit_grow.avg'].std())]
dataset = dataset[np.abs(dataset['ppe_vs_asset']-dataset['ppe_vs_asset'].mean())<=(2*dataset['ppe_vs_asset'].std())]
dataset = dataset[np.abs(dataset['rev_vs_cost']-dataset['rev_vs_cost'].mean())<=(2*dataset['rev_vs_cost'].std())]
dataset = dataset[np.abs(dataset['rev_vs_ebit']-dataset['rev_vs_ebit'].mean())<=(2*dataset['rev_vs_ebit'].std())]
dataset = dataset[np.abs(dataset['ebit_vs_cost']-dataset['ebit_vs_cost'].mean())<=(2*dataset['ebit_vs_cost'].std())]
dataset = dataset[np.abs(dataset['ebit_vs_sga']-dataset['ebit_vs_sga'].mean())<=(2*dataset['ebit_vs_sga'].std())]

dataset.shape

dataset.describe()

# Inserting bubble_size column based on deal frequency with minimum value of 5 (screen unit)
dataset.is_copy = False
dataset.loc[:, 'frequency_bubble'] = dataset['ma_per_year']*10 + 5
dataset.loc[:, 'intensity_bubble'] = dataset['Deal intensity']*10 + 5
organic = dataset[(dataset['Acquisitions']==0)]
inorganic = dataset[((dataset['Acquisitions']!=0))]
vertical = inorganic[(inorganic['sector_percentage']>0.66) & (inorganic['sector_percentage']<=1)]
horizontal = inorganic[(inorganic['sector_percentage']<0.66) & (inorganic['Acquisitions']!=0)]
local = inorganic[(inorganic['domestic_percentage']>0.66)]
offshore = inorganic[(inorganic['domestic_percentage']<0.66) & (inorganic['Acquisitions']!=0)]
periodic = inorganic[(inorganic['ma_per_year']<=0.2) & (inorganic['ma_per_year']>0)]
frequent = inorganic[(inorganic['ma_per_year'])>0.2]
small = inorganic[(inorganic['Deal intensity']<0.1) & (inorganic['Acquisitions']!=0)]
large = inorganic[inorganic['Deal intensity']>=0.1]
print("vertical:", vertical.shape[0], "\n",
      "horizontal:", horizontal.shape[0], "\n",
      "inorganic:", inorganic.shape[0], "\n", 
      "organic:", organic.shape[0], "\n", 
      "local:", local.shape[0], "\n", 
      "offshore:", offshore.shape[0], "\n", 
      "periodic:", periodic.shape[0], "\n", 
      "frequent:", frequent.shape[0], "\n",
      "small:", small.shape[0], "\n",
      "large:", large.shape[0], "\n")

# Output the scatter plot of TSR and deal intensity of inorganic growths.

source1 = ColumnDataSource(inorganic)
source2 = ColumnDataSource(organic)
in_organic_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="Deal Intensity", y_axis_label="TSR", x_range=(0,15))
in_organic_plot.scatter(y='avg_TSR', x='Deal intensity', size='frequency_bubble', alpha=0.6, source=source1)
in_organic_plot.scatter(y='avg_TSR', x='Deal intensity', size=5, alpha=0.6, color='YellowGreen', source=source2)
#in_organic_plot.scatter(y='avg_TSR', x='Deal intensity', alpha=0.6, color='blue', source=source2)
hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR")])
in_organic_plot.add_tools(hover)
in_organic_plot.ray(x=0, y=organic['avg_TSR'].mean(), angle=0, color='red', legend="Organic Avg. TSR", length=900, )
in_organic_plot.ray(x=0, y=inorganic['avg_TSR'].mean(), angle=0, color='green', legend="Inorganic Avg. TSR", length=900)
organic_label = Label(x=800, y=-0.05, x_units='screen', text=str('{0:.2f}'.format(organic['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
in_organic_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(inorganic['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
in_organic_plot.add_layout(organic_label)
in_organic_plot.add_layout(in_organic_label)
show(in_organic_plot)

frequent['avg_TSR'].mean(), periodic['avg_TSR'].mean(), organic['avg_TSR'].mean()

# Does frequency matter?

source1 = ColumnDataSource(organic)
source2 = ColumnDataSource(periodic)
source3 = ColumnDataSource(frequent)
frequency_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="Deal Intensity", y_axis_label="TSR", x_range=(0,15))
frequency_plot.scatter(y='avg_TSR', x='Deal intensity', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
frequency_plot.scatter(y='avg_TSR', x='Deal intensity', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)
frequency_plot.scatter(y='avg_TSR', x='Deal intensity', size='frequency_bubble', alpha=0.6, color='Gold', source=source3)
#in_organic_plot.scatter(y='avg_TSR', x='Deal intensity', alpha=0.6, color='blue', source=source2)
hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR")])
frequency_plot.add_tools(hover)
frequency_plot.ray(x=0, y=organic['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Organic Avg. TSR", length=900, )
frequency_plot.ray(x=0, y=periodic['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Periodic Avg. TSR", length=900)
frequency_plot.ray(x=0, y=frequent['avg_TSR'].mean(), angle=0, color='Gold', legend="Periodic Avg. TSR", length=900)
organic_label = Label(x=800, y=-0.05, x_units='screen', text=str('{0:.2f}'.format(organic['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
periodic_label = Label(x=800, y=0.01, x_units='screen', text=str('{0:.2f}'.format(periodic['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
frequenct_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(frequent['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
frequency_plot.add_layout(organic_label)
frequency_plot.add_layout(periodic_label)
frequency_plot.add_layout(frequenct_label)
show(frequency_plot)

# Does geography matter?

source1 = ColumnDataSource(local)
source2 = ColumnDataSource(offshore)

geo_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="Deal Intensity", y_axis_label="TSR", x_range=(0,15))
geo_plot.scatter(y='avg_TSR', x='Deal intensity', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
geo_plot.scatter(y='avg_TSR', x='Deal intensity', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR")])
geo_plot.add_tools(hover)
geo_plot.ray(x=0, y=local['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Local Avg. TSR", length=900, )
geo_plot.ray(x=0, y=offshore['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Offshore Avg. TSR", length=900)
local_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(local['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
offshore_label = Label(x=800, y=0, x_units='screen', text=str('{0:.2f}'.format(offshore['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
geo_plot.add_layout(local_label)
geo_plot.add_layout(offshore_label)
show(geo_plot)

vertical['avg_TSR'].mean(), horizontal['avg_TSR'].mean()

# Does sector matter?

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

sector_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="Deal Intensity", y_axis_label="TSR", x_range=(0,15))
sector_plot.scatter(y='avg_TSR', x='Deal intensity', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
sector_plot.scatter(y='avg_TSR', x='Deal intensity', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR")])
sector_plot.add_tools(hover)
sector_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
sector_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
sector_plot.add_layout(vertical_label)
sector_plot.add_layout(horizontal_label)
show(sector_plot)

# Do frequent acquirers take their eyes off the cost reduction ball? (Does frequency distract from cost synergies?)

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

cost_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="EBIT Growth / Cost Growth", y_axis_label="TSR", x_range=(0,5))
cost_plot.scatter(y='avg_TSR', x='ebit_vs_cost', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
cost_plot.scatter(y='avg_TSR', x='ebit_vs_cost', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("EBIT Growth / Cost Growth", "@ebit_vs_cost")])
cost_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# cost_plot.add_layout(vertical_label)
# cost_plot.add_layout(horizontal_label)
show(cost_plot)

# Does deal intensity distract from cost reduction?

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

cost_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="EBIT Growth / Cost Growth", y_axis_label="TSR", x_range=(0,5))
cost_plot.scatter(y='avg_TSR', x='ebit_vs_cost', size='intensity_bubble', alpha=0.8, color='YellowGreen', source=source1)
cost_plot.scatter(y='avg_TSR', x='ebit_vs_cost', size='intensity_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("EBIT Growth / Cost Growth", "@ebit_vs_cost")])
cost_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# cost_plot.add_layout(vertical_label)
# cost_plot.add_layout(horizontal_label)
show(cost_plot)

# Do frequent acquirers take their eyes off the SG&A cost reduction ball? (Does frequency distract from cost synergies?)

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

sga_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="EBIT Growth / SG&A Growth", y_axis_label="TSR", x_range=(0,5))
sga_plot.scatter(y='avg_TSR', x='ebit_vs_sga', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
sga_plot.scatter(y='avg_TSR', x='ebit_vs_sga', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("EBIT Growth / SG&A Growth", "@ebit_vs_sga")])
sga_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(sga_plot)

# Does deal intensity distract from SG&A cost reduction?

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

sga_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="EBIT Growth / SG&A Growth", y_axis_label="TSR", x_range=(0,5))
sga_plot.scatter(y='avg_TSR', x='ebit_vs_sga', size='intensity_bubble', alpha=0.8, color='YellowGreen', source=source1)
sga_plot.scatter(y='avg_TSR', x='ebit_vs_sga', size='intensity_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("EBIT Growth / SG&A Growth", "@ebit_vs_sga")])
sga_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(sga_plot)

# Does frequent acquirer achieve more revenue synergies over cost synergies?

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

revenue_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="EBIT Growth / Cost Growth", y_axis_label="EBIT Growth / Revenue Growth", x_range=(0,5), y_range=(0,5))
revenue_plot.scatter(y='avg_cost_growh', x='Rev_grow.Avg', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
revenue_plot.scatter(y='avg_cost_growh', x='Rev_grow.Avg', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("EBIT Growth / SG&A Growth", "@ebit_vs_sga")])
revenue_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(revenue_plot)

# by frequency

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

asset_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="ROA", y_axis_label="TSR", x_range=(-1,1))
asset_plot.scatter(y='avg_TSR', x='avg_ROA', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
asset_plot.scatter(y='avg_TSR', x='avg_ROA', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("Average ROA", "@avg_ROA")])
asset_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(asset_plot)

# by intensity

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

asset_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="ROA", y_axis_label="TSR", x_range=(-1,1))
asset_plot.scatter(y='avg_TSR', x='avg_ROA', size='intensity_bubble', alpha=0.8, color='YellowGreen', source=source1)
asset_plot.scatter(y='avg_TSR', x='avg_ROA', size='intensity_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("Average ROA", "@avg_ROA")])
asset_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(asset_plot)

# by frequency

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

ppe_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="PPE Growth / Assets Growth", y_axis_label="TSR", x_range=(-5,10))
ppe_plot.scatter(y='avg_TSR', x='ppe_vs_asset', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
ppe_plot.scatter(y='avg_TSR', x='ppe_vs_asset', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("Average ROA", "@avg_ROA")])
ppe_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(ppe_plot)

# by intensity

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

ppe_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="PPE Growth / Assets Growth", y_axis_label="TSR", x_range=(0,1))
ppe_plot.scatter(y='avg_TSR', x='ppe_vs_asset', size='intensity_bubble', alpha=0.8, color='YellowGreen', source=source1)
ppe_plot.scatter(y='avg_TSR', x='ppe_vs_asset', size='intensity_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("Average ROA", "@avg_ROA")])
ppe_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(ppe_plot)

# by frequency

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

cost_asset_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="Assets Growth", y_axis_label="EBIT Growth / Cost Growth", x_range=(-1,5), y_range=(-10,10))
cost_asset_plot.scatter(y='ebit_vs_cost', x='assets_growth', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
cost_asset_plot.scatter(y='ebit_vs_cost', x='assets_growth', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("Average ROA", "@avg_ROA")])
cost_asset_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(cost_asset_plot)

# by intensity

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

cost_asset_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="Assets Growth", y_axis_label="EBIT Growth / Cost Growth", x_range=(-1,5), y_range=(-10,10))
cost_asset_plot.scatter(y='ebit_vs_cost', x='assets_growth', size='intensity_bubble', alpha=0.8, color='YellowGreen', source=source1)
cost_asset_plot.scatter(y='ebit_vs_cost', x='assets_growth', size='intensity_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("Average ROA", "@avg_ROA")])
cost_asset_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(cost_asset_plot)

# by frequency

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

rev_asset_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="Assets Growth", y_axis_label="EBIT Growth / Cost Growth", x_range=(-1,5), y_range=(-10,10))
rev_asset_plot.scatter(y='rev_vs_cost', x='assets_growth', size='frequency_bubble', alpha=0.8, color='YellowGreen', source=source1)
rev_asset_plot.scatter(y='rev_vs_cost', x='assets_growth', size='frequency_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("Average ROA", "@avg_ROA")])
rev_asset_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(rev_asset_plot)

# by intensity

source1 = ColumnDataSource(vertical)
source2 = ColumnDataSource(horizontal)

rev_asset_plot = figure(tools=TOOLS, width=900, height=500, x_axis_label="Assets Growth", y_axis_label="EBIT Growth / Cost Growth", x_range=(-1,5), y_range=(-10,10))
rev_asset_plot.scatter(y='rev_vs_cost', x='assets_growth', size='intensity_bubble', alpha=0.8, color='YellowGreen', source=source1)
rev_asset_plot.scatter(y='rev_vs_cost', x='assets_growth', size='intensity_bubble', alpha=0.6, color='DeepPink', source=source2)

hover = HoverTool(tooltips=[("Company", "@Company"), ("ma_per_year", "@ma_per_year"), ("Average TSR", "@avg_TSR"), ("Average ROA", "@avg_ROA")])
rev_asset_plot.add_tools(hover)
# cost_plot.ray(x=0, y=vertical['avg_TSR'].mean(), angle=0, color='YellowGreen', legend="Vertical Avg. TSR", length=900, )
# cost_plot.ray(x=0, y=horizontal['avg_TSR'].mean(), angle=0, color='DeepPink', legend="Horizontal Avg. TSR", length=900)
# vertical_label = Label(x=800, y=0.05, x_units='screen', text=str('{0:.2f}'.format(vertical['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# horizontal_label = Label(x=800, y=-0.03, x_units='screen', text=str('{0:.2f}'.format(horizontal['avg_TSR'].mean()*100)) + "%", text_font_size='10pt')
# sga_plot.add_layout(vertical_label)
# sga_plot.add_layout(horizontal_label)
show(rev_asset_plot)



