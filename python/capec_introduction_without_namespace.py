import lxml.etree
tree = lxml.etree.parse('capec2.9.xml')
root = tree.getroot()
for table in root: 
    print table.tag

weakness_table = root[2]
for row in weakness_table[0]: 
    column = row.tag.split('}')[-1]
    print column

histogram = {}
for row in weakness_table: 
    for column in row: 
        column_label = column.tag.split('}')[-1]
        if column_label not in histogram: 
            histogram[column_label] = 0
        else:
            histogram[column_label] += 1
print histogram

import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import Range1d
from bokeh.io import output_notebook
output_notebook() #So plots are shown inline instead of new tab. 

data = {}
data['Entries'] = histogram

df_data = pd.DataFrame(data).sort_values(by='Entries', ascending=True)
series = df_data.loc[:,'Entries']

p = figure(width=800, y_range=series.index.tolist(), title="Attack Pattern Histogram")

p.xaxis.axis_label = 'Frequency'
p.xaxis.axis_label_text_font_size = '10pt'
p.xaxis.major_label_text_font_size = '8pt'

p.yaxis.axis_label = 'Field'
p.yaxis.axis_label_text_font_size = '10pt'
p.yaxis.major_label_text_font_size = '8pt'

j = 1
for k,v in series.iteritems():
  
  #Print fields, values, orders
  #print (k,v,j) 
  p.rect(x=v/2, y=j, width=abs(v), height=0.4,
    width_units="data", height_units="data")
  j += 1

show(p)

