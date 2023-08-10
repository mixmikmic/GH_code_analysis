from bokeh.charts import Line, output_notebook, show
from bokeh.models import HoverTool
from bokeh.plotting import figure, ColumnDataSource

import pandas as pd
import numpy as np
from itertools import cycle
from bokeh.palettes import Spectral6
output_notebook()
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])

colors = Spectral6
colorcycler = cycle(colors)

source = ColumnDataSource(df)
p = figure(plot_width=800, plot_height=400, title="Bokeh Line Example")

for column in df.columns:
    line = p.line(x=df.index, y=column, source=source, line_color=next(colorcycler),
                  line_width=5, line_cap='round', line_join='round'
                 )
    p.add_tools(HoverTool(tooltips=[("x,y","$x,$y"),("column",column),], line_policy="interp", renderers=[line]))

show(p)

from bokeh.charts import Line, output_notebook, show
from bokeh.models import HoverTool
from bokeh.plotting import figure, ColumnDataSource
import pandas as pd
import numpy as np
from itertools import cycle
from bokeh.palettes import Spectral6
output_notebook()

df = pd.read_clipboard().applymap(lambda x: float(str(x).replace('%','')))

colors = Spectral6
colorcycler = cycle(colors)

source = ColumnDataSource(df)
p = figure(plot_width=800, plot_height=500, title="Bokeh Line Example")

for column in df.columns:
    line = p.line(x=df.index, y=column, source=source, line_color=next(colorcycler),
                  line_width=5, line_cap='round', line_join='round'
                 )
    p.add_tools(HoverTool(tooltips=[("x,y","$x,$y"),("column",column),], line_policy="interp", renderers=[line]))

show(p)

import pandas as pd
from bokeh.charts import Bar, output_notebook, show

df = pd.read_clipboard()

df_long = pd.melt(df, id_vars=['BUILD_MTH']).dropna()

df_long
df_long.value = df_long.value.str.replace('%','').replace('$','').replace(',','')

df_long.rename(columns={'variable':'Model','value':'Defect_Rate'}, inplace=True)

df_long.set_index('BUILD_MTH', inplace=True)

df_long.Defect_Rate = df_long.Defect_Rate.astype(float)

bar = Bar(df_long, df_long.index, group='Model', values='Defect_Rate', width=700, height=500,
         legend='top_right', xlabel='Build Month', ylabel='Defect Rate', title='Defect Rate by Build Month')
output_notebook()
show(bar)

df_long

