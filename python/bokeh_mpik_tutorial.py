from IPython.core.display import HTML, display
display(HTML('<iframe src=https://demo.bokehplots.com/apps/crossfilter width=1000 height=650></iframe>')
)

import numpy as np

x = np.arange(10)
y1 = np.random.rand(10)
y2 = np.random.rand(10)

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt

# Create a figure
fig = plt.figure(figsize=(13, 6))

# Create 2 subplots in the figure
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# Plot the data
ax1.plot(x, y1, color='purple')
ax2.plot(x, y2)
ax2.plot(x, y1, 'o') # Plot as circles

# Add axis titles
ax2.set_xlabel("x_points")
ax2.set_ylabel("y_data")

# Show the plot
plt.show()

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import column, gridplot
output_notebook()

# Create 2 figures
fig1 = figure(width=700, height=200)
fig2 = figure(width=700, height=200)

# Add glyphs to the figures
fig1.line(x, y1, color='purple')
fig2.line(x, y2)
fig2.circle(x, y1, color='orange')

# Add axis titles
fig2.xaxis.axis_label = "x_points"
fig2.yaxis.axis_label = "y_data"

# Put the figures into a grid, and show them
show(gridplot([[fig1], [fig2]]))

import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import column, gridplot
output_notebook()

data = np.random.normal(size=10000)
histogram, edges = np.histogram(data, bins=30)
bottom = np.zeros(edges.size - 1)
left = edges[:-1]
right = edges[1:]

fig = figure(width=700, height=300)
fig.quad(bottom=bottom, left=left, right=right, top=histogram, alpha=0.5)
show(fig)

from bokeh.plotting import figure, output_notebook, show

output_notebook()

# Add hover to this comma-separated string and see what changes
TOOLS = 'hover'

p = figure(plot_width=400, plot_height=400, title="Mouse over the dots", tools=TOOLS)
p.circle([1, 2, 3, 4, 5], [2, 5, 8, 2, 7], size=10)
show(p)

from bokeh.plotting import figure, output_notebook, show
from bokeh.models import HoverTool

output_notebook()
TOOLS = [HoverTool()]

p = figure(plot_width=400, plot_height=400, title="Mouse over the dots", tools=TOOLS)
p.circle([1, 2, 3, 4, 5], [2, 5, 8, 2, 7], size=10)
show(p)

from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.models import HoverTool

output_notebook()

source = ColumnDataSource(
        data=dict(
            x=[1, 2, 3, 4, 5],
            y=[2, 5, 8, 2, 7],
            desc=['A', 'b', 'C', 'd', 'E'],
        )
    )

hover = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]
    )

p = figure(plot_width=400, plot_height=400, tools=[hover],
           title="Mouse over the dots")

p.circle('x', 'y', size=20, source=source)

show(p)

from bokeh.io import output_notebook, show
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Button, CheckboxButtonGroup, RadioGroup, Select

output_notebook()

button = Button(label="Foo", button_type="success")
checkbox_button_group = CheckboxButtonGroup(labels=["Option 1", "Option 2", "Option 3"], active=[0, 1])
radio_group = RadioGroup(labels=["Option 1", "Option 2", "Option 3"], active=0)
select = Select(title="Option:", value="foo", options=["foo", "bar", "baz", "quux"])

show(widgetbox(button, checkbox_button_group, radio_group, select))

from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import Figure, output_notebook, show

output_notebook()

x = [x*0.005 for x in range(0, 200)]
y = x

source = ColumnDataSource(data=dict(x=x, y=y))

plot = Figure(plot_width=400, plot_height=400)
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

callback = CustomJS(args=dict(source=source), code="""
    var data = source.data;
    var f = cb_obj.value
    x = data['x']
    y = data['y']
    for (i = 0; i < x.length; i++) {
        y[i] = Math.pow(x[i], f)
    }
    source.trigger('change');
""")

slider = Slider(start=0.1, end=4, value=1, step=.1, title="power")
slider.js_on_change('value', callback)

layout = column(slider, plot)

show(layout)

get_ipython().run_cell_magic('writefile', 'interactions.py', 'from bokeh.layouts import column\nfrom bokeh.models import CustomJS, ColumnDataSource, Slider, Select\nfrom bokeh.plotting import figure, output_file, show\nfrom bokeh.io import curdoc\n\noutput_file("callback.html")\n\nx = [x*0.005 for x in range(0, 200)]\ny = x\n\nsource = ColumnDataSource(data=dict(x=x, y=y))\n\nplot = figure(plot_width=400, plot_height=400)\nl = plot.line(\'x\', \'y\', source=source, line_width=3, line_alpha=0.6)\n\n\n# Javascript callback function\ncallback = CustomJS(args=dict(source=source), code="""\n        var data = source.data;\n        var f = cb_obj.value\n        x = data[\'x\']\n        y = data[\'y\']\n        for (i = 0; i < x.length; i++) {\n            y[i] = Math.pow(x[i], f)\n        }\n        source.trigger(\'change\');\n    """)\n\n# Python callback function\ndef update(attr, old, new):\n    l.glyph.line_color = new\n\n# Javascript widget\nslider = Slider(start=0.1, end=4, value=1, step=.1, title="power", callback=callback)\n\n# Python widget\nselect = Select(title="Color:", value="blue", options=["blue","orange", "purple", "green"])\nselect.on_change(\'value\', update)\n\nlayout = column(slider, select, plot)\n\nshow(layout)\n\ncurdoc().add_root(layout)')

get_ipython().system('python interactions.py')

get_ipython().system('ls -lt')

get_ipython().system('bokeh serve --show interactions.py')

from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.io import output_notebook, push_notebook
from bokeh.layouts import column, gridplot
import numpy as np
import time
output_notebook()

x1 = np.arange(10)
y1 = np.random.rand(10)

# Create the ColumnDataSource
cdsource_d = dict(x=x1, y=y1)
cdsource = ColumnDataSource(data=cdsource_d)

# Create 2 figures
fig = figure(width=700, height=200)

# Add glyphs to the figures
fig.line('x', 'y', source=cdsource)

# Plot the figure
handle = show(fig, notebook_handle=True)

# Loop where the data inside the ColumnDataSource is changed
for i in range(100):
    y1 = np.random.rand(10)
    cdsource_d = dict(x=x1, y=y1)
    cdsource.data = cdsource_d
    push_notebook(handle=handle)
    time.sleep(0.1)

# This command will only work on my laptop
get_ipython().system('/Users/Jason/anaconda3/envs/py27/bin/bokeh serve --show spectrogram/')

from IPython.display import Image
Image(url='live.gif')

# This command will only work on my laptop
get_ipython().system('/Users/Jason/anaconda3/envs/cta/bin/bokeh serve --show /Users/Jason/Software/TargetIO/source/script/targetpipe/scripts/bokeh_live_camera/')

