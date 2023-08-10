import numpy as np
from bokeh.plotting import *
from bokeh.models import ColumnDataSource
output_notebook()
output_file("linked_brushing.html", title='Bokeh demo 1')

# prepare some date
N = 300
x = np.linspace(0, 4*np.pi, N)
y0 = np.sin(x)
y1 = np.cos(x)

# create a column data source for the plots to share
source = ColumnDataSource(data=dict(x=x, y0=y0, y1=y1))

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

# create a new plot and add a renderer
left = figure(tools=TOOLS, width=350, height=350, title=None)
left.line('x', 'y0', source=source, color='red', line_width=2, line_alpha=0.4)
left.circle('x', 'y0', source=source, color='red')

# create another new plot and add a renderer
right = figure(tools=TOOLS, width=350, height=350, title=None)
right.line('x', 'y1', source=source, color='blue', line_width=2, line_alpha=0.4)
right.circle('x', 'y1', source=source, color='blue')

# put the subplots in a gridplot
p = gridplot([[left, right]])

# show the results
show(p)
#

import numpy as np
from bokeh.plotting import *
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource, CustomJS, Slider
output_notebook()
output_file("linked_brushing_slider.html", title='Bokeh demo 2')

# prepare some date
N = 300
x = np.linspace(0, 4*np.pi, N)
y0 = np.sin(x)
y1 = np.cos(x)

# create a column data source for the plots to share
source = ColumnDataSource(data=dict(x=x, y0=y0, y1=y1))

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

# create a new plot and add a renderer
left = figure(tools=TOOLS, width=350, height=350, title=None)
left.line('x', 'y0', source=source, color='red', line_width=2, line_alpha=0.4)
left.circle('x', 'y0', source=source, color='red')

# create another new plot and add a renderer
right = figure(tools=TOOLS, width=350, height=350, title=None)
right.line('x', 'y1', source=source, color='blue', line_width=2, line_alpha=0.4)
right.circle('x', 'y1', source=source, color='blue')

# put the subplots in a gridplot
p = gridplot([[left, right]])

callback = CustomJS(args=dict(source=source), code="""
    var data = source.data;
    var A = amp.value;
    var k = freq.value;
    var phi = phase.value;
    var B = offset.value;
    x = data['x']
    y0 = data['y0']
    y1 = data['y1']
    for (i = 0; i < x.length; i++) {
        y0[i] = B + A*Math.sin(k*x[i]+phi);
        y1[i] = B + A*Math.cos(k*x[i]+phi);
    }
    source.trigger('change');
""")

amp_slider = Slider(start=0.1, end=10, value=1, step=.1,
                    title="Amplitude", callback=callback)
callback.args["amp"] = amp_slider

freq_slider = Slider(start=0.1, end=10, value=1, step=.1,
                     title="Frequency", callback=callback)
callback.args["freq"] = freq_slider

phase_slider = Slider(start=0, end=6.4, value=0, step=.1,
                      title="Phase", callback=callback)
callback.args["phase"] = phase_slider

offset_slider = Slider(start=-5, end=5, value=0, step=.1,
                       title="Offset", callback=callback)
callback.args["offset"] = offset_slider

layout = row(
    p,
    widgetbox(amp_slider, freq_slider, phase_slider, offset_slider),
)

show(layout)



