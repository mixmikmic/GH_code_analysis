from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
if (code_show){
$('div.input').hide();
} else {
$('div.input').show();
}
code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')

import numpy as np
import pandas as pd

import scipy.interpolate
import scipy.stats 

import bokeh.plotting
import bokeh.models  # LinearAxis 
from bokeh.palettes import Category10_10 as palette

bokeh.plotting.output_notebook()

m1 = 6.5
m2 = 12.0

s1 = .75
s2 = 3.0

n1 = 20
n2 = 80
n = n1 + n2

x_min = 2.0
x_max = 25.0
x = np.linspace(x_min, x_max, 200)
y1 = scipy.stats.norm.pdf(x, loc=m1, scale=s1) * n1 / n

y2 = scipy.stats.norm.pdf(x, loc=m2, scale=s2) * n2 / n

y = y1 + y2

p = bokeh.plotting.figure(width=400, height=300, 
                          x_axis_label='Time (m)',
                          y_axis_label='% of People')
p.line(x, y, legend='combined', color=palette[0], line_width=2)
p.line(x, y1, legend='fast group', color=palette[1])
p.line(x, y2, legend='slow group', color=palette[2])

p.legend.location = 'top_right'

bokeh.plotting.show(p)

dx = x[1] - x[0]

a = dx * y.sum()
print(a)

a1 = dx * y1.sum()
print(a1)

a2 = dx * y2.sum()
print(a2)

p1_v1 = np.zeros_like(x)
p1_v1[x < m1] = 1

d1 = np.abs(x - m1)
d2 = np.abs(x - m2)

mask = np.logical_and(m1 < x, x < m2)
p1_v1[mask] = 1 - s1 * d1[mask] / (s1 * d1[mask] + s2 * d2[mask])

p = bokeh.plotting.figure(width=400, height=300, toolbar_location='above',
                          x_axis_label='Time (m)',
                          y_axis_label='% of People')
p.extra_y_ranges = {'prob': bokeh.models.Range1d(start=0, end=1.02)}
p.add_layout(bokeh.models.LinearAxis(y_range_name='prob', 
                                     axis_label='Probability'), 'right')

p.y_range = bokeh.models.Range1d(start=0, end=0.15)
p.line(x, y, legend='combined', color=palette[0], line_width=2)
p.line(x, y1, legend='fast group', color=palette[1])
p.line(x, y2, legend='slow group', color=palette[2])
p.line(x, p1_v1, legend='p fast', color=palette[3], y_range_name='prob')
p.line(x, 1-p1_v1, legend='p slow', color=palette[4], y_range_name='prob')

p.legend.location = 'top_right'

bokeh.plotting.show(p)

p1_v1[mask] = 1 - s2 * d1[mask] / (s2 * d1[mask] + s1 * d2[mask])
p2_v1 = 1 - p1_v1

p = bokeh.plotting.figure(width=400, height=300, toolbar_location='above',
                          x_axis_label='Time (m)',
                          y_axis_label='% of People')
p.extra_y_ranges = {'prob': bokeh.models.Range1d(start=0, end=1.02)}
p.add_layout(bokeh.models.LinearAxis(y_range_name='prob', 
                                     axis_label='Probability'), 'right')

p.y_range = bokeh.models.Range1d(start=0, end=0.15)
p.line(x, y, legend='combined', color=palette[0], line_width=2)
p.line(x, y1, legend='fast group', color=palette[1])
p.line(x, y2, legend='slow group', color=palette[2])
p.line(x, p1_v1, legend='p fast', color=palette[3], y_range_name='prob')
p.line(x, p2_v1, legend='p slow', color=palette[4], y_range_name='prob')

p.legend.location = 'top_right'

bokeh.plotting.show(p)

p1_v1[mask] = 1 - s2 * d1[mask] * n1 / n / (s2 * d1[mask] * n1 / n + s1 * d2[mask] * n2 / n)
p2_v1 = 1 - p1_v1

p = bokeh.plotting.figure(width=400, height=300, toolbar_location='above',
                          x_axis_label='Time (m)',
                          y_axis_label='% of People')
p.extra_y_ranges = {'prob': bokeh.models.Range1d(start=0, end=1.02)}
p.add_layout(bokeh.models.LinearAxis(y_range_name='prob', 
                                     axis_label='Probability'), 'right')

p.y_range = bokeh.models.Range1d(start=0, end=0.15)
p.line(x, y, legend='combined', color=palette[0], line_width=2)
p.line(x, y1, legend='fast group', color=palette[1])
p.line(x, y2, legend='slow group', color=palette[2])
p.line(x, p1_v1, legend='p fast', color=palette[3], y_range_name='prob')
p.line(x, p2_v1, legend='p slow', color=palette[4], y_range_name='prob')

p.legend.location = 'top_right'

bokeh.plotting.show(p)

m2 = 15.0

y1 = scipy.stats.norm.pdf(x, loc=m1, scale=s1) * n1 / n
y2 = scipy.stats.norm.pdf(x, loc=m2, scale=s2) * n2 / n
y = y1 + y2

p1_v1 = np.zeros_like(x)
p1_v1[x < m1] = 1

d1 = np.abs(x - m1)
d2 = np.abs(x - m2)

mask = np.logical_and(m1 < x, x < m2)
p1_v1[mask] = 1 - s1 * d1[mask] / (s1 * d1[mask] + s2 * d2[mask])
p1_v1[mask] = 1 - s2 * d1[mask] * n1 / n / (s2 * d1[mask] * n1 / n + s1 * d2[mask] * n2 / n)
p2_v1 = 1 - p1_v1

p = bokeh.plotting.figure(width=400, height=300, toolbar_location='above',
                          x_axis_label='Time (m)',
                          y_axis_label='% of People')
p.extra_y_ranges = {'prob': bokeh.models.Range1d(start=0, end=1.02)}
p.add_layout(bokeh.models.LinearAxis(y_range_name='prob', 
                                     axis_label='Probability'), 'right')

p.y_range = bokeh.models.Range1d(start=0, end=0.15)
p.line(x, y, legend='combined', color=palette[0], line_width=2)
p.line(x, y1, legend='fast group', color=palette[1])
p.line(x, y2, legend='slow group', color=palette[2])
p.line(x, p1_v1, legend='p fast', color=palette[3], y_range_name='prob')
p.line(x, p2_v1, legend='p slow', color=palette[4], y_range_name='prob')

p.legend.location = 'top_right'

bokeh.plotting.show(p)

m2 = 12.0
n1 = 40.0
n2 = 60.0
n = n1 + n2

y1 = scipy.stats.norm.pdf(x, loc=m1, scale=s1) * n1 / n
y2 = scipy.stats.norm.pdf(x, loc=m2, scale=s2) * n2 / n
y = y1 + y2

p1_v1 = np.zeros_like(x)
p1_v1[x < m1] = 1

d1 = np.abs(x - m1)
d2 = np.abs(x - m2)

mask = np.logical_and(m1 < x, x < m2)
p1_v1[mask] = 1 - s1 * d1[mask] / (s1 * d1[mask] + s2 * d2[mask])
p1_v1[mask] = 1 - s2 * d1[mask] * n1 / n / (s2 * d1[mask] * n1 / n + s1 * d2[mask] * n2 / n)
p2_v1 = 1 - p1_v1

p = bokeh.plotting.figure(width=400, height=300, toolbar_location='above',
                          x_axis_label='Time (m)',
                          y_axis_label='% of People')
p.extra_y_ranges = {'prob': bokeh.models.Range1d(start=0, end=1.02)}
p.add_layout(bokeh.models.LinearAxis(y_range_name='prob', 
                                     axis_label='Probability'), 'right')

p.y_range = bokeh.models.Range1d(start=0, end=0.15)
p.line(x, y, legend='combined', color=palette[0], line_width=2)
p.line(x, y1, legend='fast group', color=palette[1])
p.line(x, y2, legend='slow group', color=palette[2])
p.line(x, p1_v1, legend='p fast', color=palette[3], y_range_name='prob')
p.line(x, p2_v1, legend='p slow', color=palette[4], y_range_name='prob')

p.legend.location = 'top_right'

bokeh.plotting.show(p)

n1 = 20.0
n2 = 80.0
n = n1 + n2

y1 = scipy.stats.norm.pdf(x, loc=m1, scale=s1) * n1 / n
y2 = scipy.stats.norm.pdf(x, loc=m2, scale=s2) * n2 / n
y = y1 + y2

p1_v1 = np.zeros_like(x)
p1_v1[x < m1] = 1

d1 = np.abs(x - m1)
d2 = np.abs(x - m2)

mask = np.logical_and(m1 < x, x < m2)
p1_v1[mask] = 1 - s1 * d1[mask] / (s1 * d1[mask] + s2 * d2[mask])
p1_v1[mask] = 1 - s2 * d1[mask] * n1 / n / (s2 * d1[mask] * n1 / n + s1 * d2[mask] * n2 / n)
p2_v1 = 1 - p1_v1

p = bokeh.plotting.figure(width=400, height=300, toolbar_location='above',
                          x_axis_label='Time (m)',
                          y_axis_label='% of People')
p.extra_y_ranges = {'prob': bokeh.models.Range1d(start=0, end=1.02)}
p.add_layout(bokeh.models.LinearAxis(y_range_name='prob', 
                                     axis_label='Probability'), 'right')

p.y_range = bokeh.models.Range1d(start=0, end=0.15)
p.line(x, y, legend='combined', color=palette[0], line_width=2)
p.line(x, y1, legend='fast group', color=palette[1])
p.line(x, y2, legend='slow group', color=palette[2])
p.line(x, p1_v1, legend='p fast', color=palette[3], y_range_name='prob')
p.line(x, p2_v1, legend='p slow', color=palette[4], y_range_name='prob')

p.legend.location = 'top_right'

bokeh.plotting.show(p)

x_lower = x - dx / 2
x_upper = x + dx / 2
pd1 = (scipy.stats.norm.cdf(x_upper, loc=m1, scale=s1) - 
       scipy.stats.norm.cdf(x_lower, loc=m1, scale=s1)) * n1 / n
pd2 = (scipy.stats.norm.cdf(x_upper, loc=m2, scale=s2) - 
       scipy.stats.norm.cdf(x_lower, loc=m2, scale=s2)) * n2 / n

atol = 0.001  # absolute tolerance
assert np.allclose(pd1.sum(), n1 / n, atol=atol)
assert np.allclose(pd2.sum(), n2 / n, atol=atol)

p1_v2 = pd1 / (pd1 + pd2)
p2_v2 = 1 - p1_v2

p = bokeh.plotting.figure(width=400, height=300, toolbar_location='above',
                          x_axis_label='Time (m)',
                          y_axis_label='% of People')
p.extra_y_ranges = {'prob': bokeh.models.Range1d(start=0, end=1.02)}
p.add_layout(bokeh.models.LinearAxis(y_range_name='prob', 
                                     axis_label='Probability'), 'right')

p.y_range = bokeh.models.Range1d(start=0, end=0.15)
p.line(x, y, legend='combined', color=palette[0], line_width=2)
p.line(x, y1, legend='fast group', color=palette[1])
p.line(x, y2, legend='slow group', color=palette[2])
p.line(x, p1_v2, legend='p fast', color=palette[3], y_range_name='prob')
p.line(x, p2_v2, legend='p slow', color=palette[4], y_range_name='prob')

p.legend.location = 'top_right'

bokeh.plotting.show(p)

p_per = bokeh.plotting.figure(width=400, height=300,
                              y_axis_label='% of People')
p_per.line(x, y, legend='combined', color=palette[0], line_width=2)
p_per.line(x, y1, legend='fast group', color=palette[1])
p_per.line(x, y2, legend='slow group', color=palette[2])

p_prob = bokeh.plotting.figure(width=400, height=300,
                               y_axis_label='Probability',
                               x_range=p_per.x_range)
p_prob.line(x, p1_v1, legend='p_v1 fast', color=palette[3])
p_prob.line(x, p2_v1, legend='p_v1 slow', color=palette[4])
p_prob.line(x, p1_v2, legend='p_v2 fast', color=palette[3], line_dash='dashed')
p_prob.line(x, p2_v2, legend='p_v2 slow', color=palette[4], line_dash='dashed')

p_diff = bokeh.plotting.figure(width=400, height=100,
                               x_axis_label='Time (m)',
                               y_axis_label='Residuals',
                               x_range=p_per.x_range)
p_diff.line(x, p1_v2-p1_v1, color=palette[3])
p_diff.line(x, p2_v2-p2_v1, color=palette[4])

p = bokeh.layouts.gridplot([[p_per], [p_prob], [p_diff]])
bokeh.plotting.show(p)



