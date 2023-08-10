get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import mpld3

mpld3.enable_notebook()

fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
ax.grid(color='white', linestyle='solid')

N = 50
scatter = ax.scatter(np.random.normal(size=N),
                     np.random.normal(size=N),
                     c=np.random.random(size=N),
                     s = 1000 * np.random.random(size=N),
                     alpha=0.3,
                     cmap=plt.cm.jet)

ax.set_title("D3 Scatter Plot", size=18);

from mpld3 import plugins

fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
ax.grid(color='white', linestyle='solid')

N = 50
scatter = ax.scatter(np.random.normal(size=N),
                     np.random.normal(size=N),
                     c=np.random.random(size=N),
                     s = 1000 * np.random.random(size=N),
                     alpha=0.3,
                     cmap=plt.cm.jet)

ax.set_title("D3 Scatter Plot (with tooltips!)", size=20)

labels = ['point {0}'.format(i + 1) for i in range(N)]
fig.plugins = [plugins.PointLabelTooltip(scatter, labels)]

np.random.seed(0)

P = np.random.random(size=10)
A = np.random.random(size=10)

x = np.linspace(0, 10, 100)
data = np.array([[x, Ai * np.sin(x / Pi)]
                 for (Ai, Pi) in zip(A, P)])

fig, ax = plt.subplots(2)

points = ax[1].scatter(P, A, c=P + A,
                       s=200, alpha=0.5)
ax[1].set_xlabel('Period')
ax[1].set_ylabel('Amplitude')

colors = plt.cm.ScalarMappable().to_rgba(P + A)

for (x, l), c in zip(data, colors):
    ax[0].plot(x, l, c=c, alpha=0.5, lw=3)

from mpld3.plugins import PluginBase
import jinja2
import json


class LinkedView(PluginBase):
    """A simple plugin showing how multiple axes can be linked"""
    
    FIG_JS = jinja2.Template("""
    var linedata{{ id }} = {{ linedata }};

    ax{{ axid }}.axes.selectAll(".paths{{ collid }}")
	    .on("mouseover", function(d, i){
             line{{ elid }}.data = linedata{{ id }}[i];
             line{{ elid }}.lineobj.transition()
                .attr("d", line{{ elid }}.line(line{{ elid }}.data))
                .style("stroke", this.style.fill);})
    """)

    def __init__(self, points, line, linedata):
        self.points = points
        self.line = line
        self.linedata = linedata
        self.id = self.generate_unique_id()

    def _fig_js_args(self):
        points = self._get_d3obj(self.points)
        line = self._get_d3obj(self.line)
        return dict(id=self.id,
                    axid=points.axid,
                    collid=points.collid,
                    elid=line.elid,
                    lineaxid=line.axid,
                    lineid=line.lineid,
                    linedata=json.dumps(self.linedata))

fig, ax = plt.subplots(2)

# scatter periods and amplitudes
np.random.seed(0)
P = np.random.random(size=10)
A = np.random.random(size=10)
x = np.linspace(0, 10, 100)
data = np.array([[x, Ai * np.sin(x / Pi)]
                 for (Ai, Pi) in zip(A, P)])
points = ax[1].scatter(P, A, c=P + A,
                       s=200, alpha=0.5)
ax[1].set_xlabel('Period')
ax[1].set_ylabel('Amplitude')

# create the line object
lines = ax[0].plot(x, 0 * x, '-w', lw=3, alpha=0.5)
ax[0].set_ylim(-1, 1)

# transpose line data and add plugin
linedata = data.transpose(0, 2, 1).tolist()
fig.plugins = [LinkedView(points, lines[0], linedata)]

