get_ipython().magic('matplotlib inline')

import pandas as pd
import seaborn as sns
import os

import ipywidgets as widgets
from traitlets import Unicode, List, Instance, link
from IPython.display import display, clear_output, HTML, Javascript
import jinja2

features_df = pd.read_pickle('./datasets/features.dataframe')
sim_df = pd.read_pickle('./datasets/sims.dataframe')

features_df = features_df.drop('cluster', axis=1)

"""
Example of creating a radar chart (a.k.a. a spider or star chart) [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    def rand_jitter(arr):
        stdev = .01*(max(arr)-min(arr))
        return arr + np.random.randn(len(arr)) * stdev
        
    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def radar(df):
    theta = radar_factory(len(df.columns), frame='polygon')
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(2, 2, 1, projection='radar')
    for d, color in zip(df.itertuples(), sns.color_palette()):
        ax.plot(theta, d[1:], color=color, alpha=0.7)
        ax.fill(theta, d[1:], facecolor=color, alpha=0.5)
    ax.set_varlabels(df.columns)
    plt.rgrids([1, 2, 3, 4])
    legend = plt.legend(df.index, loc=(0.9, .95))

class RadarWidget(widgets.DOMWidget):
    _view_name = Unicode('RadarView').tag(sync=True)
    _view_module = Unicode('radar').tag(sync=True)
    factors_keys = List(['Aberfeldy']).tag(sync=True)
    
    def __init__(self, df, **kwargs):
        self.df = df
        super(RadarWidget, self).__init__(**kwargs)
        self._factors_keys_changed('', self.factors_keys, self.factors_keys)
    
    def _factors_keys_changed(self, name, old_value, new_value):
        clear_output(wait=True)
        return radar(self.df.loc[new_value])

get_ipython().run_cell_magic('javascript', '', 'require.undef(\'radar\');\n\ndefine(\'radar\', ["jupyter-js-widgets", "base/js/events"], function(widgets, events) {\n    var RadarView = widgets.DOMWidgetView.extend({\n        render: function() {\n            var that = this;\n            events.on(\'select.factors_keys\', function(event, data) {\n                if(data.factors_keys) {\n                    that.model.set(\'factors_keys\', data.factors_keys);\n                    that.touch();\n                }\n            });\n        }\n    });\n    return {\n        RadarView: RadarView\n    }\n});')

get_ipython().run_cell_magic('javascript', '', "$(document).off('click', 'a.scotch');\n$(document).on('click', 'a.scotch', function(event) {\n    var data = $(event.target).data();\n    IPython.notebook.events.trigger('select.factors_keys', data);\n});")

get_ipython().run_cell_magic('html', '', '<style>\ntable.dataframe {\n    width: 100%\n}\niframe.wiki {\n    width: 100%;\n    height: 400px;\n}\n</style>')

tmpl = jinja2.Template('''<p>If you like {{name}} you might want to try these five brands. Click one to see how its taste profile compares.</p>''')

def get_similar(name, n, top=True):
    a = sim_df[name].order(ascending=False)
    a.name = 'Similarity'
    df = pd.DataFrame(a) #.join(features_df).iloc[start:end]
    return df.head(n) if top else df.tail(n)

def on_pick_scotch(Scotch):
    name = Scotch
    # Get top 6 similar whiskeys, and remove this one
    top_df = get_similar(name, 6).iloc[1:]
    # Get bottom 5 similar whiskeys
#     bottom_df = get_similar(name, 5, False)
#     df = pd.concat([top_df, bottom_df])
    df = top_df
    
    # Make table index a set of links that the radar widget will watch
    df.index = ['''<a class="scotch" href="#" data-factors_keys='["{}","{}"]'>{}</a>'''.format(name, i, i) for i in df.index]
    
    prompt_w.value = tmpl.render(name=name)
    html = HTML(df.to_html(escape=False))
    js = Javascript("IPython.notebook.events.trigger('select.factors_keys', {factors_keys: ['%s']});" % name)
    
    return display(html, js)

prompt_w = widgets.HTML(value=tmpl.render(name='Aberfeldy'))
prompt_w

picker_w = widgets.interact(on_pick_scotch, Scotch=list(sim_df.index))

radar_w = RadarWidget(df=features_df)
radar_w

