import pandas as pd
import numpy as np 
import numpy as np
import holoviews as hv
from holoviews import streams
hv.extension('bokeh')

get_ipython().run_cell_magic('opts', "Points [tools=['box_select', 'lasso_select', 'tap']]", "\n# Declare two sets of points generated from multivariate distribution\npoints = hv.Points(np.random.multivariate_normal((0, 0), [[1, 0.1], [0.1, 1]], (1000,)))\npoints2 = hv.Points(np.random.multivariate_normal((3, 3), [[1, 0.1], [0.1, 1]], (1000,)))\n\n# Declare two selection streams and set points and points2 as the source of each\nsel1 = streams.Selection1D(source=points)\nsel2 = streams.Selection1D(source=points2)\n\n# Declare DynamicMaps to show mean y-value of selection as HLine\nhline1 = hv.DynamicMap(lambda index: hv.HLine(points['y'][index].mean() if index else -10), streams=[sel1])\nhline2 = hv.DynamicMap(lambda index: hv.HLine(points2['y'][index].mean() if index else -10), streams=[sel2])\n\ndef selcallback(index):\n    strindex = index.__repr__()\n    return hv.Div(strindex)\n\ndiv = hv.DynamicMap(selcallback, streams=[sel2])\n\n# Combine points and dynamic HLines\npoints * points2 * hline1 * hline2 << div")

#df = pd.read_csv("/Users/volkerhilsenstein/Dropbox/F508delplate7.csv")
df = pd.read_csv("df.csv")
df.keys()

get_ipython().run_cell_magic('opts', "Scatter[tools=['box_select', 'lasso_select', 'tap']]", 'scat = hv.Scatter(df, kdims=["value1", "value2"])\nsel  = streams.Selection1D(source=scat)\n\ndef selcallback(index):\n    \n    strindex = ""\n    for f in df["giffile"][index]:\n        strindex += f\'<img src="{f}" width=50 height=50>\'\n    return hv.Div(strindex)\n\ndiv = hv.DynamicMap(selcallback, streams=[sel])\n\nscat << div')



