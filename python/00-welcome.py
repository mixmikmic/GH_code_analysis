from IPython.core import page
with open('../README.md', 'r') as f:
    page.page(f.read())

import holoviews as hv
hv.__version__

hv.extension('bokeh', 'matplotlib')

import bokeh
import matplotlib
import pandas
import geoviews
import holoext

import os
lines = ['import holoviews as hv', 'hv.extension.case_sensitive_completion=True',
         "hv.Dataset.datatype = ['dataframe']+hv.Dataset.datatype"]
print('\n'.join(lines))

rcpath = os.path.join(os.path.expanduser('~'), '.holoviews.rc')
if not os.path.isfile(rcpath):
    with open(rcpath, 'w') as f:
        f.write('\n'.join(lines))

