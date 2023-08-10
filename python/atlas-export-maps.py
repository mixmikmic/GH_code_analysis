import pandas as pd
import sys
sys.path.append("./modules")
import vistk

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')

def generate_map(name):
    df = pd.read_json("sourcedata/%s_partners_exports_2012.json" % (name))
    year = 2013
    geomap = vistk.Geomap(id='name', color='sum_export', color_range=['#F2ECB0', '#E64B22'], name='name', year=year, title='Export partners for %s (%s)' % (name, year))
    geomap.draw(df)

countries = ['usa', 'bra', 'mex', 'chn', 'tur', 'rus', 'jpn', 'ind']

[generate_map(c) for c in countries];

