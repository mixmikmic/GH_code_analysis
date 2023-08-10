from ipywidgets import interact, IntSlider
import ipywidgets as widgets

import pandas as pd
import krisk.plot as kk
# Use this when you want to nbconvert the notebook (used by nbviewer)
from krisk import init_notebook; init_notebook()

df = pd.read_csv('../krisk/tests/data/gapminderDataFiveYear.txt',sep='\t')

p = kk.bar(df[df.year == 1952],'continent',y='pop', how='mean')
p.set_size(width=800)

p.resync_data(df[df.year == 2007])

def resync(year):
    return p.resync_data(df[df.year == year])
interact(resync,year=IntSlider(min=df.year.min(),max=df.year.max(),step=5,value=1952))

p.replot(kk.line(df,'continent'))

p.read_df(df)

