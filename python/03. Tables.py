import numpy as np
import pandas as pd

from astropy import units as u
from astropy.table import Table

tab = Table(masked=True)

tab['name'] = ['source 1', 'source 2', 'source 3']
tab['flux'] = [1.2, 2.2, 3.1]
tab

tab['flux'].unit = u.F
tab

tab['name']

tab[0]

tab.meta = {'name': 'first table'}
tab.meta

tab.show_in_browser(jsviewer=True)

from astropy.table import join

tab2 = Table()
tab2['name'] = ['source 1', 'source 3']
tab2['flux2'] = [1,9]

tab3 = join(tab, tab2, join_type='outer')
tab3

df = tab.to_pandas()

t = Table.from_pandas(df)
t

