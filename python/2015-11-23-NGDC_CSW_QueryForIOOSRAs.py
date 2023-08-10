"""
The original notebook is NGDC_CSW_QueryForIOOSRAs.ipynb

Created by Emilio Mayorga, 2/10/2014
"""

title = 'Catalog based search for the IOOS Regional Associations acronyms'
name = '2015-11-23-NGDC_CSW_QueryForIOOSRAs'

get_ipython().magic('matplotlib inline')
import seaborn
seaborn.set(style='ticks')

import os
from datetime import datetime
from IPython.core.display import HTML

import warnings
warnings.simplefilter("ignore")

# Metadata and markdown generation.
hour = datetime.utcnow().strftime('%H:%M')
comments = "true"

date = '-'.join(name.split('-')[:3])
slug = '-'.join(name.split('-')[3:])

metadata = dict(title=title,
                date=date,
                hour=hour,
                comments=comments,
                slug=slug,
                name=name)

markdown = """Title: {title}
date:  {date} {hour}
comments: {comments}
slug: {slug}

{{% notebook {name}.ipynb cells[2:] %}}
""".format(**metadata)

content = os.path.abspath(os.path.join(os.getcwd(), os.pardir,
                                       os.pardir, '{}.md'.format(name)))

with open('{}'.format(content), 'w') as f:
    f.writelines(markdown)


html = """
<small>
<p> This post was written as an IPython notebook.  It is available for
<a href="http://ioos.github.com/system-test/downloads/
notebooks/%s.ipynb">download</a>.  You can also try an interactive version on
<a href="http://mybinder.org/repo/ioos/system-test/">binder</a>.</p>
<p></p>
""" % (name)

from owslib.csw import CatalogueServiceWeb

endpoint = 'http://www.ngdc.noaa.gov/geoportal/csw'
csw = CatalogueServiceWeb(endpoint, timeout=30)

ioos_ras = ['AOOS',      # Alaska
            'CaRA',      # Caribbean
            'CeNCOOS',   # Central and Northern California
            'GCOOS',     # Gulf of Mexico
            'GLOS',      # Great Lakes
            'MARACOOS',  # Mid-Atlantic
            'NANOOS',    # Pacific Northwest 
            'NERACOOS',  # Northeast Atlantic 
            'PacIOOS',   # Pacific Islands 
            'SCCOOS',    # Southern California
            'SECOORA']   # Southeast Atlantic

from owslib.fes import PropertyIsEqualTo

def query_ra(csw, ra='SECOORA'):
    q = PropertyIsEqualTo(propertyname='apiso:Keywords', literal=ra)
    csw.getrecords2(constraints=[q], maxrecords=100, esn='full')
    return csw

for ra in ioos_ras:
    csw = query_ra(csw, ra)
    ret = csw.results['returned']
    word = 'records' if ret > 1 else 'record'
    print("{0:>8} has {1:>3} {2}".format(ra, ret, word))
    csw.records.clear()

csw = query_ra(csw, 'SECOORA')
key = csw.records.keys()[0]

print(key)

station = csw.records[key]

station.type, station.title, station.modified

station.subjects

print(station.xml)

HTML(html)

