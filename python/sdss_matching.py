import numpy as np
from astropy.table import Table
from astropy import coordinates as coords
from astropy import units as u

from astroquery.sdss import SDSS

# Take a list of targets and search for them in the SDSS database. Add the spectroscopic IDs to the table.
def sdssquery(targets):
    specobjid = []
    for pos in targets['coords']:
        try: 
            xid = SDSS.query_region(pos,data_release=12,spectro=True,radius=10*u.arcsec)
            specobjid += [xid['specobjid'][0]]
        except:
            print("Not found: ",pos)
    targets['specobjid'] = specobjid
    return targets

# Create a URL to the Explorer from a spectroscopic ID.
def sdss_url(specobjid):
    urlbase = "http://skyserver.sdss.org/dr12/en/tools/explore/summary.aspx?"
    url = urlbase+"sid=%d" % (specobjid)
    href = '<a href=%s> sdss </a>' % url
    return href

# Run the two functions defined above for all the targets
def sdss(targets):
    targets = sdssquery(targets)
    targets['url'] = [sdss_url(sid) for sid in zip(targets['specobjid'])]
    return targets

targets = Table()
targets['ID'] = [133,229,222]
targets['RA'] = [158.43144,195.36813,192.14461]
targets['Dec'] = [63.888338,51.080956,12.567532]
targets['coords']=coords.SkyCoord(targets['RA'],targets['Dec'],unit='deg',frame='icrs')
targets

newtable = sdss(targets)
ipy_html = newtable.show_in_notebook()
# the show_in_notebook format has changed the HTML brackets to &lt and &gt; we need to change them back
ipy_html.data = ipy_html.data.replace('&lt;','<')
ipy_html.data = ipy_html.data.replace('&gt;','>')
ipy_html

