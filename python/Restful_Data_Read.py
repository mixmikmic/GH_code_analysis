from DataDL import download_source as dld

url = "http://www2.census.gov/acs2013_1yr/pums/csv_hny.zip"
files = dld(url, pth='Trial/', zp=True)

print files

get_ipython().magic('pinfo dld')

import os

for fil in files:
    os.remove(fil)
    os.removedirs(fil)

