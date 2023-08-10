clears = []
clears.append( {"hash": "7KUBIUXQYGNTAS4J6AEG7BNCSQNFPQZN", "names": ["visit.gif"], "total": 406 } )
clears.append( {"hash": "D2VQOLYTNWE6VT3MMPGIQANFPXGIS4SW", "names": ["c.gif"], "total": 814 } )
clears.append( {"hash": "FWXKVC27DHYLYIE5S5WAFPLKZNI3ACYK", "names": ["n.gif", "spaceball.gif"], "total": 180895 } )
clears.append( {"hash": "GF2JNIEW23EGJBVHDVCSDGKLZULRU25T", "names": ["clear.gif"], "total": 16949 } )
clears.append( {"hash": "GKHEOJZBVEZULAA62VJTEQHKYLI7QSMM", "names": ["dot_clear.gif","pixel.gif"], "total": 203802 } )
clears.append( {"hash": "K3KF7CQX6UDYUIFPTFRMTEWKIZ4EKB3F", "names": ["cleardot.gif", "cleardot.gif"], "total": 2365898 } )
clears.append( {"hash": "TUA4YXOI4BBMBVFNNT5YWOWDR2CKL347", "names": ["blank.gif"], "total": 29252 } )

extra = {"hash": "EXMAIMZW5N4Z4UVRUDQV75VZLYE4ETRV", "names": ["ANJcron.php.gif"], "total": 196746 } 

print(clears[5])

def json_name(k,y):
    return "json/%s-for-%s.json" % (k,y)

import json, sys, codecs, hashlib
import urllib, datetime, re
from pprint import pprint
    
#
#urlo = urllib.FancyURLopener({"http":"http://explorer.bl.uk:3127"})
#
urlo=urllib.URLopener()

q = "http://192.168.1.181:8983/solr/jisc5/select?q=hash%%3A%%22sha1%%3A%s%%22&fq=crawl_years%%3A%s&rows=100&wt=json&indent=true&facet=true&facet.field=domain&facet.mincount=1&sort=crawl_date+asc"

for c in clears:
    for y in range(1996,2011):
        k = c['hash']
        yq = q % ( k, y )
        print("GET %s %s - %s" % (k,y, yq) )
        # Currently disabled as the data has been downloaded already:
        #urlo.retrieve(yq , json_name(k,y) )

print("DONE")

import json
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#%pylab inline

# Loop over the items and years:
td = {}
for c in clears:
    k = c['hash']
    n = c['names']
    values = []
    years = []
    for y in range(1996,2011):
        fn = json_name(k,y)
        #print("PARSE %s %s - %s" % (k,y, f) )
        with open( fn ) as data_file:
            data = json.load(data_file)
            years.append(y)
            values.append(data['response']['numFound'])
            #pprint(data['facet_counts']['facet_fields']['domain'][0:10])
    # And add:
    td[", ".join(n)] = pd.Series(values,index=years)

df = pd.DataFrame(td)       
print(df)

# Update the matplotlib configuration parameters:
matplotlib.rcParams.update({'font.size': 16, 'font.family': 'STIXGeneral', 
                            'mathtext.fontset': 'stix', 'axes.titlesize': 'medium' })

# Plot:
axs = df.plot(kind='bar', subplots=True, figsize=(16,20), legend=False, sharex=False)

# No border on the legend, please:
#leg = plt.legend(loc="best")
#leg.get_frame().set_linewidth(0.0)

# Use a logarithmic Y-axis (doesn't look great):
#ax.set_yscale("log")

# Dark Magic to get commas to show up in the integers on the Y axis:
for ax in axs:
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))


axs = df.plot(kind='bar', subplots=True, figsize=(16,20), legend=False, sharex=False)

# No border on the legend, please:
#leg = plt.legend(loc="best")
#leg.get_frame().set_linewidth(0.0)

# Use a logarithmic Y-axis (doesn't look great):

# Dark Magic to get commas to show up in the integers on the Y axis:
for ax in axs:
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_yscale("log")



