import pandas as pd
import math

df = pd.read_csv("volcan-tsunami-samecountry-8days.csv", na_values=[""])
cdf = pd.read_csv("countries.csv", sep=";")

tsevent = pd.read_csv("tsevent.csv", sep=";", na_values=[""])
volerup = pd.read_csv("volerup.csv", sep=";", na_values=[""])

results = pd.DataFrame(columns=('ID', 'LATITUDE', 'LONGITUDE', 'GROUP'))
for row in df.itertuples():
    id = row.ID
    group = row.GROUP
    if id and id[0] == 'T':
        f = tsevent.query("ID==@id")
    else:
        f = volerup.query("ID==@id")
    for frow in f.itertuples():
        results.loc[len(results)] = [frow.ID, frow.LATITUDE, frow.LONGITUDE, group]
results

import json
import numpy
import locale

array = []
locale.setlocale(locale.LC_NUMERIC, 'English_Canada.1252')
for row in results.itertuples():
    if not isinstance(row.LATITUDE, float):
        latitude = float(row.LATITUDE.replace(',', '.'))
    else:
        latitude = row.LATITUDE
        
    if not isinstance(row.LONGITUDE, float):
        longitude = float(row.LONGITUDE.replace(',', '.'))
    else:
        longitude = row.LONGITUDE
    color = "ff8888" if row.ID[0] =='V' else "8888ff"
    if numpy.isnan(latitude) or numpy.isnan(longitude):
        array.append([0, 0, color])
    else:
        array.append([latitude, longitude, color])
json.dumps(array)

import json
import numpy
import locale

array = []
locale.setlocale(locale.LC_NUMERIC, 'English_Canada.1252')
count = 0
for row in results.itertuples():
    # if count >= 9:
    #     break
    count+=1
    
    if not isinstance(row.LATITUDE, float):
        latitude = float(row.LATITUDE.replace(',', '.'))
    else:
        latitude = row.LATITUDE

    if not isinstance(row.LONGITUDE, float):
        longitude = float(row.LONGITUDE.replace(',', '.'))
    else:
        longitude = row.LONGITUDE
        
    if row.ID[0] == 'V':
        vlat = latitude
        vlong = longitude
        
    color = "ff8888" if row.ID[0] == 'V' else "8888ff"
    if numpy.isnan(latitude) or numpy.isnan(longitude):
        array.append([0, 0, 0, 0, color])
    else:
        array.append([vlat, vlong, latitude, longitude, color])

json.dumps(array)

import json
import numpy
import locale

array = []
locale.setlocale(locale.LC_NUMERIC, 'English_Canada.1252')
count = 0
for row in results.itertuples():
    # if count >= 9:
    #     break
    count+=1
    
    if not isinstance(row.LATITUDE, float):
        latitude = float(row.LATITUDE.replace(',', '.'))
    else:
        latitude = row.LATITUDE

    if not isinstance(row.LONGITUDE, float):
        longitude = float(row.LONGITUDE.replace(',', '.'))
    else:
        longitude = row.LONGITUDE
        
    if row.ID[0] == 'V':
        vlat = latitude
        vlong = longitude
        
    color = "ff8888" if row.ID[0] == 'V' else "8888ff"
    if numpy.isnan(latitude) or numpy.isnan(longitude):
        array.append([0, 0, 0, 0, color])
    else:
        array.append([vlat, vlong, latitude, longitude, color])

json.dumps(array)



