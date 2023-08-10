import os, sys
import requests
import pandas as pd
import numpy as np
from fbprophet import Prophet
# needed for display in notebook
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import pyplot as plt

PROTOCOL        = "http"
API_LANG        = "en"
API_FMT         = "json"
API_DOMAIN      = 'ec.europa.eu/eurostat/wdds'
API_VERS        = 2.1
API_URL         = "{}://{}/rest/data/v{}/{}/{}".format(
                  PROTOCOL, API_DOMAIN, API_VERS, API_FMT, API_LANG
                  )
print(API_URL)

GEO             = "EU28"
# TIME : all
INDICATOR       = (u'tour_occ_nim', "Tour accomodation")
UNIT            = (u'NR', "Number of nights")
NACE_R2         = (u'I551', "Hotels; holiday and other short-stay accommodation...")
INDIC_TO        = (u'B006', "Nights spent, total")

url             = "{}/{}?geo={}&unit={}&nace_r2={}&indic_to={}".format(
                  API_URL, INDICATOR[0], GEO, UNIT[0], NACE_R2[0], INDIC_TO[0])
print(url)

session = requests.session()
try:
    response = session.head(url)
    response.raise_for_status()
except:
    raise IOError("ERROR: wrong request formulated")  
else:
    print ("OK: status={}".format(response.status_code))
    
try:    
    response = session.get(url)
except:
    raise IOError('error retrieveing response from URL')

resp = response.json()
lbl2idx = resp['dimension']['time']['category']['index']
idx2lbl = {v:k for (k,v) in lbl2idx.items()}
data = resp['value']
data = {idx2lbl[int(k)]:v for (k,v) in data.items()}
table = {k.replace('M','-'):v for (k,v) in data.items()}

df = pd.DataFrame(list(table.items()), columns=['ds','y'])
# df = df[df['y'].notnull()] # Prophet can deal with NaN data!

df.sort_values('ds', inplace=True)
ds_last = df['ds'].values[-1] # we keep that for later

df.head()

df.tail()

df['ds'] = pd.to_datetime(df['ds']) # note that this will add a day date
df.tail()

xlabel = "Time"
ylabel = "{} : {} - {}".format(INDICATOR[0], INDIC_TO[1], GEO)  
plt.plot(df['ds'], df['y'], 'k.')
plt.plot(df['ds'], df['y'], ls='-', c='#0072B2')
plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
plt.xlabel(xlabel, fontsize=10); plt.ylabel(ylabel, fontsize=10)
plt.title("Historical data (last: {})".format(ds_last), fontsize=16)
plt.show()

m = Prophet(growth = "linear", yearly_seasonality=True, weekly_seasonality=False)

m.fit(df)

nyears = 5
future = m.make_future_dataframe(periods=12*nyears, freq='M')
fcst = m.predict(future)

fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(fcst, uncertainty=True) 
plt.axvline(pd.to_datetime(ds_last), color='r', linestyle='--', lw=2)
plt.xlabel(xlabel, fontsize=10); plt.ylabel(ylabel, fontsize=10)
plt.title("Forecast data ({} years)".format(nyears), fontsize=16)

m.plot_components(fcst, uncertainty=True);
# plt.title("Forecast components", fontsize=16)

