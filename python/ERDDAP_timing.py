get_ipython().magic('matplotlib inline')
import urllib, json
import pandas as pd

url = 'http://www.neracoos.org/erddap/tabledap/B01_accelerometer_all.csv?time,significant_wave_height&time>"now-7days"'
print(url)

get_ipython().run_cell_magic('timeit', '', "df_sb = pd.read_csv(url,index_col='time',parse_dates=True,skiprows=[1])  # skip the units row ")

df_sb = pd.read_csv(url,index_col='time',parse_dates=True,skiprows=[1])  # skip the units row 
df_sb.plot(figsize=(12,4),grid='on');

url = 'http://www.neracoos.org/erddap/tabledap/B01_accelerometer_all.csv?time,significant_wave_height&time>"now-365days"'
print(url)

get_ipython().run_cell_magic('timeit', '', "df_sb = pd.read_csv(url,index_col='time',parse_dates=True,skiprows=[1])  # skip the units row ")

df_sb = pd.read_csv(url,index_col='time',parse_dates=True,skiprows=[1])  # skip the units row 
df_sb.plot(figsize=(12,4),grid='on');

url = 'http://www.neracoos.org/erddap/tabledap/B01_accelerometer_all.json?time,significant_wave_height&time>"now-7days"'
print(url)

get_ipython().run_cell_magic('timeit', '', 'response = urllib.urlopen(url)\ndata = json.loads(response.read())')

response = urllib.urlopen(url)
data = json.loads(response.read())

data

