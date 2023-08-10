get_ipython().run_line_magic('matplotlib', 'inline')

import urllib

import pandas as pd

data = ('messw_beg=01.01.2016&messw_end=31.12.2016&'
        'felder[]=Temp2m&felder[]=TempWasser&felder[]=Windchill&'
        'felder[]=LuftdruckQFE&felder[]=Regen&felder[]=Taupunkt&'
        'felder[]=Strahlung&felder[]=Feuchte&felder[]=Pegel&'
        'auswahl=2&combilog=mythenquai&suchen=Werte anzeigen')
data = data.encode('ascii')

req = urllib.request.Request(
    'https://www.tecson-data.ch/zurich/mythenquai/uebersicht/messwerte.php',
    method='POST',
    data=data,
    headers={"Content-Type": "application/x-www-form-urlencoded",
             'User-Agent': 'http://github.com/wildtreetech/explore-open-data'
            },
    )

with urllib.request.urlopen(req) as web:
    with open('weather-2016.html', 'w') as local:
        local.write(web.read().decode('iso-8859-1'))

df = pd.read_html('weather-2016.html',
                  # a little hint to find the data in the file
                  attrs={'border': '1'},
                  # skip the first row of the data, it is junk
                  skiprows=1,
                  # convert all dates we find into datetime objects
                  parse_dates=True,
                  # and use the first column as the index, this corresponds to the date
                  index_col=0)
df = df[0]

df.head()

df.columns = ['Temp', 'WaterTemp', 'Windchill', 'Pressure', 'Rain',
              'Dewpoint', 'Radiation', 'Humidity', 'Waterlevel']
df.head()

temperature = df['Temp']
temperature.resample('W').mean().plot();

temperature[:"20160115"].plot();

pd.to_datetime('01.08.2016')

pd.to_datetime('01.08.2016', dayfirst=True)

df = pd.read_html('weather-2016.html',
                  attrs={'border': '1'},
                  skiprows=1)
df = df[0]

df[0] = pd.to_datetime(df[0], dayfirst=True)

df.columns = ['Date', 'Temp', 'WaterTemp', 'Windchill', 'Pressure', 'Rain',
              'Dewpoint', 'Radiation', 'Humidity', 'Waterlevel']
df = df.set_index('Date')

df.head(5)

temperature = df['Temp']
temperature[:"20160115"].plot()

temperature.resample('W').mean().plot();

