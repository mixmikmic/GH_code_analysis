import pandas as pd

# Adapted from http://holoext.readthedocs.io/en/latest/examples/gallery.html
DATA_URL_FMT = (
    'http://mesonet.agron.iastate.edu/'
    'cgi-bin/request/daily.py?'
    'network=IL_ASOS&stations={0}&'
    'year1=2000&month1=1&day1=1&year2=2018&month2=1&day2=1'
)
STATIONS = ['CMI', 'DEC', 'MDW', 'ORD', 'BMI']

df_list = []
for station in STATIONS:
    data_url = DATA_URL_FMT.format(station)
    df = pd.read_csv(data_url, index_col='day', parse_dates=True)
    df.iloc[:, 1:] = df[df.columns[1:]].apply(
        pd.to_numeric, errors='coerce').fillna(0)
    df.index.name = 'date'
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['precip_cumsum_in'] = df['precip_in'].cumsum()
    df = df.reset_index()  # 7-day rolling average
    df_list.append(df)
df = pd.concat(df_list)

df.head()

df.tail()

df.describe()

df.info()

df.to_parquet('datasets/weather_station_data.parquet')

