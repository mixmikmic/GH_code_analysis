import pandas as pd
from sqlalchemy import create_engine # database connection
import datetime as dt
import csv # read csv files, dont write your own parser!
from pathlib import Path
import os
import string

import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10


def process_files(in_dir):
    in_path = Path(in_dir)
    for nfile, file in enumerate(in_path.iterdir()):
        if file.name.endswith('.csv'):
            with file.open('r') as f:
                csv_reader = csv.DictReader(f)
                header = next(csv_reader)
                fields = header[' Site Name'].split(';')
                header['Site Name'] = fields[0]
                header['GPS Ref'] = fields[1]

                df = pd.read_csv(str(file), skiprows=3)
                df['LegacyID'] = int(header['Legacy ID'])
                if nfile % 20 == 0:
                    print('Done %d files' % nfile)
                yield df
                

#disk_engine = create_engine('sqlite:///traffic.db')
disk_engine = create_engine('mysql+mysqldb://root:%s@localhost/traffic' % os.environ['MYSQL_PASSWORD'])

not_done_index = True
for df in process_files('c:/dev/lse/midas'):
    df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})
    df['LocalDateTime'] = pd.to_datetime(df['LocalTime']+' '+df['LocalDate'], format='%H:%M:%S %d/%m/%Y')
    columns = ['LegacyID','LocalDateTime','TotalCarriagewayFlow','SpeedValue']
    for c in df.columns:
        if c not in columns:
            df = df.drop(c, axis=1)
    df['hour'] = df['LocalDateTime'].dt.hour
    df.to_sql('data', disk_engine, if_exists='append')
    if not_done_index:
        disk_engine.execute('create index idx_hour on data (hour)')
        disk_engine.execute('create index idx_lid on data (LegacyID)')
        not_done_index = False
    

df = pd.read_sql_query('SELECT LegacyID, TotalCarriagewayFlow, LocalDateTime, hour '
                       'FROM data '
                       'WHERE hour > 22 or hour < 5', disk_engine)
df.head()

df2 = df.set_index(['LegacyID', 'LocalDateTime'])
df2['TotalCarriagewayFlow'].plot()

df3 = pd.read_sql_query('''SELECT LegacyID, TotalCarriagewayFlow, LocalDateTime, hour
                       FROM data 
                       WHERE LocalDateTime between %(dt1)s and %(dt2)s  and LegacyID=%(id)s''', disk_engine,
                       params={'dt1': dt.datetime(2014,1,1), 'dt2': dt.datetime(2014,1,31), 'id': 120})

df3 = df3.set_index(['LocalDateTime'])

df3.info()

plt.scatter(df3['hour'],df3['TotalCarriagewayFlow'])


df3_byhour = df3.groupby(['hour'])
df3_byhour['TotalCarriagewayFlow'].mean().plot()

df4 = pd.read_sql_query('SELECT LegacyID, TotalCarriagewayFlow, LocalDateTime, hour, '
                        'DAYOFYEAR(LocalDateTime) %% 29  as lunar_day '
                       'FROM data '
                       'WHERE (hour > %(h1)s OR hour < %(h2)s) AND LegacyID=%(id)s', disk_engine,
                       params={'h1': 24, 'h2': 5, 'id': 120})

df4.info()

plt.scatter(df4['lunar_day'],df4['TotalCarriagewayFlow'])

df4_by = df4.groupby(['lunar_day'])

df4_by['TotalCarriagewayFlow'].mean().plot()

df5 = pd.read_sql_query('SELECT LegacyID, TotalCarriagewayFlow, LocalDateTime, hour, '
                        'DAYOFYEAR(LocalDateTime) %% 29  as lunar_day '
                       'FROM data '
                       'WHERE (hour > %(h1)s OR hour < %(h2)s) AND LegacyID=%(id)s', disk_engine,
                       params={'h1': 24, 'h2': 5, 'id': 101})

df6 = pd.read_sql_query('SELECT LegacyID, TotalCarriagewayFlow, LocalDateTime, hour, '
                        'DAYOFYEAR(LocalDateTime) %% 29  as lunar_day '
                       'FROM data '
                       'WHERE (hour > %(h1)s OR hour < %(h2)s) AND LegacyID=%(id)s', disk_engine,
                       params={'h1': 24, 'h2': 5, 'id': 1030}).groupby(['lunar_day'])

df5_by=df5.groupby(['lunar_day'])

df4_by['TotalCarriagewayFlow'].mean().plot()
df5_by['TotalCarriagewayFlow'].mean().plot()
df6['TotalCarriagewayFlow'].mean().plot()

