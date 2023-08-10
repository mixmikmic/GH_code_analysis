import pandas as pd
import numpy as np

df_ebola = pd.read_csv("data/in/ebola-outbreaks-before-2014.csv")

df_ebola.head()

start, end = df_ebola["Start date"][0], df_ebola["End date"][0]

abs(pd.to_datetime(start) - pd.to_datetime(end))

start_datetime, end_datetime = [], []
for i in range(len(df_ebola)):
    start_datetime.append(pd.to_datetime(df_ebola["Start date"][i]))
    end_datetime.append(pd.to_datetime(df_ebola["End date"][i]))

print start_datetime[:2]

df_ebola.insert(0, "Start_Datetime", start_datetime)
df_ebola.insert(2, "End_Datetime", end_datetime)

df_ebola.head()

duration = []
for i in range(len(df_ebola)):
    duration.append(abs(df_ebola["Start_Datetime"][i] - df_ebola["End_Datetime"][i]))
    
print duration[:3]

#print duration[0]

## Get the number of days from thee "duration" time deltas:
duration_d = [x / np.timedelta64(1, 'D') for x in duration]

#print duration_d[0]

df_ebola.insert(4, "Duration (days)", duration_d)

df_ebola.head()

df_ebola.to_csv("data/out/ebola-outbreaks-before-2014-dates.csv")



