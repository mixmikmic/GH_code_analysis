get_ipython().magic('pylab --no-import-all')
from os import path
import pandas as pd

try:
    file = path.join("..", "data", "raw", "london.csv")
except OSError:
    print("This repository does not host the data. "
          "Put the csv in ../data/raw/")
    raise
df = pd.read_csv(file, na_values=0)

df.rename(columns={'DONNA': 'ID',
                   'P_SPEZZ': 'SEGMENT_ID',
                   'P_CICLO': 'CYCLE_ID',
                   'ANNO_NAS': 'BIRTH_YR',
                   'DATA': 'BEGIN_DATE',
                   'T_SPEZZ': 'N_SEGMENTS',
                   'T_CICLI': 'N_CYCLES',
                   'QUALIFI': 'DESC',
                   'TIPOTEMP': 'TEMP_SCALE',
                   'L_CICLO': 'L_CYCLE',
                   'L_PREOV': 'L_PREOVULATION',
                   'L_PERIOD': 'L_PERIOD',
                   'FIGLI': 'CHILDREN'
                  },
         inplace=True)

df.dropna(subset=['L_PREOVULATION'], inplace=True)

df = df[df.DESC == 1]
df.drop('DESC', 1, inplace=True)

FAHR = 1
CELS = 2
for i in range (1,100):
    df.loc[df.TEMP_SCALE == FAHR, 'TEMP' + str(i)] =  90 + df.loc[df.TEMP_SCALE == FAHR, 'TEMP' + str(i)]/10
    df.loc[df.TEMP_SCALE == CELS, 'TEMP' + str(i)] =  30 + df.loc[df.TEMP_SCALE == CELS , 'TEMP' + str(i)]/10
    #Convert celsius temps to fahrenheit
    df.loc[df.TEMP_SCALE == CELS, 'TEMP' + str(i)] =  32 + ((9/5) *  df.loc[df.TEMP_SCALE == CELS , 'TEMP' + str(i)])

#Display medians one at a time    
#df[df.TEMP_SCALE==FAHR].median()
#df[df.TEMP_SCALE==CELS].median()

#NEED TO FIX SO THAT WE HAVE ONE AGE PER ID NOT CYCLE
df['AGE'] = (df.BEGIN_DATE.apply(lambda s: int(s.split('/')[-1]) % 100 if isinstance(s, str) else s) - df.BIRTH_YR)

import errno
import os

destination = ["..", "data", "interim"]
df.to_csv(path.join(*destination, "df.csv"))

