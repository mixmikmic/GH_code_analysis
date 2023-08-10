# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os

# Plot settings
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
fontsize = 20 # size for x and y ticks
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams.update({'font.size': fontsize})

# Connect to the database
fn = os.path.join('data','eicu_demo.sqlite3')
con = sqlite3.connect(fn)
cur = con.cursor()

query = """
SELECT patientunitstayid
FROM patient
"""

df = pd.read_sql_query(query,con)

query = """
SELECT pasthistorypath, count(*) as n
FROM pasthistory
WHERE pasthistorypath LIKE '%Renal  (R)%'
GROUP BY pasthistorypath
ORDER BY n DESC;
"""

ph = pd.read_sql_query(query,con)

for row in ph.iterrows():
    r = row[1]
    print('{:3g} - {:20s}'.format(r['n'],r['pasthistorypath'][48:]))

# identify patients with insufficiency

query = """
SELECT DISTINCT patientunitstayid
FROM pasthistory
WHERE pasthistorypath LIKE '%Renal  (R)%'
"""

df_have_crf = pd.read_sql_query(query,con)
df_have_crf['crf'] = 1

# merge the above data into our original dataframe

df = df.merge(df_have_crf, how='left',
              left_on='patientunitstayid', right_on='patientunitstayid')
df.head()

# impute 0s for the missing CRF values
df.fillna(value=0,inplace=True)
df.head()

# set patientunitstayid as the index - convenient for indexing later
df.set_index('patientunitstayid',inplace=True)

query = """
SELECT patientunitstayid, labresult
FROM lab
WHERE labname = 'creatinine'
"""

lab = pd.read_sql_query(query, con)

# set patientunitstayid as the index
lab.set_index('patientunitstayid',inplace=True)

# get first creatinine by grouping by the index (level=0)
cr_first = lab.groupby(level=0).first()

# similarly get maximum creatinine
cr_max = lab.groupby(level=0).max()

plt.figure(figsize=[10,6])

xi = np.arange(0,10,0.1)

# get patients who had CRF and plot a histogram
idx = df.loc[df['crf']==1,:].index
plt.hist( cr_first.loc[idx,'labresult'].dropna(), bins=xi, label='With CRF' )

# get patients who did not have CRF
idx = df.loc[df['crf']==0,:].index
plt.hist( cr_first.loc[idx,'labresult'].dropna(), alpha=0.5, bins=xi, label='No CRF' )

plt.legend()

plt.show()

plt.figure(figsize=[10,6])

xi = np.arange(0,10,0.1)

# get patients who had CRF and plot a histogram
idx = df.loc[df['crf']==1,:].index
plt.hist( cr_first.loc[idx,'labresult'].dropna(), bins=xi, normed=True,
         label='With CRF' )

# get patients who did not have CRF
idx = df.loc[df['crf']==0,:].index
plt.hist( cr_first.loc[idx,'labresult'].dropna(), alpha=0.5, bins=xi, normed=True,
         label='No CRF' )

plt.legend()

plt.show()

plt.figure(figsize=[10,6])

xi = np.arange(0,10,0.1)

# get patients who had CRF and plot a histogram
idx = df.loc[df['crf']==1,:].index
plt.hist( cr_max.loc[idx,'labresult'].dropna(), bins=xi, normed=True,
         label='With CRF' )

# get patients who did not have CRF
idx = df.loc[df['crf']==0,:].index
plt.hist( cr_max.loc[idx,'labresult'].dropna(), alpha=0.5, bins=xi, normed=True,
         label='No CRF' )

plt.legend()

plt.show()

