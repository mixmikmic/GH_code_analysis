import pandas as pd

df = pd.read_csv('/home/ubuntu/.jupyter/Notebooks/terrorsave.csv')
count_year_max_unbias = df.apply(lambda row: df[(df.iyear <= row.iyear) & (df.perpo_new == row.perpo_new)].current_year_count.max(), axis = 1)
df['count_year_max_unbias'] = count_year_max_unbias

def shift_temp(row, num):
    try:
        return df[(df.iyear == row.iyear - num) & (df.perpo_new == row.perpo_new)].avgtemp.iloc[0]
    except:
        return np.nan

df['tempone'] = df.apply(lambda x: shift_temp(x, 1), axis =1)
df['temptwo'] = df.apply(lambda x: shift_temp(x, 2), axis =1)
df['tempthree'] = df.apply(lambda x: shift_temp(x, 3), axis =1)
df['tempfour'] = df.apply(lambda x: shift_temp(x, 4), axis =1)

# df[['avgtemp','tempone','iyear','perpo_new']].sort(['perpo_new','iyear'])
print(df.tempone.isnull().sum())
print(df.temptwo.isnull().sum())
print(df.tempthree.isnull().sum())
print(df.tempfour.isnull().sum())

df['tempdiffone'] = df.avgtemp - df.tempone
df['tempdifftwo'] = df.avgtemp - df.temptwo
df['tempdiffthree'] = df.avgtemp - df.tempthree
df['tempdifffour'] = df.avgtemp - df.tempfour

def t_year(y, x):
    try:
        if y/5 <= x:
            return 1
        else:
            return 0
    except:
        return np.nan

def t_20(x):
    try:
        if 20 <= x:
            return 1
        else:
            return 0
    except:
        return np.nan

df['t1_year'] = df.apply(lambda row: t_year(row.count_year_max_unbias, row.diffone), axis = 1)
df['t2_year'] = df.apply(lambda row: t_year(row.count_year_max_unbias, row.difftwo), axis = 1)
df['t1_20'] = df.diffone.map(lambda x: t_20(x))
df['t2_20'] = df.difftwo.map(lambda x: t_20(x))

def t1_year_f(row, num):
    try:
        return df[(df.iyear == row.iyear + num) & (df.perpo_new == row.perpo_new)].t1_year.iloc[0]
    except:
        return np.nan
    
def t2_year_f(row, num):
    try:
        return df[(df.iyear == row.iyear + num) & (df.perpo_new == row.perpo_new)].t2_year.iloc[0]
    except:
        return np.nan
    
def t1_20_f(row, num):
    try:
        return df[(df.iyear == row.iyear + num) & (df.perpo_new == row.perpo_new)].t1_20.iloc[0]
    except:
        return np.nan

def t2_20_f(row, num):
    try:
        return df[(df.iyear == row.iyear + num) & (df.perpo_new == row.perpo_new)].t2_20.iloc[0]
    except:
        return np.nan

df['t1_year_f1'] = df.apply(lambda row: t1_year_f(row, 1), axis = 1)
df['t2_year_f1'] = df.apply(lambda row: t2_year_f(row, 1), axis = 1)

df['t1_year_f2'] = df.apply(lambda row: t1_year_f(row, 2), axis = 1)
df['t2_year_f2'] = df.apply(lambda row: t2_year_f(row, 2), axis = 1)

df['t1_year_f1'] = df.apply(lambda row: t1_year_f(row, 1), axis = 1)
df['t2_year_f1'] = df.apply(lambda row: t2_year_f(row, 1), axis = 1)
df['t1_20_f1'] = df.apply(lambda row: t1_20_f(row, 1), axis = 1)
df['t2_20_f1'] = df.apply(lambda row: t2_20_f(row, 1), axis = 1)

df['t1_year_f2'] = df.apply(lambda row: t1_year_f(row, 2), axis = 1)
df['t2_year_f2'] = df.apply(lambda row: t2_year_f(row, 2), axis = 1)
df['t1_20_f2'] = df.apply(lambda row: t1_20_f(row, 2), axis = 1)
df['t2_20_f2'] = df.apply(lambda row: t2_20_f(row, 2), axis = 1)

print(df.t1_year_f1.value_counts())
print(df.t2_year_f1.value_counts())
print(df.t1_20_f1.value_counts())
print(df.t2_20_f1.value_counts())
print(df.t1_year_f2.value_counts())
print(df.t2_year_f2.value_counts())
print(df.t1_20_f2.value_counts())
print(df.t2_20_f2.value_counts())

