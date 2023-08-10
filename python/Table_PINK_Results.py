import pandas as pd
import numpy as np
import glob
import os
pd.options.display.float_format = '{:,.2f}'.format

mode = '../Results/Segments'
mode = '../Results/Fraction'
mode = '../Results/initial_results'

files = glob.glob(f'{mode}/*/*json')
dfs   = [pd.read_json(i) for i in files] 
print(len(dfs))

df = pd.concat(dfs)
df = df.reset_index(drop=True)

# Removes lists from the data, as this can kill the pivot

def transform(x):
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return str(x)
    else:
        return '_'.join([str(i) for i in x])

df['log'] = df['log'].apply(transform)
df.rename(columns={'som-width':'width', 'som-height':'height',
                    '1_1_accuracy':'1_1', '1_2_accuracy':'1_2', '1_3_accuracy':'1_3',
                    '2_2_accuracy':'2_2', '2_3_accuracy':'2_3', 
                    '3_3_accuracy':'3_3',
                    'total_accuracy':'total'}, inplace=True)
df['WISE'] = df['validate_path'].str.contains('WISE_W1')

mask = df['WISE'] == False
mdf = pd.pivot_table(df[mask], index=['WISE' ,'norm','sigma','log','width','height', 'convex'],
                     values=['total', '1_1', '1_2', '1_3', '2_2', '2_3', '3_3'],
                     fill_value=0)

mdf.to_latex('Tables/No_WISE_pivot.tex', multirow=True, bold_rows=True)
mdf

mdf = pd.pivot_table(df[~mask], index=['WISE' ,'norm','sigma','log','convex','width','height'],
                     values=['total', '1_1', '1_2', '1_3', '2_2', '2_3', '3_3'],
                     fill_value=0)

mdf.to_latex('Tables/WISE_pivot.tex', multirow=True, bold_rows=True)
mdf

mdf = pd.pivot_table(df, index=['WISE' ,'norm','sigma','log','convex','width','height'],
                     values=['total', '1_1', '1_2', '1_3', '2_2', '2_3', '3_3'],
                     fill_value=0)

mdf.to_latex('Tables/Complete_pivot.tex', multirow=True, bold_rows=True)
mdf

df.columns.tolist()

def reduce_funct(v):
    return f'{np.mean(v):.2f}, {np.std(v):.2f}'
    return f'{np.mean(v):.2f}, {np.min(v):.2f}, {np.max(v):.2f}'
    return (np.mean(v), np.min(v), np.max(v))

mdf = pd.pivot_table(df, index=['WISE' ,'norm','sigma','log','convex','width','height'],
                     values=['total', '1_1', '1_2', '1_3', '2_2', '2_3', '3_3'],
                     fill_value=0,
                    aggfunc=reduce_funct)
mdf

def reduce_funct(v):
    return len(v)
    return (np.mean(v), np.min(v), np.max(v))

mdf = pd.pivot_table(df, index=['WISE' ,'norm','sigma','log','convex','width','height'],
                     values=['total', '1_1', '1_2', '1_3', '2_2', '2_3', '3_3'],
                     fill_value=0,
                    aggfunc=reduce_funct)
mdf

df







mode = '../Results/initial_results'
files = glob.glob(f'{mode}/*/*json')
dfs   = [pd.read_json(i) for i in files] 



