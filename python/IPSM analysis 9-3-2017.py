import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

df_6 = pd.read_csv('Data_6MV.csv')
df_6

friedmanchisquare(df_6['AAA_Gy'].values, df_6['AXB_Gy'].values, df_6['Ref_Gy'].values)

df_10 = pd.read_csv('Data_10MVFFF.csv')
df_10

friedmanchisquare(df_10['AAA_Gy'].values, df_10['AXB_Gy'].values, df_10['Ref_Gy'].values)

wilcoxon(df_10['AAA_Gy'].values, df_10['AXB_Gy'].values)

df_6['AAA-ref'] = df_6['AAA_Gy'] - df_6['Ref_Gy']
df_6['AXB-ref'] = df_6['AXB_Gy'] - df_6['Ref_Gy']
df_6

df_6['AXB-ref-diff'] = 100.0 * df_6['AXB-ref'] / df_6['Ref_Gy']
df_6['AXB-ref-diff'].mean()

df_6['AXB-ref-diff'].std()

wilcoxon(df_6['AAA_Gy'].values, df_6['AXB_Gy'].values)



